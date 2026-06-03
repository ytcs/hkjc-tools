"""
Harville Race Model Optimizer

Estimates horse racing outcome probabilities using the Harville model via dynamic
programming. Fits latent strength parameters from observed betting market odds across
multiple pool types (Win, Qin, Quinella, Banker).

The optimizer uses O(N * 2^N) complexity DP with Numba JIT compilation for speed.
Suitable for races with up to ~20 horses.

Example:
    >>> optimizer = HarvilleOptimizer(n_horses=14)
    >>> results = optimizer.fit(W_obs=win_odds, Qin_obs=qin_odds, 
    ...                          Q_obs=quinella_odds, b_obs=banker_odds)
    >>> print(results['theta'])  # Fitted strength parameters
"""

import numpy as np
import itertools
import collections
from functools import lru_cache

from scipy.optimize import minimize
from numba import njit
from typing import Tuple, List, Optional

LAMBDA_DEFAULTS = {
    'WIN': 1.0,
    'QIN': 2.0,
    'QPL': 1.5,
    'PLA': 0.7
}


@njit(cache=True)
def _popcount(mask: int) -> int:
    count = 0
    while mask:
        count += 1
        mask &= mask - 1
    return count


@njit(cache=True)
def _build_mask_tables(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the subset-lattice structure that depends only on ``n``.

    Returns:
        mask_popcount: number of set bits in each mask, shape (2**n,).
        low_bit_index: index of the lowest set bit in each mask (-1 for mask 0),
            used to fill subset strengths incrementally without an (2**n, n) matmul.
        order: all masks sorted ascending by popcount, shape (2**n,).
        pop_offsets: offsets into ``order`` for each popcount, shape (n + 2,);
            masks of popcount k are ``order[pop_offsets[k]:pop_offsets[k + 1]]``.
    """
    max_mask = 1 << n
    mask_popcount = np.zeros(max_mask, dtype=np.int64)
    low_bit_index = np.full(max_mask, -1, dtype=np.int64)

    for mask in range(max_mask):
        mask_popcount[mask] = _popcount(mask)
        for i in range(n):
            if mask & (1 << i):
                low_bit_index[mask] = i
                break

    # Counting sort of masks by popcount.
    pop_offsets = np.zeros(n + 2, dtype=np.int64)
    for mask in range(max_mask):
        pop_offsets[mask_popcount[mask] + 1] += 1
    for k in range(n + 1):
        pop_offsets[k + 1] += pop_offsets[k]

    order = np.zeros(max_mask, dtype=np.int64)
    cursor = pop_offsets.copy()
    for mask in range(max_mask):
        k = mask_popcount[mask]
        order[cursor[k]] = mask
        cursor[k] += 1

    return mask_popcount, low_bit_index, order, pop_offsets


@lru_cache(maxsize=24)
def _mask_tables(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache the N-only mask tables so they are built once per field size."""
    return _build_mask_tables(n)


@njit(cache=True)
def _compute_probabilities_kernel(
    theta: np.ndarray,
    low_bit_index: np.ndarray,
    order: np.ndarray,
    pop_offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Harville DP given precomputed mask tables. Hot path (called per loss eval).

    All work here depends on ``theta``; everything that depends only on ``n`` is
    passed in via the precomputed tables, so nothing is rebuilt across the
    hundreds of evaluations a single fit performs.
    """
    n = len(theta)
    max_mask = 1 << n

    # Subset strengths via the lowest-set-bit recurrence: a mask's strength is
    # its lowest member's strength plus the strength of the remaining subset
    # (already computed, since that subset is a smaller integer).
    mask_strength = np.zeros(max_mask)
    for mask in range(1, max_mask):
        lb = low_bit_index[mask]
        mask_strength[mask] = mask_strength[mask & (mask - 1)] + theta[lb]

    dp = np.zeros((n + 1, max_mask))
    dp[0, 0] = 1.0

    for k in range(n):
        for idx in range(pop_offsets[k], pop_offsets[k + 1]):
            mask = order[idx]
            prob_current = dp[k, mask]
            if prob_current == 0:
                continue

            remaining = 1.0 - mask_strength[mask]
            if remaining < 1e-12:
                continue

            for i in range(n):
                if not (mask & (1 << i)):
                    dp[k + 1, mask | (1 << i)] += prob_current * theta[i] / remaining

    # T[i, k] = P(horse i is among the top k finishers).
    T = np.zeros((n, n + 1))
    for k in range(1, n + 1):
        for idx in range(pop_offsets[k], pop_offsets[k + 1]):
            mask = order[idx]
            prob = dp[k, mask]
            if prob == 0:
                continue
            for i in range(n):
                if mask & (1 << i):
                    T[i, k] += prob

    # P[i, j] = P(horse i finishes exactly in position j) = T[i, j+1] - T[i, j].
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = T[i, j + 1] - T[i, j]

    # Pair-in-top-k matrices for Qin (top 2) and Quinella place (top 3).
    Qin = np.zeros((n, n))
    Q = np.zeros((n, n))
    for k, M in ((2, Qin), (3, Q)):
        for idx in range(pop_offsets[k], pop_offsets[k + 1]):
            mask = order[idx]
            prob = dp[k, mask]
            if prob == 0:
                continue
            horses = np.empty(k, dtype=np.int64)
            count = 0
            for i in range(n):
                if mask & (1 << i):
                    horses[count] = i
                    count += 1
            for a in range(k):
                for b_ in range(k):
                    if horses[a] != horses[b_]:
                        M[horses[a], horses[b_]] += prob

    W = P[:, 0].copy()
    b = T[:, 3].copy()

    return W, Qin, Q, b, P


def _compute_probabilities(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Harville outcome probabilities for a strength vector ``theta``.

    Thin wrapper that supplies cached N-only mask tables to the JIT kernel.
    """
    _, low_bit_index, order, pop_offsets = _mask_tables(len(theta))
    return _compute_probabilities_kernel(theta, low_bit_index, order, pop_offsets)


@njit(cache=True)
def _kl_divergence(p_obs: np.ndarray, p_model: np.ndarray) -> float:
    eps = 1e-10

    p_obs_flat = np.maximum(p_obs.ravel(), eps)
    p_model_flat = np.maximum(p_model.ravel(), eps)

    sum_obs = p_obs_flat.sum()
    sum_model = p_model_flat.sum()

    if sum_obs > eps:
        p_obs_flat = p_obs_flat / sum_obs
    if sum_model > eps:
        p_model_flat = p_model_flat / sum_model

    return np.sum(p_obs_flat * np.log(p_obs_flat / p_model_flat))


class HarvilleModel:
    """
    Fits Harville race model to betting market odds using dynamic programming.

    The Harville model assigns each horse a latent strength parameter theta_i, where
    the probability of finishing next among remaining horses is proportional to 
    relative strength. This optimizer estimates theta from observed betting odds
    across multiple pool types.

    Default lambda weights reflect that early Win odds are biased by informed 
    traders waiting until closing, while exotic pools provide more stable 
    signals for ensemble estimation.

    Attributes:
        n (int): Number of horses
        lambda_win (float): Weight for Win pool loss
        lambda_qin (float): Weight for Qin pool loss
        lambda_quinella (float): Weight for Quinella pool loss
        lambda_banker (float): Weight for Banker pool loss
        takeout_rate (float): House take out rate (e.g., 0.175 = 17.5%)
    """

    def __init__(self, n_horses: int, lambda_win: float = LAMBDA_DEFAULTS['WIN'], lambda_qin: float = LAMBDA_DEFAULTS['QIN'],
                 lambda_quinella: float = LAMBDA_DEFAULTS['QPL'], lambda_banker: float = LAMBDA_DEFAULTS['PLA'],
                 takeout_rate: float = 0.175) -> None:
        """
        Initialize model.

        Args:
            n_horses: Number of horses in race (recommend <= 20 for speed)
            lambda_win: Weight for Win odds (prob horse finishes 1st)
            lambda_qin: Weight for Qin odds (prob pair finishes 1st-2nd)
            lambda_quinella: Weight for Quinella odds (prob pair in top 3)
            lambda_banker: Weight for Banker odds (prob horse in top 3)
            takeout_rate: House take out rate as decimal (default 0.175 = 17.5%).
                         The observed odds include the house's take, which makes
                         them higher than true odds. This parameter adjusts for that.

        Raises:
            ValueError: If n_horses > 20 (exponential complexity warning)
        """
        if n_horses > 20:
            raise ValueError("N > 20 may be too slow (2^N complexity)")

        self.n = n_horses
        self.lambda_win = lambda_win
        self.lambda_qin = lambda_qin
        self.lambda_quinella = lambda_quinella
        self.lambda_banker = lambda_banker
        self.takeout_rate = takeout_rate
        self._eval_count = 0
        self.result = None

    def _adjust_for_takeout(self, probs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Adjust observed probabilities to remove house takeout rate.

        Observed odds from the betting market include the house's take, causing
        the sum of implied probabilities to exceed 1.0. This method adjusts them
        to represent true probabilities.

        Args:
            probs: Observed probabilities (can be 1D or 2D array)

        Returns:
            Adjusted probabilities with takeout removed, or None if input is None
        """
        if probs is None:
            return None

        # Multiply by (1 - takeout_rate) to remove the house edge
        adjusted = probs * (1.0 - self.takeout_rate)

        return adjusted

    def _probs_to_market_odds(self, probs: np.ndarray) -> np.ndarray:
        """
        Convert fitted probabilities to market odds including takeout rate.

        This converts true probabilities (which sum to 1.0) back to decimal odds
        as they would appear in the betting market, which includes the house's
        takeout rate. The resulting odds can be directly compared to observed odds.

        Args:
            probs: Fitted probabilities (1D or 2D array)

        Returns:
            Market odds (decimal format) with takeout reintroduced
        """

        # Convert true probabilities to market odds with takeout
        # Market odds are worse (higher) than fair odds due to house edge
        return (1.0 - self.takeout_rate) / probs

    def _loss(self, theta: np.ndarray, W_obs: Optional[np.ndarray],
              Qin_obs: Optional[np.ndarray], Q_obs: Optional[np.ndarray],
              b_obs: Optional[np.ndarray]) -> float:
        """
        Compute weighted KL divergence loss between observed and model odds.

        Args:
            theta: Strength parameters (will be normalized to simplex)
            W_obs: Observed Win probabilities (n,) or None
            Qin_obs: Observed Qin probabilities (n, n) or None
            Q_obs: Observed Quinella probabilities (n, n) or None
            b_obs: Observed Banker probabilities (n,) or None

        Returns:
            Scalar loss value (sum of weighted KL divergences)
        """
        self._eval_count += 1

        theta = np.abs(theta) + 1e-10
        theta = theta / theta.sum()

        W_model, Qin_model, Q_model, b_model, P_model = _compute_probabilities(
            theta)

        loss = 0.0

        if W_obs is not None:
            loss += self.lambda_win * _kl_divergence(W_obs, W_model)

        if Qin_obs is not None:
            loss += self.lambda_qin * _kl_divergence(Qin_obs, Qin_model)

        if Q_obs is not None:
            loss += self.lambda_quinella * _kl_divergence(Q_obs, Q_model)

        if b_obs is not None:
            loss += self.lambda_banker * _kl_divergence(b_obs, b_model)

        return loss

    def fit(self, W_obs: Optional[np.ndarray] = None,
            Qin_obs: Optional[np.ndarray] = None,
            Q_obs: Optional[np.ndarray] = None,
            b_obs: Optional[np.ndarray] = None,
            theta_init: Optional[np.ndarray] = None,
            method: str = 'L-BFGS-B') -> dict:
        """
        Fit Harville model to observed betting odds.

        At least one odds type must be provided. All odds should be probabilities
        (not decimal/fractional odds). Matrices should be symmetric where applicable.

        Args:
            W_obs: Win probabilities, shape (n,). W_obs[i] = prob horse i wins
            Qin_obs: Qin probabilities, shape (n, n). Qin_obs[i,j] = prob horses
                     i,j finish 1st-2nd in any order
            Q_obs: Quinella probabilities, shape (n, n). Q_obs[i,j] = prob horses
                   i,j both finish in top 3
            b_obs: Banker probabilities, shape (n,). b_obs[i] = prob horse i 
                   finishes in top 3
            theta_init: Initial strength guess (default: W_obs if available, else uniform)
            method: Scipy optimizer ('L-BFGS-B' or 'SLSQP')

        Returns:
            Dictionary containing:
                - success: Whether optimization converged
                - message: Optimizer status message
                - n_eval: Number of loss function evaluations
                - loss: Final loss value
                - prob_fit: Dictionary of fitted probabilities (sum to 1.0)
                    - theta: Fitted strength parameters (n,)
                    - W: Win probabilities (n,)
                    - Qin: Qin probabilities (n, n)
                    - Q: Quinella probabilities (n, n)
                    - b: Banker probabilities (n,)
                    - P: Full place probability matrix (n, n), P[i,j] = 
                         prob horse i finishes in position j
                - odds_fit: Dictionary of fitted market odds (directly comparable to observed)
                    - WIN: Win odds (n,)
                    - QIN: Qin odds (n, n)
                    - QPL: Quinella Place odds (n, n)
                    - PLA: Place odds (n,)

        Raises:
            ValueError: If no odds provided or shapes don't match n_horses

        Example:
            >>> opt = HarvilleModel(n_horses=10, takeout_rate=0.175)
            >>> results = opt.fit(W_obs=win_probs, Q_obs=quinella_probs)
            >>> print(f"Fitted strengths: {results['prob_fit']['theta']}")
            >>> print(f"Converged: {results['success']}")
            >>> # Compare fitted odds to observed odds
            >>> diff = results['odds_fit']['WIN'] - observed_win_odds
        """
        if W_obs is None and Qin_obs is None and Q_obs is None and b_obs is None:
            raise ValueError("At least one type of odds must be provided")

        if W_obs is not None and W_obs.shape != (self.n,):
            raise ValueError(f"W_obs must be ({self.n},)")
        if Qin_obs is not None and Qin_obs.shape != (self.n, self.n):
            raise ValueError(f"Qin_obs must be ({self.n}, {self.n})")
        if Q_obs is not None and Q_obs.shape != (self.n, self.n):
            raise ValueError(f"Q_obs must be ({self.n}, {self.n})")
        if b_obs is not None and b_obs.shape != (self.n,):
            raise ValueError(f"b_obs must be ({self.n},)")

        # Adjust observed probabilities for house takeout rate
        W_obs = self._adjust_for_takeout(W_obs)
        Qin_obs = self._adjust_for_takeout(Qin_obs)
        Q_obs = self._adjust_for_takeout(Q_obs)
        b_obs = self._adjust_for_takeout(b_obs)

        if theta_init is None:
            if W_obs is not None:
                theta_init = W_obs / W_obs.sum()
            else:
                theta_init = np.ones(self.n) / self.n
        else:
            theta_init = theta_init / theta_init.sum()

        self._eval_count = 0

        if method == 'L-BFGS-B':
            result = minimize(
                fun=lambda x: self._loss(x, W_obs, Qin_obs, Q_obs, b_obs),
                x0=theta_init,
                method='L-BFGS-B',
                bounds=[(1e-6, 1.0) for _ in range(self.n)],
                options={'maxiter': 500, 'ftol': 1e-9, 'maxls': 50}
            )
        else:
            result = minimize(
                fun=lambda x: self._loss(x, W_obs, Qin_obs, Q_obs, b_obs),
                x0=theta_init,
                method='SLSQP',
                bounds=[(1e-6, 1.0) for _ in range(self.n)],
                constraints={'type': 'eq', 'fun': lambda x: x.sum() - 1},
                options={'maxiter': 500, 'ftol': 1e-9}
            )

        theta_opt = np.abs(result.x) + 1e-10
        theta_opt = theta_opt / theta_opt.sum()

        W_fitted, Qin_fitted, Q_fitted, b_fitted, P_fitted = _compute_probabilities(
            theta_opt)

        # Convert fitted probabilities to market odds (with takeout reintroduced)
        WIN_odds_fitted = self._probs_to_market_odds(W_fitted)
        PLA_odds_fitted = self._probs_to_market_odds(b_fitted)
        QIN_odds_fitted = self._probs_to_market_odds(Qin_fitted)
        QPL_odds_fitted = self._probs_to_market_odds(Q_fitted)

        self.result = {
            'success': result.success,
            'message': result.message,
            'n_eval': self._eval_count,
            'loss': result.fun,
            'prob_fit': {
                'theta': theta_opt,
                'W': W_fitted,
                'Qin': Qin_fitted,
                'Q': Q_fitted,
                'b': b_fitted,
                'P': P_fitted
            },
            'odds_fit': {
                'WIN': WIN_odds_fitted,
                'QIN': QIN_odds_fitted,
                'QPL': QPL_odds_fitted,
                'PLA': PLA_odds_fitted
            }
        }

        return self.result


def fit_harville_to_odds(odds: dict[str, np.ndarray], lambdas: dict[str, float] = None, takeout_rate: float = 0.175) -> dict:
    """
    Fit Harville model to observed betting odds.

    At least one odds type must be provided. All odds should be decimal odds
    (not probabilities). Matrices should be symmetric where applicable.

    Args:
        odds: Dictionary of odds arrays with types as keys.:
                'WIN' (n,), 'QIN' (n,n), 'QPL' (n,n), 'PLA' (n,)
        lambdas: Optional dictionary of lambda weights for each odds type.
                    Keys can be 'WIN', 'QIN', 'QPL', 'PLA'. Defaults to
                    {'WIN': 1.0, 'QIN': 2.0, 'QPL': 1.5, 'PLA': 0.7}
        takeout_rate: House take out rate as decimal (default 0.175 = 17.5%).
                     The house keeps this percentage of the betting pool, causing
                     observed odds to be higher than fair odds.

    Returns:
        Dictionary containing:
            - success: Whether optimization converged
            - message: Optimizer status message
            - n_eval: Number of loss function evaluations
            - loss: Final loss value
            - prob_fit: Dictionary of fitted probabilities (sum to 1.0)
                - theta: Fitted strength parameters (n,)
                - W: Win probabilities (n,)
                - Qin: Qin probabilities (n, n)
                - Q: Quinella probabilities (n, n)
                - b: Banker probabilities (n,)
                - P: Full place probability matrix (n, n), P[i,j] = 
                     prob horse i finishes in position j
            - odds_fit: Dictionary of fitted market odds (directly comparable to observed)
                - WIN: Win odds (n,)
                - QIN: Qin odds (n, n)
                - QPL: Quinella Place odds (n, n)
                - PLA: Place odds (n,)

    Example:
        >>> odds = {'WIN': np.array([3.5, 4.2, 5.0, 8.5, 12.0])}
        >>> result = fit_harville_to_odds(odds, takeout_rate=0.175)
        >>> print(result['prob_fit']['theta'])  # True winning probabilities
        >>> print(result['odds_fit']['WIN'])  # Fitted market odds (compare to input)
    """
    n_horses = None
    W_obs = None
    Qin_obs = None
    Q_obs = None
    b_obs = None

    if 'WIN' in odds:
        W_odds = odds['WIN']
        if n_horses is None:
            n_horses = len(W_odds)
        elif n_horses != len(W_odds):
            raise ValueError("Inconsistent number of horses in WIN odds")
        W_obs = np.nan_to_num(1.0 / W_odds, 0)

    if 'QIN' in odds:
        Qin_odds = odds['QIN']
        if n_horses is None:
            n_horses = Qin_odds.shape[0]
        elif n_horses != Qin_odds.shape[0]:
            raise ValueError("Inconsistent number of horses in QIN odds")
        Qin_obs = np.nan_to_num(1.0 / Qin_odds, 0)

    if 'QPL' in odds:
        Q_odds = odds['QPL']
        if n_horses is None:
            n_horses = Q_odds.shape[0]
        elif n_horses != Q_odds.shape[0]:
            raise ValueError("Inconsistent number of horses in QPL odds")
        Q_obs = np.nan_to_num(1.0 / Q_odds, 0)

    if 'PLA' in odds:
        b_odds = odds['PLA']
        if n_horses is None:
            n_horses = len(b_odds)
        elif n_horses != len(b_odds):
            raise ValueError("Inconsistent number of horses in PLA odds")
        b_obs = np.nan_to_num(1.0 / b_odds, 0)

    merged_lambdas = {**LAMBDA_DEFAULTS, **(lambdas or {})}
    ho = HarvilleModel(
        n_horses,
        lambda_win=merged_lambdas['WIN'],
        lambda_qin=merged_lambdas['QIN'],
        lambda_quinella=merged_lambdas['QPL'],
        lambda_banker=merged_lambdas['PLA'],
        takeout_rate=takeout_rate
    )
    result = ho.fit(W_obs=W_obs, Qin_obs=Qin_obs, Q_obs=Q_obs, b_obs=b_obs)
    return result


def find_top_outcomes(p: np.ndarray, k: int, m: int, ordered: bool = True) -> Tuple[List, List]:
    """
    Calculates the k most likely outcomes for the first m places given a
    probability matrix using optimized NumPy operations.

    Args:
        p (np.ndarray): An (N, N) NumPy array where p[i, j] is the probability
                        of runner i (0-indexed) finishing in j-th place (0-indexed).
        k (int): The number of top outcomes to return.
        m (int): The number of top places to consider (e.g., 4 for first 4 places).
        ordered (bool): If True (default), calculates for ordered permutations
                        (e.g., [1, 2] is different from [2, 1]).
                        If False, calculates for unordered combinations
                        (e.g., [1, 2] and [2, 1] are summed).

    Returns:
        Tuple[List, List]: A tuple containing two lists:
            - top_outcomes (List[List[int]]): 
              A list of the top k outcomes (e.g., [[5, 1, 9, 2], ...]).
              Runner indices are 0-indexed.
            - top_probabilities (List[float]): 
              A list of the corresponding probabilities for each outcome.

    Raises:
        ValueError: If p is not a 2D square NumPy array, if k or m are non-positive,
                    or if N < m.
    """
    # --- 1. Input Validation ---
    if not isinstance(p, np.ndarray) or p.ndim != 2:
        raise ValueError("Input 'p' must be a 2D NumPy array.")

    if p.shape[0] != p.shape[1]:
        raise ValueError(
            f"Input matrix 'p' must be square (N, N), but has shape {p.shape}.")

    N = p.shape[0]

    if N < m:
        raise ValueError(
            f"Input matrix 'p' must have N >= m, but N={N} and m={m}.")

    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"Input 'k' must be a positive integer, but got {k}.")

    if not isinstance(m, int) or m <= 0:
        raise ValueError(f"Input 'm' must be a positive integer, but got {m}.")

    # --- 2. Generate All Permutations ---
    runner_indices = range(N)
    # We are finding permutations for the first 'm' places
    permutations_iter = itertools.permutations(runner_indices, m)

    # Convert iterator to a NumPy array for vectorized operations
    permutations = np.array(list(permutations_iter), dtype=np.uint16)
    num_outcomes = permutations.shape[0]

    # --- 3. Vectorized Probability Calculation ---
    # Calculate: p[r1, 0] * p[r2, 1] * ... * p[rm, m-1] for each permutation

    # Initialize probabilities
    all_probs = np.ones(num_outcomes, dtype=np.float64)

    # Multiply probabilities for each place
    for j in range(m):
        all_probs *= p[permutations[:, j], j]

    # --- 4. Branch Logic based on 'ordered' ---
    if ordered:
        # --- 4a. Efficiently Find Top K Ordered ---
        k_clamped = min(k, num_outcomes)

        # Use np.argpartition to find the k largest elements without a full sort
        top_k_indices_unsorted = np.argpartition(
            all_probs, -k_clamped)[-k_clamped:]

        # Sort just the top k elements by probability
        top_k_probs = all_probs[top_k_indices_unsorted]
        sort_order = np.argsort(top_k_probs)[::-1]  # Descending order
        top_k_indices = top_k_indices_unsorted[sort_order]

        # --- 5a. Extract and Format Ordered Results ---
        top_outcomes_np = permutations[top_k_indices]+1 # switch to 1-index
        top_probabilities = all_probs[top_k_indices]

        return top_outcomes_np.astype(int).tolist(), top_probabilities.tolist()

    else:
        # --- 4b. Group by Unordered Combination ---
        # We group all permutations by their content, summing their probabilities.
        # A sorted tuple (e.g., (1, 2, 5, 9)) is used as the key.
        grouped_probs = collections.defaultdict(float)
        for i in range(num_outcomes):
            # Sort the runners to create a unique key for the combination
            key = tuple(np.sort(permutations[i]))
            grouped_probs[key] += all_probs[i]

        # --- 5b. Sort Unordered Groups and Format ---

        # Convert dict to list of (probability, outcome_tuple)
        all_unordered = [(prob, outcome)
                         for outcome, prob in grouped_probs.items()]

        # Sort by probability (descending)
        all_unordered.sort(key=lambda x: x[0], reverse=True)

        # Get the top k
        k_clamped = min(k, len(all_unordered))
        top_k_unordered = all_unordered[:k_clamped]

        top_outcomes = [[int(runner)+1 for runner in outcome]
                        for prob, outcome in top_k_unordered]
        top_probabilities = [prob for prob, outcome in top_k_unordered]

        return top_outcomes, top_probabilities
