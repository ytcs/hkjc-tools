"""PoolPricer — the joint order distribution → every pool price.

Every HKJC pool price is a FUNCTIONAL of the joint finishing-order distribution (Layer 2 output):

    win        P(i finishes 1st)                       — a marginal of the order dist
    place      P(i in top-k)                           — top-k marginal (k by field size, HKJC rule)
    quinella   P({i,j} are the top 2, unordered)       — 2nd-order joint
    quin-place P({i,j} both in top-k, unordered)       — pairwise top-k joint (QPL)
    forecast   P(i 1st AND j 2nd, ordered)             — ordered 2nd-order joint
    tierce     P(i 1st, j 2nd, k 3rd, ordered)         — ordered 3rd-order joint (TCE)
    trio       P({i,j,k} are the top 3, unordered)     — 3rd-order set joint (TRI)

All are Monte-Carlo functionals of the :class:`~hkjc.model.generative.OrderDistribution`
sample, so a single sampler prices the whole POOL BOOK consistently (the prices cannot disagree —
they are marginals of ONE law). This is what the canonical framework means by "every pool price"
and what the GATE scores across the efficiency-weighted pool book.

NOTE: place top-k and the exotic min-leg rules are HKJC facts — the BUILD-WAVE must source k and the
dead-heat/refund handling from ``hkjc.core.hkjc_facts`` rather than hard-coding (the win/place skeleton
below uses the simple rules to keep the smoke path dependency-light).
"""
from __future__ import annotations

import numpy as np

from .generative import OrderDistribution


class PoolPricer:
    """Price the full pool book from one joint order-distribution sample.

    Win and place are implemented (the Stage-1/2 needs); the exotics are wired as Monte-Carlo
    functionals with the trifecta deferred to the BUILD-WAVE only for the dead-heat/box accounting
    (the core counting is shown for quinella/forecast).
    """

    def __init__(self, order_dist: OrderDistribution) -> None:
        self.od = order_dist

    def win(self) -> np.ndarray:
        """P(i wins) — the win pool marginal."""
        return self.od.win_probs()

    def place(self, k: int | None = None) -> np.ndarray:
        """P(i places) — the top-k marginal.

        ``k`` defaults to the HKJC field-size rule (:func:`hkjc.core.hkjc_facts.places_paid`):
        3 places for 7+ runners, 2 for 4-6, 0 for tiny fields — SOURCED, not hard-coded. Pass an
        explicit ``k`` to override (e.g. when the caller already resolved the rule).
        """
        if k is None:
            from hkjc.core.hkjc_facts import places_paid
            k = places_paid(self.od.n)
        return self.od.topk_probs(k) if k > 0 else np.zeros(self.od.n)

    def forecast(self) -> np.ndarray:
        """P(i 1st AND j 2nd) — the ORDERED exacta/forecast matrix, shape ``(n, n)``.

        Entry [i, j] = P(runner i first, runner j second); diagonal is 0. A pure Monte-Carlo
        functional of the order sample.
        """
        first, second = self.od.orders[:, 0], self.od.orders[:, 1]
        n, m = self.od.n, len(self.od.orders)
        mat = np.zeros((n, n))
        np.add.at(mat, (first, second), 1.0)
        return mat / m

    def quinella(self) -> np.ndarray:
        """P({i, j} are the top two, UNORDERED) — symmetric matrix, shape ``(n, n)``.

        The forecast symmetrised: quinella[i,j] = forecast[i,j] + forecast[j,i].
        """
        f = self.forecast()
        q = f + f.T
        np.fill_diagonal(q, 0.0)
        return q

    def trifecta(self):
        """P(i 1st, j 2nd, k 3rd) — the ordered 3rd-order joint (full tensor; rarely materialised).

        For the pool-book gate we never need the full ``(n,n,n)`` tensor — only the probability of the
        ONE realized combination (:meth:`prob_realized`). Materialising the dense tensor is avoided;
        the dead-heat / refund accounting is a BUILD-WAVE concern for the trading seam, not the gate.
        """
        raise NotImplementedError("Dense trifecta tensor not needed for the gate; use prob_realized().")

    # ---------------------------------------------------------- realized-combination probabilities
    # For the POOL-BOOK GATE we score only the ONE realized winning combination per race+pool against
    # its de-vigged market prob. Each is a Laplace-smoothed Monte-Carlo functional of the order sample:
    # count the sampled orders matching the realized combo, smoothed so a zero-sample combo never gives
    # an infinite log-loss (the production sampler's discipline). Indices below are 0-based runner
    # positions in the field (the caller maps the dividend's saddle numbers to these positions).
    #
    # DEAD-HEAT / REFUND (HKJC rule, SOURCED not hard-coded). A dead-heat splits the dividend across the
    # tying runners and a scratching refunds the leg — these affect the PAYOUT, not the order-probability
    # the gate scores (the realized combination's probability under the order law). The dead-heat/refund
    # accounting therefore lives in the betting/settlement seam (hkjc.core.hkjc_facts.places_paid for the
    # place-rule k; the takeout/refund schedule for settlement), NOT in these order-probability functionals.

    def _smooth(self, hits: int) -> float:
        """Laplace-smoothed P(realized combo) = (hits + 1) / (n_samples + n)."""
        m = len(self.od.orders)
        return (hits + 1.0) / (m + self.od.n)

    def prob_win(self, i: int) -> float:
        """P(runner ``i`` finishes 1st) on the realized winner — Laplace-smoothed."""
        return self._smooth(int(np.count_nonzero(self.od.orders[:, 0] == i)))

    def prob_place(self, placers: list[int], k: int) -> float:
        """P(the realized top-``k`` SET is exactly ``placers``) — the unordered place combination.

        HKJC pays each placed runner separately, but the realized PLA *combination* is the set of the
        top-k finishers; its joint probability is the cleanest single-number benchmark against the
        de-vigged place dividends (which, multiplied across legs, price that same set).
        """
        topk = np.sort(self.od.orders[:, :k], axis=1)
        target = np.sort(np.asarray(placers))
        hits = int(np.count_nonzero((topk == target).all(axis=1))) if len(target) == k else 0
        return self._smooth(hits)

    def prob_forecast(self, i: int, j: int) -> float:
        """P(runner ``i`` 1st AND runner ``j`` 2nd) — the ORDERED top-2 (forecast/exacta)."""
        o = self.od.orders
        return self._smooth(int(np.count_nonzero((o[:, 0] == i) & (o[:, 1] == j))))

    def prob_quinella(self, i: int, j: int) -> float:
        """P({i, j} are the top two, UNORDERED) — quinella = forecast(i,j) + forecast(j,i)."""
        o = self.od.orders
        a = (o[:, 0] == i) & (o[:, 1] == j)
        b = (o[:, 0] == j) & (o[:, 1] == i)
        return self._smooth(int(np.count_nonzero(a | b)))

    def prob_quinella_place(self, i: int, j: int, k: int | None = None) -> float:
        """P({i, j} are BOTH in the top-``k``, UNORDERED) — the quinella-place (QPL) pool.

        QPL pays any pair that both finish in the paying places (the HKJC field-size place rule —
        :func:`hkjc.core.hkjc_facts.places_paid`, SOURCED), so it is the pairwise top-k set joint,
        the natural QPL analogue of :meth:`prob_quinella` (which is the pairwise top-2 set).
        """
        if k is None:
            from hkjc.core.hkjc_facts import places_paid
            k = places_paid(self.od.n)
        if k < 2:
            return self._smooth(0)
        topk = self.od.orders[:, :k]
        a = (topk == i).any(axis=1)
        b = (topk == j).any(axis=1)
        return self._smooth(int(np.count_nonzero(a & b)))

    def prob_tierce(self, i: int, j: int, k: int) -> float:
        """P(i 1st, j 2nd, k 3rd) — the ORDERED top-3 (tierce/TCE). Softer, noisier; down-weighted."""
        o = self.od.orders
        return self._smooth(int(np.count_nonzero((o[:, 0] == i) & (o[:, 1] == j) & (o[:, 2] == k))))

    def prob_trio(self, runners: list[int]) -> float:
        """P({i,j,k} are the top three, UNORDERED) — the trifecta box (TRI). Down-weighted."""
        top3 = np.sort(self.od.orders[:, :3], axis=1)
        target = np.sort(np.asarray(runners))
        hits = int(np.count_nonzero((top3 == target).all(axis=1))) if len(target) == 3 else 0
        return self._smooth(hits)
