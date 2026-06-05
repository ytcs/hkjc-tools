"""JAX port of the interacting traffic simulation — JIT + batched, for fast fitting.

The numpy :class:`~hkjc.model.traffic.TrafficRaceModel` runs a Python loop over ~200
time steps; that loop is the calibration bottleneck (the Layer-2 ``bend_curvature``
likelihood simulates thousands of races). This module expresses the *same* dynamics
— car-following, gap-acceptance lane changes, bend ground-loss, finish-time
interpolation — as a single :func:`jax.lax.scan`, fully batched over Monte-Carlo
samples and JIT-compiled. ``bend_curvature`` and ``speed_noise`` are traced arguments,
so sweeping the calibration grid reuses one compiled kernel (no recompile per value);
only the field size / bend count are static, so the kernel is compiled once per shape.

The result matches the numpy model in distribution (not RNG-for-RNG). Heavy deps
(``jax``) are imported lazily — install with ``uv sync --group bayes``.
"""
from __future__ import annotations

import numpy as np

from ..core.track import HAPPY_VALLEY, TrackGeometry


def _finish_times(v_target, off0, key, bend_curvature, speed_noise, pace_bias, pace_gain,
                  dist, circ, dt, follow_gap, lane_band, lane_change_prob, lane_step,
                  size, max_steps, bend_lo, bend_hi):
    import jax
    import jax.numpy as jnp

    n = v_target.shape[0]
    k_noise, k_steps = jax.random.split(key)
    vt = v_target[None, :] * (1.0 + speed_noise * jax.random.normal(k_noise, (size, n)))
    step_keys = jax.random.split(k_steps, max_steps)
    # pace policy: front-runners (pace_bias>0) spend speed early to take up position,
    # closers (pace_bias<0) hold for late. The shape (1−2·progress) is zero-mean over the
    # trip's distance, so total work is conserved — only the TIMING of effort differs.
    pb = pace_gain * pace_bias[None, :]                       # (1, n)

    def on_bend(s):
        dm = jnp.mod(dist - s, circ)
        out = jnp.zeros(s.shape, dtype=bool)
        for a, b in zip(bend_lo, bend_hi):
            out = out | ((dm >= a) & (dm < b))
        return out

    def step(carry, key):
        s, off, ft, t = carry
        t = t + dt
        live = jnp.isinf(ft)
        ob = on_bend(s)
        vt_eff = vt * (1.0 + pb * (1.0 - 2.0 * jnp.clip(s / dist, 0.0, 1.0)))   # (size, n)
        gap = s[:, None, :] - s[:, :, None]                  # gap[ ,i,j] = s_j - s_i
        blocking = (gap > 0) & (gap < follow_gap) & \
                   (jnp.abs(off[:, None, :] - off[:, :, None]) < lane_band)
        vj = jnp.broadcast_to(vt_eff[:, None, :], blocking.shape)
        v_cap = jnp.min(jnp.where(blocking, vj, jnp.inf), axis=2)
        v = jnp.minimum(vt_eff, v_cap)
        blocked = v_cap < vt_eff - 1e-9
        r = jax.random.uniform(key, (size, n))
        move_out = blocked & (r < lane_change_prob)
        off_new = off + move_out * lane_step
        drift_in = (~blocked) & (off_new > 0)
        off_new = jnp.maximum(0.0, off_new - drift_in * 0.3 * lane_step)
        ground = 1.0 + off_new * bend_curvature * ob
        ds = jnp.where(live, v * dt / ground, 0.0)
        s_new = s + ds
        crossed = live & (s_new >= dist)
        frac = jnp.where(crossed, (dist - s) / jnp.maximum(ds, 1e-9), 1.0)
        ft = jnp.where(crossed, t - dt + frac * dt, ft)
        return (jnp.minimum(s_new, dist), off_new, ft, t), None

    s0 = jnp.zeros((size, n))
    off0b = jnp.broadcast_to(off0[None, :], (size, n))
    ft0 = jnp.full((size, n), jnp.inf)
    (s, off, ft, t), _ = jax.lax.scan(step, (s0, off0b, ft0, 0.0), step_keys)
    return jnp.where(jnp.isinf(ft), t + (dist - s), ft)


_CACHED: dict = {}


def _kernel():
    import jax
    fn = _CACHED.get("fn")
    if fn is None:
        fn = jax.jit(_finish_times,
                     static_argnames=("size", "max_steps", "bend_lo", "bend_hi"))
        _CACHED["fn"] = fn
    return fn


def finish_times(v_c, w_prime, draw, distance, *, track: TrackGeometry = HAPPY_VALLEY,
                 course: str = "A", size: int = 2000, bend_curvature: float = 1 / 120,
                 speed_noise: float = 0.02, pace_bias=None, pace_gain: float = 0.0,
                 seed: int = 0, dt: float = 1.0,
                 follow_gap: float = 6.0, lane_band: float = 1.5,
                 lane_change_prob: float = 0.3, lane_step: float = 1.2,
                 lane_sep: float = 1.2, max_steps: int = 220) -> np.ndarray:
    """Simulate ``size`` finishing times per horse with the JAX kernel — shape ``(size, n)``.

    ``pace_bias`` (per horse, centred ~0) with ``pace_gain`` enables the early/late pace
    policy: >0 spends speed early to take up forward position, <0 conserves for a late
    run. ``pace_gain=0`` (default) reproduces the constant-speed model exactly.
    """
    import jax
    import jax.numpy as jnp

    v_c = np.asarray(v_c, float)
    w_prime = np.asarray(w_prime, float)
    draw = np.asarray(draw, float)
    v_target = jnp.asarray(v_c * (1.0 + w_prime / float(distance)))
    n = int(v_target.shape[0])
    pb = jnp.zeros(n) if pace_bias is None else jnp.asarray(np.asarray(pace_bias, float))
    width = max(track.width(course) - 1.0, lane_sep)
    off0 = jnp.asarray(np.clip((draw - 1.0) * lane_sep, 0.0, width))
    bends = track.bend_segments or ()
    bend_lo = tuple(float(a) for a, _ in bends)
    bend_hi = tuple(float(b) for _, b in bends)

    ft = _kernel()(
        v_target, off0, jax.random.PRNGKey(seed),
        jnp.float32(bend_curvature), jnp.float32(speed_noise), pb, jnp.float32(pace_gain),
        float(distance), float(track.circumference), float(dt), float(follow_gap),
        float(lane_band), float(lane_change_prob), float(lane_step),
        size=int(size), max_steps=int(max_steps), bend_lo=bend_lo, bend_hi=bend_hi)
    return np.asarray(ft)


def sample_orders_jax(v_c, w_prime, draw, distance, **kw) -> np.ndarray:
    """Finishing orders (``argsort`` of JAX finish times) — drop-in for calibration."""
    return np.argsort(finish_times(v_c, w_prime, draw, distance, **kw), axis=1).astype(np.int64)


def win_probs(v_c, w_prime, draw, distance, **kw) -> np.ndarray:
    """P(each horse wins) — shape ``(n,)``, sums to 1 (Laplace-smoothed)."""
    ft = finish_times(v_c, w_prime, draw, distance, **kw)
    n = ft.shape[1]
    winners = np.argmin(ft, axis=1)
    counts = np.bincount(winners, minlength=n).astype(float)
    return (counts + 1.0) / (ft.shape[0] + n)


def topk_probs(v_c, w_prime, draw, distance, k: int = 3, **kw) -> np.ndarray:
    """P(each horse finishes in the top ``k``) — shape ``(n,)`` (Laplace-smoothed).

    The place pool (top-3 for 7+ runners) is a more *robust* target than win: it
    averages over the split-second 1st/2nd noise, so it leans on whole-field ordering
    rather than the knife-edge for first.
    """
    ft = finish_times(v_c, w_prime, draw, distance, **kw)
    size, n = ft.shape
    ranks = ft.argsort(1).argsort(1)               # ranks[s, h]: 0 = fastest
    hits = (ranks < k).sum(axis=0).astype(float)
    return (hits + k / n) / (size + 1.0)


def _auto_steps(distance: float, dt: float) -> int:
    """Enough steps to finish even with heavy ground-loss (~12 m/s floor) + margin."""
    return int(min(260, distance / (12.0 * dt) + 24))


_GRID_CACHED: dict = {}


def _grid_kernel():
    import jax
    fn = _GRID_CACHED.get("fn")
    if fn is None:
        def grid(v_target, off0, key, bend_grid, speed_noise, dist, circ, dt, follow_gap,
                 lane_band, lane_change_prob, lane_step, size, max_steps, bend_lo, bend_hi):
            import jax.numpy as jnp
            zpb = jnp.zeros(v_target.shape[0])          # calibration is pace-neutral
            core = lambda bc: _finish_times(  # noqa: E731
                v_target, off0, key, bc, speed_noise, zpb, jnp.float32(0.0), dist, circ, dt,
                follow_gap, lane_band, lane_change_prob, lane_step, size, max_steps, bend_lo, bend_hi)
            return jax.vmap(core)(bend_grid)            # (n_grid, size, n)
        fn = jax.jit(grid, static_argnames=("size", "max_steps", "bend_lo", "bend_hi"))
        _GRID_CACHED["fn"] = fn
    return fn


def win_prob_grid(v_c, w_prime, draw, distance, winner_idx, bend_grid, *,
                  track: TrackGeometry = HAPPY_VALLEY, course: str = "A", size: int = 1500,
                  speed_noise: float = 0.02, seed: int = 0, dt: float = 1.0,
                  follow_gap: float = 6.0, lane_band: float = 1.5,
                  lane_change_prob: float = 0.3, lane_step: float = 1.2,
                  lane_sep: float = 1.2) -> np.ndarray:
    """P(realised winner wins) for every ``bend_curvature`` in ``bend_grid`` — one call.

    vmaps the simulation over the bend grid (and runs only as many steps as the trip
    needs), so the whole Layer-2 likelihood for a race is a single compiled kernel.
    Returns a Laplace-smoothed probability per grid value, shape ``(len(bend_grid),)``.
    """
    import jax
    import jax.numpy as jnp

    v_c = np.asarray(v_c, float)
    w_prime = np.asarray(w_prime, float)
    draw = np.asarray(draw, float)
    n = v_c.shape[0]
    v_target = jnp.asarray(v_c * (1.0 + w_prime / float(distance)))
    width = max(track.width(course) - 1.0, lane_sep)
    off0 = jnp.asarray(np.clip((draw - 1.0) * lane_sep, 0.0, width))
    bends = track.bend_segments or ()
    bend_lo = tuple(float(a) for a, _ in bends)
    bend_hi = tuple(float(b) for _, b in bends)

    ft = _grid_kernel()(
        v_target, off0, jax.random.PRNGKey(seed), jnp.asarray(bend_grid, jnp.float32),
        jnp.float32(speed_noise), float(distance), float(track.circumference), float(dt),
        float(follow_gap), float(lane_band), float(lane_change_prob), float(lane_step),
        size=int(size), max_steps=_auto_steps(distance, dt), bend_lo=bend_lo, bend_hi=bend_hi)
    wins = (np.asarray(jnp.argmin(ft, axis=-1)) == winner_idx).sum(axis=1)   # (n_grid,)
    return (wins + 1.0) / (size + n)
