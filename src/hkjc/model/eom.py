"""Canonical power-balance equation of motion — one JAX implementation.

The per-step physics is written **once** as :func:`eom_step` and integrated under
``jax.lax.scan`` by :func:`integrate_jax`. Having a single source for the mass-aware
dynamics is what lets body mass enter the (JAX) production sim — closing the divergence
that left mass out and showed up as the bodywt residual. ``eom_step`` keeps an ``xp``
parameter (the array module) so the algebra reads backend-neutrally, but the only
integrator is JAX; heavy deps are imported lazily inside :func:`integrate_jax`.

Physics, integrated each step by Newton's law (see ``model/dynamics.py`` for the full
derivation)::

    m·dv/dt = F_prop − F_resist
    F_prop  = P(t) / v ,   P(t) = P_c + p_anaer drawn from a finite reserve W′
    F_resist = K_LOCO·m·g                     (locomotion: carrying/moving the mass)
             + ½·ρ·cda·v²·(1−draft)           (aerodynamic drag; drafting cuts it)
             + bend ground-loss               (geometric, longer path off the rail)

Mass enters as a real term (inertia ``m·dv/dt`` and locomotion cost ``K_LOCO·m·g``), so a
horse's sustainable velocity ``v_c = P_c/(K_LOCO·m·g)`` falls mechanistically with weight.
"""
from __future__ import annotations

import numpy as np

G = 9.81            # m/s²
RHO = 1.2           # air density, kg/m³
K_LOCO = 0.10       # cost-of-transport coefficient (net propulsive force per unit weight)


def bend_mask(d_from_finish, bend_lo, bend_hi, circ, xp):
    """Boolean mask: is each runner currently on a bend? (track position mod circumference)."""
    dm = xp.mod(d_from_finish, circ)
    out = dm < -1.0  # all-False seed of the right shape/backend
    for a, b in zip(bend_lo, bend_hi):
        out = out | ((dm >= a) & (dm < b))
    return out


def eom_step(s, off, v, w_bal, finish_t, t, rand, *, vt0, pb, kick, v_hold, v_kick, kf,
             p_c, floco, mass, cda, draft, tau, dist, follow_gap, lane_band,
             lane_change_prob, lane_step, bend_curvature, bend_lo, bend_hi, circ, dt, xp):
    """One integration step of the power-balance EOM. Pure; works under numpy or jax.numpy.

    ``rand`` is the per-step uniform draw (size, n) for the gap-acceptance lane change. All
    velocity targets (``vt0`` for pace mode, ``v_hold``/``v_kick`` for kick mode) arrive
    pre-noised so the function itself is deterministic given its inputs.
    """
    t = t + dt
    live = xp.isinf(finish_t)
    on_bend = bend_mask(dist - s, bend_lo, bend_hi, circ, xp)
    # --- velocity target: open-loop pace tilt, or state-feedback kick ---
    if kick:
        v_tgt = xp.where((dist - s) <= kf * dist, v_kick, v_hold)
    else:
        v_tgt = vt0 * (1.0 + pb * (1.0 - 2.0 * xp.clip(s / dist, 0.0, 1.0)))
    # --- traffic: car-following velocity cap + drafting (drag relief) ---
    gap = s[:, None, :] - s[:, :, None]                       # gap[ ,i,j] = s_j - s_i
    same_lane = xp.abs(off[:, None, :] - off[:, :, None]) < lane_band
    blocking = (gap > 0) & (gap < follow_gap) & same_lane
    vj = xp.broadcast_to(v[:, None, :], blocking.shape)
    v_cap = xp.min(xp.where(blocking, vj, xp.inf), axis=2)
    drafting = xp.any(blocking, axis=2)
    v_goal = xp.minimum(v_tgt, v_cap)
    # --- power balance ---
    draft_relief = xp.where(drafting, draft, 0.0)
    f_drag = 0.5 * RHO * cda * v * v * (1.0 - draft_relief)
    a_des = (v_goal - v) / tau
    p_need = (floco + f_drag + mass * a_des) * xp.maximum(v, 1.0)
    p_avail = p_c + xp.where(w_bal > 0, p_c, 0.0)             # anaerobic ≤ aerobic ceiling
    p = xp.clip(p_need, 0.0, p_avail)
    f_prop = p / xp.maximum(v, 1.0)
    a = (f_prop - floco - f_drag) / mass
    v = xp.maximum(0.5, v + a * dt)
    w_bal = xp.maximum(0.0, w_bal - xp.maximum(p - p_c, 0.0) * dt)
    # --- lateral: blocked move out (gap-acceptance), else drift to rail ---
    blocked = v_cap < v_tgt - 1e-9
    move_out = blocked & (rand < lane_change_prob)
    off = off + move_out * lane_step
    drift_in = (~blocked) & (off > 0)
    off = xp.maximum(0.0, off - drift_in * 0.3 * lane_step)
    # --- advance with bend ground-loss; interpolate the finish crossing ---
    ground = 1.0 + off * bend_curvature * on_bend
    ds = xp.where(live, v * dt / ground, 0.0)
    s_new = s + ds
    crossed = live & (s_new >= dist)
    frac = xp.where(crossed, (dist - s) / xp.maximum(ds, 1e-9), 1.0)
    finish_t = xp.where(crossed, t - dt + frac * dt, finish_t)
    s = xp.minimum(s_new, dist)
    return s, off, v, w_bal, finish_t, t


_JAX_CACHE: dict = {}


def integrate_jax(size, seed, *, v_c, v_target, w_reserve, p_c, floco, mass, off0,
                  kick, kick_frac, kick_overspeed, pace_bias, pace_gain, speed_noise,
                  cda, draft, tau, dist, follow_gap, lane_band, lane_change_prob,
                  lane_step, bend_curvature, bend_lo, bend_hi, circ, dt, max_steps):
    """JAX integrator: the identical :func:`eom_step` under ``lax.scan``. Returns (size, n)."""
    import jax
    import jax.numpy as jnp

    n = int(v_c.shape[0])

    def run(v_c, v_target, w_reserve, p_c, floco, mass, off0, pace_bias, kick_frac, key,
            cda, draft, tau, bend_curvature, speed_noise, pace_gain, kick_overspeed):
        k_noise, k_kick, k_steps = jax.random.split(key, 3)
        vt0 = v_target[None, :] * (1.0 + speed_noise * jax.random.normal(k_noise, (size, n)))
        pb = (pace_gain * pace_bias)[None, :]
        if kick:
            kn = 1.0 + speed_noise * jax.random.normal(k_kick, (size, n))
            v_hold = v_c[None, :] * kn
            v_kick = v_hold * (1.0 + kick_overspeed)
            kf = kick_frac[None, :]
        else:
            v_hold = v_kick = kf = None
        step_keys = jax.random.split(k_steps, max_steps)

        def body(carry, key):
            s, off, v, w_bal, finish_t, t = carry
            rand = jax.random.uniform(key, (size, n))
            carry = eom_step(
                s, off, v, w_bal, finish_t, t, rand, vt0=vt0, pb=pb, kick=kick,
                v_hold=v_hold, v_kick=v_kick, kf=kf, p_c=p_c[None, :], floco=floco[None, :],
                mass=mass[None, :], cda=cda, draft=draft, tau=tau, dist=dist,
                follow_gap=follow_gap, lane_band=lane_band, lane_change_prob=lane_change_prob,
                lane_step=lane_step, bend_curvature=bend_curvature, bend_lo=bend_lo,
                bend_hi=bend_hi, circ=circ, dt=dt, xp=jnp)
            return carry, None

        s0 = jnp.zeros((size, n))
        off0b = jnp.broadcast_to(off0[None, :], (size, n))
        v0 = jnp.broadcast_to(v_c[None, :], (size, n)).astype(float)
        w0 = jnp.broadcast_to(w_reserve[None, :], (size, n)).astype(float)
        ft0 = jnp.full((size, n), jnp.inf)
        (s, off, v, w_bal, ft, t), _ = jax.lax.scan(
            body, (s0, off0b, v0, w0, ft0, 0.0), step_keys)
        return jnp.where(jnp.isinf(ft), t + (dist - s), ft)

    cache_key = (n, size, max_steps, kick, bend_lo, bend_hi, float(dist), float(circ),
                 float(dt), float(follow_gap), float(lane_band), float(lane_change_prob),
                 float(lane_step))
    fn = _JAX_CACHE.get(cache_key)
    if fn is None:
        fn = jax.jit(run)
        _JAX_CACHE[cache_key] = fn
    ft = fn(jnp.asarray(v_c), jnp.asarray(v_target), jnp.asarray(w_reserve), jnp.asarray(p_c),
            jnp.asarray(floco), jnp.asarray(mass), jnp.asarray(off0), jnp.asarray(pace_bias),
            jnp.asarray(kick_frac if kick_frac is not None else np.zeros(n)),
            jax.random.PRNGKey(seed), jnp.float32(cda), jnp.float32(draft), jnp.float32(tau),
            jnp.float32(bend_curvature), jnp.float32(speed_noise), jnp.float32(pace_gain),
            jnp.float32(kick_overspeed))
    return np.asarray(ft)
