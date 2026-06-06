"""The coherent barrier-trial fusion: precision-weighted v_c posterior, leak-free, count-fading."""
import numpy as np
import polars as pl

from hkjc.model.calibration import fit_energetics_population
from hkjc.model.fit import fit_predict_traffic, fit_trial_vc_map


def _frame(n_races=40, seed=0):
    """A small synthetic race panel carrying the columns the production fit reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        nf = 8
        for run in range(nf):
            rows.append({
                "race_id": f"d{r}", "race_no": r % 10, "runner": run + 1,
                "venue": "ST", "distance": 1200, "horse_id": f"H{r*nf+run:03d}",
                "v_c": 16.8 + rng.normal(0, 0.3), "w_prime": 200.0 + rng.normal(0, 20),
                "rtg": 60.0 + rng.normal(0, 8), "draw": run + 1,
                "form": float(rng.uniform(0, 1)), "jky_sr": float(rng.uniform(0, 0.2)),
                "trial_fig": float(rng.normal(0, 0.3)), "trial_m": float(rng.integers(0, 3)),
                "won": 1 if run == 0 else 0, "field_size": nf, "bodywt": 1100.0,
            })
    return pl.DataFrame(rows)


def test_fusion_is_a_noop_at_zero_ratio():
    # ρ=0 ⇒ the precision-fusion contributes nothing; v_c is exactly the additive-trial path's v_c.
    # (group_by ordering is nondeterministic, so compare per-horse after sorting.)
    df = _frame()
    tr, te = df.head(280), df.tail(40)
    base = fit_predict_traffic(tr, te, sim="wald", trial_pi_ratio=0.0).sort("horse_id")
    # a positive ratio must MOVE some v_c (so the no-op below is a real assertion, not a tautology)
    fused = fit_predict_traffic(tr, te, sim="wald", trial_pi_ratio=2.0).sort("horse_id")
    base_again = fit_predict_traffic(tr, te, sim="wald", trial_pi_ratio=0.0).sort("horse_id")
    assert np.allclose(base["v_c_eb"].to_numpy(), base_again["v_c_eb"].to_numpy())   # deterministic
    moved = ~np.isclose(base["v_c_eb"].to_numpy(), fused["v_c_eb"].to_numpy())
    assert moved.any()                                                              # ρ>0 is NOT a no-op


def _w_norm(n, m, rho, pi0, pi_race):
    """The fusion's race-count-adaptive trial weight, normalised to 1 at debut (the fit's law)."""
    m_pi = m * rho * pi_race
    w = m_pi / (pi0 + n * pi_race + m_pi)
    w0 = (rho * pi_race) / (pi0 + rho * pi_race)
    return w / w0 if w0 > 0 else 0.0


def test_fusion_weight_fades_with_race_count():
    # The defining property (the user's intent): the trial's effective contribution decays as race
    # evidence accumulates — full at debut (n=0), tiny once established. This is the exact weight law
    # the fit applies (w_norm = w_trial(n,m)/w_trial(0,1)); assert it on representative precisions.
    pi0, pi_race, rho, m = 11.7, 3.6, 2.0, 1   # ≈ the real-data v_c variance components, ST ρ
    ns = [0, 1, 2, 5, 10, 30]
    w = [_w_norm(n, m, rho, pi0, pi_race) for n in ns]
    assert w[0] == 1.0                                   # debut: full additive coupling recovered
    assert all(w[i + 1] < w[i] for i in range(len(w) - 1))        # strictly fading with race count
    assert w[3] < 0.6 and w[5] < 0.2                     # n=5 < 0.6, n=30 < 0.2: established discounted
    assert w[0] > 5 * w[-1]                              # a debut (n=0) moves ≫ an established (n=30)
    # more trials at the same race count carry more weight (m scales the trial precision)
    assert _w_norm(3, 3, rho, pi0, pi_race) > _w_norm(3, 1, rho, pi0, pi_race)


def test_fusion_recovers_additive_for_a_debutant():
    # At n=0 (debut) the normalised weight is 1, so the fused v_c shift equals the additive TRIAL_GAIN
    # shift exactly (tg_v · trial_fig) — the fusion does not under-serve debutants.
    rng = np.random.default_rng(2)
    rows = []
    for h in range(40):
        rows.append({"race_id": f"p{h}", "race_no": h, "runner": 1, "venue": "ST", "distance": 1200,
                     "horse_id": f"P{h}", "v_c": 16.8 + rng.normal(0, 0.3), "w_prime": 200.0,
                     "rtg": 60.0, "draw": 1, "form": 0.5, "jky_sr": 0.1, "trial_fig": 0.0,
                     "trial_m": 0.0, "won": 0, "field_size": 2, "bodywt": 1100.0})
    train = pl.DataFrame(rows)
    # a debut runner (horse_id NOT in train ⇒ n=0) carrying one trial
    deb = {"race_id": "T", "race_no": 99, "runner": 1, "venue": "ST", "distance": 1200,
           "horse_id": "DEBUT", "v_c": 16.8, "w_prime": 200.0, "rtg": 60.0, "draw": 1, "form": 0.5,
           "jky_sr": 0.1, "trial_fig": 0.4, "trial_m": 1.0, "won": 0, "field_size": 2, "bodywt": 1100.0}
    test = pl.DataFrame([deb])
    add = fit_predict_traffic(train, test, sim="wald", trial_pi_ratio=0.0, trial_gain=1.5)
    fus = fit_predict_traffic(train, test, sim="wald", trial_pi_ratio=2.0, trial_gain=1.5)
    assert abs(add["v_c_eb"][0] - fus["v_c_eb"][0]) < 1e-9   # debut: fusion == additive (w_norm=1)


def test_trial_vc_map_is_leak_free_and_sane():
    # The affine map fits trial_fig→race-v_c-residual on TRAIN only and returns a finite (a, b).
    df = _frame(seed=3)
    pop = fit_energetics_population(df, rating_col="rtg")
    a, b = fit_trial_vc_map(df, pop, signal="trial_fig")
    assert np.isfinite(a) and np.isfinite(b)
    # too few trials ⇒ no-op (a, b) = (0, 0)
    a0, b0 = fit_trial_vc_map(df.head(5), pop, signal="trial_fig")
    assert (a0, b0) == (0.0, 0.0)
