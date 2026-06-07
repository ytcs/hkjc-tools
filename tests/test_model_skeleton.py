"""Smoke test for the hkjc.model scaffold — proves the frame RUNS end-to-end.

The build wave inherits a working frame: this test constructs every core object and asserts the
generative latent state-space produces a valid win distribution + pool prices, and that the
fit→predict adapter is leak-free under the production no-lookahead check.
"""
import numpy as np
import polars as pl

from hkjc.backtesting import assert_no_lookahead
from hkjc.model import (
    CouplingCalibrator,
    CouplingGains,
    GenerativeRaceModel,
    LatentState,
    PathLikelihood,
    PoolPricer,
    RaceConditions,
    build_fit_predict,
    predict_race,
    smoke_race,
)


def test_latent_state_roundtrips():
    theta = LatentState(v_c=16.0, w_prime=60.0, rho=0.1, a=0.95)
    back = LatentState.from_array(theta.as_array())
    assert np.allclose(theta.as_array(), back.as_array())
    assert back.v_c == 16.0 and back.a == 0.95


def test_smoke_race_produces_valid_win_distribution():
    out = smoke_race(seed=0, n=8)
    win = out["win"]
    assert win.shape == (8,)
    assert np.all(win > 0) and np.all(win < 1)
    assert abs(win.sum() - 1.0) < 1e-9          # win is a proper distribution
    # place(top-3) marginals sum to ~3 (three placings shared across the field)
    assert abs(out["place"].sum() - 3.0) < 0.05
    # forecast is an (n,n) ordered joint with zero diagonal, summing to ~1
    fc = out["forecast"]
    assert fc.shape == (8, 8)
    assert np.allclose(np.diag(fc), 0.0)
    assert abs(fc.sum() - 1.0) < 1e-9


def test_predict_race_matches_pricer_win():
    thetas = [LatentState(v_c=16.0 + 0.1 * i, w_prime=60.0, a=1.0) for i in range(6)]
    cond = RaceConditions(race_id="R1", distance=1200, venue="ST", surface="turf",
                          draws=np.arange(1, 7.0), field_size=6, band="sprint")
    od, win = predict_race(thetas, cond)
    assert np.allclose(win, PoolPricer(od).win())
    # the faster horse (higher v_c) should win more often
    assert win.argmax() == 5


def test_coupling_a_off_matches_bare_law():
    """a_start_gain=0 must reproduce the bare (production-equivalent) drift — the honest baseline."""
    thetas = [LatentState(v_c=16.0, w_prime=60.0, a=0.8 + 0.05 * i) for i in range(5)]
    cond = RaceConditions(race_id="R2", distance=1200, venue="ST", surface="turf",
                          draws=np.arange(1, 6.0), field_size=5, band="sprint")
    model = GenerativeRaceModel(n_samples=2000)
    off = CouplingCalibrator(CouplingGains(a_start_gain=0.0))
    mu_off, _ = off.couple(thetas, cond, model)
    mu_bare = model._drift_times(thetas, cond)
    assert np.allclose(mu_off, mu_bare)
    # a non-zero gain DOES move the drift (the coupling is live)
    on = CouplingCalibrator(CouplingGains(a_start_gain=0.5))
    mu_on, _ = on.couple(thetas, cond, model)
    assert not np.allclose(mu_on, mu_bare)


def test_field_pace_war_off_is_byte_identical_baseline():
    """GUARD (journal 0065): field_pace_war=OFF reproduces the physiology-only drift BYTE-IDENTICALLY.

    OFF (the default) must not change μ even when a pace-war burn is present on conditions — the win
    delta is measured ON vs OFF, and OFF == baseline keeps the 181 existing results unchanged.
    """
    thetas = [LatentState(v_c=16.0, w_prime=60.0, a=1.0) for _ in range(5)]
    burn = np.array([0.0, 8.0, 6.0, 0.0, 0.0])   # a contested front burns reserve from two runners
    cond_off = RaceConditions(race_id="FW1", distance=1200, venue="ST", surface="turf",
                              draws=np.arange(1, 6.0), field_size=5, band="sprint")
    cond_on = RaceConditions(race_id="FW1", distance=1200, venue="ST", surface="turf",
                             draws=np.arange(1, 6.0), field_size=5, band="sprint",
                             pace_war_burn=burn)
    base = GenerativeRaceModel(n_samples=2000)
    # OFF model ignores the burn entirely (byte-identical with and without the burn on conditions)
    off = GenerativeRaceModel(n_samples=2000, field_pace_war=False)
    mu_base = base._drift_times(thetas, cond_off)
    assert np.array_equal(off._drift_times(thetas, cond_off), mu_base)
    assert np.array_equal(off._drift_times(thetas, cond_on), mu_base)
    # ON with NO burn on conditions also reproduces baseline (additive guard)
    on = GenerativeRaceModel(n_samples=2000, field_pace_war=True)
    assert np.array_equal(on._drift_times(thetas, cond_off), mu_base)


def test_field_pace_war_on_reduces_reserve_and_slows_burnt_runners():
    """ON: a burnt front-runner carries LESS reserve ⇒ slower target ⇒ LARGER μ; others unchanged."""
    thetas = [LatentState(v_c=16.0, w_prime=60.0, a=1.0) for _ in range(5)]
    burn = np.array([0.0, 12.0, 9.0, 0.0, 0.0])
    cond = RaceConditions(race_id="FW2", distance=1200, venue="ST", surface="turf",
                          draws=np.arange(1, 6.0), field_size=5, band="sprint",
                          pace_war_burn=burn)
    base = GenerativeRaceModel(n_samples=2000)
    on = GenerativeRaceModel(n_samples=2000, field_pace_war=True)
    mu_base = base._drift_times(thetas, cond)
    mu_on = on._drift_times(thetas, cond)
    # the burnt runners (1,2) lose reserve ⇒ slower v_target ⇒ a LONGER expected time
    assert mu_on[1] > mu_base[1] and mu_on[2] > mu_base[2]
    # the un-burnt runners (0,3,4) are byte-identical
    assert mu_on[0] == mu_base[0] and mu_on[3] == mu_base[3] and mu_on[4] == mu_base[4]
    # the bigger burn (runner 1) slows more than the smaller (runner 2)
    assert (mu_on[1] - mu_base[1]) > (mu_on[2] - mu_base[2])


def test_coupling_rho_off_matches_bare_law_and_moves_body():
    """rho_gain=0 reproduces the bare drift; a non-zero gain moves μ for off-mean ρ (a body-mover).

    ρ is the RECOVERY/closer latent (RQ-rho/0044). A high-ρ closer banks reserve ⇒ more effective
    usable reserve late ⇒ smaller effective μ (faster v_target). ρ=OFF must be byte-exact baseline.
    """
    thetas = [LatentState(v_c=16.0, w_prime=60.0, a=1.0, rho=0.05 * (i - 2)) for i in range(5)]
    cond = RaceConditions(race_id="RHO1", distance=1400, venue="ST", surface="turf",
                          draws=np.arange(1, 6.0), field_size=5, band="mile")
    model = GenerativeRaceModel(n_samples=2000)
    off = CouplingCalibrator(CouplingGains(rho_gain=0.0))
    mu_off, _ = off.couple(thetas, cond, model)
    assert np.allclose(mu_off, model._drift_times(thetas, cond))
    on = CouplingCalibrator(CouplingGains(rho_gain=4.0))
    mu_on, _ = on.couple(thetas, cond, model)
    assert not np.allclose(mu_on, mu_off)
    # the highest-ρ closer (last) gets a SMALLER μ than the lowest-ρ front-runner relative to baseline
    rel = (mu_on - mu_off)
    assert rel[-1] < 0.0 < rel[0]
    # a uniform ρ shift cancels (within-race centred): all-equal ρ ⇒ no change
    flat = [LatentState(v_c=16.0, w_prime=60.0, a=1.0, rho=0.3) for _ in range(5)]
    mu_flat_on, _ = on.couple(flat, cond, model)
    assert np.allclose(mu_flat_on, model._drift_times(flat, cond))


def test_pool_pricer_realized_combo_probs():
    """The realized-combination pricers are consistent Monte-Carlo functionals of one order dist."""
    thetas = [LatentState(v_c=16.0 + 0.1 * i, w_prime=60.0, a=1.0) for i in range(8)]
    cond = RaceConditions(race_id="PB1", distance=1200, venue="ST", surface="turf",
                          draws=np.arange(1, 9.0), field_size=8, band="sprint")
    od, _ = predict_race(thetas, cond, model=GenerativeRaceModel(n_samples=8000))
    pr = PoolPricer(od)
    # prob_win matches the win marginal (same Laplace smoothing)
    assert abs(pr.prob_win(0) - od.win_probs()[0]) < 1e-12
    # quinella = forecast(i,j)+forecast(j,i) up to the single shared Laplace term
    m = len(od.orders)
    laplace = 1.0 / (m + od.n)
    assert abs(pr.prob_quinella(0, 1) - (pr.prob_forecast(0, 1) + pr.prob_forecast(1, 0) - laplace)) < 1e-9
    # ordered tierce ⊆ unordered trio (== place-set for k=3); all in (0,1)
    assert 0.0 < pr.prob_tierce(0, 1, 2) <= pr.prob_trio([0, 1, 2]) < 1.0
    assert abs(pr.prob_place([0, 1, 2], 3) - pr.prob_trio([0, 1, 2])) < 1e-12


def test_pricer_qpl_and_place_field_size_rule():
    """QPL (both in top-k) is a valid pairwise top-k joint; place() sources k from hkjc_facts."""
    from hkjc.core.hkjc_facts import places_paid
    thetas = [LatentState(v_c=16.0 + 0.1 * i, w_prime=60.0, a=1.0) for i in range(8)]
    cond = RaceConditions(race_id="QPL1", distance=1200, venue="ST", surface="turf",
                          draws=np.arange(1, 9.0), field_size=8, band="sprint")
    od, _ = predict_race(thetas, cond, model=GenerativeRaceModel(n_samples=8000))
    pr = PoolPricer(od)
    # place() with no k uses the HKJC field-size rule (8 runners ⇒ 3 places)
    assert places_paid(8) == 3
    assert pr.place().shape == (8,) and abs(pr.place().sum() - 3.0) < 0.05
    # QPL ≥ QIN (top-3 set ⊇ top-2 set) and QPL ≤ either runner's place marginal sum
    qpl = pr.prob_quinella_place(0, 1, k=3)
    qin = pr.prob_quinella(0, 1)
    assert qin <= qpl < 1.0
    # QPL(i,j) ≤ min place marginals (both-in-top-k can't exceed either-in-top-k)
    place_marg = pr.place(k=3)
    assert qpl <= min(place_marg[0], place_marg[1]) + 1e-6


def test_pool_book_score_pct_of_market():
    """pool_book_score returns the exact %market per pool on the realized combination."""
    from hkjc.model import pool_book_score
    # one pool, two races: construct so model_LL between uniform and market ⇒ 0<%market<1
    rows = [
        {"race_id": "r1", "pool": "FCT", "p_model": 0.05, "p_market": 0.08, "p_uniform": 0.01},
        {"race_id": "r2", "pool": "FCT", "p_model": 0.04, "p_market": 0.06, "p_uniform": 0.01},
    ]
    sc = pool_book_score(pl.DataFrame(rows))["FCT"]
    assert sc["n_races"] == 2
    expect = (sc["uniform_ll"] - sc["model_ll"]) / (sc["uniform_ll"] - sc["market_ll"])
    assert abs(sc["pct_of_market"] - expect) < 1e-12
    assert 0.0 < sc["pct_of_market"] < 1.0


def test_path_likelihood_profile_term_runs():
    """The Stage-1 profile term reuses energetics_dynamic and returns a finite log-likelihood."""
    pl_obj = PathLikelihood.stage1()
    cond = RaceConditions(race_id="R3", distance=1200, venue="ST", surface="turf",
                          draws=np.array([1.0]), field_size=1, band="sprint")
    obs = {"section_times": [12.0, 11.5, 11.8, 12.4],
           "section_dists": [300.0, 300.0, 300.0, 300.0]}
    theta = pl_obj.identify(cond, obs)                 # closed-form #41 recovery
    assert np.isfinite(theta.v_c) and np.isfinite(theta.a)
    ll = pl_obj.log_prob(theta, cond, obs)
    assert np.isfinite(ll)


def _synth_frame(n_races: int = 30, field: int = 8, seed: int = 0) -> pl.DataFrame:
    """Synthetic frame with PRE-COMPUTED strictly-prior θ columns + outcomes (for the leak check)."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        vc = 16.0 + rng.normal(0, 0.4, field)
        winner = int(vc.argmax())
        for i in range(field):
            rows.append(dict(
                race_id=f"2026-01-01_{r}", runner=i + 1, venue="ST", distance=1200,
                draw=i + 1, v_c_prior=float(vc[i]), w_prime_prior=60.0,
                a_prior=float(0.95 + rng.normal(0, 0.05)),
                won=int(i == winner), placed=int(i in vc.argsort()[-3:]),
                finish_pos=int((-vc).argsort().argsort()[i] + 1),
            ))
    return pl.DataFrame(rows)


def test_fit_predict_is_leak_free():
    """The fit→predict adapter passes the production adversarial no-lookahead check."""
    df = _synth_frame()
    train, test = df, df          # the default builder ignores train (pre-computed prior θ)
    fp = build_fit_predict(n_samples=2000)
    preds = fp(train, test)
    assert "p" in preds.columns
    # per-race win probs are proper
    for _k, g in preds.group_by("race_id"):
        assert abs(g["p"].sum() - 1.0) < 1e-9
    drift = assert_no_lookahead(fp, train, test, key_cols=("race_id", "runner"))
    assert drift < 1e-12


def _production_wald_win(vc, wp, distance, *, n_samples, race_id,
                         diff_coef=170.0):
    """The production wald win-prob law (turf, no-bend, no extra couplings) computed directly.

    Mirrors ``hkjc.model.fit.fit_predict_traffic``'s ``wald`` branch for the bare case the
    state-space a=OFF path must reproduce: μ_i = L/(v_c·(1+W′/L)), λ_i = L²/(2D), sampled with the
    SAME race-id seed and Laplace smoothing. The full-panel byte-exactness on real data is shown by
    ``the Acceptance Gate A study (since removed)`` (Acceptance Gate A); this is the data-free unit guard.
    """
    n = len(vc)
    mu = distance / (vc * (1.0 + wp / distance))
    lam = np.full(n, distance * distance / (2.0 * diff_coef))
    seed = abs(hash(race_id)) % (2**31)
    rng = np.random.default_rng(seed)
    t = rng.wald(np.broadcast_to(mu, (n_samples, n)), np.broadcast_to(lam, (n_samples, n)))
    counts = np.bincount(t.argmin(axis=1), minlength=n).astype(float)
    return (counts + 1.0) / (n_samples + n)


def test_a_off_reproduces_production_wald():
    """ACCEPTANCE GATE A (unit): with a_start_gain=0 the generative core IS the production wald law.

    Given the production-computed (v_c_eb, w_prime_eb), the state-space a=OFF μ/λ + sampler must
    reproduce the production wald win probs byte-for-byte (same seed, same n_samples). This proves
    the generative core is faithful before `a` is added — the architecture's first acceptance.
    """
    rng = np.random.default_rng(7)
    # ST 1000m straight: no bend (bend_fraction=0) ⇒ no draw penalty ⇒ the bare wald law applies.
    n, distance, n_samples = 9, 1000, 6000
    vc = 16.0 + rng.normal(0, 0.4, n)
    wp = 60.0 + rng.normal(0, 8.0, n)
    a_vec = 0.85 + 0.03 * np.arange(n)          # varied across the field so centring is non-trivial
    thetas = [LatentState(v_c=float(vc[i]), w_prime=float(wp[i]), a=float(a_vec[i]))
              for i in range(n)]
    cond = RaceConditions(race_id="ACC1", distance=distance, venue="ST", surface="turf",
                          draws=np.arange(1, n + 1.0), field_size=n, band="sprint")
    model = GenerativeRaceModel(n_samples=n_samples, diff_coef=170.0)
    off = CouplingCalibrator(CouplingGains(a_start_gain=0.0))
    mu_off, lam_off = off.couple(thetas, cond, model)
    od = model.order_distribution(thetas, cond, mu=mu_off, lam=lam_off)
    p_ss = od.win_probs()
    p_prod = _production_wald_win(vc, wp, distance, n_samples=n_samples, race_id="ACC1")
    assert np.max(np.abs(p_ss - p_prod)) < 1e-12       # byte-exact: a=OFF == production wald
    # a non-zero gain MUST move the win probs (the coupling is live)
    on = CouplingCalibrator(CouplingGains(a_start_gain=0.4))
    mu_on, lam_on = on.couple(thetas, cond, model)
    p_on = model.order_distribution(thetas, cond, mu=mu_on, lam=lam_on).win_probs()
    assert np.max(np.abs(p_on - p_prod)) > 1e-6
