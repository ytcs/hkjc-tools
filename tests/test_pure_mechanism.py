"""Unit tests for the PureMechanismThetaBuilder (``hkjc.model.pure_mechanism``, journal 0050).

These pin the CAUSAL-REBUILD CONTRACT, not a fit:
  * the module imports NO meta-signal table from ``hkjc.model.features`` (the discipline — feeding a
    form/jockey/collateral table would break the payoff test);
  * the per-race θ builder reads ONLY path-physiology / population columns and the drift v_c follows
    the source priority Hill → de-confounded cruise → population mean;
  * the builder leaves the stripped meta-signal geometry (gate-break, fig-disp) OFF the conditions;
  * the strictly-prior Hill panel never lets a run's OWN path enter its anchor (leak-free).
No regression, no market/win/pool.
"""
from __future__ import annotations

import polars as pl
import pytest

from hkjc.model import pure_mechanism as pm
from hkjc.model.conditions import RaceConditions
from hkjc.model.latent import LatentState


def test_no_meta_signal_imports():
    """The module must NOT import any meta-signal table (form/jockey/collateral/...) — the discipline."""
    import inspect

    # only the IMPORT lines + executable body matter (docstrings legitimately name the stripped set).
    src_lines = [ln for ln in inspect.getsource(pm).splitlines()
                 if ln.strip().startswith(("import ", "from "))]
    src = "\n".join(src_lines)
    banned = ["form_table", "recency_form_table", "jockey_table", "trainer_table",
              "trainer_hot_table", "collateral_form_table", "speed_figure_table",
              "barrier_trial_table", "gallop_vc_table", "trial_vc_table",
              "rating_trajectory", "early_position_table", "gate_break_table",
              "trajectory_trait_table", "finishing_sectional", "features"]
    for name in banned:
        assert name not in src, f"pure_mechanism imports a STRIPPED meta-signal: {name}"


def _race_frame(**cols):
    n = len(next(iter(cols.values())))
    base = dict(date=["2024-01-01"] * n, venue=["ST"] * n, race_no=[1] * n,
                runner=list(range(1, n + 1)), distance=[1600] * n, draw=list(range(1, n + 1)),
                race_id=["2024-01-01_1"] * n, course=["A"] * n)
    base.update(cols)
    return pl.DataFrame(base)


def test_builder_fuses_instruments_by_precision():
    """v_c is a PRECISION-WEIGHTED fusion of the instruments (RQ-vc1): a precise Hill slope dominates,
    a NOISY Hill slope (large cv_se) yields to a tight de-confounded cruise level (the 0050 fix), and
    a debut runner (no path) falls to the rating-conditioned population mean."""
    g = _race_frame(
        vc_hill=[16.0, 18.0, None],
        vc_hill_se=[0.1, 5.0, None],        # runner 0 precise; runner 1 very noisy
        vc_decon=[None, 15.0, None],
        vc_decon_n=[None, 4, None],
        vc_decon_sd=[None, 0.4, None],      # runner 1 has a tight de-confounded level
        vc_pop=[14.5, 14.5, 14.8],
        wprime_hill=[30.0, None, None],
        wprime_pop=[20.0, 20.0, 20.0],
    )
    thetas, cond = pm.pure_mechanism_theta_builder(None, g)
    assert isinstance(cond, RaceConditions)
    assert all(isinstance(t, LatentState) for t in thetas)
    assert thetas[0].v_c == pytest.approx(16.0, abs=0.2)   # precise Hill dominates
    assert thetas[1].v_c == pytest.approx(15.0, abs=0.4)   # noisy Hill yields to the tight decon level
    assert thetas[2].v_c == pytest.approx(14.8)            # debut → population mean
    # W' uses Hill where present, else the population mean
    assert thetas[0].w_prime == pytest.approx(30.0)
    assert thetas[2].w_prime == pytest.approx(20.0)


def test_builder_strips_metasignal_geometry():
    """The conditions must NOT carry the stripped gate-break / fig-disp meta-signal geometry."""
    g = _race_frame(vc_hill=[15.0, 15.5], vc_decon=[None, None], vc_pop=[14.0, 14.0],
                    wprime_hill=[25.0, 25.0], wprime_pop=[20.0, 20.0],
                    gate_break=[0.5, -0.5], fig_disp=[1.0, 2.0])
    _thetas, cond = pm.pure_mechanism_theta_builder(None, g)
    assert cond.gate_break_shrunk is None, "gate-break is a STRIPPED meta-signal — must stay OFF"
    assert cond.fig_disp is None, "fig-disp is a STRIPPED meta-signal — must stay OFF"


def test_builder_clamps_drift_to_physical_window():
    """A wild Hill read is clamped to the generative cruise window (the Wald law stays physical)."""
    g = _race_frame(vc_hill=[3.0, 40.0], vc_hill_se=[0.1, 0.1], vc_decon=[None, None],
                    vc_decon_n=[None, None], vc_decon_sd=[None, None], vc_pop=[15.0, 15.0],
                    wprime_hill=[20.0, 20.0], wprime_pop=[20.0, 20.0])
    thetas, _ = pm.pure_mechanism_theta_builder(None, g)
    for t in thetas:
        assert pm._VC_DRIFT_LO <= t.v_c <= pm._VC_DRIFT_HI


def test_strictly_prior_hill_panel_excludes_own_run():
    """The strictly-prior Hill anchor must be NULL on a horse's first runs (no prior distinct dists)."""
    # one horse, 4 runs at 4 distinct distances on distinct dates
    dists = [1200, 1400, 1600, 1800]
    # shorter trips run FASTER (the critical-velocity law): d = CV·t + D' with CV=15, D'=40.
    cv, d_prime = 15.0, 40.0
    rows = []
    for i, d in enumerate(dists):
        nsec = 4
        sd = [d / nsec] * nsec
        total_t = (d - d_prime) / cv          # invert d = CV·t + D'
        st = [total_t / nsec] * nsec
        for s in range(nsec):
            rows.append(dict(date=f"2024-0{i+1}-01", venue="ST", race_no=1, runner=1,
                             horse_id="H1", distance=d, section=s, section_dist=sd[s],
                             section_time=st[s], velocity=sd[s] / st[s], position=1))
    traj = pl.DataFrame(rows)
    rr = pl.DataFrame(dict(date=[f"2024-0{i+1}-01" for i in range(4)], Venue=["ST"] * 4,
                           RaceNumber=[1] * 4, HorseNo=[1] * 4, Going=["GOOD"] * 4,
                           DeclarHorseWt=[1100.0] * 4, ActWt=[125.0] * 4))
    panel = pm.build_pure_mechanism_panel(traj, rr).sort("date")
    vc = panel["vc_hill"].to_list()
    # first three runs have <3 distinct PRIOR distances → null; the 4th has 3 priors → a Hill line
    assert vc[0] is None and vc[1] is None and vc[2] is None
    assert vc[3] is not None and 8.0 < vc[3] < 22.0
