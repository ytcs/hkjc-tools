"""Validation against the OFFICIAL outcome. The gates score the VIDEO-ONLY product
(``paths_video.parquet``, which never uses official outcome data); the FINAL product (``paths.parquet``)
includes the official records and is only checked for how closely it honours them.

Fixed acceptance gates (set before looking at numbers; never loosened):
  leader_timing_mae   <= 0.25 s   leader crossing time at each sectional boundary vs official
  sectional_mae       <= 0.25 s   per-horse crossing times at sectional boundaries vs official
  finish_rho          >= 0.90     Spearman, reconstructed vs official finish times
  winner              == official
  behind_winner_mae   <= 0.20 s   per-horse time behind the winner at the line
Declared before the held-out races (no retuning on them): sectional coverage >= 0.80, a fused path
for >= N-1 runners, finish rho over >= N-2 runners, lateral scale: the racing->fixed rail gap on the
map vs the course table within 0.75 m (where the fixed rail is visible). N counts every runner for the
path gate and the runners with an official finish time for the finish gate (the two differ only when a
horse did not finish); coverage counts only the official times that exist.
A crossing is only scored where the fused posterior is tight (sd_s < ``SD_MAX`` m) — coverage is
reported alongside, so gaps are visible rather than smoothed over.
"""

from __future__ import annotations

import numpy as np

SD_MAX = 1.5
GATES = {
    "sectional_coverage": (">=", 0.80),
    "horses_with_path_short": ("<=", 1),  # at most one runner without a fused path
    "finish_n_short": ("<=", 2),  # finish order scored on at least N-2 runners
    "lateral_scale_err": ("<=", 0.75),  # |fixed-rail gap - course table| (m), where measurable
    "leader_timing_mae": ("<=", 0.25),
    "sectional_mae": ("<=", 0.25),
    "finish_rho": (">=", 0.90),
    "winner_ok": ("==", True),
    "behind_winner_mae": ("<=", 0.20),
}


def _cum_official(info):
    """The FULL official record's (boundaries, {runner: cumulative times}) of a ``RaceInfo``."""
    from ..data.race import official_boundaries

    return official_boundaries(info)


def crossing(t, d, sd, mark):
    """Time a path first reaches distance-to-go ``mark`` (NaN if not tightly observed there)."""
    k = np.where(d <= mark)[0]
    if len(k) == 0 or k[0] == 0:
        return np.nan
    a, b = k[0] - 1, k[0]
    if sd is not None and max(sd[a], sd[b]) > SD_MAX:
        return np.nan
    return t[a] + (d[a] - mark) / max(d[a] - d[b], 1e-9) * (t[b] - t[a])


def validate(run, truth) -> dict:
    """Gates for ``run``'s artifacts against ``truth``, the full (unmasked) ``RaceInfo``."""
    import polars as pl
    from scipy.stats import spearmanr

    from ..data.race import official_finish, official_placing

    info = truth
    bnd, cum = _cum_official(info)
    # gates score the VIDEO-ONLY posterior (the final product contains the official records)
    P = pl.read_parquet(run.p("paths_video.parquet"))
    res = {}
    # identity-free leader timing (raw detections)
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    tu = np.unique(T["t"])
    lead = np.minimum.accumulate(np.array([T["d"][T["t"] == x].min() for x in tu]))
    off_lead = np.nanmin(np.array(list(cum.values())), axis=0)
    lt = np.array([crossing(tu, lead, None, b) for b in bnd]) - off_lead
    res["leader_timing_err"] = lt.tolist()
    res["leader_timing_mae"] = float(np.nanmean(np.abs(lt)))
    # per-horse sectionals from the fused paths
    errs, fin = {}, []
    for h, g in P.sort("horse_no").group_by("horse_no", maintain_order=True):
        h = int(h[0]) if isinstance(h, tuple) else int(h)
        g = g.sort("t")
        t, d, sd = g["t"].to_numpy(), g["d"].to_numpy(), g["sd_s"].to_numpy()
        rec = np.array([crossing(t, d, sd, b) for b in bnd])
        if h in cum:
            errs[h] = (rec - cum[h]).tolist()
            fin.append((h, rec[-1], cum[h][-1]))
    E = np.array(list(errs.values()))
    res["sectional_err"] = errs
    res["sectional_mae"] = float(np.nanmean(np.abs(E)))
    # coverage over the official times that exist (a non-finisher has none after its last complete section)
    off = np.array([cum[h] for h in errs], float)
    res["sectional_coverage"] = float(np.isfinite(E).sum() / max(np.isfinite(off).sum(), 1))
    fin = [(h, r, o) for h, r, o in fin if np.isfinite(r) and np.isfinite(o)]
    res["finish_n"] = len(fin)
    if len(fin) >= 3:
        rr = np.array([f[1] for f in fin])
        oo = np.array([f[2] for f in fin])
        res["finish_rho"] = float(spearmanr(rr, oo).statistic)
        res["behind_winner_mae"] = float(np.abs((rr - rr.min()) - (oo - oo.min())).mean())
    res["horses_with_path_short"] = info.n - P["horse_no"].n_unique()
    # final product (with the official records as observations): how closely it honours them
    Pf = pl.read_parquet(run.p("paths.parquet"))
    fe = []
    for h, g in Pf.sort("horse_no").group_by("horse_no", maintain_order=True):
        h = int(h[0]) if isinstance(h, tuple) else int(h)
        if h in cum:
            g = g.sort("t")
            fe += list(
                np.array([crossing(g["t"].to_numpy(), g["d"].to_numpy(), None, b) for b in bnd])
                - cum[h]
            )
    res["final_official_resid_mae"] = float(np.nanmean(np.abs(fe))) if fe else float("nan")
    # N for the finish gate: the runners that finished (an official finish time; equals info.n without non-finishers)
    res["finish_n_short"] = len(official_finish(info)) - res["finish_n"]
    m = np.load(run.p("metric.npz"))
    if "fixed_gap" in m.files and np.isfinite(float(m["fixed_gap"])):
        res["fixed_gap_m"] = float(m["fixed_gap"])
        res["lateral_scale_err"] = abs(float(m["fixed_gap"]) - float(m["fixed_gap_expected"]))
    off_winners = [h for h, p in official_placing(info).items() if p == 1]  # two on a dead heat for first
    rec_winner = min(fin, key=lambda f: f[1])[0] if fin else None
    res["winner_ok"] = bool(rec_winner in off_winners)
    res["winner"] = {"official": off_winners[0] if len(off_winners) == 1 else off_winners, "reconstructed": rec_winner}
    res["gates"] = {}
    for k, (op, thr) in GATES.items():
        v = res.get(k)
        ok = (
            None
            if v is None
            else (v <= thr if op == "<=" else v >= thr if op == ">=" else v == thr)
        )
        res["gates"][k] = ok
    return res


def summary_line(key, res) -> str:
    g = res["gates"]
    mark = {True: "PASS", False: "FAIL", None: "n/a"}
    extra = (
        f" | coverage [{mark[g['sectional_coverage']]}] paths-short {res['horses_with_path_short']} "
        f"[{mark[g['horses_with_path_short']]}] finish-short {res['finish_n_short']} "
        f"[{mark[g['finish_n_short']]}] lateral-scale {res.get('lateral_scale_err', float('nan')):.2f}m "
        f"[{mark[g['lateral_scale_err']]}]"
    )
    return (
        f"{key}: leader {res['leader_timing_mae']:.3f}s [{mark[g['leader_timing_mae']]}]  "
        f"sectional {res['sectional_mae']:.3f}s @ {res['sectional_coverage']:.0%} "
        f"[{mark[g['sectional_mae']]}]  finish rho {res.get('finish_rho', float('nan')):.2f} "
        f"(n={res['finish_n']}) [{mark[g['finish_rho']]}]  winner {res['winner']} "
        f"[{mark[g['winner_ok']]}]  behind-winner {res.get('behind_winner_mae', float('nan')):.3f}s "
        f"[{mark[g['behind_winner_mae']]}]" + extra
    )
