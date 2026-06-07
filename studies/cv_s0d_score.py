"""S0d re-scoring — derive finish/pole orders from the SAVED panels (no re-OCR) and score
each view against OFFICIAL data. Separated from the (slow) extraction so the scorecard can be
re-derived cheaply once the parquets exist.

  AERIAL  — bound to saddlecloth numbers, so we score finish order + per-pole longitudinal
            order vs the official running-position sequence directly.
  DRONE   — ANONYMOUS silk clusters (numbers illegible), so we score only the STRUCTURE: does
            it recover a clear leader pulling clear / a closer from the back? We CANNOT bind
            numbers, so finish-order vs official is N/A (honest).

    uv run --group vision python studies/cv_s0d_score.py
"""
from __future__ import annotations

import html
import re

import polars as pl

OFFICIAL = "data_cache/race_official/20260603R1_results.html"
AERIAL = "data_cache/cv_s0d_aerial.parquet"
DRONE = "data_cache/cv_s0d_drone.parquet"
DIST = 1650.0
OFFICIAL_SECTIONS = [27.65, 23.30, 23.75, 24.15]


def parse_official(path: str) -> dict:
    t = open(path, encoding="utf-8", errors="ignore").read()
    rest = t[t.find("RACE 1 (745)"):]
    out: dict = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", rest, re.S):
        cc = [re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html.unescape(x))).strip()
              for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cc) == 12 and re.match(r"^\d+$", cc[0]) and re.match(r"^\d+$", cc[1]):
            out[int(cc[1])] = dict(pla=int(cc[0]), horse=cc[2], dr=int(cc[7]),
                                   rp=[int(x) for x in cc[9].split() if x.isdigit()])
    return out


def _rank(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(a, b):
    import numpy as np

    if len(a) < 3:
        return float("nan")
    ra, rb = np.array(_rank(a), float), np.array(_rank(b), float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def time_to_reach(df, runner, xq):
    """Elapsed time (s) at which ``runner`` first reaches arc-length ``xq`` (interp)."""
    import numpy as np

    s = df.filter(pl.col("runner") == runner).sort("x_arclen")
    xa, t = s["x_arclen"].to_numpy(), s["t"].to_numpy()
    if len(xa) < 2 or xa.max() < xq - 60 or xa.min() > xq + 60:
        return None
    j = int(np.argmin(np.abs(xa - xq)))
    return float(t[j])


def finish_order(df, runners):
    """Finish order = runners ranked by time to reach the latest common arc-length pole."""
    import numpy as np

    # latest arc-length most runners reach
    maxes = {r: df.filter(pl.col("runner") == r)["x_arclen"].max() for r in runners}
    xq = float(np.percentile(list(maxes.values()), 30))   # a late pole most runners reach
    times = {}
    for r in runners:
        tt = time_to_reach(df, r, xq)
        if tt is not None:
            times[r] = tt
    return sorted(times, key=lambda r: times[r]), xq


def pole_orders(df, runners, poles_frac=(0.25, 0.50, 0.75, 0.97)):
    """Per official pole, CV order leader->tail (by time to reach that pole's arc-length)."""
    orders = []
    for fr in poles_frac:
        xq = fr * DIST
        times = {}
        for r in runners:
            tt = time_to_reach(df, r, xq)
            if tt is not None:
                times[r] = tt
        orders.append(sorted(times, key=lambda r: times[r]))
    return orders


def score_aerial(official):
    import numpy as np

    df = pl.read_parquet(AERIAL)
    runners = sorted(df["runner"].unique().to_list())
    fo, xq = finish_order(df, runners)
    winner_ok = bool(fo and fo[0] == next(n for n in official if official[n]["pla"] == 1))
    # finish-rank corr: CV finish position vs official placing (bound runners only)
    common = [r for r in fo if r in official]
    fcorr = (float(np.corrcoef([fo.index(r) for r in common],
                               [official[r]["pla"] for r in common])[0, 1])
             if len(common) >= 3 else float("nan"))
    # per-pole running-position corr (cv leader->tail rank vs official rp where 1=leader)
    orders = pole_orders(df, runners)
    pole_corrs = []
    for p, order in enumerate(orders):
        cm = [r for r in order if r in official and len(official[r]["rp"]) > p]
        if len(cm) < 3:
            continue
        cv_rank = [order.index(r) for r in cm]          # 0 = CV leader
        off_rank = [official[r]["rp"][p] for r in cm]   # 1 = official leader
        pole_corrs.append(spearman([float(x) for x in cv_rank],
                                    [float(x) for x in off_rank]))
    rp_corr = float(np.mean([c for c in pole_corrs if c == c])) if pole_corrs else float("nan")
    # trips: y_lat median per bound runner (rail vs wide), cross-check with RP trajectory
    trips = {r: float(df.filter(pl.col("runner") == r)["y_lat"].median()) for r in runners}
    return dict(df=df, runners=runners, finish_order=fo, finish_pole=xq, winner_ok=winner_ok,
                fcorr=fcorr, rp_corr=rp_corr, pole_orders=orders, trips=trips,
                pole_corrs=pole_corrs)


def score_drone(official):
    """Drone is ANONYMOUS: report only the longitudinal STRUCTURE (leader separation, spread),
    not a number-bound finish order."""
    import numpy as np

    df = pl.read_parquet(DRONE)
    runners = sorted(df["runner"].unique().to_list())
    fo, xq = finish_order(df, runners)
    # structure metrics: how strung-out is the field at the late pole (s between leader & tail)?
    times = {}
    for r in runners:
        tt = time_to_reach(df, r, xq)
        if tt is not None:
            times[r] = tt
    spread_s = (max(times.values()) - min(times.values())) if len(times) >= 3 else float("nan")
    # lateral: per-cluster y_lat median spread (does the recon place runners across the track?)
    y_meds = [float(df.filter(pl.col("runner") == r)["y_lat"].median()) for r in runners]
    return dict(df=df, runners=runners, anon_order=fo, leader_tail_spread_s=spread_s,
                y_lat_spread_m=(float(np.max(y_meds) - np.min(y_meds)) if y_meds else float("nan")),
                n_clusters=len(runners))


def main():
    official = parse_official(OFFICIAL)
    off_winner = next(n for n in official if official[n]["pla"] == 1)
    print("=" * 78)
    print("S0d RE-SCORE — CV views vs OFFICIAL (ST 2026-06-03 R1, 1650M, \"C\" course)")
    print("=" * 78)
    print(f"OFFICIAL winner #{off_winner} {official[off_winner]['horse']} (rp {official[off_winner]['rp']})")
    print("OFFICIAL finish: " + ", ".join(
        f"{official[n]['pla']}:#{n}" for n in sorted(official, key=lambda n: official[n]['pla'])))

    a = score_aerial(official)
    print("\n[AERIAL] bound:", a["runners"])
    print(f"  CV finish order (by time@{a['finish_pole']:.0f}m): {a['finish_order']}")
    print(f"  winner == #{off_winner}? {a['winner_ok']}")
    print(f"  finish-rank corr vs official placing: {a['fcorr']:+.2f}")
    print(f"  per-pole running-pos Spearman (mean): {a['rp_corr']:+.2f}   (poles: "
          + ", ".join(f"{c:+.2f}" for c in a["pole_corrs"]) + ")")
    print("  per-pole CV order leader->tail:")
    for p, o in enumerate(a["pole_orders"]):
        print(f"    pole {p+1}: {o}")
    print("  official rp by bound runner (1=leader):")
    for r in a["runners"]:
        if r in official:
            print(f"    #{r}: rp {official[r]['rp']}  finish {official[r]['pla']}  "
                  f"y_lat_med {a['trips'][r]:.1f}m")

    d = score_drone(official)
    print("\n[DRONE] anonymous silk clusters:", d["n_clusters"])
    print(f"  leader->tail time spread @late pole: {d['leader_tail_spread_s']:.1f}s")
    print(f"  lateral y_lat spread across clusters: {d['y_lat_spread_m']:.1f}m")
    print("  number binding: N/A (saddlecloth numbers illegible from altitude)")
    print(f"  anonymous along-rail order (cluster ids): {d['anon_order']}")

    # ---- VERDICT ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SCORECARD")
    print("=" * 78)
    nb = len(a["runners"])
    x3 = float(a["df"].filter(pl.col("runner") == 3)["x_arclen"].max()) if 3 in a["runners"] else 0.0
    # NOTE: a['finish_order'] here is derived from the FIELD-SHARED odometer arc-timing, which
    # cannot resolve intra-field rank (journal 0068/0069) — it is a NOISE order. The reliable
    # finish/order is the LEGEND overlay read in the recon (cv_finish_order = official top-8).
    print(f"{'view':9} {'winner?':8} {'finishρ':8} {'RP-Spearman':12} {'reid':24}")
    reidlab = f"legend#-bound ({nb}/12)"
    print(f"{'AERIAL':9} {str(a['winner_ok']):8} {a['fcorr']:+.2f}    "
          f"{a['rp_corr']:+.2f}        {reidlab:24}")
    print(f"{'DRONE':9} {'N/A':8} {'  -':8} {'  -':12} {'anon silk-colour':24}")
    print("\n  HONEST READING (journal 0069 reframe):")
    print("   - finish ORDER comes from the LEGEND overlay (=official chart), NOT geometry.")
    print(f"   - winner #3 now TRACKED through the run-in to x_arclen {x3:.0f}m "
          f"({'reaches finish' if x3 > DIST - 60 else 'stops short'}).")
    print("   - the per-pole arc-timing ρ above is NOISE-dominated (field-shared odometer).")
    print("   - y_lat trip vs RP is a near-NULL (see cv_s0d_qc.py): sprite noise swamps it.")
    print("DONE.")


if __name__ == "__main__":
    main()
