"""S0d AERIAL #82 — the FAITHFULNESS VERDICT analysis on the recovered render panel.

Reads ``data_cache/cv_s0d_aerial_render.parquet`` (written by ``cv_s0d_aerial_render.py``) and
runs the disambiguation the coarse run flagged: the recovered y_lat is SMOOTH but a flat
rail-perpendicular offset is depth-confounded on the OBLIQUE bend (a single drawn rail can't
give the full ground-plane homography — under-determined). The cleanest window to read the
render's lateral truth is therefore the RUN-IN STRAIGHT (last ~300 m), where the field is at
similar depth and the projection is near-affine, so perpendicular px ~ true lateral.

Tests, per PHILOSOPHY 4b:
 (a) DYNAMICS — within/between ratio + lag-1 autocorr, on the straight segment.
 (b) RP-CONSISTENCY — does the straight-segment y_lat ordering track the official RP?
 (c) COMMENTARY — the sharp stewards calls: #12 rail leader tightest? #3 winner shifts OUT
     near the finish? #6/#2/#7 wide? #8 inside?

No model/lib edits; reads the parquet + official HTML only.

    PYTHONHASHSEED=0 uv run --group vision python studies/cv_s0d_aerial_render_verdict.py
"""
from __future__ import annotations

import html
import re

import numpy as np
import polars as pl

PANEL = "data_cache/cv_s0d_aerial_render.parquet"
RESULTS_HTML = "data_cache/race_official/20260603R1_results.html"
DIST = 1650.0

COMMENTARY = {
    12: ("rail",       "rp 1-1-1-2 WIRE-TO-WIRE LEADER (rail/inside) -> expect TIGHT/low y_lat"),
    3:  ("late-wide",  "rp 7-7-6-1 WINNER: 150M shifted OUTSIDE #12 -> expect LATE outward move"),
    6:  ("early-wide", "rp 11-11-10-3 caught WIDE early -> expect wide"),
    2:  ("early-wide", "rp 9-9-9-4 WIDE barrier, shifted across early -> expect wide early"),
    7:  ("early-wide", "rp 12-12-11-9 WIDE barrier, shifted across early -> expect wide early"),
    8:  ("inside",     "rp 6-2-2-6 improved to INSIDE of #10, raced tight -> expect tightish"),
}


def parse_official(path: str) -> dict:
    t = open(path, encoding="utf-8", errors="ignore").read()
    rest = t[t.find("RACE 1 (745)"):]
    out: dict = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", rest, re.S):
        cc = [re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html.unescape(x))).strip()
              for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cc) == 12 and re.match(r"^\d+$", cc[0]) and re.match(r"^\d+$", cc[1]):
            out[int(cc[1])] = dict(pla=int(cc[0]),
                                   rp=[int(x) for x in cc[9].split() if x.isdigit()])
    return out


def ac1(y):
    y = np.asarray(y, float)
    if len(y) < 4 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(y[:-1], y[1:])[0, 1])


def corr(a, b):
    a, b = np.array(a, float), np.array(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or a[m].std() < 1e-9 or b[m].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def segment_table(df: pl.DataFrame, official: dict, lo: float, hi: float, ycol="y_lat"):
    rows = []
    for r in sorted(df["runner"].unique().to_list()):
        if r not in official:
            continue
        s = df.filter((pl.col("runner") == r) & pl.col("x_arclen").is_finite()
                      & (pl.col("x_arclen") >= lo) & (pl.col("x_arclen") < hi)).sort("x_arclen")
        if s.height < 4:
            continue
        y = s[ycol].to_numpy()
        rows.append(dict(no=r, n=s.height, ymed=float(np.median(y)), ystd=float(np.std(y)),
                         ac1=ac1(y), rp_mean=float(np.mean(official[r]["rp"])),
                         place=official[r]["pla"]))
    return rows


def main() -> None:
    official = parse_official(RESULTS_HTML)
    df = pl.read_parquet(PANEL)
    print("=" * 80)
    print("S0d AERIAL #82 — FAITHFULNESS VERDICT analysis")
    print("=" * 80)
    fin = df.filter(pl.col("x_arclen").is_finite())
    print(f"rows {df.height}  arclen-finite {fin.height}  runners {sorted(df['runner'].unique().to_list())}")
    print(f"arclen range [{fin['x_arclen'].min():.0f}, {fin['x_arclen'].max():.0f}] m")

    for name, lo, hi in [("WHOLE RACE", 0.0, DIST + 1),
                         ("RUN-IN STRAIGHT (>=1350m, near-affine)", 1350.0, DIST + 1),
                         ("EARLY (<900m)", 0.0, 900.0)]:
        rows = segment_table(df, official, lo, hi)
        if len(rows) < 4:
            print(f"\n[{name}] too few runners ({len(rows)})")
            continue
        within = float(np.median([r["ystd"] for r in rows]))
        between = float(np.std([r["ymed"] for r in rows]))
        acm = float(np.nanmedian([r["ac1"] for r in rows]))
        cm = corr([r["ymed"] for r in rows], [r["rp_mean"] for r in rows])
        print(f"\n[{name}]  n={len(rows)}  within {within:.2f}m  between {between:.2f}m  "
              f"ratio {within / (between + 1e-9):.2f}  ac1_med {acm:+.2f}  "
              f"corr(y_med,rp) {cm:+.2f}")
        for r in sorted(rows, key=lambda x: x["rp_mean"]):
            tag = COMMENTARY.get(r["no"], ("", ""))[0]
            print(f"    #{r['no']:>2} rp {r['rp_mean']:4.1f} pl {r['place']:>2} | "
                  f"y_med {r['ymed']:5.1f}m std {r['ystd']:4.1f} ac1 {r['ac1']:+.2f} "
                  f"n={r['n']:>3}  {tag}")

    # sharp commentary calls on the run-in straight + the #3 late-out move on the full race
    print("\n[COMMENTARY — sharp calls]")
    for no, (tag, txt) in COMMENTARY.items():
        s = df.filter((pl.col("runner") == no) & pl.col("x_arclen").is_finite()).sort("x_arclen")
        if not s.height:
            print(f"  #{no:>2} [{tag:10s}] NOT BOUND | {txt}")
            continue
        e = s.filter(pl.col("x_arclen") < 0.5 * DIST)["y_lat"].to_numpy()
        m = s.filter((pl.col("x_arclen") >= 0.5 * DIST) & (pl.col("x_arclen") < 0.85 * DIST))["y_lat"].to_numpy()
        late = s.filter(pl.col("x_arclen") >= 0.85 * DIST)["y_lat"].to_numpy()
        def md(a):
            return float(np.median(a)) if len(a) else float("nan")
        print(f"  #{no:>2} [{tag:10s}] early {md(e):5.1f} mid {md(m):5.1f} late {md(late):5.1f} m | {txt}")


if __name__ == "__main__":
    main()
