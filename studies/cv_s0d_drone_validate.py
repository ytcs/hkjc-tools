"""S0d DRONE validation — scale check + number-bridge check + the y_lat lateral-trip finding.

Validates the NUMBERED, scale-fixed drone reconstruction (``cv_s0d_drone.py`` parquet) against the
official ST 2026-06-03 R1 result, NON-CIRCULARLY: the numbers come from the finish-zoom OCR bridge,
not from the official chart. Reports, per §4b honestly:

  SCALE   — does per-runner arc-length reach ~1650 m? is y_lat within [0, 21.3] m?
  BRIDGE  — which runners got numbered; do the numbered finish/running orders track official?
  TRIP    — the rail-relative y_lat per numbered runner (rail-saved vs wide) vs the official RP
            trajectory (rail-hugger #12 should read tight; wide closers should read wider), and
            WHICH race segment the drone reconstructs trustworthily (top-down back-straight vs the
            gate-load / crowd-clutter starts and the run-in).

    uv run --group vision python studies/cv_s0d_drone_validate.py
"""
from __future__ import annotations

import html
import re

import numpy as np
import polars as pl

OFFICIAL = "data_cache/race_official/20260603R1_results.html"
DRONE = "data_cache/cv_s0d_drone.parquet"
DIST = 1650.0
TRACK_W = 21.3


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


def main() -> None:
    official = parse_official(OFFICIAL)
    off_winner = next(n for n in official if official[n]["pla"] == 1)
    d = pl.read_parquet(DRONE)
    numbered = d.filter(pl.col("number").is_not_null())
    runners = sorted(d["runner"].unique().to_list())

    print("=" * 80)
    print('S0d DRONE VALIDATION — ST 2026-06-03 R1, 1650M "C" course (running width 21.3 m)')
    print("=" * 80)
    print(f"OFFICIAL winner #{off_winner} {official[off_winner]['horse']} "
          f"(rp {official[off_winner]['rp']}); #12 wire-to-wire {official[12]['rp']}; "
          f"#6 closer {official[6]['rp']}")

    # ---- SCALE ----------------------------------------------------------------------------
    print("\n[SCALE]")
    arc_max = {r: float(d.filter(pl.col("runner") == r)["x_arclen"].max()) for r in runners}
    arcs = np.array(list(arc_max.values()))
    print(f"  per-track arc-length max: median {np.median(arcs):.0f} m, "
          f"range [{arcs.min():.0f}, {arcs.max():.0f}] m  (target ~{DIST:.0f} m)")
    ylat = d["y_lat"].to_numpy()
    print(f"  y_lat: min {ylat.min():.2f}, max {ylat.max():.2f}, median {np.median(ylat):.2f} m "
          f"(must be in [0, {TRACK_W}])  -> in-bounds: {bool(ylat.max() <= TRACK_W + 1e-6)}")
    # honest read: the drone tracks only the SEGMENT it sees top-down, not the whole 1650 m
    print("  NOTE: the drone tracks only the segment(s) the camera holds top-down; arc-length is")
    print("  the reconstructed running length of THAT segment, not the full 1650 m race.")

    # ---- BRIDGE ---------------------------------------------------------------------------
    print("\n[NUMBER BRIDGE — non-circular finish-zoom OCR]")
    bind = (numbered.group_by("runner").agg(pl.col("number").first()).sort("runner"))
    nums = sorted(numbered["number"].unique().to_list())
    print(f"  numbered tracks: {len(bind)} / {len(runners)}   numbers bound: {nums}")
    dup = [n for n in nums if numbered.filter(pl.col("number") == n)["runner"].n_unique() > 1]
    print(f"  duplicate-number tracks (bind collision): {dup or 'none'}")

    # ---- TRIP: y_lat per numbered runner, on the trustworthy (top-down) segment -----------
    # Trust window: the high near-top-down back-straight / bend where BOTH rails calibrate cleanly.
    print("\n[LATERAL TRIP — y_lat median per numbered runner vs official RP]")
    seg = numbered.filter((pl.col("t") >= 35) & (pl.col("t") <= 95))
    if len(seg) == 0:
        seg = numbered
    print("  (trust segment t in [35,95]s: high near-top-down back-straight/bend; "
          "gate-load t<10s and run-in clutter EXCLUDED)")
    print(f"  {'#':>3} {'rp (1=lead)':>16} {'finish':>6} {'y_lat_med':>9} {'n_obs':>6}  read")
    trip = {}
    for n in nums:
        s = seg.filter(pl.col("number") == n)
        if len(s) == 0:
            continue
        ym = float(s["y_lat"].median())
        trip[n] = ym
    # rank runners rail->wide and compare to official rp-leader (rail) tendency
    for n in sorted(trip, key=lambda n: trip[n]):
        off = official.get(n, {})
        rp = off.get("rp", [])
        pla = off.get("pla", "?")
        ns = int(seg.filter(pl.col("number") == n).height)
        read = "RAIL/tight" if trip[n] < np.median(list(trip.values())) else "WIDE"
        print(f"  {n:>3} {str(rp):>16} {str(pla):>6} {trip[n]:>9.1f} {ns:>6}  {read}")

    # correlation: does y_lat (lateral) relate to the official running position (longitudinal)?
    # honest: rail-hugging is a CHOICE not a rank, so we report the #12-rail anchor explicitly.
    if 12 in trip:
        rail_rank = sorted(trip, key=lambda n: trip[n]).index(12) + 1
        print(f"\n  #12 (wire-to-wire rail leader) y_lat rank rail->wide: {rail_rank}/{len(trip)} "
              f"(y_lat {trip[12]:.1f} m) — expected NEAR-RAIL (low rank) if the trip is real.")
    if 6 in trip:
        print(f"  #6 (closer from near-last 11-11-10-3) y_lat {trip[6]:.1f} m.")

    print("\n[HONEST SEGMENT RELIABILITY]")
    tcov = numbered.group_by("number").agg(
        pl.col("t").min().alias("t0"), pl.col("t").max().alias("t1"), pl.len().alias("n"))
    print("  per-number temporal coverage (t0..t1, n_obs):")
    for row in tcov.sort("number").iter_rows(named=True):
        print(f"    #{int(row['number']):>2}: t {row['t0']:.0f}-{row['t1']:.0f}s  n={row['n']}")


if __name__ == "__main__":
    main()
