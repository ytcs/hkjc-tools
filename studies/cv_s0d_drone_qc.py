"""S0d DRONE QC FIGURE — eyeball review of the numbered, scale-fixed real-horse reconstruction.

Three panels on ST 2026-06-03 R1 (official winner #3 7->1, #12 wire-to-wire, #6 closer):

  A. arc-length(t) per NUMBERED track — the scale fix: tracked-segment running length lands in a
     physical range (vs the old ~3-4x inflation), with the trustworthy top-down window shaded.
  B. the LATERAL TRIP — y_lat (rail-relative offset, 0 = inner rail) per numbered runner on the
     trust segment, ordered rail->wide, annotated with the official running position. This is the
     NOVEL signal: ground saved/lost, invisible to the chart/form.
  C. the SILK->NUMBER bridge — cluster silk-LAB (a*,b*) scatter, each point the bound number, the
     finish-zoom OCR read that makes the drone non-anonymous.

Writes data_cache/figs/cv_s0d_drone_qc.png. Disposable scratch.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

OUT = Path("data_cache/figs/cv_s0d_drone_qc.png")
DRONE = "data_cache/cv_s0d_drone.parquet"
OFFICIAL = "data_cache/race_official/20260603R1_results.html"
DIST = 1650.0
TRACK_W = 21.3
TRUST = (35.0, 95.0)


def parse_official(path: str) -> dict:
    t = open(path, encoding="utf-8", errors="ignore").read()
    rest = t[t.find("RACE 1 (745)"):]
    out: dict = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", rest, re.S):
        cc = [re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html.unescape(x))).strip()
              for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cc) == 12 and re.match(r"^\d+$", cc[0]) and re.match(r"^\d+$", cc[1]):
            out[int(cc[1])] = dict(pla=int(cc[0]), rp=[int(x) for x in cc[9].split() if x.isdigit()])
    return out


def main() -> None:
    official = parse_official(OFFICIAL)
    d = pl.read_parquet(DRONE)
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    numd = d.filter(pl.col("number").is_not_null())
    nums = sorted(numd["number"].unique().to_list())

    # ---- A: arc-length(t) per numbered track ----------------------------------------------
    ax = axes[0]
    for n in nums:
        s = numd.filter(pl.col("number") == n).sort("t")
        ax.plot(s["t"].to_numpy(), s["x_arclen"].to_numpy(),
                color=cmap(int(n) % 20), lw=1.6, alpha=0.85, label=f"#{n}")
    ax.axvspan(*TRUST, color="0.85", zorder=0, label="trust window")
    ax.set_xlabel("race time t (s)")
    ax.set_ylabel("CV arc-length x (m) — scale-fixed")
    amax = float(numd.group_by("runner").agg(pl.col("x_arclen").max())["x_arclen"].median())
    ax.set_title(f"A · DRONE arc-length(t), {len(nums)} NUMBERED tracks\n"
                 f"per-track median reach {amax:.0f} m (no ~3-4x inflation); "
                 "tracks the SEGMENT held top-down", fontsize=10)
    ax.legend(ncol=3, fontsize=6, loc="upper left")
    ax.grid(alpha=0.25)

    # ---- B: lateral trip y_lat per numbered runner (rail->wide) ----------------------------
    ax = axes[1]
    seg = numd.filter((pl.col("t") >= TRUST[0]) & (pl.col("t") <= TRUST[1]))
    if len(seg) == 0:
        seg = numd
    trip = {}
    for n in nums:
        s = seg.filter(pl.col("number") == n)
        if len(s):
            trip[n] = (float(s["y_lat"].median()),
                       float(s["y_lat"].quantile(0.25)), float(s["y_lat"].quantile(0.75)))
    order = sorted(trip, key=lambda n: trip[n][0])
    ys = np.arange(len(order))
    meds = [trip[n][0] for n in order]
    los = [trip[n][0] - trip[n][1] for n in order]
    his = [trip[n][2] - trip[n][0] for n in order]
    ax.errorbar(meds, ys, xerr=[los, his], fmt="o", color="C0", capsize=3)
    for y, n in zip(ys, order):
        rp = official.get(n, {}).get("rp", [])
        pla = official.get(n, {}).get("pla", "?")
        ax.annotate(f"  #{n}  rp{rp} fin{pla}", (trip[n][0], y), fontsize=8, va="center")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(TRACK_W, color="r", ls="--", lw=1)
    ax.text(TRACK_W, -0.6, "outer rail 21.3 m", color="r", fontsize=8, ha="right")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"#{n}" for n in order])
    ax.set_xlim(-0.5, TRACK_W + 1)
    ax.set_xlabel("y_lat: rail-relative offset (m), 0 = inner rail")
    ax.set_title("B · LATERAL TRIP (the novel signal)\nrail-saved (low) -> wide (high); "
                 "annotated with official running position", fontsize=10)
    ax.grid(alpha=0.25, axis="x")

    # ---- C: the silk->number bridge --------------------------------------------------------
    ax = axes[2]
    for n in nums:
        s = numd.filter(pl.col("number") == n)
        a = float(s["silk_lab_a"].median())
        b = float(s["silk_lab_b"].median())
        ax.scatter([a], [b], s=90, color=cmap(int(n) % 20), edgecolor="k", zorder=3)
        ax.annotate(f"#{n}", (a, b), fontsize=9, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(128, color="0.7", lw=0.8)
    ax.axvline(128, color="0.7", lw=0.8)
    ax.set_xlabel("silk LAB a*  (green<->red)")
    ax.set_ylabel("silk LAB b*  (blue<->yellow)")
    ax.set_title(f"C · SILK->NUMBER bridge ({len(nums)} bound)\n"
                 "finish-zoom OCR makes the drone NON-anonymous (non-circular)", fontsize=10)
    ax.grid(alpha=0.25)

    fig.suptitle("S0d · DRONE real-horse reconstruction (PRIMARY lateral signal) — "
                 "ST 2026-06-03 R1 (#3 7->1, #12 wire-to-wire, #6 closer): "
                 "scale-fixed + numbered + lateral trip", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125)
    print(f"wrote {OUT}")
    print(f"numbered: {nums}")
    print("trip rail->wide:", [(n, round(trip[n][0], 1)) for n in order])


if __name__ == "__main__":
    main()
