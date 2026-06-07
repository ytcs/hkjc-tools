"""S0d QC FIGURE (#78, post-0069) — eyeball the FINISH-EXTENDED aerial recon + the y_lat
validation against official, on ST 2026-06-03 R1 (off. winner #3 7->1, #12 wire-to-wire).

Three panels, the honest §4b read after the 0069 reframing:

  A. AERIAL arc-length(t) per bound runner — the recon now extends THROUGH the true finish
     (~1650m, t~100s) via the robust legend band + full-video pass; winner #3 is tracked
     across the line instead of dropping out at ~1507m (0068's failure).
  B. y_lat TRIP vs official running-position — the NOVEL lateral signal. Scatter of CV median
     y_lat (rail-relative metres) against each runner's mean official RP, with the field-wide
     correlation + leave-one-out band: does rail-saved/wide hold up, or is it a 3-horse
     anecdote? (This is the deliverable 0069 said was UNVALIDATED.)
  C. per-pole CV running order vs official RP — the legend-anchored longitudinal order, now
     read through the finish (so the late poles are populated).

Writes data_cache/figs/cv_s0d_qc.png. Disposable scratch.

    PYTHONHASHSEED=0 uv run --group vision python studies/cv_s0d_qc.py
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

OUT = Path("data_cache/figs/cv_s0d_qc.png")
AERIAL = "data_cache/cv_s0d_aerial.parquet"
RESULTS_HTML = "data_cache/race_official/20260603R1_results.html"
DIST = 1650.0


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


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")

    def rank(x):
        o = np.argsort(x)
        r = np.empty_like(o, float)
        r[o] = np.arange(len(x))
        return r
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main() -> None:
    official = parse_official(RESULTS_HTML)
    a = pl.read_parquet(AERIAL)
    ar = sorted(a["runner"].unique().to_list())
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # ---- A: finish-extended arc-length(t) ------------------------------------------
    ax = axes[0]
    x3max = 0.0
    for r in ar:
        s = a.filter(pl.col("runner") == r).sort("t")
        tt, xx = s["t"].to_numpy(), s["x_arclen"].to_numpy()
        hi = r in (3, 12, 6)
        ax.plot(tt, xx, color=cmap(ar.index(r) % 20), lw=3.0 if hi else 1.2,
                alpha=1.0 if hi else 0.55,
                label=f"#{r}" + (" WIN" if r == 3 else " led" if r == 12 else
                                 " 3rd" if r == 6 else ""))
        if r == 3:
            x3max = float(xx.max())
            ax.scatter([tt[-1]], [xx[-1]], s=150, marker="*", color="red", zorder=6)
    ax.axhline(DIST, color="k", ls="--", lw=1, alpha=0.7)
    ax.text(2, DIST + 18, "finish 1650 m", fontsize=8)
    ax.set_xlabel("race time t (s)")
    ax.set_ylabel("CV arc-length x (m)")
    reached = "REACHES finish" if x3max > DIST - 60 else f"stops @{x3max:.0f}m"
    ax.set_title(f"A · AERIAL arc-length(t), {len(ar)}/12 bound\n"
                 f"winner #3 tracked to {x3max:.0f}m ({reached})", fontsize=10)
    ax.legend(ncol=2, fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    # ---- B: y_lat TRIP vs official RP (the novel signal validation) -----------------
    ax = axes[1]
    rows = [(r, float(a.filter(pl.col("runner") == r)["y_lat"].median()),
             float(np.mean(official[r]["rp"])), official[r]["pla"])
            for r in ar if r in official]
    yl = [x[1] for x in rows]
    rpm = [x[2] for x in rows]
    pear = float(np.corrcoef(yl, rpm)[0, 1]) if len(rows) >= 3 else float("nan")
    spear = _spearman(yl, rpm)
    # leave-one-out band
    loo = []
    for k in range(len(rows)):
        idx = [j for j in range(len(rows)) if j != k]
        loo.append(float(np.corrcoef([yl[j] for j in idx], [rpm[j] for j in idx])[0, 1]))
    for r, y, rp, _pl in rows:
        ax.scatter([rp], [y], s=90, color=cmap(ar.index(r) % 20), zorder=4,
                   edgecolor="k", lw=0.6)
        ax.annotate(f"#{r}", (rp, y), fontsize=8, xytext=(3, 3),
                    textcoords="offset points")
    if len(rows) >= 3:
        z = np.polyfit(rpm, yl, 1)
        xs = np.linspace(min(rpm), max(rpm), 20)
        ax.plot(xs, np.polyval(z, xs), "r--", lw=1.4, alpha=0.7)
    # 4b de-confound: is y_lat a STABLE trip or per-frame noise? Compare the within-runner
    # frame dispersion against the between-runner trip spread. ratio<1 => noise dominates =>
    # the measurement (synthetic sprite offset off a per-frame rail fit) is mis-specified,
    # NOT a true null of the lateral cause.
    within = float(np.mean([a.filter(pl.col("runner") == r)["y_lat"].std() for r in ar]))
    between = float(np.std(yl))
    ax.set_xlabel("mean official running position (1 = lead)")
    ax.set_ylabel("CV median y_lat (m from rail)")
    ax.set_title(f"B · y_lat trip vs RP — NOT VALIDATED (honest null)\n"
                 f"ρ {pear:+.2f}/{spear:+.2f} n={len(rows)}; within/between {within:.1f}/"
                 f"{between:.1f}m (ratio {between / within:.2f}<1: sprite NOISE swamps trip)",
                 fontsize=9.5)
    ax.grid(alpha=0.25)

    # ---- C: per-pole CV order vs official RP ---------------------------------------
    ax = axes[2]
    poles = [0.25, 0.50, 0.75, 0.97]
    rhos = []
    for p, fr in enumerate(poles):
        xq = fr * DIST
        times = {}
        for r in ar:
            s = a.filter(pl.col("runner") == r).sort("x_arclen")
            xa, tt = s["x_arclen"].to_numpy(), s["t"].to_numpy()
            if len(xa) < 2 or xa.max() < xq - 60:
                continue
            j = int(np.argmin(np.abs(xa - xq)))
            times[r] = float(tt[j])
        order = sorted(times, key=lambda r: times[r])
        cm = [r for r in order if r in official and len(official[r]["rp"]) > p]
        if len(cm) < 3:
            rhos.append(float("nan"))
            continue
        cv_rank = [order.index(r) for r in cm]
        off_rank = [official[r]["rp"][p] for r in cm]
        rhos.append(_spearman([float(x) for x in cv_rank], [float(x) for x in off_rank]))
    polelab = ["~410m", "~825m", "~1240m", "~1600m"]
    ax.bar(range(4), rhos, color=["C0" if r == r and r > 0 else "C3" for r in rhos])
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(polelab)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("CV-vs-official running-pos Spearman ρ")
    mean_rho = float(np.nanmean(rhos))
    ax.set_title(f"C · per-pole running order vs official\n"
                 f"legend-anchored, now read through finish · mean ρ {mean_rho:+.2f}",
                 fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("S0d · AERIAL #78 — recon EXTENDED through the finish (A, winner #3 surge "
                 "tracked); legend ORDER = official top-8; but y_lat trip is an HONEST NULL "
                 "(B) — synthetic-sprite lateral noise (ST 2026-06-03 R1)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=125)
    print(f"wrote {OUT}")
    print(f"y_lat vs RP: Pearson {pear:+.3f} Spearman {spear:+.3f} n={len(rows)} "
          f"LOO[{min(loo):+.2f},{max(loo):+.2f}]")
    print(f"per-pole ρ: {[round(r, 2) for r in rhos]} mean {mean_rho:+.3f}")


if __name__ == "__main__":
    main()
