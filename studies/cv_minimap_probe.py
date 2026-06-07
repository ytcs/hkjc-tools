"""TASK #85 — AERIAL MINIMAP probe: is the AVV top-down inset a clean, foreshortening-free,
number-identified per-horse LATERAL source?

CONTEXT (journals 0073/0075/0077). The Aerial Virtual View (data_cache/video/aerial_20260603_01.mp4,
ST 2026-06-03 R1) is a SPLIT SCREEN: left = real broadcast camera, right = the deterministic 3D
render (the render-inversion lateral, #82/0075). The render's lateral is FAITHFUL on the body but
DEGRADES on the bend/run-in (foreshortening, 0075/0077) — exactly the overlap window where the only
aerial<->patrol cross-feed test could be run, so 0077's cross-feed coherence came back null on
GEOMETRY. The minimap inset (flagged unexploited since 0073, never tested) was hypothesised to be a
(near-)orthographic 2nd view: foreshortening-FREE and in the same frame as the legend (number-
identified). IF it resolved 12 per-horse lateral positions it would be a CLEAN whole-race lateral and
could become the fusion backbone.

STEP A — SCRUTINIZE FIRST (this file). Locate the inset, crop+upscale across the race, and answer
concretely whether it carries PER-HORSE lateral. This is cheap and decides everything.

FINDING (see journal): the top-right inset is NOT a per-horse positional minimap. It is a
PROGRESS-INDICATOR — a small (~136x66 px) schematic of the FULL TRACK LOOP with:
  * a "<NNN>M" distance-to-finish caption (yellow),
  * the COVERED arc drawn RED and the REMAINING arc drawn GREEN (red shrinks monotonically as the
    race runs: red-px 911 -> 0 from t=20s to t=100s),
  * a SINGLE yellow dot marking the field's current along-track position on the loop.
There is ONE field-position marker, not 12 horse markers; no lateral (rail-offset) axis at all; the
loop is an along-track progress bar shaped like the course. So STEP B/C (per-horse minimap lateral)
are MOOT — the data does not exist in this inset. A clean NEGATIVE that closes the minimap option.

Study-only. Reads the video + the existing aerial render panel (READ-ONLY) for the figure context.
src/hkjc untouched. Official data is VALIDATION-ONLY and not used here (STEP A needs none).

    PYTHONHASHSEED=0 uv run --group vision python studies/cv_minimap_probe.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

VID = "data_cache/video/aerial_20260603_01.mp4"
OUTDIR = Path("/tmp/minimap_probe")
FIG = "data_cache/figs/cv_minimap_probe_qc.png"

# Inset panel bbox in the 1920x1080 frame (located via the dark-green panel rectangle; stable).
INSET = (1620, 292, 1760, 360)  # x0,y0,x1,y1  (~140x68 px)
# The loop interior (where a per-horse marker WOULD live), past the left-edge distance text.
LOOP = (1690, 295, 1820, 360)

# Frames spanning the race (the inset only renders once the race is underway, ~t>=15s).
SAMPLE_TS = [20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def grab(ts: int) -> np.ndarray | None:
    out = OUTDIR / f"_grab_{ts}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(ts), "-i", VID, "-frames:v", "1", "-q:v", "2",
         str(out), "-y"],
        check=True,
    )
    return cv2.imread(str(out))


def red_remaining_px(panel_bgr: np.ndarray) -> int:
    """Pixels of the RED 'covered' arc in the inset — a monotone progress proxy."""
    hsv = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2HSV)
    rm = (((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 110))
    return int(rm.sum())


def count_position_markers(loop_bgr: np.ndarray) -> int:
    """How many DISTINCT bright-yellow position dots inside the loop? (per-horse would be ~12).

    The yellow dot is small (~3-12 px). We mask bright yellow, drop the green track-outline that
    leaks in, and count connected components in the dot size band.
    """
    hsv = cv2.cvtColor(loop_bgr, cv2.COLOR_BGR2HSV)
    ym = ((hsv[:, :, 0] > 22) & (hsv[:, :, 0] < 36) & (hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 170)).astype(np.uint8)
    ym = cv2.morphologyEx(ym, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(ym, 8)
    # dot-sized blobs only (3..200 px); larger blobs are the green outline / text bleed
    return sum(1 for i in range(1, n) if 3 <= stats[i, 4] <= 200)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    Path("data_cache/figs").mkdir(parents=True, exist_ok=True)

    x0, y0, x1, y1 = INSET
    lx0, ly0, lx1, ly1 = LOOP
    rows = []
    crops = []
    for ts in SAMPLE_TS:
        img = grab(ts)
        if img is None:
            continue
        panel = img[y0:y1, x0:x1]
        loop = img[ly0:ly1, lx0:lx1]
        red = red_remaining_px(panel)
        nmark = count_position_markers(loop)
        rows.append((ts, red, nmark))
        # 6x nearest-neighbour upscale for the figure
        up = cv2.resize(panel, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        crops.append((ts, up))
        print(f"t={ts:>3}s  inset={panel.shape[1]}x{panel.shape[0]}px  "
              f"red_covered_px={red:>4}  position_dots_in_loop={nmark}")

    print(f"\nInset size: {x1 - x0}x{y1 - y0} px  "
          f"({100 * (x1 - x0) * (y1 - y0) / (1920 * 1080):.2f}% of frame)")
    reds = [r for _, r, _ in rows]
    print(f"RED-covered arc px over the race: {reds}  (monotone-decreasing => progress bar)")
    print("VERDICT: single field-position dot + covered/remaining loop = PROGRESS indicator, "
          "NOT per-horse lateral. STEP B/C moot.")

    _figure(crops, rows)


def _figure(crops, rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(crops)
    ncol = 5
    nrow = (n + ncol - 1) // ncol + 1
    fig = plt.figure(figsize=(16, 3.2 * nrow))
    for i, (ts, up) in enumerate(crops):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        ax.imshow(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
        ax.set_title(f"t={ts}s (6x)", fontsize=10)
        ax.axis("off")
    # progress-bar panel
    ax = fig.add_subplot(nrow, 1, nrow)
    ts = [r[0] for r in rows]
    red = [r[1] for r in rows]
    nmark = [r[2] for r in rows]
    ax.plot(ts, red, "o-", color="crimson", label="RED 'covered' arc px (progress)")
    ax2 = ax.twinx()
    ax2.plot(ts, nmark, "s--", color="goldenrod", label="position dots in loop")
    ax2.axhline(12, color="gray", ls=":", lw=1)
    ax2.text(ts[0], 12.3, "12 = per-horse would need this", color="gray", fontsize=8)
    ax.set_xlabel("race video t (s)")
    ax.set_ylabel("red covered-arc px", color="crimson")
    ax2.set_ylabel("# yellow position dots", color="goldenrod")
    ax.set_title("STEP A VERDICT: the inset is a PROGRESS INDICATOR (single field dot; red covered "
                 "arc shrinks monotonically) — NOT a per-horse lateral minimap", fontsize=10)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=110)
    print(f"[fig] {FIG}")


if __name__ == "__main__":
    main()
