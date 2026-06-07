"""S0d — read the AERIAL legend running-order at each odometer pole and score it against the
official running-position sequence. The legend lists the running order DIRECTLY every frame, so
it is the aerial view's strongest longitudinal signal (stronger than arc-length timing, which is
field-shared via the odometer). Cheap: OCRs only the legend + odometer at a coarse cadence.

    uv run --group vision python studies/cv_s0d_legend_poles.py
"""
from __future__ import annotations

import html
import re

VIDEO = "data_cache/video/aerial_20260603_01.mp4"
OFFICIAL = "data_cache/race_official/20260603R1_results.html"
DIST = 1650.0


def parse_official(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    rest = t[t.find("RACE 1 (745)"):]
    out = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", rest, re.S):
        cc = [re.sub(r"\s+", " ", re.sub("<[^>]+>", "", html.unescape(x))).strip()
              for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cc) == 12 and re.match(r"^\d+$", cc[0]) and re.match(r"^\d+$", cc[1]):
            out[int(cc[1])] = dict(pla=int(cc[0]),
                                   rp=[int(x) for x in cc[9].split() if x.isdigit()])
    return out


def main():
    import av
    import numpy as np

    from hkjc.vision.mode_b.layout import find_layout
    from hkjc.vision.mode_b.ocr import read_legend, read_odometer

    official = parse_official(OFFICIAL)
    valid = frozenset(range(1, 13))
    # capture legend running order keyed by distance-to-go, at a coarse cadence
    by_d2go = {}
    c = av.open(VIDEO)
    na = 0
    for i, fr in enumerate(c.decode(video=0)):
        if i % 15:
            continue
        img = fr.to_ndarray(format="bgr24")
        L = find_layout(img)
        if not L.aerial_active:
            continue
        na += 1
        reads = read_legend(img[L.legend_y0:L.legend_y1, L.panel_x0:L.panel_x1], valid_set=valid)
        order, seen = [], set()
        for r in reads:
            if r.number not in seen:
                order.append(r.number)
                seen.add(r.number)
        oy0, oy1, ox0, ox1 = L.odo_roi
        d2go = read_odometer(img[oy0:oy1, ox0:ox1], max_m=int(DIST + 200))
        if order and d2go is not None:
            by_d2go[round(d2go)] = order
    c.close()
    print(f"captured {len(by_d2go)} legend frames over {na} aerial frames")

    # official RP is sampled at 4 poles; HKJC 1650m poles ~ at d-to-go 1200, 800, 400, finish.
    # we map each official pole-index to the nearest captured distance-to-go.
    poles_d2go = [1250, 850, 450, 80]
    print("\nLEGEND running order vs OFFICIAL running position, by pole:")
    corrs = []
    for p, pd in enumerate(poles_d2go):
        near = min(by_d2go, key=lambda d: abs(d - pd)) if by_d2go else None
        if near is None:
            continue
        leg = by_d2go[near]
        # official order at this pole: runners sorted by rp[p]
        cm = [n for n in leg if n in official and len(official[n]["rp"]) > p]
        if len(cm) < 3:
            continue
        cv_rank = [leg.index(n) for n in cm]                  # 0 = legend leader
        off_rank = [official[n]["rp"][p] for n in cm]         # 1 = official leader
        c_ = np.corrcoef(_rank(cv_rank), _rank(off_rank))[0, 1]
        corrs.append(float(c_))
        print(f"  pole {p+1} (d2go~{near}m): legend {leg}")
        print("           official rp:   " + str({n: official[n]["rp"][p] for n in cm}))
        print(f"           Spearman={c_:+.2f}")
    if corrs:
        print(f"\nMEAN legend-vs-official running-position Spearman: {np.mean(corrs):+.2f}")


def _rank(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


if __name__ == "__main__":
    main()
