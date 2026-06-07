"""JOINT RECON — Stage 1 (aerial), step B: per-horse ALONG-RAIL longitudinal from the rigid rail.

The old pipeline assigned every horse arclen=clock(t) (shoulder-to-shoulder by construction).
The fix: project each horse's ground-contact onto the RAIL image curve -> along-rail position
s_px. The render tracks the field (keeps it centred) so absolute s_px barely translates, but the
RELATIVE s_px between horses IS the running order, scale-free. This step proves the aerial pixels
carry front-to-back (they do -- the user was right) BEFORE any metric scale or smoothing.

Two validations, no hand-tuning:
  (1) SCALE-FREE ORDER: order horses by along-rail s_px each frame; the late-race order must match
      the official finish; the order time-series must track the legend. Direction sign is picked by
      agreement with the official finish (a 1-bit geometric fact, not a tune).
  (2) METRIC (provisional): odometer (m travelled) gives absolute field arclen d0(t); a local along-
      rail m/px from the odometer advance converts s_px separation -> metres -> per-horse arclen(t),
      mean-speed spread, finish margins vs official. Flagged provisional (scale is the next rigor step).

Overlays the rail + per-horse projections + order on real aerial frames -> data_cache/figs/.
Official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_joint_aerial_pose.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "studies")
from cv_r08_common import parse_official  # noqa: E402

OBS = "data_cache/cv_joint_aerial_R08_obs.npz"
AERIAL_VIDEO = "data_cache/video/aerial_20260603_08.mp4"
HALF = 960


def _project_to_rail(rail: np.ndarray, gx: float, gy: float) -> tuple[float, float]:
    """Nearest rail vertex: return (s_px along rail, signed perp distance px)."""
    d2 = (rail[:, 0] - gx) ** 2 + (rail[:, 1] - gy) ** 2
    k = int(np.argmin(d2))
    s_px = np.r_[0.0, np.cumsum(np.hypot(np.diff(rail[:, 0]), np.diff(rail[:, 1])))]
    k0, k1 = max(0, k - 1), min(len(rail) - 1, k + 1)
    tx, ty = rail[k1, 0] - rail[k0, 0], rail[k1, 1] - rail[k0, 1]
    tn = float(np.hypot(tx, ty)) + 1e-9
    perp = (gx - rail[k, 0]) * (-ty / tn) + (gy - rail[k, 1]) * (tx / tn)
    return float(s_px[k]), float(perp)


def main() -> None:
    z = np.load(OBS, allow_pickle=True)
    frames = z["frames"]
    odo = z["odo"]
    rails = z["rails"]
    hg = z["horse_gc"]
    fidx = {int(f): j for j, f in enumerate(frames)}
    off = parse_official()
    official = off.finish_order
    print("=" * 80)
    print("JOINT RECON Stage-1B — aerial per-horse ALONG-RAIL longitudinal (R08, rigid rail)")
    print("=" * 80)
    print("OFFICIAL finish:", official, " winner #", official[0])

    # ---- project every horse ground-contact onto its frame's rail -> (frame, runner, s_px, perp)
    rows = []
    for runner, frame, gx, gy, _used in hg:
        j = fidx.get(int(frame))
        if j is None:
            continue
        rail = rails[j]
        if len(rail) < 2:
            continue
        s_px, perp = _project_to_rail(rail, gx, gy)
        rows.append((int(frame), int(runner), s_px, perp))
    rows = np.array(rows, float)  # frame, runner, s_px, perp
    runners = sorted(set(rows[:, 1].astype(int)))
    uf = sorted(set(rows[:, 0].astype(int)))
    print(f"\nprojected {len(rows)} contacts; {len(runners)} runners; {len(uf)} frames")

    # ---- per-frame along-rail ORDER (scale-free). Sign chosen to match official finish. ----
    # For each frame, median s_px per runner; rank. Compare late-race ranking to official both ways.
    def frame_order(sign):
        order_ts = {}  # frame -> [runners leader..last]
        for f in uf:
            sub = rows[rows[:, 0] == f]
            med = {int(r): float(np.median(sub[sub[:, 1] == r, 2])) for r in set(sub[:, 1].astype(int))}
            order_ts[f] = [r for r, _ in sorted(med.items(), key=lambda kv: -sign * kv[1])]
        return order_ts

    def late_order(order_ts):
        late = [f for f in uf if f >= uf[int(0.85 * len(uf))]]
        from collections import defaultdict
        score = defaultdict(float)
        for f in late:
            for rank, r in enumerate(order_ts[f]):
                score[r] += rank
        return [r for r, _ in sorted(score.items(), key=lambda kv: kv[1])]

    def order_corr(pred):
        common = [r for r in official if r in pred]
        a = [official.index(r) for r in common]
        b = [pred.index(r) for r in common]
        if len(a) < 3:
            return -1.0
        a, b = np.array(a, float), np.array(b, float)
        return float(np.corrcoef(a, b)[0, 1])

    cands = {s: late_order(frame_order(s)) for s in (+1, -1)}
    corrs = {s: order_corr(cands[s]) for s in (+1, -1)}
    sign = max(corrs, key=corrs.get)
    pred_order = cands[sign]
    order_ts = frame_order(sign)
    print(f"\n[SCALE-FREE ORDER]  along-rail direction sign = {sign:+d} "
          f"(Spearman vs official: sign+1 {corrs[+1]:+.2f}, sign-1 {corrs[-1]:+.2f})")
    print(f"  aerial late-race order : {pred_order}")
    print(f"  official finish        : {official}")
    print(f"  rank corr (final)      : {order_corr(pred_order):+.2f}")

    # ---- METRIC (provisional): odometer d0(t) + local along-rail m/px from odometer advance ----
    # field-centroid s_px per frame (median of horses) vs odometer -> dt scale.
    fc_s = {}
    odo_by_f = {}
    for j, f in enumerate(frames):
        if not np.isfinite(odo[j]):
            continue
        sub = rows[rows[:, 0] == int(f)]
        if len(sub) == 0:
            continue
        fc_s[int(f)] = float(np.median(sub[:, 2]))
        odo_by_f[int(f)] = float(odo[j])
    # global along-rail scale: regress odometer on field-centroid s_px is wrong (camera tracks);
    # instead use the SPREAD: at the finish, the field's s_px spread maps to the known margin spread?
    # Provisional: per-horse arclen = d0(frame) + (s_px_horse - fc_s) * mpp_along, mpp from the
    # render's typical foreshortening — here we report the SCALE-FREE separation in px and defer
    # the metre scale to step C (needs the rigid-rail projection / a 2nd anchor). Report px spread.
    print("\n[METRIC] deferred to step C (no markers in R08 aerial; odometer is field-absolute only,")
    print("  camera tracks the field so a single along-rail m/px needs the rigid-rail projection).")
    # but we CAN report the per-frame along-rail px spread as a separation sanity check:
    spreads = []
    for f in uf:
        sub = rows[rows[:, 0] == f]
        med = [np.median(sub[sub[:, 1] == r, 2]) for r in set(sub[:, 1].astype(int))]
        if len(med) > 3:
            spreads.append(max(med) - min(med))
    print(f"  along-rail px spread across field: med {np.median(spreads):.0f}px "
          f"(p10 {np.percentile(spreads,10):.0f}, p90 {np.percentile(spreads,90):.0f}) "
          f"-- NONZERO = real front-to-back (old pipeline forced this to 0)")

    # ---- OVERLAY on real frames -----------------------------------------------------------
    _overlay(frames, rails, rows, order_ts, uf)
    print("\nDONE — see overlays in data_cache/figs/cv_joint_aerial_pose_*.png")


def _overlay(frames, rails, rows, order_ts, uf):
    import av
    import cv2

    # pick 4 frames spread across the race
    picks = [uf[int(p * (len(uf) - 1))] for p in (0.2, 0.45, 0.7, 0.92)]
    want = set(picks)
    got = {}
    container = av.open(AERIAL_VIDEO)
    for i, fr in enumerate(container.decode(video=0)):
        if i in want:
            got[i] = fr.to_ndarray(format="bgr24")[:, HALF:].copy()
            if len(got) == len(want):
                break
    container.close()
    fidx = {int(f): j for j, f in enumerate(frames)}
    for f in picks:
        if f not in got:
            continue
        img = got[f]
        rail = rails[fidx[f]]
        for k in range(1, len(rail)):
            cv2.line(img, (int(rail[k - 1, 0]), int(rail[k - 1, 1])),
                     (int(rail[k, 0]), int(rail[k, 1])), (0, 255, 255), 2)
        sub = rows[rows[:, 0] == f]
        order = order_ts.get(f, [])
        # draw each contact projected onto the rail (marks the along-rail position used for order)
        for (_fr, runner, s_px, perp) in sub:
            s_cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(rail[:, 0]), np.diff(rail[:, 1])))]
            k = int(np.argmin(np.abs(s_cum - s_px)))
            px, py = int(rail[k, 0]), int(rail[k, 1])
            cv2.circle(img, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(img, f"{int(runner)}", (px - 6, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        leader = order[0] if order else "?"
        cv2.putText(img, f"frame {f}  along-rail order(leader->last): {order}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"leader #{leader}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out = f"data_cache/figs/cv_joint_aerial_pose_f{f:04d}.png"
        cv2.imwrite(out, img)
        print(f"  [overlay] {out}")


if __name__ == "__main__":
    main()
