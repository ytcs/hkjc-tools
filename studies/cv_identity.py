"""UNIFIED, race-agnostic IDENTITY module — solve identity ONCE, propagate to all feeds (#92).

The #91 R03 generalization test (journal 0088) localised the dominant cross-feed weakness to
IDENTITY + OCR: each feed independently OCR'd saddlecloth numbers — slow (the aerial render's
stylised numerals barely OCR; the R1 cadence ran ~2.4hr and STALLED on R03) and fragile (patrol
silk->number PERMUTED inside the red/pink family -> the rail-vs-wide gap came out WRONG SIGN
-5.7m; drone fell to 5/12 with no race-agnostic bridge).

THE DESIGN — solve-once-then-propagate (replacing per-feed OCR):

  1. ANCHOR (numbers, once, from the crispest source). The RF/race-film side-on saddlecloths are
     razor-sharp real footage. We OCR number<->RF-track by MULTI-FRAME VOTING (RapidOCR over many
     finish-window frames, majority vote per number), PLUS the RF legend-strip NAMES->number LUT
     (0086 — names OCR cleanly even when the tiny number boxes do not), which also gives a per-horse
     running-ORDER over the WHOLE race (a rank(race_t) table). Target >=10/12 numbered on RF.

  2. SILK SIGNATURE per runner. Where number+silk are co-visible (RF finish-cam, and the patrol
     finish-cam), compute each runner's jockey-silk colour signature as the SATURATION-WEIGHTED HUE
     HISTOGRAM (the 0082 method that separated silks; a LAB/BGR median collapses them -> we do NOT
     use it). Store number->silk-signature.

  3. PROPAGATE to the other feeds (aerial render sprites, patrol head-on, drone). For each feed's
     anonymous tracks compute the same silk signature and assign runner numbers by ONE-TO-ONE
     Hungarian matching (circular hue-EMD) to the silk LUT — CROSS-CONSTRAINED BY THE SHARED CLOCK:
     a track's along-track ORDER/position at race_t must be consistent with that runner's RF-known
     rank at race_t. This clock/order constraint breaks the silk-degeneracy (the red/pink family)
     that PERMUTED R03's patrol identity and flipped the lateral sign. Identity is solved as
     silk-cost + order-consistency JOINTLY.

  4. The slow per-feed OCR is ELIMINATED (aerial especially): aerial identity now comes from
     silk-propagation + the legend strip, NOT re-OCR of stylised numerals.

DISCIPLINE: study-only. Imports existing studies READ-ONLY. Does NOT edit src/hkjc/{model,core,
vision,...}. Writes only NEW _identity-suffixed parquets. Official data is VALIDATION-ONLY (the
number<->silk map and the name->number LUT are fixed PRE-RACE facts, never outcomes).

CANDIDATE FOR PROMOTION: once stable across >=R01+R03(+R08), this module is the natural home of a
`hkjc.vision` identity service (anchor + silk-LUT + clock-order propagation). Flagged here, not yet
promoted.

    PYTHONHASHSEED=0 uv run --group vision python studies/cv_identity.py --race R03
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, "studies")

# silk hue-histogram primitives (0082) — reused READ-ONLY
from cv_patrol_identity import _hist_dist, _hue_hist  # noqa: E402

NBINS = 18  # hue histogram bins (10deg each over OpenCV 0..180 hue)

# ============================================================================= consensus weights
# THE SINGLE FIXED CONSENSUS SCHEME (#94). identity = a one-to-one (Hungarian) assignment
# minimising a weighted sum of FOUR independent identity-vote costs. These weights are FIXED —
# the SAME values are used on R01 AND R03 (no per-race tuning; per-race tuning would be overfitting).
# Each silk cost is normalised by its own median (scale-free); the order/draw costs are normalised
# |Delta-rank| in [0,1]. Rationale for the balance:
#   * OWN finish-cam LUT is the highest-fidelity per-track silk (R01's dedicated 102.5-123s LUT
#     earned +4.1m) -> the largest silk weight, but down-weighted per-runner by silk-DISTINCTNESS
#     (a degenerate red/pink silk contributes little, an unambiguous blue/yellow contributes fully).
#   * RF anchor LUT is the cross-feed shared backbone (lead-group biased, lower per-track fidelity)
#     -> a smaller, steady silk weight.
#   * the shared-clock ORDER constraint is gated by rank RELIABILITY (down to ~0 on bunched windows,
#     where 0089 showed it amplifies error) so it only helps where the RF rank is unbunched.
#   * the DRAW ORDER is RACE-INDEPENDENT and strong (a far better order proxy than head-on depth):
#     a steady mid weight, the magnitude anchor that breaks the silk degeneracy at the start.
W_OWN_SILK = 1.00     # feed's own finish-cam LUT silk-EMD (x per-runner distinctness)
W_RF_SILK = 0.45      # RF anchor LUT silk-EMD (cross-feed backbone)
W_ORDER = 0.60        # shared-clock along-track order inconsistency (x rank reliability)
W_DRAW = 0.70         # barrier-draw START-order inconsistency (race-independent)


# ============================================================================= silk descriptor
def silk_hist(hsv_pixels: np.ndarray) -> np.ndarray | None:
    """Sat-weighted hue histogram of an (N,3) HSV silk-pixel array (the 0082 descriptor)."""
    if hsv_pixels is None or hsv_pixels.size < 24:
        return None
    p = hsv_pixels.reshape(-1, 3)
    return _hue_hist(p[:, 0], p[:, 1], p[:, 2])


def hist_emd(a: np.ndarray, b: np.ndarray) -> float:
    """Circular hue-histogram EMD (NOT rotation-invariant — silks are absolute hues)."""
    return _hist_dist(a, b)


def _match_name(txt: str, name_lut: dict[str, int]) -> int | None:
    """Map an OCR'd legend english NAME to a runner number (exact, then best bigram Jaccard)."""
    q = re.sub(r"[^A-Z]", "", txt.upper())
    if len(q) < 5:
        return None
    if q in name_lut:
        return name_lut[q]

    def bigrams(s):
        return {s[i:i + 2] for i in range(len(s) - 1)}

    qb = bigrams(q)
    best, bj = None, 0.0
    for name, num in name_lut.items():
        nb = bigrams(name)
        j = len(qb & nb) / max(1, len(qb | nb))
        if j > bj:
            bj, best = j, num
    return best if bj >= 0.6 else None


def _decode_window(video: str, t0: float, t1: float, dt: float):
    """Yield (vid_t, bgr_frame) over [t0,t1) by SEQUENTIAL av decode (NOT cap.set random-seek).

    The #91 backlog (journal 0088, item 6) flagged cv2 cap.set seeking as pathologically slow on
    H.264 (it nearly blocked the R03 run). Sequential av decode + sampling at ~dt is the tractable
    path used by the aerial feed. We decode forward, emitting roughly one frame per dt seconds.
    """
    import av

    container = av.open(video)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 25.0
    next_emit = t0
    for i, frame in enumerate(container.decode(video=0)):
        t = i / fps
        if t < t0:
            continue
        if t >= t1:
            break
        if t + 1e-6 >= next_emit:
            yield t, frame.to_ndarray(format="bgr24")
            next_emit += dt
    container.close()


def _silk_patch_hsv(fr, cx, cy, dx=16, dy_up=40, dy_lo=6):
    """HSV pixels of the jockey-silk torso patch ABOVE an OCR'd saddlecloth number at (cx,cy)."""
    import cv2
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    y0 = max(0, int(cy - dy_up))
    y1 = max(0, int(cy - dy_lo))
    x0 = max(0, int(cx - dx))
    x1 = min(fr.shape[1], int(cx + dx))
    patch = hsv[y0:y1, x0:x1]
    return patch.reshape(-1, 3) if patch.size else None


# ============================================================================= the anchor
@dataclass
class Identity:
    """The solved-once identity: silk LUT + the RF rank(race_t) order constraint.

    v2 (#94): identity is a CONSENSUS of FOUR independent votes fused with FIXED weights
    (no per-race tuning). See ``consensus_assign``. The Identity carries the RF anchor LUT
    (``silk_lut``) + the RF rank backbone (``rank_seq``) + the barrier DRAW (``draw``); the
    feed's OWN finish-cam LUT is supplied per-call (it lives in the feed's own module).
    """

    silk_lut: dict[int, np.ndarray] = field(default_factory=dict)      # runner -> hue hist (RF anchor)
    silk_votes: dict[int, int] = field(default_factory=dict)           # runner -> # OCR votes
    rank_seq: list[tuple[float, list[int]]] = field(default_factory=list)  # (race_t, lead->back)
    draw: dict[int, int] = field(default_factory=dict)                 # runner -> barrier draw (1=rail)
    field_size: int = 12

    # ---- the order/clock constraint ----
    def rank_at(self, race_t: float) -> dict[int, int] | None:
        """RF-known running rank (1=lead) at the nearest legend timepoint to race_t."""
        if not self.rank_seq:
            return None
        _, order = min(self.rank_seq, key=lambda s: abs(s[0] - race_t))
        return {n: i + 1 for i, n in enumerate(order)}

    def known_numbers(self) -> list[int]:
        return sorted(self.silk_lut)

    # ---- caching (so the re-run harness need not redo the slow OCR) ----
    def save(self, path: str) -> None:
        np.savez(
            path,
            lut_nums=np.array(sorted(self.silk_lut), dtype=int),
            lut_hists=np.array([self.silk_lut[n] for n in sorted(self.silk_lut)]),
            votes_nums=np.array(sorted(self.silk_votes), dtype=int),
            votes_n=np.array([self.silk_votes[n] for n in sorted(self.silk_votes)], dtype=int),
            seq_t=np.array([t for t, _ in self.rank_seq], dtype=float),
            seq_orders=np.array([",".join(map(str, o)) for _, o in self.rank_seq], dtype=object),
            draw_nums=np.array(sorted(self.draw), dtype=int),
            draw_vals=np.array([self.draw[n] for n in sorted(self.draw)], dtype=int),
            field_size=self.field_size,
        )

    @classmethod
    def load(cls, path: str) -> Identity:
        z = np.load(path, allow_pickle=True)
        lut = {int(n): h for n, h in zip(z["lut_nums"], z["lut_hists"], strict=True)}
        votes = {int(n): int(v) for n, v in zip(z["votes_nums"], z["votes_n"], strict=True)}
        seq = [(float(t), [int(x) for x in o.split(",") if x])
               for t, o in zip(z["seq_t"], z["seq_orders"], strict=True)]
        draw = ({int(n): int(v) for n, v in zip(z["draw_nums"], z["draw_vals"], strict=True)}
                if "draw_nums" in z.files else {})
        return cls(silk_lut=lut, silk_votes=votes, rank_seq=seq, draw=draw,
                   field_size=int(z["field_size"]))

    # ---- the propagation ----
    def propagate(
        self,
        track_sig: dict[int, np.ndarray],
        track_order: dict[int, float],
        track_race_t: dict[int, float],
        lam_order: float = 0.6,
        conf_floor: float = 0.0,
    ) -> dict[int, tuple[int, float]]:
        """Assign each anonymous track a runner number by JOINT silk-EMD + order-consistency.

        track_sig[ti]      : the track's silk hue-histogram signature.
        track_order[ti]    : a SCALAR proxy for the track's along-track position at its
                             reference time (e.g. mean arclen, or -lateral-rank). Larger ==
                             further along / more advanced. Used only for RANK agreement, so
                             any monotone proxy works.
        track_race_t[ti]   : the track's reference race_t (to look up the RF rank).

        Returns {track: (runner, confidence)} via Hungarian on
            cost = silk_EMD(track, runner)  +  lam_order * order_inconsistency
        where order_inconsistency penalises assigning a track whose ALONG-TRACK rank (among the
        tracks) disagrees with the runner's RF-known rank at that race_t. This is the key fix:
        when two red/pink silks are EMD-degenerate, the clock/order term picks the assignment
        whose running-order matches the RF backbone, instead of permuting.
        """
        from scipy.optimize import linear_sum_assignment

        tis = [ti for ti in track_sig if track_sig[ti] is not None]
        nums = self.known_numbers()
        if not tis or not nums:
            return {}

        # 1) silk EMD cost, normalised to [0,1] by its own median (scale-free vs lam_order)
        silk = np.array([[hist_emd(track_sig[ti], self.silk_lut[n]) for n in nums] for ti in tis])
        silk_norm = silk / (np.median(silk) + 1e-9)

        # 2) order-consistency cost. Rank the tracks by their along-track proxy (1=most advanced);
        #    rank the runners by the RF backbone at the track's race_t; penalise |Δrank| (normalised
        #    by field size). Robust to missing RF ranks (fall back to 0 penalty for that cell).
        order_vals = np.array([track_order[ti] for ti in tis])
        # tracks ranked by along-track position (descending -> rank 1 = furthest along)
        trk_rank = {ti: r + 1 for r, ti in
                    enumerate(sorted(tis, key=lambda t: -track_order[t]))}
        ntrk = len(tis)
        ordc = np.zeros_like(silk_norm)
        for i, ti in enumerate(tis):
            rf = self.rank_at(track_race_t[ti])
            if rf is None:
                continue
            # rescale the RF rank (over the field) onto the local track-rank scale (over ntrk tracks)
            for j, n in enumerate(nums):
                if n in rf:
                    rf_local = (rf[n] - 1) / max(1, self.field_size - 1) * (ntrk - 1) + 1
                    ordc[i, j] = abs(trk_rank[ti] - rf_local) / max(1, ntrk - 1)
        _ = order_vals  # (kept for clarity; trk_rank derives from it)

        cost = silk_norm + lam_order * ordc
        ri, ci = linear_sum_assignment(cost)
        out: dict[int, tuple[int, float]] = {}
        med = np.median(cost) + 1e-9
        for r, c in zip(ri, ci, strict=True):
            ti, num = tis[r], nums[c]
            row = cost[r]
            second = np.partition(row, 1)[1] if len(row) > 1 else row[0]
            conf = float(np.clip((second - row[c]) / med, 0.0, 1.0))
            if conf >= conf_floor:
                out[ti] = (num, conf)
        return out

    # ---- the v2 CONSENSUS (#94) ----
    def _rank_reliability(self, race_t: float) -> float:
        """How TRUSTWORTHY is the RF rank at race_t? (0=bunched/absent, 1=clean/unbunched).

        0089: the order constraint helps on R03's late reliable window but AMPLIFIES error on
        R01's bunched window. We gate the constraint by a reliability scalar so a single FIXED
        weight is safe on both. Reliability = how SPREAD the RF rank-sequence is around race_t
        (a clean ladder is reliable; a bunched/degenerate order is not), AND whether race_t falls
        in the body of the race (the RF legend backbone validated rho +1.00 at the 1/4-1/2 calls,
        ~0 late when the field bunches — 0086/0088). We combine a body-window taper with the local
        rank dispersion. No race-specific constants: derived from this race's own backbone.
        """
        if not self.rank_seq:
            return 0.0
        # local backbone density + spread: pick the nearest few timepoints, measure how many
        # distinct runners and how monotone the order is. A reliable window has a FULL, stable order.
        ts = np.array([t for t, _ in self.rank_seq])
        i0 = int(np.argmin(np.abs(ts - race_t)))
        near = self.rank_seq[max(0, i0 - 1): i0 + 2]
        lens = [len(o) for _, o in near]
        depth = float(np.mean(lens)) if lens else 0.0
        # depth/field: a backbone showing most of the field is a reliable ladder; a thin late call
        # (only the lead group still distinguishable) is unreliable.
        rel_depth = float(np.clip(depth / max(3.0, 0.6 * self.field_size), 0.0, 1.0))
        # body-window taper: trust the RF rank up to ~80% of the race, decay after (the bunched
        # come-from-behind finish — 0088 late-call rho ~0). race_t normalised by the last legend t.
        t_last = float(ts.max()) if ts.size else 1.0
        frac = race_t / max(1e-6, t_last)
        taper = float(np.clip(1.0 - max(0.0, frac - 0.80) / 0.20, 0.0, 1.0))
        return rel_depth * taper

    def consensus_assign(
        self,
        track_sig: dict[int, np.ndarray],
        track_order: dict[int, float],
        track_lat: dict[int, float],
        track_race_t: dict[int, float],
        own_lut: dict[int, np.ndarray] | None = None,
        own_distinct: dict[int, float] | None = None,
    ) -> dict[int, tuple[int, float]]:
        """THE SINGLE FIXED CONSENSUS (#94): one-to-one Hungarian over 4 weighted votes.

        track_sig[ti]    : the track's silk hue-histogram signature.
        track_order[ti]  : along-track position proxy (larger == further advanced). For the ORDER
                           vote (vs the RF clock rank).
        track_lat[ti]    : the track's median lateral (metres from rail). For the DRAW vote (the
                           draw gives the START lateral order: low draw == rail == small lateral).
        track_race_t[ti] : the track's reference race_t (to look up RF rank + its reliability).
        own_lut          : the FEED'S OWN finish-cam silk LUT (number->hist). Restores per-feed
                           fidelity (e.g. R01's dedicated 102.5-123s LUT). If None, falls back to
                           the RF anchor LUT for the own-silk term too.
        own_distinct     : per-runner silk DISTINCTNESS in [0,1] (1=unambiguous colour, 0=degenerate
                           red/pink). Scales the own-silk term per runner. If None, all 1.0.

        cost[t,r] = W_OWN_SILK*distinct[r]*own_emd_norm[t,r] + W_RF_SILK*rf_emd_norm[t,r]
                  + W_ORDER*reliability(t)*order_incons[t,r] + W_DRAW*draw_incons[t,r]
        """
        from scipy.optimize import linear_sum_assignment

        tis = [ti for ti in track_sig if track_sig[ti] is not None]
        nums = self.known_numbers()
        if not tis or not nums:
            return {}
        own_lut = own_lut or self.silk_lut
        if own_distinct is None:
            own_distinct = {n: 1.0 for n in nums}
        ntrk = len(tis)

        def _norm_emd(lut: dict[int, np.ndarray]) -> np.ndarray:
            m = np.full((ntrk, len(nums)), np.nan)
            for i, ti in enumerate(tis):
                for j, n in enumerate(nums):
                    if n in lut and lut[n] is not None:
                        m[i, j] = hist_emd(track_sig[ti], lut[n])
            med = np.nanmedian(m) if np.isfinite(m).any() else 1.0
            m = m / (med + 1e-9)
            return np.nan_to_num(m, nan=float(np.nanmax(m) + 1.0) if np.isfinite(m).any() else 1.0)

        own = _norm_emd(own_lut)
        rf = _norm_emd(self.silk_lut)
        dist_w = np.array([own_distinct.get(n, 1.0) for n in nums])[None, :]

        # ORDER vote: tracks ranked by along-track proxy vs RF rank, gated by reliability(race_t).
        trk_rank = {ti: r + 1 for r, ti in enumerate(sorted(tis, key=lambda t: -track_order[t]))}
        ordc = np.zeros((ntrk, len(nums)))
        for i, ti in enumerate(tis):
            rel = self._rank_reliability(track_race_t[ti])
            rf_rank = self.rank_at(track_race_t[ti])
            if rf_rank is None or rel <= 0:
                continue
            for j, n in enumerate(nums):
                if n in rf_rank:
                    rf_local = (rf_rank[n] - 1) / max(1, self.field_size - 1) * (ntrk - 1) + 1
                    ordc[i, j] = rel * abs(trk_rank[ti] - rf_local) / max(1, ntrk - 1)

        # DRAW vote (race-INDEPENDENT): the draw gives the START lateral order (low draw == rail ==
        # small lateral). Rank tracks by their observed lateral (1 = most rail); rank runners by
        # draw (1 = lowest draw == rail). Penalise |Delta-rank|, normalised. This breaks the silk
        # degeneracy with a strong, always-on order proxy (a far better proxy than head-on depth).
        drawc = np.zeros((ntrk, len(nums)))
        if self.draw:
            trk_lat_rank = {ti: r + 1 for r, ti in
                            enumerate(sorted(tis, key=lambda t: track_lat[t]))}  # small lat = rail = 1
            draw_runners = sorted(nums, key=lambda n: self.draw.get(n, 999))
            draw_rank = {n: r + 1 for r, n in enumerate(draw_runners)}  # low draw = rail = rank 1
            for i, ti in enumerate(tis):
                for j, n in enumerate(nums):
                    dr_local = (draw_rank[n] - 1) / max(1, len(nums) - 1) * (ntrk - 1) + 1
                    drawc[i, j] = abs(trk_lat_rank[ti] - dr_local) / max(1, ntrk - 1)

        cost = (W_OWN_SILK * dist_w * own + W_RF_SILK * rf
                + W_ORDER * ordc + W_DRAW * drawc)
        ri, ci = linear_sum_assignment(cost)
        out: dict[int, tuple[int, float]] = {}
        med = float(np.median(cost)) + 1e-9
        for r, c in zip(ri, ci, strict=True):
            row = cost[r]
            second = np.partition(row, 1)[1] if len(row) > 1 else row[0]
            conf = float(np.clip((second - row[c]) / med, 0.0, 1.0))
            out[tis[r]] = (nums[c], conf)
        return out


def silk_distinctness(lut: dict[int, np.ndarray]) -> dict[int, float]:
    """Per-runner silk DISTINCTNESS in [0,1] from a LUT: how far this silk's nearest neighbour is.

    A degenerate red/pink silk (close to many others) scores ~0; an unambiguous blue/yellow scores
    ~1. Used to confidence-weight the own-silk vote (the task's 'confidence-weight by silk-
    distinctness'). Race-agnostic: derived from the LUT's own pairwise EMD spread.
    """
    nums = sorted(lut)
    if len(nums) < 2:
        return {n: 1.0 for n in nums}
    nn = {}
    for n in nums:
        ds = [hist_emd(lut[n], lut[m]) for m in nums if m != n and lut[m] is not None]
        nn[n] = min(ds) if ds else 1.0
    vals = np.array(list(nn.values()))
    lo, hi = float(np.percentile(vals, 10)), float(np.percentile(vals, 90))
    rng = max(1e-9, hi - lo)
    return {n: float(np.clip((nn[n] - lo) / rng, 0.0, 1.0)) for n in nums}


# ============================================================================= draw -> y_lat(0)
N_STALLS_HV = 14   # HKJC Happy Valley standard barrier (14 stalls); draw d sits in stall d.


def draw_to_metres(draw: dict[int, int], track_width_m: float,
                   n_stalls: int = N_STALLS_HV) -> dict[int, float]:
    """Map each runner's barrier DRAW -> its START lateral y_lat(t=0) in METRES from the rail.

    PHYSICS (race-agnostic, no per-race tuning): the barrier stalls are UNIFORMLY spaced across
    the course; the field at the gate sits one horse per stall. So
        stall_width   = track_width / n_stalls
        y_lat(draw d) = 0.5*stall_width + (d-1)*stall_width      [m from the inside rail]
    i.e. stall-1's centre is half a stall off the rail and each higher draw steps out one stall.
    This is the SECOND metric anchor (the start) that the patrol mid-race anchor lacked; it pins
    the lateral state exactly where the feeds are weakest (race_t~0, before drone/patrol coverage).

    Scale is set by the C-course width (21.3m) + the standard HV stall count. The CALIBRATION is
    cross-checked against (a) the aerial gate-frame field fan and (b) the validated patrol metric
    field span in the harness — reported, never tuned to the outcome. The draw is a PRE-RACE
    INITIAL CONDITION (philosophy-clean, same class as distance/weight), not an outcome.
    """
    sw = track_width_m / n_stalls
    return {n: 0.5 * sw + (d - 1) * sw for n, d in draw.items()}


# ============================================================================= RF anchor build
def build_rf_anchor(
    rf_video: str,
    name2num: dict[str, int],
    rf_offset: float,
    fin_window: tuple[float, float, float],
    legend_window: tuple[float, float, float] = (4.0, 95.0, 2.0),
    field_size: int = 12,
    ocr=None,
) -> Identity:
    """Solve identity ONCE from the RF/race-film feed (the crispest side-on source).

    * MULTI-FRAME VOTING saddlecloth OCR over the finish window -> number->silk hue-hist LUT.
    * legend NAME->number bridge over the whole race -> the rank(race_t) order backbone.

    REUSES cv_rf_longitudinal's detect_horses / ocr_numbers / read_legend_order machinery via a
    thin re-point (no edits to that module). Falls back gracefully if RapidOCR is unavailable.
    """
    import cv2

    import cv_rf_longitudinal as RF  # noqa: N812

    if ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        except Exception:
            print("[identity] RapidOCR unavailable — RF anchor empty")
            return Identity(field_size=field_size)

    # re-point the RF toolkit at this race (read-only over the module; no file edits)
    RF.NAME2NUM = {k.replace(" ", "").upper(): v for k, v in name2num.items()}
    RF.RF_OFFSET = rf_offset
    name_lut = {k.replace(" ", "").upper(): v for k, v in name2num.items()}

    # ---- STEP A: number->silk LUT by multi-frame voting over the finish window ----
    fin_t0, fin_t1, fin_dt = fin_window
    votes: dict[int, list[np.ndarray]] = {}
    nseen: dict[int, int] = {}
    for _vid_t, fr in _decode_window(rf_video, fin_t0, fin_t1, fin_dt):
        h = fr.shape[0]
        dets = [d for d in RF.ocr_numbers(fr, ocr) if not RF.in_legend_strip(d["cy"], h)]
        best: dict[int, dict] = {}
        for d in dets:
            if d["num"] not in best or d["conf"] > best[d["num"]]["conf"]:
                best[d["num"]] = d
        for num, d in best.items():
            if not 1 <= num <= field_size:
                continue
            nseen[num] = nseen.get(num, 0) + 1
            patch = _silk_patch_hsv(fr, d["cx"], d["cy"])
            if patch is not None:
                votes.setdefault(num, []).append(patch)

    silk_lut: dict[int, np.ndarray] = {}
    silk_votes: dict[int, int] = {}
    for num, plist in votes.items():
        if nseen.get(num, 0) < 2:        # majority gate: drop single-frame OCR flukes
            continue
        sig = silk_hist(np.vstack(plist))
        if sig is not None:
            silk_lut[num] = sig
            silk_votes[num] = nseen[num]

    # ---- STEP B: legend NAME->number rank backbone over the whole race (sequential decode) ----
    rank_seq: list[tuple[float, list[int]]] = []
    lt0, lt1, ldt = legend_window
    for vid_t, fr in _decode_window(rf_video, lt0, lt1, ldt):
        h, w = fr.shape[:2]
        strip = fr[int(h * 0.855): int(h * 0.93), : int(w * 0.92)]
        big = cv2.resize(strip, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(big)
        if not res:
            continue
        hits = []
        for box, txt, conf in res:
            num = _match_name(txt, name_lut) if conf > 0.40 else None
            if num is not None:
                hits.append((float(np.mean([p[0] for p in box])), num))
        hits.sort()
        order: list[int] = []
        for _, num in hits:
            if num not in order:
                order.append(num)
        if len(order) >= 3:
            rank_seq.append((vid_t - rf_offset, order))

    print(f"[identity] RF anchor: silk LUT {sorted(silk_lut)} ({len(silk_lut)}/{field_size}); "
          f"votes {dict(sorted(silk_votes.items()))}; legend timepoints {len(rank_seq)}")
    return Identity(silk_lut=silk_lut, silk_votes=silk_votes, rank_seq=rank_seq,
                    field_size=field_size)


def augment_silk_lut_from_patrol(
    ident: Identity,
    patrol_video: str,
    fin_window: tuple[float, float, float],
    field_size: int = 12,
    ocr=None,
) -> Identity:
    """Add silk signatures from the PATROL finish-cam (a 2nd crisp side-on number source).

    Merges into the existing LUT (mean of normalised hists) so numbers RF missed at the line can
    still get a silk signature, and the shared ones get more votes. Same multi-frame voting gate.
    """
    import cv2

    if ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        except Exception:
            return ident

    fin_t0, fin_t1, fin_dt = fin_window
    votes: dict[int, list[np.ndarray]] = {}
    nseen: dict[int, int] = {}
    for _vid_t, fr in _decode_window(patrol_video, fin_t0, fin_t1, fin_dt):
        hh = fr.shape[0]
        crop = fr[int(hh * 0.25): int(hh * 0.85), :]
        best: dict[int, dict] = {}
        for scale in (1.6, 2.2):
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr(big)
            if not res:
                continue
            for box, txt, conf in res:
                txt = txt.strip()
                if not (re.fullmatch(r"\d{1,2}", txt) and conf > 0.45):
                    continue
                num = int(txt)
                if not 1 <= num <= field_size:
                    continue
                cx = float(np.mean([p[0] for p in box])) / scale
                cy = float(np.mean([p[1] for p in box])) / scale + int(hh * 0.25)
                if num not in best or conf > best[num]["conf"]:
                    best[num] = dict(cx=cx, cy=cy, conf=conf)
        for num, d in best.items():
            nseen[num] = nseen.get(num, 0) + 1
            patch = _silk_patch_hsv(fr, d["cx"], d["cy"])
            if patch is not None:
                votes.setdefault(num, []).append(patch)

    added = []
    for num, plist in votes.items():
        if nseen.get(num, 0) < 2:
            continue
        sig = silk_hist(np.vstack(plist))
        if sig is None:
            continue
        if num in ident.silk_lut:
            merged = ident.silk_lut[num] + sig
            ident.silk_lut[num] = merged / (merged.sum() + 1e-9)
            ident.silk_votes[num] = ident.silk_votes.get(num, 0) + nseen[num]
        else:
            ident.silk_lut[num] = sig
            ident.silk_votes[num] = nseen[num]
            added.append(num)
    if added:
        print(f"[identity] patrol finish-cam ADDED numbers {sorted(added)}; "
              f"LUT now {sorted(ident.silk_lut)} ({len(ident.silk_lut)}/{field_size})")
    return ident


# ============================================================================= the R03 driver
def _run_r03():
    """Build the unified identity for HV R03 and report LUT + rank-backbone coverage."""
    from cv_r03_common import FIELD_SIZE, WINNER_FINISH_RACE_T, parse_official

    off = parse_official()
    name2num = {v: k for k, v in off.horse.items()}

    rf_video = "data_cache/video/full_20260603_03.mp4"
    patrol_video = "data_cache/video/patrol_20260603_03.mp4"
    rf_offset = 72.0 - WINNER_FINISH_RACE_T   # 2.71 (0088)

    print("=" * 80)
    print("UNIFIED IDENTITY (#92) — HV 2026-06-03 R03 1200m sprint")
    print("=" * 80)
    print(f"winner #{off.finish_order[0]} {off.horse[off.finish_order[0]]}")

    # one OCR instance shared across both crisp sources
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except Exception:
        ocr = None

    ident = build_rf_anchor(
        rf_video, name2num, rf_offset,
        fin_window=(68.5, 71.2, 0.04),      # R03 legible approach (0088 backlog #4)
        legend_window=(4.0, 68.0, 2.0),
        field_size=FIELD_SIZE, ocr=ocr,
    )
    ident = augment_silk_lut_from_patrol(
        ident, patrol_video, fin_window=(72.0, 100.0, 0.2),
        field_size=FIELD_SIZE, ocr=ocr,
    )
    ident.save("data_cache/cv_identity_R03.npz")
    print("[identity] cached -> data_cache/cv_identity_R03.npz")

    # rank backbone validation vs official RP (validation only)
    from scipy.stats import spearmanr
    print("\n[rank backbone vs official RP — validation only]")
    sec_t = np.concatenate([[0.0], np.cumsum(off.sections)])
    sec_s = np.array([0.0, 400.0, 800.0, 1200.0])
    for name, arc, ci in [("1/4", 400.0, 0), ("1/2", 800.0, 1), ("late", 1100.0, 2)]:
        rt = float(np.interp(arc, sec_s, sec_t))
        rk = ident.rank_at(rt)
        if not rk:
            continue
        common = [n for n in rk if n in off.rp and len(off.rp[n]) > ci]
        if len(common) >= 3:
            rho, _ = spearmanr([rk[n] for n in common],
                               [off.rp[n][ci] for n in common])
            print(f"   {name:5s} race_t {rt:4.1f}s  rho(rank,RP)={rho:+.2f}  n={len(common)}")

    return ident, off


if __name__ == "__main__":
    _run_r03()
