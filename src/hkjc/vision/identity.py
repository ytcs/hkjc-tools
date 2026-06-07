"""Unified, race-agnostic IDENTITY — solve once, propagate to all feeds.

Identity is solved ONCE from the crispest source (the RF side-on saddlecloths + legend NAMES),
producing a number<->silk-signature LUT + a per-horse running-ORDER backbone, then PROPAGATED
to every feed's anonymous tracks by a single FIXED consensus of four independent votes
(no per-race tuning):

    cost = W_OWN_SILK · distinct · own_finish_cam_silk_EMD   (the feed's own crisp LUT)
         + W_RF_SILK  · rf_anchor_silk_EMD                   (the cross-feed backbone)
         + W_ORDER    · reliability · clock_order_inconsistency
         + W_DRAW     · draw_order_inconsistency             (race-independent start order)

assigned one-to-one (Hungarian). The clock-order and draw votes break the silk degeneracy of
the red/pink silk family. The silk descriptor is ``vision.silk`` (NOT a private copy); the
saddlecloth OCR is ``vision.ocr``. The draw VOTE uses the draw ORDER the caller supplies; the
draw->metres START anchor lives in ``hkjc.reconstruction.geometry`` (vision does not import it).

See ``docs/CV_RECON.md`` §4 (the consensus) + §5 (the draw anchor).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .silk import silk_distance, silk_distinctness, silk_signature

# ============================================================ the single fixed consensus weights
# Each silk cost is normalised by its own median (scale-free); the order/draw costs are normalised
# |Delta-rank| in [0,1]. The SAME weights are used on every race (per-race tuning would overfit).
#   * OWN finish-cam LUT — the highest-fidelity per-track silk -> the largest weight, down-weighted
#     per runner by silk DISTINCTNESS (a degenerate silk contributes little).
#   * RF anchor LUT — the cross-feed shared backbone (lead-group biased) -> a smaller steady weight.
#   * shared-clock ORDER — gated by rank reliability (~0 on bunched windows where it amplifies error).
#   * DRAW ORDER — race-independent and strong; the magnitude anchor that breaks silk degeneracy.
W_OWN_SILK = 1.00
W_RF_SILK = 0.45
W_ORDER = 0.60
W_DRAW = 0.70


@dataclass
class Identity:
    """The solved-once identity: silk LUT + the RF rank(race_t) order backbone + the draw.

    The feed's OWN finish-cam LUT is supplied per call (it lives in the feed module); this
    carries the RF anchor LUT, the rank backbone, and the barrier draw for the consensus votes.
    """

    silk_lut: dict[int, np.ndarray] = field(default_factory=dict)      # runner -> hue hist (RF anchor)
    silk_votes: dict[int, int] = field(default_factory=dict)           # runner -> # OCR votes
    rank_seq: list = field(default_factory=list)                       # [(race_t, [lead..back]), ...]
    draw: dict[int, int] = field(default_factory=dict)                 # runner -> barrier draw (1=rail)
    field_size: int = 12

    # ---- order / clock constraint ----
    def rank_at(self, race_t: float) -> dict[int, int] | None:
        """RF-known running rank (1=lead) at the nearest legend timepoint to ``race_t``."""
        if not self.rank_seq:
            return None
        _, order = min(self.rank_seq, key=lambda s: abs(s[0] - race_t))
        return {n: i + 1 for i, n in enumerate(order)}

    def known_numbers(self) -> list[int]:
        return sorted(self.silk_lut)

    # ---- caching ----
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

    # ---- rank reliability gate ----
    def _rank_reliability(self, race_t: float) -> float:
        """How trustworthy is the RF rank at ``race_t`` (0=bunched/absent, 1=clean ladder)?

        The order constraint amplifies error on bunched windows, so we gate it: reliability is the
        local backbone DEPTH (a full visible field is a reliable ladder) tapered by a body-window
        (trust up to ~80% of the race, decaying after as the field bunches). No race-specific
        constants — derived from this race's own backbone.
        """
        if not self.rank_seq:
            return 0.0
        ts = np.array([t for t, _ in self.rank_seq])
        i0 = int(np.argmin(np.abs(ts - race_t)))
        near = self.rank_seq[max(0, i0 - 1): i0 + 2]
        lens = [len(o) for _, o in near]
        depth = float(np.mean(lens)) if lens else 0.0
        rel_depth = float(np.clip(depth / max(3.0, 0.6 * self.field_size), 0.0, 1.0))
        t_last = float(ts.max()) if ts.size else 1.0
        frac = race_t / max(1e-6, t_last)
        taper = float(np.clip(1.0 - max(0.0, frac - 0.80) / 0.20, 0.0, 1.0))
        return rel_depth * taper

    # ---- the consensus assignment ----
    def consensus_assign(
        self,
        track_sig: dict[int, np.ndarray],
        track_order: dict[int, float],
        track_lat: dict[int, float],
        track_race_t: dict[int, float],
        own_lut: dict[int, np.ndarray] | None = None,
        own_distinct: dict[int, float] | None = None,
    ) -> dict[int, tuple[int, float]]:
        """One-to-one Hungarian over the 4 weighted identity votes -> ``{track: (runner, conf)}``.

        track_sig    : track silk hue-histogram signature.
        track_order  : along-track position proxy (larger == further advanced) for the ORDER vote.
        track_lat    : track median lateral (m from rail) for the DRAW vote (low draw = rail = small).
        track_race_t : track reference ``race_t`` (to look up the RF rank + its reliability).
        own_lut      : the feed's OWN finish-cam silk LUT; falls back to the RF anchor LUT.
        own_distinct : per-runner silk distinctness in [0,1]; scales the own-silk term (default 1.0).
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
                        m[i, j] = silk_distance(track_sig[ti], lut[n])
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

        # DRAW vote (race-INDEPENDENT): the draw gives the START lateral order (low draw = rail =
        # small lateral). Rank tracks by observed lateral, runners by draw; penalise |Delta-rank|.
        drawc = np.zeros((ntrk, len(nums)))
        if self.draw:
            trk_lat_rank = {ti: r + 1 for r, ti in
                            enumerate(sorted(tis, key=lambda t: track_lat[t]))}
            draw_runners = sorted(nums, key=lambda n: self.draw.get(n, 999))
            draw_rank = {n: r + 1 for r, n in enumerate(draw_runners)}
            for i, ti in enumerate(tis):
                for j, n in enumerate(nums):
                    dr_local = (draw_rank[n] - 1) / max(1, len(nums) - 1) * (ntrk - 1) + 1
                    drawc[i, j] = abs(trk_lat_rank[ti] - dr_local) / max(1, ntrk - 1)

        cost = (W_OWN_SILK * dist_w * own + W_RF_SILK * rf + W_ORDER * ordc + W_DRAW * drawc)
        ri, ci = linear_sum_assignment(cost)
        out: dict[int, tuple[int, float]] = {}
        med = float(np.median(cost)) + 1e-9
        for r, c in zip(ri, ci, strict=True):
            row = cost[r]
            second = np.partition(row, 1)[1] if len(row) > 1 else row[0]
            conf = float(np.clip((second - row[c]) / med, 0.0, 1.0))
            out[tis[r]] = (nums[c], conf)
        return out

    def propagate(
        self,
        track_sig: dict[int, np.ndarray],
        track_order: dict[int, float],
        track_race_t: dict[int, float],
        lam_order: float = 0.6,
        conf_floor: float = 0.0,
    ) -> dict[int, tuple[int, float]]:
        """Silk-only + clock-order propagation (the lighter assign used where the draw is absent).

        Hungarian over ``silk_EMD + lam_order · order_inconsistency``. Where two silks are
        EMD-degenerate the clock/order term picks the running-order-consistent assignment.
        """
        from scipy.optimize import linear_sum_assignment

        tis = [ti for ti in track_sig if track_sig[ti] is not None]
        nums = self.known_numbers()
        if not tis or not nums:
            return {}
        silk = np.array([[silk_distance(track_sig[ti], self.silk_lut[n]) for n in nums]
                         for ti in tis])
        silk_norm = silk / (np.median(silk) + 1e-9)
        trk_rank = {ti: r + 1 for r, ti in enumerate(sorted(tis, key=lambda t: -track_order[t]))}
        ntrk = len(tis)
        ordc = np.zeros_like(silk_norm)
        for i, ti in enumerate(tis):
            rf = self.rank_at(track_race_t[ti])
            if rf is None:
                continue
            for j, n in enumerate(nums):
                if n in rf:
                    rf_local = (rf[n] - 1) / max(1, self.field_size - 1) * (ntrk - 1) + 1
                    ordc[i, j] = abs(trk_rank[ti] - rf_local) / max(1, ntrk - 1)
        cost = silk_norm + lam_order * ordc
        ri, ci = linear_sum_assignment(cost)
        out: dict[int, tuple[int, float]] = {}
        med = np.median(cost) + 1e-9
        for r, c in zip(ri, ci, strict=True):
            row = cost[r]
            second = np.partition(row, 1)[1] if len(row) > 1 else row[0]
            conf = float(np.clip((second - row[c]) / med, 0.0, 1.0))
            if conf >= conf_floor:
                out[tis[r]] = (nums[c], conf)
        return out


def lut_distinctness(lut: dict[int, np.ndarray]) -> dict[int, float]:
    """Per-runner silk distinctness in [0,1] from a LUT (re-exported from ``vision.silk``)."""
    return silk_distinctness(lut)


# ============================================================ anchor builders (RF + patrol)
def _silk_patch_hsv(fr, cx, cy, dx=16, dy_up=40, dy_lo=6):
    """HSV pixels of the silk torso patch ABOVE an OCR'd saddlecloth number at ``(cx, cy)``."""
    import cv2

    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    y0, y1 = max(0, int(cy - dy_up)), max(0, int(cy - dy_lo))
    x0, x1 = max(0, int(cx - dx)), min(fr.shape[1], int(cx + dx))
    patch = hsv[y0:y1, x0:x1]
    return patch.reshape(-1, 3) if patch.size else None


def build_rf_anchor(
    rf_frames_fin,
    rf_frames_legend,
    name2num: dict[str, int],
    rf_offset: float,
    *,
    field_size: int = 12,
    ocr=None,
    legend_conf: float = 0.40,
) -> Identity:
    """Solve identity ONCE from the RF feed (the crispest side-on source).

    ``rf_frames_fin`` / ``rf_frames_legend`` are iterables of ``(vid_t, BGR)`` over the finish
    window and the legend window. Builds the number->silk LUT by multi-frame voting (>=2 OCR hits
    per number drops flukes) and the rank(race_t) backbone from the legend NAME->number bridge.
    ``name2num`` keys are raw names (normalised here). Returns an empty :class:`Identity` if no OCR.
    """
    from .feeds.rf import match_name, ocr_saddlecloths

    if ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        except Exception:
            return Identity(field_size=field_size)

    name_lut = {k.replace(" ", "").upper(): v for k, v in name2num.items()}

    # STEP A — number->silk LUT by multi-frame voting over the finish window
    votes: dict[int, list[np.ndarray]] = {}
    nseen: dict[int, int] = {}
    for _vid_t, fr in rf_frames_fin:
        dets = ocr_saddlecloths(fr, ocr, field_size)
        best: dict[int, dict] = {}
        for d in dets:
            if d["num"] not in best or d["conf"] > best[d["num"]]["conf"]:
                best[d["num"]] = d
        for num, d in best.items():
            nseen[num] = nseen.get(num, 0) + 1
            patch = _silk_patch_hsv(fr, d["cx"], d["cy"])
            if patch is not None:
                votes.setdefault(num, []).append(patch)

    silk_lut: dict[int, np.ndarray] = {}
    silk_votes: dict[int, int] = {}
    for num, plist in votes.items():
        if nseen.get(num, 0) < 2:
            continue
        sig = silk_signature(np.vstack(plist))
        if sig is not None:
            silk_lut[num] = sig
            silk_votes[num] = nseen[num]

    # STEP B — legend NAME->number rank backbone over the whole race
    import cv2

    rank_seq: list[tuple[float, list[int]]] = []
    for vid_t, fr in rf_frames_legend:
        h, w = fr.shape[:2]
        strip = fr[int(h * 0.855): int(h * 0.93), : int(w * 0.92)]
        big = cv2.resize(strip, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(big)
        if not res:
            continue
        hits = []
        for box, txt, conf in res:
            num = match_name(txt, name_lut) if conf > legend_conf else None
            if num is not None:
                hits.append((float(np.mean([p[0] for p in box])), num))
        hits.sort()
        order: list[int] = []
        for _, num in hits:
            if num not in order:
                order.append(num)
        if len(order) >= 3:
            rank_seq.append((vid_t - rf_offset, order))

    return Identity(silk_lut=silk_lut, silk_votes=silk_votes, rank_seq=rank_seq,
                    field_size=field_size)


def augment_silk_lut_from_patrol(
    ident: Identity,
    patrol_frames_fin,
    *,
    field_size: int = 12,
    ocr=None,
    crop_band: tuple[float, float] = (0.25, 0.85),
) -> Identity:
    """Add silk signatures from the PATROL finish-cam (a 2nd crisp side-on number source).

    Merges into the existing LUT (mean of normalised hists) so numbers RF missed at the line still
    get a signature; same multi-frame voting gate (>=2 hits). ``patrol_frames_fin`` is an iterable
    of ``(vid_t, BGR)``.
    """
    import re

    import cv2

    if ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        except Exception:
            return ident

    votes: dict[int, list[np.ndarray]] = {}
    nseen: dict[int, int] = {}
    for _vid_t, fr in patrol_frames_fin:
        hh = fr.shape[0]
        crop = fr[int(hh * crop_band[0]): int(hh * crop_band[1]), :]
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
                cy = float(np.mean([p[1] for p in box])) / scale + int(hh * crop_band[0])
                if num not in best or conf > best[num]["conf"]:
                    best[num] = dict(cx=cx, cy=cy, conf=conf)
        for num, d in best.items():
            nseen[num] = nseen.get(num, 0) + 1
            patch = _silk_patch_hsv(fr, d["cx"], d["cy"])
            if patch is not None:
                votes.setdefault(num, []).append(patch)

    for num, plist in votes.items():
        if nseen.get(num, 0) < 2:
            continue
        sig = silk_signature(np.vstack(plist))
        if sig is None:
            continue
        if num in ident.silk_lut:
            merged = ident.silk_lut[num] + sig
            ident.silk_lut[num] = merged / (merged.sum() + 1e-9)
            ident.silk_votes[num] = ident.silk_votes.get(num, 0) + nseen[num]
        else:
            ident.silk_lut[num] = sig
            ident.silk_votes[num] = nseen[num]
    return ident
