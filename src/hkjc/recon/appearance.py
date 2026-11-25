"""Render appearance term of the core (silk render): the learned avatar model's per-detection runner likelihoods,
discounted for the correlation of repeated detections of one avatar, as a (segments x runners) cost in J.

Input: ``appearance.npz`` in the run dir, rows = dets.npz rows (an upstream artifact of the frozen M3-R2 model:
``logp`` the calibrated per-detection log p(runner | crop) over the race's runners ``nos``, ``ok``, ``has_gif``).

Per detection d and runner h, with P_d the box's P(horse) (``core.p_horse``): the softmax has no "not a horse"
outcome, so a box that is not a horse reads as uniform:

  log r_dh = log(P_d G p_dh + (1 - P_d))     (G runners with an official silk; r = 1 for a runner without one and
                                              for rows the model did not score)

The unit of correlated evidence is the avatar's tracker tracklet tau (observe's ``members``; a detection in no
tracklet is its own): the same avatar renders the same way frame after frame, so each detection carries weight
w = 1 / (1 + (n_tau - 1) RHO), n_tau the tracklet's P(horse)-weighted count of scored rows, and the tracklet's pooled
evidence saturates at about 1 / RHO detections. The tracklet's read is

  L_tau,h = sum_{d in tau} w log r_dh,   q_tau = (1 - EPS) softmax(L_tau) + EPS / H,   cost -2 log(H q_tau,h)

EPS is the share of avatars that render like another runner's silk (systematic, not averaged away by pooling). The
core's segments cut tracklets, so the tracklet's cost is split over its segments additively: segment k of tracklet
tau carries its own rows' evidence and its share a_k (of the tracklet's P(horse)-weighted scored rows) of the
tracklet's normaliser and floor,

  C_kh = -2 L_kh + a_k [2 (logsumexp(L_tau) - log H) - 2 log((1 - EPS) + EPS / (H softmax_h(L_tau)))],

so a tracklet whose segments share one label costs exactly -2 log(H q_tau,h), however it is cut, and segments of a
tracklet that changes avatar each keep their own read. RHO and EPS were fitted by maximum likelihood on development
races the model never trained on, against the stored product's confident tracklet labels.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

RHO = 0.48
EPS = 0.031


def log_ratios(app, det_idx, horses, p_horse):
    """Per tracks row: (log r_dh (rows x runners), scored mask)."""
    horses = np.asarray(horses, int)
    H = len(horses)
    det_idx = np.asarray(det_idx, int)
    nos = np.asarray(app["nos"], int)
    gif = np.asarray(app["has_gif"], bool)
    lp = np.asarray(app["logp"], float)[det_idx]
    valid = np.asarray(app["ok"], bool)[det_idx] & np.isfinite(lp[:, gif]).all(1)
    col = np.searchsorted(horses, nos)
    use = gif & (col < H) & (horses[np.minimum(col, H - 1)] == nos)
    G = int(use.sum())
    P = np.asarray(p_horse, float)
    logr = np.zeros((len(det_idx), H))
    if G and valid.any():
        q = np.exp(np.where(valid[:, None], lp[:, use], 0.0))
        q /= q.sum(1, keepdims=True)
        logr[:, col[use]] = np.log(P[:, None] * G * q + (1.0 - P[:, None]))
        logr[~valid] = 0.0
    return logr, valid


def tracklet_ids(members, n_rows) -> np.ndarray:
    """Per tracks row: its tracklet index (rows in no tracklet get their own)."""
    trk = np.full(n_rows, -1)
    for i, m in enumerate(members):
        trk[np.asarray(m, int)] = i
    lone = trk < 0
    trk[lone] = len(members) + np.arange(lone.sum())
    return trk


def costs(app, det_idx, seg, n_seg, horses, p_horse, members) -> np.ndarray:
    """(segments x runners) appearance cost C_kh (above) for the core's segments ``seg`` (per tracks row; each
    segment lies in one tracklet)."""
    H = len(horses)
    seg = np.asarray(seg, int)
    logr, valid = log_ratios(app, det_idx, horses, p_horse)
    trk = tracklet_ids(members, len(seg))
    n_trk = int(trk.max()) + 1 if len(trk) else 0
    pv = np.asarray(p_horse, float) * valid
    cnt = np.bincount(trk, pv, n_trk)
    w = 1.0 / (1.0 + (np.maximum(cnt, 1.0)[trk] - 1.0) * RHO)
    Lt = np.zeros((n_trk, H))
    np.add.at(Lt, trk, w[:, None] * logr)
    lse = logsumexp(Lt, axis=1, keepdims=True)
    lift = 2.0 * (lse - np.log(H)) - 2.0 * np.log((1.0 - EPS) + EPS / (H * np.exp(Lt - lse)))
    L = np.zeros((n_seg, H))
    np.add.at(L, seg, w[:, None] * logr)
    seg_trk = np.zeros(n_seg, int)
    seg_trk[seg] = trk
    share = np.bincount(seg, pv, n_seg) / np.maximum(cnt[seg_trk], 1e-12)
    return -2.0 * L + share[:, None] * lift[seg_trk]
