"""jtest.py: does J prefer the omission-corrected labelling of 20240511_ST_7 (#9 3rd from ~12 s to ~88 s, TV/official)
over the core's shifted one? Relabel the video solution's segments inside [T0, T1] by the rank shift
4->9, 1->4, 7->1, 3->7, 10->3, 5->10, 6->5, 9->6 and evaluate J with and without the finish camera."""
import pickle
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core, fusion
from hkjc.recon.pipeline import RaceRun

ROOT = Path(__file__).resolve().parents[3] / "data_cache/s_stageA/runs"
key = "20240511_ST_7"
T0, T1 = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0, float(sys.argv[2]) if len(sys.argv) > 2 else 88.0
run = RaceRun(race_info(*key.split("_")[:2], 7), root=ROOT / "U", log=lambda m: print(m, flush=True), mask=Mask())
T = np.load(run.p("tracks.npz"), allow_pickle=True)
ev = run._evidence(T)
Lr = core.list_rows(run.legend_rows(), run._tick_of_tracks(), ev.horses)
V = pickle.load(open(run.p("fuse_video.pkl"), "rb"))
lab = V["seg_lab"].copy()
col = {int(h): c for c, h in enumerate(ev.horses)}
mp = {4: 9, 1: 4, 7: 1, 3: 7, 10: 3, 5: 10, 6: 5, 9: 6}
lo = np.full(ev.n_seg, np.inf)
hi = np.full(ev.n_seg, -np.inf)
np.minimum.at(lo, ev.seg, ev.t)
np.maximum.at(hi, ev.seg, ev.t)
alt = lab.copy()
inside = (lo >= T0) & (hi <= T1) & (lab >= 0)
for k in np.flatnonzero(inside):
    h = int(ev.horses[lab[k]])
    if h in mp:
        alt[k] = col[mp[h]]
print(f"relabelled {int((alt != lab).sum())} segments; segments straddling the window:",
      [(int(k), round(lo[k], 1), round(hi[k], 1), int(ev.horses[lab[k]])) for k in np.flatnonzero(
          (lab >= 0) & (lo < T1) & (hi > T0) & ~inside)])
tg = np.arange(min(T["t"].min(), Lr.tau.min()), T["t"].max() + 0.05, fusion.DT)
model = core.Model(ev, Lr, tg, run.info.width)
ex_s, ex_l = run._core_obs(V["fin"], col)
for name, extra in (("no camera", None), ("camera", ex_s)):
    a = core.evaluate(model, lab, extra)
    b = core.evaluate(model, alt, extra)
    print(f"{name}: J found {a.J:.1f} {core._fmt(a.parts)} | J shifted-back {b.J:.1f} {core._fmt(b.parts)}")
# where does the list term differ? per 10 s of list rows: -2 log of the chains restricted to that window
for name, st in (("found", a), ("shifted", b)):
    out = []
    for w0 in range(0, 100, 10):
        m = (Lr.tau >= w0) & (Lr.tau < w0 + 10)
        if not m.any():
            continue
        Lw = core.ListRows(Lr.tau[m], Lr.slots[m], Lr.full[m], Lr.same[m])
        Sw = core.statements(Lw, model.H)
        k, w = core._interp(model.tg, Sw.tau)
        X = st.S
        dl = X[Sw.a, k] * (1 - w) + X[Sw.a, k + 1] * w - X[Sw.b, k] * (1 - w) - X[Sw.b, k + 1] * w
        J, pf, om = core.list_chains(Lw, Sw, core._p_order(dl)[1], model.H, dl)
        top = [(int(ev.horses[h]), round(float(om[h].mean()), 2)) for h in range(model.H) if om[h].mean() > 0.2]
        out.append(f"{w0}s J {J:.0f} pf {pf.mean():.2f} om {top}")
    print(name, " | ".join(out))
