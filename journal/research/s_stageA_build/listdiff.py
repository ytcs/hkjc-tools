"""listdiff.py RUNROOT KEY CELL_A CELL_B: under CELL_A's evidence (unmasked list), the list facts each cell's video
labelling contradicts, per 1 s of race time: adjacent facts 'a ahead of b' with the state's b ahead by > 1 m, and
P(frozen) at those rows. Localises a J_list gap between two labellings."""
import pickle
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core, fusion
from hkjc.recon.pipeline import RaceRun

root, key, ca, cb = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=root / ca, log=lambda m: None, mask=Mask())
T = np.load(run.p("tracks.npz"), allow_pickle=True)
ev = run._evidence(T)
Lr = core.list_rows(run.legend_rows(), run._tick_of_tracks(), ev.horses)
V = pickle.load(open(run.p("fuse_video.pkl"), "rb"))
col = {int(h): i for i, h in enumerate(ev.horses)}
ex_s, ex_l = run._core_obs(V["fin"], col)
ts = [T["t"].min(), T["t"].max(), Lr.tau.min()] + ([ex_s.t.min(), ex_s.t.max()] if len(ex_s.t) else [])
tg = np.arange(min(ts), max(ts) + fusion.DT / 2, fusion.DT)
m = core.Model(ev, Lr, tg, run.info.width)
St = core.statements(Lr, m.H)
adj = ~St.unl
idx = np.arange(len(St.a))
res = {}
for c in (ca, cb):
    lab = pickle.load(open(root / c / key / "fuse_video.pkl", "rb"))["seg_lab"]
    st = core.evaluate(m, lab, ex_s, ex_l)
    Sa = np.array([np.interp(St.tau, tg, st.S[h]) for h in range(m.H)])
    gap = Sa[St.b, idx] - Sa[St.a, idx]  # > 0: b ahead while the list says a ahead
    bad = adj & (gap > 1.0)
    res[c] = (bad, st)
    print(c, "J", round(st.J, 1), "contradicted adjacent facts", int(bad.sum()), "of", int(adj.sum()))
ba, sa = res[ca]
bb, sb = res[cb]
sec = np.floor(St.tau).astype(int)


def names(sel):
    return " ".join(sorted({f"#{ev.horses[St.a[i]]}>#{ev.horses[St.b[i]]}" for i in np.flatnonzero(sel)}))


for s in np.unique(sec[ba | bb]):
    k = sec == s
    xa, xb = ba & k, bb & k
    if xa.sum() == xb.sum():
        continue
    rows = np.unique(St.row[k])
    print(f"{s:4d}s rows {len(rows):2d} bad {ca} {int(xa.sum()):3d} {cb} {int(xb.sum()):3d} pf {sa.pf[rows].mean():.2f}"
          f" | only {cb} breaks: {names(xb & ~xa)} | only {ca} breaks: {names(xa & ~xb)}")
