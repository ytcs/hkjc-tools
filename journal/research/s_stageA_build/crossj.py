"""crossj.py RUNROOT KEY CELL_A CELL_B: J of each cell's video labelling under each cell's evidence (video product,
finish camera from the cell's own fuse_video.pkl). Tells a search failure (a cell prefers the other cell's labels)
from an evidence difference."""
import pickle
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core, fusion
from hkjc.recon.pipeline import RaceRun

root, key, ca, cb = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]


def mask_of(c):
    if c == "U":
        return Mask()
    if c.startswith("lg"):
        return Mask(legend=10, legend_phase=int(c[2]))
    return Mask(discs=10, discs_phase=int(c[2]))


d, v, n = key.split("_")
labs, models = {}, {}
for c in (ca, cb):
    run = RaceRun(race_info(d, v, int(n)), root=root / c, log=lambda m: None, mask=mask_of(c))
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    ev = run._evidence(T)
    Lr = core.list_rows(run.legend_rows(), run._tick_of_tracks(), ev.horses)
    V = pickle.load(open(run.p("fuse_video.pkl"), "rb"))
    col = {int(h): i for i, h in enumerate(ev.horses)}
    ex_s, ex_l = run._core_obs(V["fin"], col)
    ts = [T["t"].min(), T["t"].max()] + ([Lr.tau.min()] if len(Lr.tau) else []) + (
        [ex_s.t.min(), ex_s.t.max()] if len(ex_s.t) else [])
    tg = np.arange(min(ts), max(ts) + fusion.DT / 2, fusion.DT)
    models[c] = (core.Model(ev, Lr, tg, run.info.width), ex_s, ex_l)
    labs[c] = V["seg_lab"]
for c in (ca, cb):
    m, ex_s, ex_l = models[c]
    row = []
    for lc in (ca, cb):
        if len(labs[lc]) != m.ev.n_seg:
            row.append(f"labels of {lc}: segment mismatch")
            continue
        st = core.evaluate(m, labs[lc], ex_s, ex_l)
        row.append(f"labels of {lc}: J {st.J:.1f}")
    print(f"evidence {c}: " + " | ".join(row))
for c in (ca, cb):
    m, ex_s, ex_l = models[c]
    if all(len(labs[lc]) == m.ev.n_seg for lc in (ca, cb)):
        pa = core.evaluate(m, labs[ca], ex_s, ex_l).parts
        pb = core.evaluate(m, labs[cb], ex_s, ex_l).parts
        print(f"evidence {c}: parts {cb} - {ca}: " + ", ".join(f"{k} {pb[k] - pa[k]:+.1f}" for k in pa))
