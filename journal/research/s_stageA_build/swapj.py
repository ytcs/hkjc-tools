"""swapj.py KEY CELL A B: J of the core's video labelling and of its abreast swap moves involving horses A and B."""
import pickle
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core, fusion
from hkjc.recon.pipeline import RaceRun

ROOT = Path(__file__).resolve().parents[3] / "data_cache/s_stageA/runs"
key, cell, A, B = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
mask = Mask() if cell == "U" else Mask(legend=10, legend_phase=int(cell[2]))
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=ROOT / cell, log=lambda m: None, mask=mask)
T = np.load(run.p("tracks.npz"), allow_pickle=True)
ev = run._evidence(T)
Lr = core.list_rows(run.legend_rows(), run._tick_of_tracks(), ev.horses)
V = pickle.load(open(run.p("fuse_video.pkl"), "rb"))
col = {int(h): c for c, h in enumerate(ev.horses)}
ex_s, ex_l = run._core_obs(V["fin"], col)
ts = [T["t"].min(), T["t"].max(), Lr.tau.min()] + ([ex_s.t.min(), ex_s.t.max()] if len(ex_s.t) else [])
tg = np.arange(min(ts), max(ts) + fusion.DT / 2, fusion.DT)
model = core.Model(ev, Lr, tg, run.info.width)
lab = V["seg_lab"]
st = core.evaluate(model, lab, ex_s, ex_l)
print(f"J kept {st.J:.1f} {core._fmt(st.parts)}")
for desc, pm in core.swap_moves(model, st, lab):
    if f"#{A}/#{B}" in desc or f"#{B}/#{A}" in desc:
        s2 = core.evaluate(model, pm, ex_s, ex_l, X0=st.S)
        print(f"{desc}: J {s2.J:.1f} (dJ {s2.J - st.J:+.1f}) {core._fmt(s2.parts)}")
