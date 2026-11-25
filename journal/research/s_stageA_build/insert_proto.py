"""insert_proto.py KEY: prototype of an insertion move (not in the library): over a window between camera cuts, a
horse the list's omission chain marks as omitted is inserted at rank r of the field and every horse from rank r down
to its old place moves one avatar back. Evaluated by J on the stored video solution (no finish camera)."""
import sys
import time
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core, fusion
from hkjc.recon.pipeline import RaceRun

ROOT = Path(__file__).resolve().parents[3] / "data_cache/s_stageA/runs"
key = sys.argv[1]
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=ROOT / "U", log=lambda m: None, mask=Mask())
T = np.load(run.p("tracks.npz"), allow_pickle=True)
ev = run._evidence(T)
Lr = core.list_rows(run.legend_rows(), run._tick_of_tracks(), ev.horses)
tg = np.arange(min(T["t"].min(), Lr.tau.min()), T["t"].max() + 0.05, fusion.DT)
model = core.Model(ev, Lr, tg, run.info.width)
lab = np.load(run.p("core_video.npz"))["seg_lab"]
st = core.evaluate(model, lab)
print(f"J start {st.J:.1f}")
lo = np.full(ev.n_seg, np.inf)
hi = np.full(ev.n_seg, -np.inf)
np.minimum.at(lo, ev.seg, ev.t)
np.maximum.at(hi, ev.seg, ev.t)
st_t, cnt = np.unique(np.round(lo[lab >= 0] * 5) / 5, return_counts=True)
cuts = np.r_[st_t[cnt >= 3], tg[-1] + 1]
best = None
t0 = time.time()
for i, B0 in enumerate(cuts):
    for B1 in cuts[i + 1:]:
        rows = (Lr.tau >= B0) & (Lr.tau < B1)
        if rows.sum() < 20:
            continue
        om = st.om[:, rows].mean(1)
        inside = (lo >= B0 - 0.15) & (hi < B1 - 0.05) & (lab >= 0)
        if not inside.any():
            continue
        k, w = core._interp(tg, Lr.tau[rows])
        ms = (st.S[:, k] * (1 - w) + st.S[:, k + 1] * w).mean(1)
        order = list(np.argsort(-ms))
        for u in np.flatnonzero(om > 0.5):
            q = order.index(u)
            for r in range(q):
                mp = {order[r]: u}
                for j in range(r + 1, q + 1):
                    mp[order[j]] = order[j - 1]
                new = lab.copy()
                for s_ in np.flatnonzero(inside):
                    new[s_] = mp.get(int(lab[s_]), lab[s_])
                st2 = core.evaluate(model, new, X0=st.S)
                tag = f"window {B0:.1f}-{B1:.1f} insert #{ev.horses[u]} at rank {r + 1} (was {q + 1}): J {st2.J:.1f}"
                print(tag, flush=True)
                if best is None or st2.J < best[0]:
                    best = (st2.J, tag)
print("best", best, f"({time.time() - t0:.0f} s)")
