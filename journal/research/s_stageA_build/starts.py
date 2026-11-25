"""starts.py RUNROOT KEY CELL [full]: under CELL's video evidence, the start labelling built from several evidence
subsets (all; reads+silk; list only; list in even / odd 10 s blocks + reads + silk), each descended by joint
assignment steps; prints the J each reaches. With 'full', a full solve from the best one."""
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core, fusion
from hkjc.recon.pipeline import RaceRun

root, key, c = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
full = len(sys.argv) > 4


def mask_of(c):
    if c == "U":
        return Mask()
    if c.startswith("lg"):
        return Mask(legend=10, legend_phase=int(c[2]))
    return Mask(discs=10, discs_phase=int(c[2]))


d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=root / c, log=lambda m: None, mask=mask_of(c))
T = np.load(run.p("tracks.npz"), allow_pickle=True)
ev = run._evidence(T)
Lr = core.list_rows(run.legend_rows(), run._tick_of_tracks(), ev.horses)
V = pickle.load(open(run.p("fuse_video.pkl"), "rb"))
col = {int(h): i for i, h in enumerate(ev.horses)}
ex_s, ex_l = run._core_obs(V["fin"], col)
ts = [T["t"].min(), T["t"].max(), Lr.tau.min(), ex_s.t.min(), ex_s.t.max()]
tg = np.arange(min(ts), max(ts) + fusion.DT / 2, fusion.DT)
m = core.Model(ev, Lr, tg, run.info.width)
print("cell's own video labelling J", round(core.evaluate(m, V["seg_lab"], ex_s, ex_l).J, 1))


def votes_block(par):
    m2 = copy.copy(m)
    keep = (np.floor(Lr.tau / 10).astype(int) % 2) == par
    m2.Lr = core.ListRows(Lr.tau, np.where(keep[:, None], Lr.slots, -1), Lr.full & keep, Lr.same & keep)
    return core.list_votes(m2, None, None)


rs = ev.read_cost + ev.silk_cost
lv = core.list_votes(m, None, None)
variants = {"all": lv + rs, "reads+silk": rs, "list": lv, "even+rs": votes_block(0) + rs, "odd+rs": votes_block(1) + rs}
best = None
for name, U0 in variants.items():
    U0 = U0.copy()
    U0[ev.forbid] = core.INF
    lab = core.milp(np.where(U0 < 0, U0, 0.0) - 1e-3 * (U0 < 0), ev, allow=U0 < 0)
    st = core.evaluate(m, lab, ex_s, ex_l)
    j0, k = st.J, 0
    while True:
        prop = core.milp(core.unaries(m, st, lab), ev)
        if np.array_equal(prop, lab):
            break
        st2 = core.evaluate(m, prop, ex_s, ex_l, X0=st.S)
        if st2.J >= st.J - core.MIN_GAIN:
            break
        lab, st, k = prop, st2, k + 1
    print(f"{name:11s} start J {j0:9.1f} ({int((lab >= 0).sum())} labelled) -> {st.J:9.1f} after {k} steps")
    if best is None or st.J < best[1]:
        best = (lab, st.J, name)
if full:
    r = core.solve(m, start=(best[0],), extra_s=ex_s, extra_l=ex_l, log=print)
    print("full solve from", best[2], "J", round(r.state.J, 1))
