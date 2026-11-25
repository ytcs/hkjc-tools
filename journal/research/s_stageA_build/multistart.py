"""multistart.py RUNROOT KEY CELL [OTHER_CELL ...]: convergence diagnostic of core.solve on one cell's video evidence
(the finish camera's crossings from the cell's stored fuse_video.pkl, as in the product's full solve).

Full solves, each forced to start from one labelling: the solver's own init; each OTHER_CELL's stored video labelling;
the init built from the list votes only and from reads + silk only;
N_PERT seeded perturbations of the init solve's end (N_SWAP random two-runner label exchanges over one segment's
span). Prints start J, end J, CPU s, and segments / detection rows differing from the lowest end.
Env NATURAL=1: each extra start is given to the product's solve as a start (its own inits kept), not forced.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core, fusion
from hkjc.recon.pipeline import RaceRun

N_PERT, N_SWAP = 2, 10
NATURAL = bool(__import__('os').environ.get('NATURAL'))  # extra starts given to the solve as is (its own inits kept)
root, key, cell, others = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]


def mask_of(c):
    if c == "U":
        return Mask()
    if c.startswith("lg"):
        return Mask(legend=10, legend_phase=int(c[2]))
    return Mask(discs=10, discs_phase=int(c[2]))


d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=root / cell, log=lambda m: None, mask=mask_of(cell))
T = np.load(run.p("tracks.npz"), allow_pickle=True)
ev = run._evidence(T)
Lr = core.list_rows(run.legend_rows(), run._tick_of_tracks(), ev.horses)
V = pickle.load(open(run.p("fuse_video.pkl"), "rb"))
col = {int(h): i for i, h in enumerate(ev.horses)}
ex_s, ex_l = run._core_obs(V["fin"], col)
ts = [T["t"].min(), T["t"].max()] + ([Lr.tau.min()] if len(Lr.tau) else []) + (
    [ex_s.t.min(), ex_s.t.max()] if len(ex_s.t) else [])
tg = np.arange(min(ts), max(ts) + fusion.DT / 2, fusion.DT)
m = core.Model(ev, Lr, tg, run.info.width)

_lv = core.list_votes
LOG = (lambda s: print(s) if (' init J' in s or 'crossover' in s or 'start 1 J' in s or 'stop at' in s or 'lowest' in s) else None) if __import__('os').environ.get('SOLVELOG') else (lambda s: None)


def solve_from(lab0):
    """Full solve whose init is lab0 (the solver's own init made empty, so lab0 wins the start comparison)."""
    if lab0 is None or NATURAL:
        core.list_votes = _lv
    else:
        core.list_votes = lambda model, st, seg_lab: (np.full((model.ev.n_seg, model.H), 1e6) if st is None
                                                      else _lv(model, st, seg_lab))
    t0 = time.process_time()
    r = core.solve(m, start=() if lab0 is None else (lab0,), extra_s=ex_s, extra_l=ex_l, log=LOG)
    core.list_votes = _lv
    return r, time.process_time() - t0


lo = np.full(ev.n_seg, np.inf)
hi = np.full(ev.n_seg, -np.inf)
np.minimum.at(lo, ev.seg, ev.t)
np.maximum.at(hi, ev.seg, ev.t)


def perturb(lab, seed):
    rng = np.random.default_rng(seed)
    lab = lab.copy()
    for _ in range(N_SWAP):
        k = rng.choice(np.flatnonzero(lab >= 0))
        a = lab[k]
        nb = np.flatnonzero((lab >= 0) & (lab != a) & (lo <= hi[k]) & (hi >= lo[k]))
        if not len(nb):
            continue
        b = lab[rng.choice(nb)]
        inside = (lo >= lo[k] - 0.15) & (hi <= hi[k] + 0.15)
        ia, ib = inside & (lab == a), inside & (lab == b)
        lab[ia], lab[ib] = b, a
    return lab


runs = []
r0, c0 = solve_from(None)
runs.append(("init", r0.log[0][1], r0, c0))
for oc in others:
    p = root / oc / key / "fuse_video.pkl"
    if not p.exists():
        continue
    ol = pickle.load(open(p, "rb"))["seg_lab"]
    if len(ol) != ev.n_seg:
        print(f"{oc}: segment count differs, skipped")
        continue
    r, c = solve_from(ol)
    runs.append((f"cell {oc}", core.evaluate(m, ol, ex_s, ex_l).J, r, c))
for name, use_list, use_app in () if NATURAL else (("list-only", True, False), ("reads+silk", False, True)):
    U0 = (_lv(m, None, None) if use_list else 0.0) + (ev.read_cost + ev.silk_cost if use_app else 0.0)
    U0 = np.broadcast_to(U0, (ev.n_seg, m.H)).copy()
    U0[ev.forbid] = core.INF
    sl = core.milp(np.where(U0 < 0, U0, 0.0) - 1e-3 * (U0 < 0), ev, allow=U0 < 0)
    r, c = solve_from(sl)
    runs.append((name, core.evaluate(m, sl, ex_s, ex_l).J, r, c))
for sd in range(N_PERT):
    pl = perturb(r0.seg_lab, sd)
    r, c = solve_from(pl)
    runs.append((f"perturb {sd}", core.evaluate(m, pl, ex_s, ex_l).J, r, c))
best = min(runs, key=lambda x: x[2].state.J)
Js = [x[2].state.J for x in runs]
print(f"{key} {cell}: end J spread {max(Js) - min(Js):.1f} (best: {best[0]})")
for name, j0, r, c in runs:
    ds = int((r.seg_lab != best[2].seg_lab).sum())
    dr = int((r.seg_lab[ev.seg] != best[2].seg_lab[ev.seg]).sum())
    print(f"  {name:10s} start J {j0:10.1f} -> end J {r.state.J:10.1f} (+{r.state.J - best[2].state.J:7.1f})  "
          f"segments differ {ds:3d}  rows differ {dr:5d}  CPU {c:6.1f} s")
