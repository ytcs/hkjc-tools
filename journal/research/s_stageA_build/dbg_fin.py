"""dbg_fin.py KEY: first core solve (no finish camera) under cProfile; finish-camera crossings vs the paths' crossings."""
import cProfile
import pstats
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core, fusion
from hkjc.recon.validate import crossing
from hkjc.recon.pipeline import RaceRun

ROOT = Path(__file__).resolve().parents[3] / "data_cache/s_stageA/runs"
key = sys.argv[1]
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=ROOT / "U", log=lambda m: print(m, flush=True), mask=Mask())
T = np.load(run.p("tracks.npz"), allow_pickle=True)
ev = run._evidence(T)
Lr = core.list_rows(run.legend_rows(), run._tick_of_tracks(), ev.horses)
tg = np.arange(min(T["t"].min(), Lr.tau.min()), T["t"].max() + 0.05, fusion.DT)
model = core.Model(ev, Lr, tg, run.info.width)
pr = cProfile.Profile()
pr.enable()
res = core.solve(model, log=run.log)
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
paths = run._core_paths(res, ev, model)
fin = run.finish_observations(paths)
D = run.info.distance
for o in fin:
    p = paths.get(o.horse)
    tc = crossing(p.t, D - p.s, None, 0.0) if p is not None else np.nan
    print(f"#{o.horse}: camera {o.t[0]:.2f} sd {o.sd_s:.2f} m | path crossing {tc:.2f}")
np.save(ROOT / "U" / key / "dbg_seg_lab.npy", res.seg_lab)
