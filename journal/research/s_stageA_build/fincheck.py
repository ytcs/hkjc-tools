"""fincheck.py KEY: the finish camera's clock source (anchor or registration) on the stored video paths, unsilenced, and
the label-free registration's tracklets near the line (are list-icon ghosts among them?)."""
import pickle
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon.pipeline import RaceRun

ROOT = Path(__file__).resolve().parents[3] / "data_cache/s_stageA/runs"
key = sys.argv[1]
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=ROOT / "U", log=lambda m: print(m, flush=True), mask=Mask())
V = pickle.load(open(run.p("fuse_video.pkl"), "rb"))
fin = run.finish_camera(V["paths"])
print("camera crossings:", sorted((o.horse, round(float(o.t[0]), 2)) for o in fin))
m, fit = run._finish_identity()
if fit is not None:
    print("registration t_w", round(fit["t_w"], 2), "C", round(fit["C"], 1), "n_close", fit["n_close"], "edge", fit["edge"])
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    for k, tc in sorted(fit["tc"].items(), key=lambda x: x[1])[:12]:
        mem = np.asarray(T["members"][k], int)
        print(f"  tracklet {k}: crossing {tc:.2f}, {len(mem)} dets, median cy {np.median(T['rows'][mem, 2]):.0f}, "
              f"median box h {np.median(T['rows'][mem, 4]):.0f}")
