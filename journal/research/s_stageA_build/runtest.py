"""runtest.py KEY CELL: the harness's call, RaceRun(info, root, mask).run(stages=replaced + outcome + validate), on a
fresh staged dir (stage.sh) under data_cache/s_stageA/runs/rt_<CELL>; then checks the label files' keys."""
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon.pipeline import RaceRun

ROOT = Path(__file__).resolve().parents[3] / "data_cache/s_stageA/runs"
key, cell = sys.argv[1], sys.argv[2]
mask = Mask() if cell == "U" else Mask(legend=10, legend_phase=int(cell[2]))
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=ROOT / f"rt_{cell}", log=lambda m: print(m, flush=True), mask=mask)
run.run(stages=("identify", "fuse_video", "fuse_final", "validate"))
for f in ("det_labels.npz", "det_labels_final.npz"):
    z = np.load(run.p(f))
    print(f, z.files, len(z["labels"]), "prob range", float(z["prob"].min()), float(z["prob"].max()))
print(open(run.p("status.json")).read())
