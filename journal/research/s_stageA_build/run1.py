"""run1.py KEY CELL [stages]: the Stage A core stages (identify, fuse_video, fuse_final, validate) on a staged dir
(stage.sh). CELL U = no mask; lg0 / lg1 = Mask(legend=10, legend_phase=0 / 1)."""
import sys
import time
from pathlib import Path

import hkjc
from hkjc.data import Mask, race_info
from hkjc.recon.pipeline import RaceRun

assert "/worktrees/" in hkjc.__file__, hkjc.__file__
ROOT = Path(__file__).resolve().parents[3] / "data_cache/s_stageA/runs"
key, cell = sys.argv[1], sys.argv[2]
stages = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("identify", "fuse_video", "fuse_final", "validate")
mask = Mask() if cell == "U" else Mask(legend=10, legend_phase=int(cell[2]))
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=ROOT / cell, log=lambda m: print(m, flush=True), mask=mask)
t0 = time.process_time()
for st in stages:
    a = time.process_time()
    getattr(run, st)()
    print(f"== {st} CPU {time.process_time() - a:.1f} s", flush=True)
print(f"== total CPU {time.process_time() - t0:.1f} s")
