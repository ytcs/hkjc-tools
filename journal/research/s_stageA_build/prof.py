"""prof.py KEY: cProfile of fuse_video (the core's video product) on a staged U dir."""
import cProfile
import pstats
import sys
from pathlib import Path

from hkjc.data import Mask, race_info
from hkjc.recon.pipeline import RaceRun

ROOT = Path(__file__).resolve().parents[3] / "data_cache/s_stageA/runs"
key = sys.argv[1]
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=ROOT / "U", log=lambda m: print(m, flush=True), mask=Mask())
pr = cProfile.Profile()
pr.enable()
run.fuse_video()
pr.disable()
st = pstats.Stats(pr)
st.sort_stats("cumulative").print_stats(30)
st.sort_stats("tottime").print_stats(20)
