"""s_subobs: local fuse_final (+ validate) replay on a reflinked copy of a stored run dir (read-only source).
usage: replay.py ARM CELLS RACES_FILE [LIMIT]
  ARM   off | on   (asserts HKJC_SUBSPLIT_OBS matches: on = 1, off = unset)
  CELLS comma list of: U (unmasked), S0 / S1 (Mask(subsplit=0|1)), H (Mask(boundary=J_INTERIOR)), F (finish withheld)
Output: data_cache/s_subobs/runs/<ARM>_<CELL>/<key>. Run from the worktree root. Resumable."""
import os
import subprocess
import sys
import time
from pathlib import Path

import hkjc
from hkjc.data import Mask, race_info
from hkjc.recon.pipeline import RaceRun

RUNS = Path("data_cache/s_subobs/runs")
SKIP = {"paths.parquet", "validation.json", "det_labels_final.npz", "audit.json", "status.json", "timings.jsonl",
        "mask.txt", "contract.json", "det_margins_final.npz", "frames.bin", "map.png"}
arm, cells = sys.argv[1], sys.argv[2].split(",")
jobs = [ln.split() for ln in open(sys.argv[3]) if ln.strip()]
if len(sys.argv) > 4:
    jobs = jobs[: int(sys.argv[4])]
want = "1" if arm == "on" else None
assert os.environ.get("HKJC_SUBSPLIT_OBS") == want, f"HKJC_SUBSPLIT_OBS={os.environ.get('HKJC_SUBSPLIT_OBS')} arm {arm}"
assert not os.environ.get("HKJC_MASK") and not os.environ.get("HKJC_FINAL_HOLDOUT")
print("hkjc from", hkjc.__file__, "arm", arm, cells, flush=True)
for setn, k, src, j, nb in jobs:
    for cell in cells:
        mask = {"U": Mask(), "H": Mask(boundary=int(j)), "F": Mask(boundary=int(nb) - 1)}.get(cell)
        if mask is None:
            mask = Mask(subsplit=int(cell[1]))
        root = RUNS / f"{arm}_{cell}"
        out = root / k
        if (out / "validation.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        for f in Path(src).iterdir():
            if f.is_file() and f.name not in SKIP and not f.name.startswith(".") and not (out / f.name).exists():
                subprocess.run(["cp", "--reflink=auto", str(f.resolve()), str(out / f.name)], check=True)
        d, v, n = k.split("_")
        t0 = time.time()
        try:
            run = RaceRun(race_info(d, v, int(n)), root=root, log=lambda m: None, mask=mask)
            run.run(("fuse_final", "validate"), ("fuse_final",))
        except Exception as e:  # noqa: BLE001
            print("FAIL", k, arm, cell, type(e).__name__, str(e)[:200], flush=True)
            continue
        print(f"{setn} {k} {arm}_{cell} mask '{mask}' {time.time() - t0:.1f}s", flush=True)
