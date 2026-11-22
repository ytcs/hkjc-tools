"""CLI: python -m hkjc.recon DATE VENUE RACE_NO [DATE VENUE RACE_NO ...] [--force a,b] [--stages a,b]
[--mask SPEC]"""

import argparse
import sys

from ..data import Mask, race_info
from .pipeline import STAGES, RaceRun


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m hkjc.recon")
    ap.add_argument("race", nargs="+", help="DATE VENUE RACE_NO triples, e.g. 20260603 HV 8")
    ap.add_argument("--force", default="", help="comma-separated stages to re-run")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated stages to run")
    ap.add_argument(
        "--mask",
        default=None,
        help="information withheld from the reconstruction: none, or ';'-joined boundary=J | outcome, "
        "dets=I,J,... (dets.npz rows), discs=P[@PHASE] (default: env HKJC_MASK / HKJC_FINAL_HOLDOUT)",
    )
    a = ap.parse_args(argv)
    if len(a.race) % 3:
        ap.error("races must be DATE VENUE RACE_NO triples")
    force = tuple(s for s in a.force.split(",") if s)
    stages = tuple(s for s in a.stages.split(",") if s)
    mask = Mask.from_env() if a.mask is None else Mask.parse(a.mask)
    ok = True
    for k in range(0, len(a.race), 3):
        d, v, n = a.race[k : k + 3]
        try:
            RaceRun(race_info(d, v, int(n)), log=lambda m: print(m, flush=True), mask=mask).run(
                stages, force
            )
        except Exception as e:  # keep going across races
            print(f"{d} {v} {n}: FAILED {type(e).__name__}: {e}", flush=True)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
