"""finaleval.py RUNROOT KEY CELL SPEC...: J parts of labellings under the FINAL product's evidence of CELL (the model,
official observations, metric shift and finish crossings fuse_final builds), from CELL's stored core_final.npz. The
first (camera-free) solve runs as in the product; the full solve's inputs are captured and each
labelling evaluated (env DROP: any of s, l, shift leaves out the extra along-track observations, the extra lateral ones, the
metric shift). SPEC as evalmoves.py ('a/b@t0-t1', '+'-joined), applied to the stored final labels; 'U' is the
U cell's stored final labelling, 'V' the cell's own video labelling."""
import sys
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core
from hkjc.recon.pipeline import RaceRun

root, key, cell, specs = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
mask = Mask() if cell == "U" else Mask(legend=10, legend_phase=int(cell[2]))
d, v, n = key.split("_")
run = RaceRun(race_info(d, v, int(n)), root=root / cell, log=lambda m: None, mask=mask)
L0 = np.load(root / cell / key / "core_final.npz")["seg_lab"]
LU = np.load(root / "U" / key / "core_final.npz")["seg_lab"]
calls = []
_solve = core.solve


def fake_solve(model, start=(), extra_s=None, extra_l=None, s_shift=None, log=print, **kw):
    calls.append(1)
    if len(calls) == 1:  # the camera-free solve as the product runs it
        return _solve(model, start=start, extra_s=extra_s, extra_l=extra_l, s_shift=s_shift, log=lambda m: None, **kw)
    import os
    drop = os.environ.get("DROP", "")
    if "s" in drop.split(","):  # the official / finish observations of s left out
        extra_s = None
    if "l" in drop.split(","):
        extra_l = None
    if "shift" in drop.split(","):
        s_shift = None
    ev = model.ev
    lo = np.full(ev.n_seg, np.inf)
    hi = np.full(ev.n_seg, -np.inf)
    np.minimum.at(lo, ev.seg, ev.t)
    np.maximum.at(hi, ev.seg, ev.t)
    col = {int(h): i for i, h in enumerate(ev.horses)}
    st0 = core.evaluate(model, L0, extra_s, extra_l, s_shift=s_shift)
    print(f"{cell} final labels: J {st0.J:.1f} {core._fmt(st0.parts)}")
    for spec in specs:
        if spec == "U":
            lab = LU.copy()
        elif spec == "V":  # the cell's own video labelling
            import pickle

            lab = pickle.load(open(root / cell / key / "fuse_video.pkl", "rb"))["seg_lab"]
        else:
            lab = L0.copy()
            for one in spec.split("+"):
                pair, span = one.split("@")
                a, b = (col[int(x)] for x in pair.split("/"))
                t0, t1 = (float(x) for x in span.split("-"))
                inside = (lo >= t0 - 0.15) & (hi <= t1 + 0.05)
                ia, ib = inside & (L0 == a), inside & (L0 == b)
                lab[ia], lab[ib] = b, a
        st = core.evaluate(model, lab, extra_s, extra_l, X0=st0.S, s_shift=s_shift)
        print(f"{spec}: J {st.J:.1f} ({st.J - st0.J:+.1f}) "
              f"{core._fmt({k: st.parts[k] - st0.parts[k] for k in st.parts})}")
    raise SystemExit(0)


core.solve = fake_solve
run._fuse_final()
