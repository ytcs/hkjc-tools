"""finaldec.py RUNROOT KEY CELL: decomposition of the final stage's departure from the video labelling. Runs CELL's
fuse_final as the product does (both solves real; inputs captured), checks the captured final model reproduces the
kept labelling's J, then evaluates the video labelling V and the final labelling F:
- under the final evidence, by term;
- with the official along-track observations left out, with the finish crossings left out, with the extra lateral
  observations left out, and with the metric shift undone (evidence rebuilt unshifted);
- by time window: F with V's labels on the segments starting in each 10 s window (final evidence).
Nothing is written to the run dir (fuse_final's outputs go to a scratch copy)."""
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from hkjc.data import Mask, race_info
from hkjc.recon import core
from hkjc.recon.pipeline import RaceRun

root, key, cell = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
mask = Mask() if cell == "U" else Mask(legend=10, legend_phase=int(cell[2]))
d, v, n = key.split("_")
tmp = Path(tempfile.mkdtemp(prefix="finaldec_", dir="/tmp/claude-1000/-home-ytc-Projects-hkjc-tools/"
                                                    "7c78352b-fd6f-4198-993e-a1bc75fe6c86/scratchpad"))
shutil.copytree(root / cell / key, tmp / cell / key)
run = RaceRun(race_info(d, v, int(n)), root=tmp / cell, log=lambda m: None, mask=mask)
V = pickle.load(open(run.p("fuse_video.pkl"), "rb"))["seg_lab"]
cap = []
_solve = core.solve


def capture(model, start=(), extra_s=None, extra_l=None, s_shift=None, log=print, **kw):
    res = _solve(model, start=start, extra_s=extra_s, extra_l=extra_l, s_shift=s_shift, log=lambda m: None, **kw)
    cap.append(dict(model=model, extra_s=extra_s, extra_l=extra_l, s_shift=s_shift, res=res, J=res.state.J))
    return res


core.solve = capture
run._fuse_final()
core.solve = _solve
first, second = cap[0], cap[1]
m, es, el, sh = second["model"], second["extra_s"], second["extra_l"], second["s_shift"]
F = second["res"].seg_lab
stF = core.evaluate(m, F, es, el, s_shift=sh)
print(f"{key} {cell}: kept final J at the solve {second['J']:.1f} (after fuse_final's metric-line re-evaluation "
      f"{second['res'].state.J:.1f}), re-evaluated {stF.J:.1f}; segments V != F "
      f"{int((V != F).sum())}")


def pair(tag, mod, xs, xl, shift):
    a = core.evaluate(mod, V, xs, xl, s_shift=shift)
    b = core.evaluate(mod, F, xs, xl, s_shift=shift)
    print(f"  {tag:34s} J(V) - J(F) {a.J - b.J:+9.1f}  "
          + core._fmt({k: a.parts[k] - b.parts[k] for k in a.parts}))


pair("final evidence", m, es, el, sh)
off = first["extra_s"]  # the first solve's extra along-track observations: the official ones
n_off = len(off.t) if off is not None else 0
fin_only = core.Obs(es.h[n_off:], es.t[n_off:], es.y[n_off:], es.sd[n_off:]) if es is not None else None
pair("without official s observations", m, fin_only, el, sh)
pair("without finish crossings", m, off, el, sh)
pair("without extra lateral observations", m, es, None, sh)
T = np.load(run.p("tracks.npz"), allow_pickle=True)
ev0 = run._evidence(T)
m0 = core.Model(ev0, m.Lr, m.tg, run.info.width)
pair("metric shift undone", m0, es, el, None)
pair("no shift, no extra observations", m0, None, None, None)
ev = m.ev
lo = np.full(ev.n_seg, np.inf)
np.minimum.at(lo, ev.seg, ev.t)
print("  by window (F with V's labels on the segments starting in it; final evidence):")
for w0 in np.arange(np.floor(lo[np.isfinite(lo)].min() / 10) * 10, lo[np.isfinite(lo)].max() + 10, 10):
    inside = (lo >= w0) & (lo < w0 + 10) & (V != F)
    if not inside.any():
        continue
    H = F.copy()
    H[inside] = V[inside]
    st = core.evaluate(m, H, es, el, X0=stF.S, s_shift=sh)
    print(f"    {w0:5.0f}-{w0 + 10:3.0f} s: {int(inside.sum()):3d} segments, J {st.J - stF.J:+8.1f} "
          + core._fmt({k: st.parts[k] - stF.parts[k] for k in st.parts}))
shutil.rmtree(tmp)
