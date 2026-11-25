"""partsdiff.py RUNROOT KEY CELL OTHER: CELL's stored video labelling against OTHER's under CELL's video evidence:
the label-connected parts of their difference (core.crossovers' parts), each part's J change applied alone to CELL's
labelling, and the J of every run of consecutive parts (what a crossover between the two would reach)."""
import pickle
import sys

import numpy as np

oth = sys.argv[4]
sys.argv = sys.argv[:4]
exec(open(__file__.replace("partsdiff.py", "multistart.py")).read().split("runs = []")[0])
a = V["seg_lab"]
b = pickle.load(open(root / oth / key / "fuse_video.pkl", "rb"))["seg_lab"]
sa = core.evaluate(m, a, ex_s, ex_l)
sb = core.evaluate(m, b, ex_s, ex_l)
print(f"J {cell} labels {sa.J:.1f}  {oth} labels {sb.J:.1f}  segments differ {int((a != b).sum())}")
H = ev.horses


def nm(x):
    return f"#{H[x]}" if x >= 0 else "-"


cands = core.crossovers(ev, a, b)

def run_of(d):
    i, j = d.split("parts ")[1].split(" ")[0].split("-")
    return int(i), int(j)


singles = [(d, x) for d, x in cands if run_of(d)[0] == run_of(d)[1]]
for d, x in singles:
    g = np.flatnonzero(x != a)
    st = core.evaluate(m, x, ex_s, ex_l, X0=sa.S)
    print(f"{d}: {lo[g].min():6.1f}-{hi[g].max():6.1f} s, labels {sorted({nm(v) for v in a[g]} | {nm(v) for v in b[g]})}"
          f" alone {st.J - sa.J:+.1f} {core._fmt({k: st.parts[k] - sa.parts[k] for k in st.parts})}")
best = None
for d, x in cands:
    st = core.evaluate(m, x, ex_s, ex_l, X0=sa.S, quick=sa)
    if best is None or st.J < best[1]:
        best = (d, st.J)
print(f"{len(cands)} runs of parts; lowest by the screen: {best[0]} ({best[1] - sa.J:+.1f})")
