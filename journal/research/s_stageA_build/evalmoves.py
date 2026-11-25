"""evalmoves.py RUNROOT KEY CELL SPEC...: J of CELL's stored video labelling after pair exchanges, under CELL's video
evidence. SPEC: 'a/b@t0-t1' exchanges horses a and b on every segment inside [t0, t1]; specs joined by '+' are
applied together."""
import pickle
import sys

import numpy as np

specs = sys.argv[4:]
sys.argv = sys.argv[:4]
exec(open(__file__.replace("evalmoves.py", "multistart.py")).read().split("runs = []")[0])
lab0 = V["seg_lab"]
st0 = core.evaluate(m, lab0, ex_s, ex_l)
print("stored labelling J", round(st0.J, 1))
for spec in specs:
    lab = lab0.copy()
    for one in spec.split("+"):
        pair, span = one.split("@")
        a, b = (col[int(x)] for x in pair.split("/"))
        t0, t1 = (float(x) for x in span.split("-"))
        inside = (lo >= t0 - 0.15) & (hi <= t1 + 0.05)
        ia, ib = inside & (lab == a), inside & (lab == b)
        lab[ia], lab[ib] = b, a
    st = core.evaluate(m, lab, ex_s, ex_l, X0=st0.S)
    print(f"{spec}: J {st.J:.1f} ({st.J - st0.J:+.1f}) {core._fmt({k: st.parts[k] - st0.parts[k] for k in st.parts})}")
