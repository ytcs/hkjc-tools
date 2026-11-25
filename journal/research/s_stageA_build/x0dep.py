"""x0dep.py RUNROOT KEY CELL: does J(labelling) depend on the state step's warm start? Solves from the init and from
the reads+silk init (multistart.py's), then evaluates each end labelling warm-started from the other end's state
and cold (X0 None)."""
import sys

import numpy as np

exec(open(__file__.replace("x0dep.py", "multistart.py")).read().split("runs = []")[0])
r0, _ = solve_from(None)
U0 = (ev.read_cost + ev.silk_cost).copy()
U0[ev.forbid] = core.INF
sl = core.milp(np.where(U0 < 0, U0, 0.0) - 1e-3 * (U0 < 0), ev, allow=U0 < 0)
r1, _ = solve_from(sl)
for name, a, b in (("init end", r0, r1), ("reads+silk end", r1, r0)):
    jc = core.evaluate(m, a.seg_lab, ex_s, ex_l).J
    jw = core.evaluate(m, a.seg_lab, ex_s, ex_l, X0=b.state.S).J
    print(f"{name}: J in solve {a.state.J:.1f}  cold {jc:.1f}  warm from the other end's state {jw:.1f}")
