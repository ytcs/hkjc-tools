"""bench.py RUNROOT KEY CELL MODE: one cell's video-evidence solve as the product runs it (the solver's own init),
end J and CPU. MODE base: that solve. MODE multi: also full solves forced from the list-only init and from the inits
built with the list rows of alternate 7 s blocks (+ reads + silk); prints each end and the lowest."""
import sys
import time

import numpy as np

mode = sys.argv[4]
sys.argv = sys.argv[:4]
exec(open(__file__.replace("bench.py", "multistart.py")).read().split("runs = []")[0])
import copy  # noqa: E402

t0 = time.process_time()
r0, c0 = solve_from(None)
ends = [("init", r0.state.J, c0)]
if mode == "multi":
    rs = ev.read_cost + ev.silk_cost

    def votes_block(par, P=7.0):
        m2 = copy.copy(m)
        keep = (np.floor(Lr.tau / P).astype(int) % 2) == par
        m2.Lr = core.ListRows(Lr.tau, np.where(keep[:, None], Lr.slots, -1), Lr.full & keep, Lr.same & keep)
        return _lv(m2, None, None)

    for name, U0 in (("list-only", _lv(m, None, None)), ("list7-0+rs", votes_block(0) + rs),
                     ("list7-1+rs", votes_block(1) + rs)):
        U0 = np.array(U0, float)
        U0[ev.forbid] = core.INF
        sl = core.milp(np.where(U0 < 0, U0, 0.0) - 1e-3 * (U0 < 0), ev, allow=U0 < 0)
        r, c = solve_from(sl)
        ends.append((name, r.state.J, c))
for name, j, c in ends:
    print(f"{key} {cell} {mode} {name:11s} end J {j:10.1f}  CPU {c:6.1f}")
best = min(ends, key=lambda x: x[1])
print(f"{key} {cell} {mode} LOWEST {best[1]:.1f} ({best[0]}) total CPU {sum(c for *_, c in ends):.1f}")
