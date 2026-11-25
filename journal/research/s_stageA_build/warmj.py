"""warmj.py RUNROOT KEY CELL: does J(labels) depend on the state step's warm start? For the cell's stored video
labelling: two from-scratch evaluations; warm starts from the states of nearby labellings (single relabels of long
segments, the solver's typical predecessor) at the default iteration budget and at a larger one; and the from-scratch
evaluation at the larger budget. Prints J and the change of J_s / J_list / J_det per case."""
import sys

import numpy as np

sys.argv = sys.argv[:4]
exec(open(__file__.replace("warmj.py", "multistart.py")).read().split("runs = []")[0])
lab = V["seg_lab"]
_ss = core.Model.solve_s


def with_budget(rounds, max_it):
    def f(self, obs, X0=None, rounds_=None, max_it_=None, **kw):
        return _ss(self, obs, X0, rounds, max_it)
    return f


def ev_(labs, X0=None, budget=None):
    if budget:
        core.Model.solve_s = lambda self, obs, X0=None, rounds=4, max_it=30: _ss(self, obs, X0, *budget)
    st = core.evaluate(m, labs, ex_s, ex_l, X0=X0)
    core.Model.solve_s = _ss
    return st


a = ev_(lab)
b = ev_(lab)
print(f"from scratch twice: {a.J:.1f} {b.J:.1f}")
c = ev_(lab, budget=(12, 200))
print(f"from scratch, budget 12 rounds x 200 Newton: {c.J:.1f} {core._fmt(c.parts)}")
cands = core.single_relabels(ev, lab, a)
rng = np.random.default_rng(0)
for i in rng.choice(len(cands), size=min(6, len(cands)), replace=False):
    x, y, nb = cands[i]
    sn = ev_(nb)  # the neighbour's own state (from scratch)
    w = ev_(lab, X0=sn.S)
    w2 = ev_(lab, X0=sn.S, budget=(12, 200))
    print(f"warm from relabel seg {x}->{y}: J {w.J:.1f} (scratch {a.J:.1f}, diff {w.J - a.J:+.1f}); "
          f"budget x: {w2.J:.1f} (diff {w2.J - c.J:+.1f}) {core._fmt({k: w.parts[k] - a.parts[k] for k in a.parts})}")
