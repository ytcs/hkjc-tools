"""refit.py RUNROOT KEY CELL [N]: the assignment step's linearised cost change of the relabel pass's candidates
(core.single_relabels at CELL's stored video labelling), as is and with a one-horse refit (the kinematic term of the
segments moved to horse y scored jointly against y's posterior with y's own observations over their time span left
out), against the exact J change of a full state step. Prints Spearman rank correlations and the share of
exact-negative relabels each ranks in its top 50. N: candidates evaluated exactly (every k-th; default 200)."""
import sys

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.stats import spearmanr

N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
sys.argv = sys.argv[:4]
exec(open(__file__.replace("refit.py", "multistart.py")).read().split("runs = []")[0])
lab = V["seg_lab"]
st = core.evaluate(m, lab, ex_s, ex_l)
U = core.unaries(m, st, lab)
K = core.kin_costs(m, st, lab)
Uc = np.c_[U, np.zeros(len(U))]
tg = m.tg


def refit_kin(segs, y):
    """-2 log predictive of the chunks of ``segs`` under horse y's posterior without y's observations in their span."""
    c = np.flatnonzero(np.isin(ev.ch_seg, segs))
    if not len(c):
        return 0.0
    t0, t1 = ev.ch_t[c].min() - 0.5, ev.ch_t[c].max() + 0.5
    tot = 0.0
    for coord in ("s", "l"):
        A = (st.As if coord == "s" else st.Al)[y]
        X = (st.S if coord == "s" else st.L)[y]
        ob = st.obs_s if coord == "s" else st.obs_l
        yv = (ev.ch_s if coord == "s" else ev.ch_l)[c]
        R = ((ev.ch_sds if coord == "s" else ev.ch_sdl)[c]) ** 2
        ab, r = A.copy(), core._band_matvec(A, X)
        mm = (ob.h == y) & (ob.t >= t0) & (ob.t <= t1) & (ob.sd < 1e3)
        if mm.any():
            core._band_add_obs(ab, r, tg, ob.t[mm], ob.y[mm], ob.sd[mm], -1.0)
        ab[0] = np.maximum(ab[0], 1e-12)
        try:
            cb = cholesky_banded(ab, lower=True)
        except np.linalg.LinAlgError:
            return core.INF
        x = cho_solve_banded((cb, True), r)
        kk, ww = core._interp(tg, ev.ch_t[c])
        Hm = np.zeros((len(c), len(tg)))
        Hm[np.arange(len(c)), kk] += 1 - ww
        Hm[np.arange(len(c)), kk + 1] += ww
        C = Hm @ cho_solve_banded((cb, True), Hm.T) + np.diag(R)
        cp = np.linalg.cholesky(C)
        z = np.linalg.solve(cp, yv - Hm @ x)
        tot += float(z @ z) + 2.0 * float(np.log(np.diag(cp)).sum()) - float(np.log(R).sum())
    return tot


cands = core.single_relabels(ev, lab, st)
step = max(1, len(cands) // N)
rows = []
for x, y, new in cands[::step]:
    ch = np.flatnonzero(new != lab)
    old_u = Uc[ch, np.where(lab[ch] >= 0, lab[ch], -1)]
    new_u = Uc[ch, np.where(new[ch] >= 0, new[ch], -1)]
    lin = float((new_u - old_u).sum())
    ref = lin
    if y >= 0:
        moved = ch[new[ch] == y]
        ref = lin - float(K[moved, y].sum()) + refit_kin(moved, y)
    ex = core.evaluate(m, new, ex_s, ex_l, X0=st.S).J - st.J
    rows.append((lin, ref, ex))
r = np.array(rows)
neg = r[:, 2] < -core.MIN_GAIN
print(f"{key} {cell}: {len(cands)} relabels, {len(r)} evaluated, exact-negative {int(neg.sum())}")
for name, col in (("linearised", 0), ("one-horse refit", 1)):
    rho = spearmanr(r[:, col], r[:, 2]).correlation
    top = np.argsort(r[:, col])[:50]
    share = neg[top].sum() / max(neg.sum(), 1)
    print(f"  {name}: Spearman {rho:.3f}; exact-negative in its top 50: {int(neg[top].sum())} of {int(neg.sum())}"
          f" ({100 * share:.0f} %)")
