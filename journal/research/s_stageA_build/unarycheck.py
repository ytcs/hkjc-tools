"""unarycheck.py RUNROOT KEY CELL SEED SEG...: at the end of the solve from multistart's perturbation SEED, each listed
segment's unaries (assignment-step costs) against the exact J change of relabelling that segment alone (overlapping
holders of the new label dropped to clutter)."""
import sys

import numpy as np

segs = [int(x) for x in sys.argv[5:]]
seed = int(sys.argv[4])
sys.argv = sys.argv[:4]
exec(open(__file__.replace("unarycheck.py", "multistart.py")).read().split("runs = []")[0])
r0, _ = solve_from(None)
r1, _ = solve_from(perturb(r0.seg_lab, seed))
lab, st = r1.seg_lab, r1.state
U = core.unaries(m, st, lab)
H = ev.horses
for k in segs:
    cur = lab[k]
    base = U[k, cur] if cur >= 0 else 0.0
    out = []
    for h in list(range(m.H)) + [-1]:
        if h == cur:
            continue
        t = lab.copy()
        t[k] = h
        if h >= 0:
            t[(t == h) & (ev.seg_lo <= ev.seg_hi[k]) & (ev.seg_hi >= ev.seg_lo[k]) & (np.arange(len(t)) != k)] = -1
        dj = core.evaluate(m, t, ex_s, ex_l, X0=st.S).J - st.J
        du = (U[k, h] if h >= 0 else 0.0) - base
        out.append((dj, du, h))
    out.sort()
    print(f"seg {k} ({lo[k]:.1f}-{hi[k]:.1f} s) now #{H[cur] if cur >= 0 else '-'}: " + "  ".join(
        f"{'#' + str(H[h]) if h >= 0 else 'clutter'} dJ {dj:+.1f} dU {du:+.1f}" for dj, du, h in out[:5]))
K = core.kin_costs(m, st, lab)
LV = core.list_votes(m, st, lab)
B = core.segment_base(ev)[:, None] + ev.read_cost + ev.silk_cost
fi = np.searchsorted(ev.view["frames"], ev.fr)
D = np.zeros_like(K)
for h in range(m.H):
    D[:, h] = np.bincount(ev.seg, st.c_lab[h, fi] - st.c_un[h, fi], ev.n_seg)
for k in segs:
    cur = lab[k]
    for h in [x for x in range(m.H) if x != cur][:0] + [-1] + list({int(np.argsort(U[k])[0]), int(np.argsort(U[k])[1])} - {int(cur)}):
        t = lab.copy()
        t[k] = h
        if h >= 0:
            t[(t == h) & (ev.seg_lo <= ev.seg_hi[k]) & (ev.seg_hi >= ev.seg_lo[k]) & (np.arange(len(t)) != k)] = -1
        p2 = core.evaluate(m, t, ex_s, ex_l, X0=st.S).parts
        dp = ", ".join(f"{q} {p2[q] - st.parts[q]:+.1f}" for q in p2)

        def comp(X):
            return (X[k, h] if h >= 0 else 0.0) - (X[k, cur] if cur >= 0 else 0.0)
        print(f"seg {k} -> {'#' + str(H[h]) if h >= 0 else 'clutter'}: exact {dp} | unary kin {comp(K):+.1f} list "
              f"{comp(LV):+.1f} base {comp(B):+.1f} det {comp(D):+.1f}")
