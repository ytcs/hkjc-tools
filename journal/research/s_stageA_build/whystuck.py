"""whystuck.py RUNROOT KEY CELL (env SEED, or OTHER=cell): solve from the init and from perturbation SEED (multistart.py's); list the
segments where the two ends differ and the J of moving the worse end toward the better one, per connected group of
differing segments (time-overlapping), with and without assignment steps after."""
import sys

import numpy as np

sys.argv = sys.argv[:4] + []  # multistart's argv: RUNROOT KEY CELL
seed = int(__import__("os").environ.get("SEED", "1"))
exec(open(__file__.replace("whystuck.py", "multistart.py")).read().split("runs = []")[0])
r0, _ = solve_from(None)
oth = __import__("os").environ.get("OTHER")
if oth:  # the other end: that cell's stored video labelling, as is (no solve)
    ol = pickle.load(open(root / oth / key / "fuse_video.pkl", "rb"))["seg_lab"]
    r1 = core.Result(ol, core.evaluate(m, ol, ex_s, ex_l), None, None, [])
else:
    r1, _ = solve_from(perturb(r0.seg_lab, seed))
good, bad = (r0, r1) if r0.state.J <= r1.state.J else (r1, r0)
print("J good", round(good.state.J, 1), "bad", round(bad.state.J, 1))
dif = np.flatnonzero(good.seg_lab != bad.seg_lab)
H = ev.horses


def nm(x):
    return f"#{H[x]}" if x >= 0 else "-"


for k in dif[np.argsort(lo[dif])]:
    print(f"  seg {k:3d} {lo[k]:6.1f}-{hi[k]:6.1f} s n={int((ev.seg == k).sum()):4d}  bad {nm(bad.seg_lab[k])} good {nm(good.seg_lab[k])}")
# groups: union of time-overlapping differing segments
order = dif[np.argsort(lo[dif])]
groups, cur, end = [], [], -1e9
for k in order:
    if cur and lo[k] > end + 0.2:
        groups.append(cur)
        cur = []
    cur.append(k)
    end = max(end, hi[k]) if len(cur) > 1 else hi[k]
if cur:
    groups.append(cur)
for g in groups:
    t = bad.seg_lab.copy()
    t[g] = good.seg_lab[g]
    j = core.evaluate(m, t, ex_s, ex_l).J
    print(f"group {lo[g].min():.1f}-{hi[g].max():.1f} s ({len(g)} segs, labels {sorted(set(nm(x) for x in bad.seg_lab[g]))}): "
          f"bad + good's labels there J {j:.1f} ({j - bad.state.J:+.1f})")
# finer: parts linked by time overlap AND a shared label (each part alone keeps exclusivity)
ch = dif
par = list(range(len(ch)))


def rt(i):
    while par[i] != i:
        i = par[i]
    return i


for i in range(len(ch)):
    for j in range(i + 1, len(ch)):
        a, b = ch[i], ch[j]
        la = {int(bad.seg_lab[a]), int(good.seg_lab[a])} - {-1}
        if ev.seg_lo[a] <= ev.seg_hi[b] and ev.seg_hi[a] >= ev.seg_lo[b] and la & {int(bad.seg_lab[b]),
                                                                                 int(good.seg_lab[b])}:
            par[rt(i)] = rt(j)
r = np.array([rt(i) for i in range(len(ch))])
tot = 0.0
for u in np.unique(r):
    g = ch[r == u]
    t = bad.seg_lab.copy()
    t[g] = good.seg_lab[g]
    j = core.evaluate(m, t, ex_s, ex_l, X0=bad.state.S).J - bad.state.J
    tot += j
    print(f"part {lo[g].min():6.1f}-{hi[g].max():6.1f} s {len(g):3d} segs "
          f"{sorted(set(nm(x) for x in bad.seg_lab[g]) | set(nm(x) for x in good.seg_lab[g]))}: alone {j:+.1f}")
print("sum of parts alone", round(tot, 1))
