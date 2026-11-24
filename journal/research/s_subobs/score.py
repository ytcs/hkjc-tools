"""s_subobs scorer (DECLARATION sections 3-5). usage: score.py check | full   (run from the worktree root)
  check: off arm only (scorer check) -> stdout
  full:  P1, P2, G1-G4, information -> data_cache/s_subobs/out/summary.txt (+ rows parquet)"""
import filecmp
import sys
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.data import race_info
from hkjc.data.race import official_boundaries, official_subsplits
from hkjc.recon.validate import crossing

RUNS, OUT = Path("data_cache/s_subobs/runs"), Path("data_cache/s_subobs/out")
OUT.mkdir(parents=True, exist_ok=True)
JOBS = [ln.split() for ln in open("data_cache/s_subobs/races.txt") if ln.strip()]
NB = 2000
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


def cross(t, s, smark):
    kk = np.where(s >= smark)[0]
    if len(kk) == 0 or kk[0] == 0:
        return np.nan, np.nan
    a, b = kk[0] - 1, kk[0]
    f = (smark - s[a]) / max(s[b] - s[a], 1e-9)
    return t[a] + f * (t[b] - t[a]), (s[b] - s[a]) / (t[b] - t[a])


def load(cell, k):
    fn = RUNS / cell / k / "paths.parquet"
    if not fn.exists():
        return None
    P = pl.read_parquet(fn, columns=["horse_no", "t", "s"]).sort("horse_no", "t")
    return {int(h): (g["t"].to_numpy(), g["s"].to_numpy()) for (h,), g in P.group_by(["horse_no"])}


RACE = {}
for setn, k, _src, j, nb in JOBS:
    d, v, n = k.split("_")
    info = race_info(d, v, int(n))
    bnd, cb = official_boundaries(info)
    pts, cs = official_subsplits(info)
    toff = {}
    for h, c in cb.items():
        for b, t in zip(bnd, c):
            if np.isfinite(t):
                toff[(int(h), float(b))] = float(t)
    for h, c in cs.items():
        for p, t in zip(pts, c):
            if np.isfinite(t):
                toff[(int(h), float(p))] = float(t)
    RACE[k] = dict(set=setn, venue=v, D=float(info.distance), bnd=bnd, cb=cb, pts=pts, cs=cs, toff=toff, j=int(j))


def crossings(G, k, points):
    r = RACE[k]
    out = {}
    for (h, p), _t in r["toff"].items():
        if p in points and h in G:
            out[(h, p)] = cross(*G[h], r["D"] - p)
    return out


def halves(G, k, hv):
    """[(race, horse, half, frac)] for halves hv = [(a, b)], race-half median over >= 3 horses, ot > 5 s"""
    r = RACE[k]
    pts = {x for ab in hv for x in ab}
    c = crossings(G, k, pts)
    out = []
    for a, b in hv:
        rows = []
        for h in {h for h, _p in r["toff"]}:
            if (h, a) in c and (h, b) in c and (h, a) in r["toff"] and (h, b) in r["toff"]:
                ot = r["toff"][(h, b)] - r["toff"][(h, a)]
                pt = c[(h, b)][0] - c[(h, a)][0]
                if np.isfinite(ot) and np.isfinite(pt) and ot > 5:
                    rows.append((h, pt - ot, ot))
        if len(rows) >= 3:
            m = np.median([x[1] for x in rows])
            out += [dict(race=k, horse=h, half=f"{a:.0f}-{b:.0f}", frac=abs(dt - m) / ot) for h, dt, ot in rows]
    return out


def point_err(G, k, p):
    """[(race, horse, point, e_rel m)], e = (t_c - t_off) * v, race-point median removed (>= 3 horses)"""
    r = RACE[k]
    c = crossings(G, k, {p})
    e = [(h, (tc - r["toff"][(h, p)]) * vv) for (h, _p), (tc, vv) in c.items() if np.isfinite(tc)]
    if len(e) < 3:
        return []
    m = np.median([x[1] for x in e])
    return [dict(race=k, horse=h, point=p, e_rel=x - m) for h, x in e]


ALLH = [(800.0, 600.0), (600.0, 400.0), (400.0, 200.0), (200.0, 0.0)]


def boot_paired(df, stat):
    a, b = df["a"].to_numpy(), df["b"].to_numpy()
    rng = np.random.default_rng(0)
    ur, inv = np.unique(df["race"].to_numpy(), return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(ur))]
    ds = []
    for _ in range(NB):
        ix = np.concatenate([idx[i] for i in rng.integers(0, len(ur), len(ur))])
        ds.append(stat(b[ix]) - stat(a[ix]))
    lo, hi = np.quantile(ds, [0.05, 0.95])
    return stat(a), stat(b), stat(b) - stat(a), lo, hi


def p90(x):
    return float(np.percentile(x, 90))


def pair(A, B, keys, val):
    return A.join(B, on=keys, suffix="_b").select("race", pl.col(val).alias("a"), pl.col(f"{val}_b").alias("b"))


def check():
    H, E = [], []
    for k in RACE:
        G = load("off_U", k)
        if G is None:
            continue
        H += halves(G, k, ALLH)
        for p in RACE[k]["pts"]:
            E += point_err(G, k, p)
    H, E = pl.DataFrame(H), pl.DataFrame(E)
    x = E["e_rel"].to_numpy()
    say(f"scorer check off_U: all-halves p90 frac {100 * p90(H['frac'].to_numpy()):.3f} % (0285 1.18 %, tol 0.3 pp) "
        f"n {H.height}; sub-split e_rel SD {x.std():.2f} m p90|e| {p90(np.abs(x)):.2f} m (0285 SD 2.0-2.2) n {len(x)}")
    return H, E


def kink_stats(G, k, points, cum):
    """kinks.py jump statistic at crossings of the given (points, {h: times}); finish (0) excluded"""
    r = RACE[k]
    D = r["D"]
    ji = []
    for h, (t, s) in G.items():
        c = cum.get(h)
        if c is None:
            continue
        v = np.diff(s) / np.diff(t)
        tm = 0.5 * (t[1:] + t[:-1])
        for b, to in zip(points, c):
            if not np.isfinite(to) or b == 0.0:
                continue
            tc = crossing(t, D - s, None, b)
            if not np.isfinite(tc):
                continue
            kk = np.searchsorted(tm, tc)
            if kk < 13 or kk + 13 > len(v):
                continue
            vb, va = v[kk - 3:kk].mean(), v[kk:kk + 3].mean()
            acc = (v[kk + 3:kk + 13].mean() - v[kk - 13:kk - 3].mean()) / (tm[kk + 7] - tm[kk - 8])
            ji.append(abs((va - vb) - acc * (tm[kk + 1] - tm[kk - 2])))
    return ji


def full():
    check()
    # P1 / P2
    P1, P2 = {"off": [], "on": []}, {"off": [], "on": []}
    for k, r in RACE.items():
        Goff = load("off_U", k)
        for kk, m in enumerate(r["pts"]):
            Gon = load(f"on_S{kk}", k)
            if Goff is None or Gon is None:
                continue
            hv = [(m + 200.0, m), (m, m - 200.0)]
            for arm, G in (("off", Goff), ("on", Gon)):
                P1[arm] += [dict(x, k=kk) for x in halves(G, k, hv)]
                P2[arm] += [dict(x, k=kk) for x in point_err(G, k, m)]
    A1, B1 = pl.DataFrame(P1["off"]), pl.DataFrame(P1["on"])
    A2, B2 = pl.DataFrame(P2["off"]), pl.DataFrame(P2["on"])
    A1.write_parquet(OUT / "p1_off.parquet"), B1.write_parquet(OUT / "p1_on.parquet")
    A2.write_parquet(OUT / "p2_off.parquet"), B2.write_parquet(OUT / "p2_on.parquet")
    W1 = pair(A1, B1, ["race", "horse", "half", "k"], "frac")
    W2 = pair(A2.with_columns(pl.col("e_rel").abs()), B2.with_columns(pl.col("e_rel").abs()), ["race", "horse", "k"],
              "e_rel")
    r1 = boot_paired(W1, p90)
    r2 = boot_paired(W2, p90)
    r2m = boot_paired(W2, lambda x: float(np.median(x)))
    say(f"P1 bracketing-halves p90 frac (off_U -> on_Sk), n {W1.height} rows, {W1['race'].n_unique()} races: "
        f"{100 * r1[0]:.3f} -> {100 * r1[1]:.3f} %  delta {100 * r1[2]:+.3f} pp [{100 * r1[3]:+.3f}, {100 * r1[4]:+.3f}]"
        f"  non-inferior (CI upper < +0.03 pp): {100 * r1[4] < 0.03}; strict (< 0): {r1[4] < 0}")
    say(f"P2 withheld-point p90 |e_rel| m, n {W2.height}: {r2[0]:.3f} -> {r2[1]:.3f}  delta {r2[2]:+.3f} "
        f"[{r2[3]:+.3f}, {r2[4]:+.3f}]  non-inferior (< +0.05 m): {r2[4] < 0.05}; strict: {r2[4] < 0}")
    say(f"   P2 median |e_rel|: {r2m[0]:.3f} -> {r2m[1]:.3f} delta {r2m[2]:+.3f} [{r2m[3]:+.3f}, {r2m[4]:+.3f}]")
    say("   by withheld point / venue (information):")
    ven = pl.DataFrame([dict(race=k, venue=r["venue"]) for k, r in RACE.items()])
    for kk in sorted(set(A1["k"].to_list())):
        for v in ("HV", "ST"):
            w1 = pair(A1.filter(pl.col("k") == kk), B1.filter(pl.col("k") == kk), ["race", "horse", "half", "k"],
                      "frac").join(ven, on="race").filter(pl.col("venue") == v)
            w2 = pair(A2.filter(pl.col("k") == kk).with_columns(pl.col("e_rel").abs()),
                      B2.filter(pl.col("k") == kk).with_columns(pl.col("e_rel").abs()), ["race", "horse", "k"],
                      "e_rel").join(ven, on="race").filter(pl.col("venue") == v)
            if w1.height and w2.height:
                say(f"   k={kk} {v}: P1 {100 * p90(w1['a'].to_numpy()):.3f} -> {100 * p90(w1['b'].to_numpy()):.3f} % "
                    f"(n {w1.height}); P2 {p90(w2['a'].to_numpy()):.3f} -> {p90(w2['b'].to_numpy()):.3f} m (n {w2.height})")
    ok_p = (100 * r1[4] < 0.03) and (r2[4] < 0.05)
    strict = r1[4] < 0 and r2[4] < 0
    # G1 / G1f
    guards = {}
    for tag, cells in (("G1 interior", ("off_H", "on_H")), ("G1f finish", ("off_F", "on_F"))):
        rows = {c: [] for c in cells}
        for k, r in RACE.items():
            jj = r["j"] if tag.startswith("G1 ") else len(r["bnd"]) - 1
            b = float(r["bnd"][jj])
            for c in cells:
                G = load(c, k)
                if G is None:
                    continue
                for (h, p), (tc, _v) in crossings(G, k, {b}).items():
                    if np.isfinite(tc):
                        rows[c].append(dict(race=k, horse=h, err=abs(tc - r["toff"][(h, p)])))
        W = pair(pl.DataFrame(rows[cells[0]]), pl.DataFrame(rows[cells[1]]), ["race", "horse"], "err")
        rm = boot_paired(W, lambda x: float(np.median(x)))
        rp = boot_paired(W, p90)
        g = rm[4] < 0.005 and rp[4] < 0.010
        guards[tag] = g
        say(f"{tag}: n {W.height}; median {rm[0]:.4f} -> {rm[1]:.4f} [{rm[3]:+.4f}, {rm[4]:+.4f}]; p90 {rp[0]:.4f} -> "
            f"{rp[1]:.4f} [{rp[3]:+.4f}, {rp[4]:+.4f}]  {'ok' if g else 'WORSE'}")
    # G2 kinks
    kj = {"off_U": {"b": [], "s": []}, "on_U": {"b": [], "s": []}}
    for k, r in RACE.items():
        for c in kj:
            G = load(c, k)
            if G is None:
                continue
            kj[c]["b"] += kink_stats(G, k, r["bnd"], {int(h): v for h, v in r["cb"].items()})
            kj[c]["s"] += kink_stats(G, k, r["pts"], {int(h): v for h, v in r["cs"].items()})
    g2 = True
    for part, nm in (("b", "boundaries"), ("s", "sub-split points")):
        a, b = np.array(kj["off_U"][part]), np.array(kj["on_U"][part])
        ok = np.median(b) <= 1.05 * np.median(a) and np.percentile(b, 95) <= 1.05 * np.percentile(a, 95)
        g2 &= bool(ok)
        say(f"G2 kinks at {nm}: median {np.median(a):.4f} -> {np.median(b):.4f}, p95 {np.percentile(a, 95):.4f} -> "
            f"{np.percentile(b, 95):.4f} m/s (n {len(a)}/{len(b)})  {'ok' if ok else 'WORSE'} (<= x1.05)")
    guards["G2"] = g2
    # G3 residuals at visible boundaries (unmasked)
    res = {"off_U": [], "on_U": []}
    fin = {"off_U": 0, "on_U": 0}
    for k, r in RACE.items():
        for c in res:
            G = load(c, k)
            if G is None:
                continue
            for b in r["bnd"]:
                cr = crossings(G, k, {float(b)})
                e = [(h, tc - r["toff"][(h, p)]) for (h, p), (tc, _v) in cr.items() if np.isfinite(tc)]
                if b == 0.0:
                    fin[c] += sum(abs(x) > 0.02 for _h, x in e)
                elif len(e) >= 3:
                    m = np.median([x for _h, x in e])
                    res[c] += [abs(x - m) for _h, x in e]
    a, b = np.array(res["off_U"]), np.array(res["on_U"])
    dm, dp = np.median(b) - np.median(a), p90(b) - p90(a)
    g3 = dm <= 0.02 and dp <= 0.02 and fin["on_U"] <= fin["off_U"]
    guards["G3"] = g3
    say(f"G3 visible interior |r| (s): median {np.median(a):.4f} -> {np.median(b):.4f} ({dm:+.4f}), p90 {p90(a):.4f} -> "
        f"{p90(b):.4f} ({dp:+.4f}); finish |t_c - t_off| > 0.02 s: {fin['off_U']} -> {fin['on_U']}  "
        f"{'ok' if g3 else 'FAIL'}")
    # G4 plumbing
    same, n4 = 0, 0
    for k, r in RACE.items():
        if r["bnd"][r["j"]] == 400.0 and (RUNS / "on_H" / k / "paths.parquet").exists():
            n4 += 1
            same += filecmp.cmp(RUNS / "on_H" / k / "paths.parquet", RUNS / "off_H" / k / "paths.parquet", shallow=False)
    k0 = JOBS[0][1]
    s0 = filecmp.cmp(RUNS / "off_S0" / k0 / "paths.parquet", RUNS / "off_U" / k0 / "paths.parquet", shallow=False)
    g4 = same == n4 and s0
    guards["G4"] = g4
    say(f"G4 on_H == off_H where the 400 m boundary is withheld: {same}/{n4}; off_S0 == off_U ({k0}): {s0}  "
        f"{'ok' if g4 else 'FAIL'}")
    # information
    I1, I2 = {"off_U": [], "on_U": []}, {"off_U": [], "on_U": []}
    for k, r in RACE.items():
        for c in I1:
            G = load(c, k)
            if G is None:
                continue
            I1[c] += halves(G, k, ALLH)
            for p in r["pts"]:
                I2[c] += point_err(G, k, p)
    w = pair(pl.DataFrame(I1["off_U"]), pl.DataFrame(I1["on_U"]), ["race", "horse", "half"], "frac")
    w2 = pair(pl.DataFrame(I2["off_U"]), pl.DataFrame(I2["on_U"]), ["race", "horse", "point"], "e_rel")
    say(f"info (circular, unmasked on_U vs off_U): all-halves p90 {100 * p90(w['a'].to_numpy()):.3f} -> "
        f"{100 * p90(w['b'].to_numpy()):.3f} % (n {w.height}); sub-split e_rel SD {w2['a'].std():.2f} -> "
        f"{w2['b'].std():.2f} m, p90|e| {p90(np.abs(w2['a'].to_numpy())):.2f} -> {p90(np.abs(w2['b'].to_numpy())):.2f} m")
    ch = []
    for k in RACE:
        fa, fb = RUNS / "off_U" / k / "det_labels_final.npz", RUNS / "on_U" / k / "det_labels_final.npz"
        if fa.exists() and fb.exists():
            la, lb = np.load(fa)["labels"], np.load(fb)["labels"]
            if la.shape == lb.shape and (la != lb).any():
                ch.append(f"{k}:{int((la != lb).sum())}")
    say(f"info det_labels_final changed (on_U vs off_U) in {len(ch)} races: {' '.join(ch)}")
    gall = all(guards.values())
    verdict = ("STRICT IMPROVEMENT" if strict and gall else "PASS" if ok_p and gall else
               "AMBIGUOUS" if gall and r1[2] <= 0 and r2[2] <= 0 else "FAIL")
    say(f"guards {guards}; VERDICT {verdict}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")


{"check": check, "full": full}[sys.argv[1]]()
