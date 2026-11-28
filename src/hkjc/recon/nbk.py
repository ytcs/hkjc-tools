"""Numba kernels for the core's hot loops (cached compilation). Bandwidth-2 lower bands (3 x n: diagonal, first and second subdiagonal) as core uses them."""
import math

import numpy as np
from numba import njit

SQRT1_2 = 1.0 / math.sqrt(2.0)


@njit(cache=True)
def band_chol(ab):
    """Lower Cholesky factor of a 3 x n band; returns (factor, ok)."""
    n = ab.shape[1]
    c = np.zeros_like(ab)
    for j in range(n):
        d = ab[0, j]
        if j >= 1:
            d -= c[1, j - 1] * c[1, j - 1]
        if j >= 2:
            d -= c[2, j - 2] * c[2, j - 2]
        if d <= 0.0:
            return c, False
        c[0, j] = math.sqrt(d)
        if j + 1 < n:
            v = ab[1, j]
            if j >= 1:
                v -= c[2, j - 1] * c[1, j - 1]
            c[1, j] = v / c[0, j]
        if j + 2 < n:
            c[2, j] = ab[2, j] / c[0, j]
    return c, True


@njit(cache=True)
def band_solve(c, r):
    n = c.shape[1]
    z = np.empty(n)
    for j in range(n):
        v = r[j]
        if j >= 1:
            v -= c[1, j - 1] * z[j - 1]
        if j >= 2:
            v -= c[2, j - 2] * z[j - 2]
        z[j] = v / c[0, j]
    x = np.empty(n)
    for j in range(n - 1, -1, -1):
        v = z[j]
        if j + 1 < n:
            v -= c[1, j] * x[j + 1]
        if j + 2 < n:
            v -= c[2, j] * x[j + 2]
        x[j] = v / c[0, j]
    return x


@njit(cache=True)
def band_matvec(ab, x):
    n = x.shape[0]
    y = ab[0] * x
    for d in range(1, 3):
        for i in range(d, n):
            y[i] += ab[d, i - d] * x[i - d]
            y[i - d] += ab[d, i - d] * x[i]
    return y


@njit(cache=True)
def add_obs(ab, r, tg0, dt, nt, t, y, sd, sign):
    for i in range(t.shape[0]):
        xx = (t[i] - tg0) / dt
        k = int(math.floor(xx))
        if k < 0:
            k = 0
        if k > nt - 2:
            k = nt - 2
        w = xx - k
        if w < 0.0:
            w = 0.0
        if w > 1.0:
            w = 1.0
        a, b = (1 - w) / sd[i], w / sd[i]
        ab[0, k] += sign * a * a
        ab[0, k + 1] += sign * b * b
        ab[1, k] += sign * a * b
        r[k] += sign * a * y[i] / sd[i]
        r[k + 1] += sign * b * y[i] / sd[i]


@njit(cache=True)
def horse_quick(A, X, ld0, tg0, dt, nt, td, yd, sdd, ta, ya, sda, inf):
    """core.Refit._horse_quick: (dJ_s, s mode)."""
    ab = A.copy()
    r = band_matvec(A, X)
    q0 = 0.0
    for i in range(X.shape[0]):
        q0 += X[i] * r[i]
    dc = 0.0
    if td.shape[0]:
        add_obs(ab, r, tg0, dt, nt, td, yd, sdd, -1.0)
        for i in range(td.shape[0]):
            dc -= (yd[i] / sdd[i]) ** 2
    if ta.shape[0]:
        add_obs(ab, r, tg0, dt, nt, ta, ya, sda, 1.0)
        for i in range(ta.shape[0]):
            dc += (ya[i] / sda[i]) ** 2
    for j in range(ab.shape[1]):
        if ab[0, j] < 1e-12:
            ab[0, j] = 1e-12
    c, ok = band_chol(ab)
    if not ok:
        return inf, X.copy()
    x = band_solve(c, r)
    rx = 0.0
    ld = 0.0
    for i in range(x.shape[0]):
        rx += r[i] * x[i]
        ld += math.log(c[0, i])
    return dc - rx + q0 + 2.0 * ld - ld0, x


@njit(cache=True)
def lat_solve(ab, r, c0, lb, ub, ld_prior):
    """core.Model.lat_one after its observations are in ab, r: (box-bounded mode, J term)."""
    n = ab.shape[1]
    pen = np.zeros(n)
    tgt = np.zeros(n)
    big = ab[0].max()
    x = np.zeros(n)
    for _ in range(8):
        abp = ab.copy()
        for j in range(n):
            abp[0, j] += pen[j]
        cf, ok = band_chol(abp)
        x = band_solve(cf, r + pen * tgt)
        any_v = False
        for j in range(n):
            if pen[j] == 0.0 and (x[j] < lb - 1e-6 or x[j] > ub + 1e-6):
                any_v = True
        if not any_v:
            break
        for j in range(n):
            if pen[j] == 0.0 and x[j] < lb - 1e-6:
                pen[j] = 1e4 * big
                tgt[j] = lb
            elif pen[j] == 0.0 and x[j] > ub + 1e-6:
                pen[j] = 1e4 * big
                tgt[j] = ub
    q = 0.0
    for j in range(n):
        q += ab[0, j] * x[j] * x[j]
    for d in range(1, 3):
        for j in range(n - d):
            q += 2.0 * ab[d, j] * x[j] * x[j + d]
    rx = 0.0
    for j in range(n):
        rx += r[j] * x[j]
    cf, ok = band_chol(ab)
    ld = 0.0
    for j in range(n):
        ld += math.log(cf[0, j])
    return x, q - 2.0 * rx + c0 + 2.0 * ld - ld_prior


@njit(cache=True)
def ndtr(x):
    return 0.5 * math.erfc(-x * SQRT1_2)


@njit(cache=True)
def view(Sr, Lr, sdSr, sdLr, k, w, A, x0, x1, y0, y1, view_px):
    """core.in_view on path rows: (rows, frames) P(in view)."""
    R, F = Sr.shape[0], k.shape[0]
    out = np.empty((R, F))
    for i in range(R):
        for f in range(F):
            kk, ww = k[f], w[f]
            s_ = Sr[i, kk] * (1 - ww) + Sr[i, kk + 1] * ww
            l_ = Lr[i, kk] * (1 - ww) + Lr[i, kk + 1] * ww
            ss = sdSr[i, kk] * (1 - ww) + sdSr[i, kk + 1] * ww
            sl = sdLr[i, kk] * (1 - ww) + sdLr[i, kk + 1] * ww
            cx = s_ * A[f, 0, 0] + l_ * A[f, 1, 0] + A[f, 2, 0]
            cy = s_ * A[f, 0, 1] + l_ * A[f, 1, 1] + A[f, 2, 1]
            ux = math.hypot(ss * A[f, 0, 0], sl * A[f, 1, 0]) + view_px
            uy = math.hypot(ss * A[f, 0, 1], sl * A[f, 1, 1]) + view_px
            out[i, f] = ndtr((cx - x0) / ux) * ndtr((x1 - cx) / ux) * ndtr((cy - y0) / uy) * ndtr((y1 - cy) / uy)
    return out


@njit(cache=True)
def _cov(W, p0, p1, w0, P, Q):
    out = np.empty((P.shape[0], Q.shape[0]))
    for i in range(P.shape[0]):
        wp, a0, a1 = w0[P[i]], p0[P[i]], p1[P[i]]
        for j in range(Q.shape[0]):
            wq, b0, b1 = w0[Q[j]], p0[Q[j]], p1[Q[j]]
            out[i, j] = ((1 - wp) * (1 - wq) * W[a0, b0] + (1 - wp) * wq * W[a0, b1] + wp * (1 - wq) * W[a1, b0]
                         + wp * wq * W[a1, b1])
    return out


@njit(cache=True)
def _chol(M):
    """Dense lower Cholesky; (L, ok)."""
    n = M.shape[0]
    L = np.zeros((n, n))
    for j in range(n):
        d = M[j, j]
        for k in range(j):
            d -= L[j, k] * L[j, k]
        if d <= 0.0:
            return L, False
        L[j, j] = math.sqrt(d)
        for i in range(j + 1, n):
            v = M[i, j]
            for k in range(j):
                v -= L[i, k] * L[j, k]
            L[i, j] = v / L[j, j]
    return L, True


@njit(cache=True)
def _fwd(L, b):
    """Solve L x = b (b: n x c)."""
    n, c = b.shape
    x = np.empty((n, c))
    for col in range(c):
        for i in range(n):
            v = b[i, col]
            for k in range(i):
                v -= L[i, k] * x[k, col]
            x[i, col] = v / L[i, i]
    return x


@njit(cache=True)
def refit_group(W, p0, p1, w0, y, R, mu, rm, ix, inf):
    """kin_costs' one-horse refit for one conflict group (core._refit_block): rm the removed chunks, ix (G, m) the
    scored segments' chunks -> (costs (G,), ok); ok False when the downdate's matrix is not positive definite."""
    r = rm.shape[0]
    G, m = ix.shape
    sr = np.sqrt(R[rm])
    Crr = _cov(W, p0, p1, w0, rm, rm)
    Sm = np.empty((r, r))
    for i in range(r):
        for j in range(r):
            Sm[i, j] = (1.0 if i == j else 0.0) - Crr[i, j] / sr[i] / sr[j]
    for i in range(r):
        for j in range(i):
            v = 0.5 * (Sm[i, j] + Sm[j, i])
            Sm[i, j] = v
            Sm[j, i] = v
    cS, ok = _chol(Sm)
    out = np.full(G, inf)
    if not ok:
        return out, False
    b = np.empty((r, 1))
    for i in range(r):
        b[i, 0] = (y[rm[i]] - mu[rm[i]]) / sr[i]
    t2 = _fwd(cS, b)[:, 0]
    flat = ix.ravel()
    Cfr = _cov(W, p0, p1, w0, flat, rm)
    B = np.empty((r, flat.shape[0]))
    for i in range(r):
        for j in range(flat.shape[0]):
            B[i, j] = Cfr[j, i] / sr[i]
    T = _fwd(cS, B)  # r x G m
    for g in range(G):
        idx = ix[g]
        C = _cov(W, p0, p1, w0, idx, idx)
        for a in range(m):
            for c in range(m):
                v = 0.0
                for q in range(r):
                    v += T[q, g * m + a] * T[q, g * m + c]
                C[a, c] += v
        P = np.empty((m, m))
        for a in range(m):
            for c in range(m):
                P[a, c] = 0.5 * (C[a, c] + C[c, a])
            P[a, a] += R[idx[a]]
        res = np.empty((m, 1))
        for a in range(m):
            v = 0.0
            for q in range(r):
                v += T[q, g * m + a] * t2[q]
            res[a, 0] = y[idx[a]] - (mu[idx[a]] - v)
        cp, okg = _chol(P)
        if not okg:
            continue
        z = _fwd(cp, res)[:, 0]
        val = 0.0
        for a in range(m):
            val += z[a] * z[a] + 2.0 * math.log(cp[a, a]) - math.log(R[idx[a]])
        out[g] = val
    return out, True
