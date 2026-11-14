"""Camera poses of the AVV render: chain turf steps, close loops, solve a similarity pose graph.

Pose ``G_k`` (3x3 similarity) maps panel px of unique frame k -> WORLD px (frame-0 panel coords,
later rescaled to metres). Edges are measured similarities ``S_ij`` mapping panel_i -> panel_j.
The graph residual compares where a few panel probe points land in the world through both ends of
an edge — expressed in world px, so every edge is weighted in the same physical unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import register


@dataclass
class Edge:
    i: int
    j: int
    S: np.ndarray  # panel_i -> panel_j
    w: float = 1.0
    kind: str = "seq"


@dataclass
class PoseGraph:
    shape: tuple  # panel (h, w)
    G: np.ndarray  # (K,3,3) world <- panel_k
    edges: list = field(default_factory=list)

    def probes(self):
        h, w = self.shape
        xs, ys = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 3), np.linspace(0.25 * h, 0.75 * h, 2))
        return np.c_[xs.ravel(), ys.ravel()]


def _p2m(p):
    """params (lnS, th, tx, ty) -> 3x3"""
    return register.sim_matrix(np.exp(p[0]), p[1], p[2], p[3])


def _m2p(M):
    s = np.hypot(M[0, 0], M[1, 0])
    return np.array([np.log(s), np.arctan2(M[1, 0], M[0, 0]), M[0, 2], M[1, 2]])


def chain(steps: list) -> np.ndarray:
    """Compose consecutive panel_{k-1}->panel_k similarities into world<-panel_k poses."""
    K = len(steps)
    G = np.zeros((K, 3, 3))
    M = np.eye(3)  # panel_0 -> panel_k
    for k in range(K):
        if k > 0:
            M = steps[k] @ M
        G[k] = np.linalg.inv(M)
    return G


def solve(
    pg: PoseGraph, *, fix: int = 0, iters: int = 30, huber: float = 3.0, rigid: bool = False
) -> PoseGraph:
    """Robust least-squares over all poses (frame ``fix`` held). Sparse Jacobian via scipy.

    ``rigid`` pins log-scale to 0 (rotation + translation only). Measured: the render camera DOES
    zoom slowly (a few %), and rigid fits ~5x worse on long-baseline edges -> similarity is default."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    K = len(pg.G)
    X = pg.probes()
    Xh = np.c_[X, np.ones(len(X))]
    P0 = np.array([_m2p(g) for g in pg.G])
    if rigid:
        P0[:, 0] = 0.0
    free = np.array([k for k in range(K) if k != fix])
    pos = -np.ones(K, int)
    pos[free] = np.arange(len(free))
    E = pg.edges
    # pre-transform probes through the edge: x in panel_i maps to S x in panel_j
    SX = [(Xh @ e.S.T) for e in E]

    npar = 3 if rigid else 4
    cols = [1, 2, 3] if rigid else [0, 1, 2, 3]

    def full(theta):
        P = P0.copy()
        P[np.ix_(free, cols)] = theta.reshape(-1, npar)
        return P

    Ii = np.array([e.i for e in E])
    Jn = np.array([e.j for e in E])
    Wt = np.array([e.w for e in E])[:, None, None]
    SXa = np.stack(SX)  # (E,n,3)

    def gmats(P):
        s, th = np.exp(P[:, 0]), P[:, 1]
        c, n = s * np.cos(th), s * np.sin(th)
        G = np.zeros((len(P), 2, 3))
        G[:, 0, 0], G[:, 0, 1], G[:, 0, 2] = c, -n, P[:, 2]
        G[:, 1, 0], G[:, 1, 1], G[:, 1, 2] = n, c, P[:, 3]
        return G

    def res(theta):
        G = gmats(full(theta))
        a = np.einsum("eab,nb->ena", G[Ii], Xh)
        b = np.einsum("eab,enb->ena", G[Jn], SXa)
        return ((a - b) * Wt).ravel()

    J = lil_matrix((len(E) * len(X) * 2, len(free) * npar), dtype=np.int8)
    for n, e in enumerate(E):
        rows = slice(n * len(X) * 2, (n + 1) * len(X) * 2)
        for k in (e.i, e.j):
            if pos[k] >= 0:
                J[rows, pos[k] * npar : pos[k] * npar + npar] = 1
    r = least_squares(
        res,
        P0[np.ix_(free, cols)].ravel(),
        jac_sparsity=J,
        loss="huber",
        f_scale=huber,
        max_nfev=iters,
        x_scale="jac",
        method="trf",
    )
    P = full(r.x)
    return PoseGraph(pg.shape, np.stack([_p2m(p) for p in P]), pg.edges)


def solve_linear(
    pg: PoseGraph, *, fix: int = 0, iters: int = 20, huber: float = 3.0, tol: float = 1e-4
) -> PoseGraph:
    """The SAME robust objective as ``solve`` (Huber on world-px probe residuals
    G_i(x) - G_j(S x), edge weights w), solved EXACTLY by IRLS on a LINEAR system: a 2D similarity is
    z -> A z + B with complex A, B, so every residual A_i x + B_i - A_j (S x) - B_j is linear in the
    unknowns (cf. complex-number planar pose-graph / similarity-synchronization formulations). One
    sparse solve per IRLS round (~ms) instead of a capped nonlinear trf (27 s on R08)."""
    from scipy.sparse import bmat, coo_matrix
    from scipy.sparse.linalg import spsolve

    K = len(pg.G)
    X = pg.probes()
    zx = X[:, 0] + 1j * X[:, 1]
    E = pg.edges
    nE, nP = len(E), len(zx)
    Ii = np.array([e.i for e in E])
    Jn = np.array([e.j for e in E])
    W = np.array([e.w for e in E], float)
    s = np.array([e.S[0, 0] + 1j * e.S[1, 0] for e in E])
    t = np.array([e.S[0, 2] + 1j * e.S[1, 2] for e in E])
    zw = s[:, None] * zx[None, :] + t[:, None]  # (nE, nP) probe images in frame j
    # unknown vector u (complex, length 2K): [A_0..A_{K-1}, B_0..B_{K-1}]
    rows = np.arange(nE * nP)
    ei = np.repeat(np.arange(nE), nP)
    cols = np.c_[Ii[ei], K + Ii[ei], Jn[ei], K + Jn[ei]]
    vals = np.c_[np.tile(zx, nE), np.ones(nE * nP), -zw.ravel(), -np.ones(nE * nP)]
    A_fix = pg.G[fix][0, 0] + 1j * pg.G[fix][1, 0]
    B_fix = pg.G[fix][0, 2] + 1j * pg.G[fix][1, 2]
    free = np.ones(2 * K, bool)
    free[[fix, K + fix]] = False
    fidx = -np.ones(2 * K, int)
    fidx[free] = np.arange(free.sum())
    known = np.zeros(2 * K, complex)
    known[fix], known[K + fix] = A_fix, B_fix
    Mfull = coo_matrix(
        (vals.ravel(), (np.repeat(rows, 4), cols.ravel())), shape=(nE * nP, 2 * K)
    ).tocsc()
    M, rhs0 = Mfull[:, free], -(Mfull[:, ~free] @ known[~free])
    w_rob = np.ones(nE * nP)
    u = None
    for _ in range(iters):
        wr = np.repeat(W, nP) * np.sqrt(w_rob)
        Mw = M.multiply(wr[:, None]).tocsc()
        # complex normal equations -> real 2x system
        N = (Mw.conj().T @ Mw).tocsc()
        b = Mw.conj().T @ (rhs0 * wr)
        Nr = bmat([[N.real, -N.imag], [N.imag, N.real]]).tocsc()
        x = spsolve(Nr, np.r_[b.real, b.imag])
        un = x[: N.shape[0]] + 1j * x[N.shape[0] :]
        r = np.abs(M @ un - rhs0) * np.repeat(W, nP)  # weighted world-px residual
        w_rob = np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-12))
        if u is not None and np.max(np.abs(un - u)) < tol:
            u = un
            break
        u = un
    full = known.copy()
    full[free] = u
    G = np.zeros((K, 3, 3))
    G[:, 0, 0], G[:, 1, 0] = full[:K].real, full[:K].imag
    G[:, 0, 1], G[:, 1, 1] = -full[:K].imag, full[:K].real
    G[:, 0, 2], G[:, 1, 2] = full[K:].real, full[K:].imag
    G[:, 2, 2] = 1.0
    return PoseGraph(pg.shape, G, pg.edges)


def edge_residuals(pg: PoseGraph) -> np.ndarray:
    """Per-edge RMS probe disagreement (world px) under the current poses."""
    X = np.c_[pg.probes(), np.ones(6)]
    out = []
    for e in pg.edges:
        a = X @ pg.G[e.i].T
        b = (X @ e.S.T) @ pg.G[e.j].T
        out.append(float(np.sqrt(((a[:, :2] - b[:, :2]) ** 2).sum(1).mean())))
    return np.array(out)


def centres(G, shape) -> np.ndarray:
    h, w = shape
    return np.einsum("kij,j->ki", G, np.array([w / 2, h / 2, 1.0]))[:, :2]


# ----------------------------------------------------------------------------- whole-race solve
_FEATS = None  # fork-shared feature list for the edge-measurement pool


def _measure(args):
    i, j, prior, gate, min_inl, kind = args
    F = _FEATS
    if kind == "loop":  # chain drift can exceed any gate: match UNGUIDED, accept if near prior
        m = register.match(F[i], F[j], prior=None)
        if m is not None and m.n_inl >= min_inl and m.resid <= 1.3:
            c = np.array([[480.0, 270.0, 1.0]])
            if np.hypot(*((c @ m.S.T) - (c @ prior.T))[0, :2]) > 400:
                m = None
    else:
        m = register.match(F[i], F[j], prior=prior, gate=gate)
    if m is None or m.n_inl < min_inl or m.resid > 1.3:
        return (i, j, kind, None)
    return (i, j, kind, m.S)


def _seq_job(args):
    k, prior = args
    return register.match(_FEATS[k - 1], _FEATS[k], prior=prior, gate=40)


def solve_race(z: dict, *, nproc: int | None = None, loop_weight: float = 3.0):
    """Feature cache (``features`` stage) -> world poses G (K,3,3) of every unique render frame.

    Edges: sequential (+ re-matched failures), long-baseline d in {2,5,12,24} (bound drift on trips
    that never revisit ground), loop closures (revisited ground, unguided). Robust similarity graph
    solve with a mid-race gauge (the first frame after the live->render dissolve is unreliable),
    then gross-outlier pruning (>10 px) and re-solve."""
    import multiprocessing as mp

    global _FEATS
    off, pts, des = z["off"], z["pts"], z["des"]
    F = [
        register.Feats(pts[off[k] : off[k + 1]], des[off[k] : off[k + 1]])
        for k in range(len(off) - 1)
    ]
    idx, steps, shape = z["idx"], z["steps"], tuple(int(x) for x in z["shape"][:2])
    K = len(idx)
    import os

    nproc = nproc or max(1, int(os.environ.get("HKJC_NPROC", "0")) or (os.cpu_count() or 4))
    # sequential edges whose guide comes from the stored step match are independent: match them in
    # the pool (identical results: OpenCV RANSAC seeds per call); only steps that need the running
    # chain as their guide stay sequential
    _FEATS = F
    pre_k = [k for k in range(1, K) if not np.isnan(steps[k][0])]
    with mp.get_context("fork").Pool(nproc) as pool:
        pre = dict(
            zip(pre_k, pool.map(_seq_job, [(k, steps[k][:9].reshape(3, 3)) for k in pre_k], 16))
        )
    S = [np.eye(3)]
    edges = []
    for k in range(1, K):
        if k in pre:
            m = pre[k]
        else:
            m = register.match(F[k - 1], F[k], prior=S[-1], gate=40)
        if m is None or m.n_inl < 60 or m.resid > 1.3:
            S.append(S[-1])
            edges.append(Edge(k - 1, k, S[-1], 0.05, "interp"))
            continue
        S.append(m.S)
        edges.append(Edge(k - 1, k, m.S, 1.0, "seq"))
    G0 = chain(S)
    jobs = [
        (k, k + d, np.linalg.inv(G0[k + d]) @ G0[k], 30 + 2 * d, 60, f"d{d}")
        for d in (2, 5, 12, 24)
        for k in range(0, K - d, 1 if d <= 5 else 2)
    ]
    C = centres(G0, shape)
    for i in range(0, K, 3):
        dd = np.hypot(*(C - C[i]).T)
        cand = np.where((np.arange(K) - i > 120) & (dd < 0.45 * shape[1]))[0]
        jobs += [(i, j, np.linalg.inv(G0[j]) @ G0[i], 120, 80, "loop") for j in cand[::4][:6]]
    _FEATS = F
    with mp.get_context("fork").Pool(nproc) as pool:
        out = pool.map(_measure, jobs, chunksize=16)
    _FEATS = None
    for i, j, kind, Sm in out:
        if Sm is not None:
            edges.append(Edge(i, j, Sm, loop_weight if kind == "loop" else 1.0, kind))
    pg = PoseGraph(shape, G0, edges)
    fix = K // 2
    lin = os.environ.get("HKJC_POSE_SOLVER") == "linear"  # opt-in until validated

    def solve_(g):
        return solve_linear(g, fix=fix) if lin else solve(g, iters=100, rigid=False, fix=fix)

    pg1 = solve_(pg)
    for _ in range(2):
        r = edge_residuals(pg1)
        keep = r < 10.0
        if keep.all():
            break
        pg = PoseGraph(shape, pg1.G, [e for e, k in zip(pg1.edges, keep) if k])
        pg1 = solve_(pg)
    r = edge_residuals(pg1)
    kinds = np.array([e.kind for e in pg1.edges])
    summ = " ".join(
        f"{kd}:{int((kinds == kd).sum())}@{np.median(r[kinds == kd]):.2f}px"
        for kd in dict.fromkeys(kinds)
    )
    return pg1.G, dict(
        edges=np.array([[e.i, e.j, e.w] for e in pg1.edges]), kinds=kinds, resid=r, summary=summ
    )
