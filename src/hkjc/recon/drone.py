"""Drone feed: a real, high, moving/zooming camera (ONE continuous shot gate -> finish; shares RF's
edit timeline, race_t = video_t - offset from RF's burnt-in clock).

Registration to the world map uses a PHYSICAL pinhole camera (7 DOF: centre C, look-at ground point
P, heading psi fixing roll, focal f) — not a free 8-DOF homography, which collapsed onto degenerate
solutions (every rail pixel snapped to one rail). Residuals are bidirectional and truncated:
  map->image  sampled world rails (racing rail, outer = racing + course width, fixed inner rail)
              projected into the frame must land on thin-white rail pixels;
  image->map  drone rail pixels must land on a projected rail;
  horses      render horse positions (same race time) projected at JOCKEY HEIGHT must land on drone
              jockey blobs, and vice versa.
Known rail spacings fix metric scale; the bidirectional terms forbid collapse. Multi-start inits look
at the field's world position; frames are then tracked with the previous camera as the start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

JOCKEY_H = 2.0  # m above ground of the jockey's silks/helmet (the drone's detected point)


# ----------------------------------------------------------------------------- detection
def turf_region(hsv):
    import cv2

    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    turf = ((h > 35) & (h < 85) & (s > 50) & (v > 50)).astype(np.uint8)
    reg = cv2.morphologyEx(turf, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(reg)
    if n < 2:
        return turf, turf
    return turf, (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8)


def jockeys(img, *, min_a=40, max_a=2500, body_r=22):
    """Jockey candidates: compact vivid-silk / white-breeches blobs on the turf next to a DARK blob
    (horse body / shadow) — isolated white dots (rail posts) fail that test. (N,2) image points."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    turf, region = turf_region(hsv)
    region = cv2.erode(region, np.ones((15, 15), np.uint8))
    vivid = ((s > 110) & (v > 90) & ((h < 30) | (h > 90))).astype(np.uint8)
    white = cv2.morphologyEx(
        ((s < 50) & (v > 185)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    darkn = cv2.dilate(
        ((v < 75) & (turf == 0)).astype(np.uint8), np.ones((2 * body_r + 1,) * 2, np.uint8)
    )
    cand = cv2.dilate(((vivid | white) & region).astype(np.uint8), np.ones((7, 7), np.uint8))
    n, _lab, st, cen = cv2.connectedComponentsWithStats(cand)
    pts = [
        cen[k]
        for k in range(1, n)
        if min_a < st[k, 4] < max_a and darkn[int(cen[k][1]), int(cen[k][0])]
    ]
    return np.array(pts).reshape(-1, 2)


def rail_pixels(img, step=3):
    """Thin white line pixels on/at the turf (rails), subsampled (N,2)."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    _turf, region = turf_region(hsv)
    white = ((s < 60) & (v > 170)).astype(np.uint8) & cv2.dilate(region, np.ones((9, 9), np.uint8))
    thick = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    ys, xs = np.nonzero((white & (thick == 0))[::step, ::step])
    return np.c_[xs * step, ys * step].astype(float)


# ----------------------------------------------------------------------------- camera
@dataclass
class Camera:
    """Look-at pinhole camera over the ground plane z=0 (world metres, map orientation)."""

    C: np.ndarray  # (3,) centre, z = height above ground (m)
    P: np.ndarray  # (2,) ground point at the image centre
    psi: float  # heading of the image "up" direction on the ground (rad)
    f: float  # focal length (px)
    size: tuple = (1920, 1080)

    @property
    def params(self):
        return np.r_[self.C, self.P, self.psi, np.log(self.f)]

    @classmethod
    def from_params(cls, p, size=(1920, 1080)):
        return cls(np.asarray(p[:3]), np.asarray(p[3:5]), float(p[5]), float(np.exp(p[6])), size)

    # The world map uses IMAGE convention (x right, y DOWN): embedded with z up that frame is
    # left-handed and a proper camera can only fit its mirror image. Internally the ground plane is
    # (x, -y) — right-handed — and C/P are stored in map coordinates.
    @staticmethod
    def _g(X):
        X = np.atleast_2d(X)
        return np.c_[X[:, 0], -X[:, 1]]

    def rot(self):
        Cg = np.r_[self.C[0], -self.C[1], self.C[2]]
        fwd = np.r_[self.P[0], -self.P[1], 0.0] - Cg
        fwd /= np.linalg.norm(fwd)
        up = np.array([np.cos(self.psi), np.sin(self.psi), 0.0])
        right = np.cross(fwd, up)
        nr = np.linalg.norm(right)
        right = right / nr if nr > 1e-9 else np.array([1.0, 0.0, 0.0])
        upc = np.cross(right, fwd)
        return np.vstack([right, -upc, fwd])  # camera x right, y down, z forward

    def project(self, X, z=0.0):
        Cg = np.r_[self.C[0], -self.C[1], self.C[2]]
        Xw = np.c_[self._g(X), np.full(len(np.atleast_2d(X)), z)] - Cg
        pc = Xw @ self.rot().T
        w, h = self.size
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = self.f * pc[:, :2] / pc[:, 2:3] + [w / 2, h / 2]
        uv[pc[:, 2] <= 0.1] = np.nan
        return uv

    def backproject(self, uv, z=0.0):
        """Image points -> ground (or height-z plane) world points."""
        w, h = self.size
        d = np.c_[(np.atleast_2d(uv) - [w / 2, h / 2]) / self.f, np.ones(len(np.atleast_2d(uv)))]
        dw = d @ self.rot()  # camera ray -> world (internal right-handed frame)
        Cg = np.r_[self.C[0], -self.C[1], self.C[2]]
        lam = (z - Cg[2]) / dw[:, 2]
        g = (Cg + lam[:, None] * dw)[:, :2]
        return np.c_[g[:, 0], -g[:, 1]]


def _inside(uv, size, m=0):
    return (
        np.isfinite(uv).all(1)
        & (uv[:, 0] > m)
        & (uv[:, 0] < size[0] - m)
        & (uv[:, 1] > m)
        & (uv[:, 1] < size[1] - m)
    )


def residuals(cam, near_w, rail_px, horses_w, jockey_px, *, tau=25.0, w_h=0.5):
    """Bidirectional truncated residuals (see module doc). FIXED length (least_squares needs it):
    ``near_w`` is the world-rail sample set frozen for this fit; out-of-view samples score tau."""
    from scipy.spatial import cKDTree

    size = cam.size
    out = []
    uv = cam.project(near_w)
    ins = _inside(uv, size, 5)
    r1 = np.ones(len(near_w))
    if ins.sum() >= 2 and len(rail_px):
        d, _ = cKDTree(rail_px).query(uv[ins])
        r1[ins] = np.minimum(d, tau) / tau
    out.append(r1)
    if len(rail_px):
        r2 = np.ones(len(rail_px))
        if ins.sum() >= 2:
            d2, _ = cKDTree(uv[ins]).query(rail_px)
            r2 = np.minimum(d2, tau) / tau
        out.append(r2)
    if len(horses_w) and len(jockey_px):
        uvh = cam.project(horses_w, z=JOCKEY_H)
        ok = np.isfinite(uvh).all(1)
        r3 = np.full(len(horses_w), w_h)
        r4 = np.full(len(jockey_px), w_h)
        if ok.sum():
            d, _ = cKDTree(jockey_px).query(uvh[ok])
            r3[ok] = w_h * np.minimum(d, 2 * tau) / (2 * tau)
            d2, _ = cKDTree(uvh[ok]).query(jockey_px)
            r4 = w_h * np.minimum(d2, 2 * tau) / (2 * tau)
        out += [r3, r4]
    return np.concatenate(out)


def _near(rails_w, P, radius=90.0, n_max=600):
    near = rails_w[np.hypot(*(rails_w - P).T) < radius]
    return near[:: max(1, len(near) // n_max)]


def fit(cam0, rails_w, rail_px, horses_w, jockey_px, *, max_nfev=120, taus=(200.0, 80.0, 25.0)):
    """Coarse-to-fine robust fit: a wide truncation first (broad basin of attraction — truncated
    residuals have no gradient beyond tau), then tighten. Returns (camera, cost at the finest tau)."""
    from scipy.optimize import least_squares

    size = cam0.size
    rp = rail_px[:: max(1, len(rail_px) // 800)]
    p = cam0.params
    for tau in taus:
        near = _near(rails_w, Camera.from_params(p, size).P)
        r = least_squares(
            lambda q: residuals(
                Camera.from_params(q, size), near, rp, horses_w, jockey_px, tau=tau
            ),
            p,
            diff_step=1e-3,
            max_nfev=max_nfev,
            x_scale=[5, 5, 5, 2, 2, 0.05, 0.05],
        )
        p = r.x
    return Camera.from_params(p, size), float(np.mean(r.fun**2))


def init_candidates(
    field_w,
    travel_w,
    *,
    heights=(50.0, 100.0),
    offsets=(0.0, 40.0),
    along=(-70.0, -30.0, 0.0, 30.0, 70.0),
    focals=(1600.0, 3200.0),
):
    """Starting cameras looking at the field from above / above-outside / ahead / behind (the camera
    tilt ALONG the track is unobservable from straight parallel rails, so it must be searched),
    image-up along any of 8 headings."""
    t = travel_w / (np.linalg.norm(travel_w) + 1e-9)
    nrm = np.array([-t[1], t[0]])
    out = []
    for hgt in heights:
        for off in offsets:
            for side in (1, -1) if off else (1,):
                for al in along:
                    C = np.r_[field_w + side * off * nrm + al * t, hgt]
                    for k in range(8):
                        psi = (
                            np.arctan2(-t[1], t[0]) + k * np.pi / 4
                        )  # heading in the internal frame
                        for f in focals:
                            out.append(Camera(C.copy(), field_w.copy(), psi, f))
    return out


def seed(rails_w, rail_px, horses_w, jockey_px, field_w, travel_w, *, keep=8):
    """Multi-start: score all candidate starts, refine the best few, return the best camera."""
    cands = init_candidates(field_w, travel_w)
    near = _near(rails_w, field_w)
    rp = rail_px[:: max(1, len(rail_px) // 800)]
    sc = [
        float(np.mean(residuals(c, near, rp, horses_w, jockey_px, tau=200.0, w_h=1.5) ** 2))
        for c in cands
    ]
    best = None
    for k in np.argsort(sc)[:keep]:
        cam, cost = fit(cands[k], rails_w, rail_px, horses_w, jockey_px)
        if best is None or cost < best[1]:
            best = (cam, cost)
    return best


def refine_icp(
    cam, rails_w, rail_px, horses_w, jockey_px, *, rounds=4, gate=120.0, w_pair=2.0, tau=25.0
):
    """ICP on horses: Hungarian-assign projected render horses (jockey height) to drone jockey blobs,
    then fit rails (truncated) + ASSIGNED pair residuals. On straights the parallel rails leave the
    along-track scale/tilt unobservable; the field (11 horses over ~25 m) is the along-track ruler."""
    from scipy.optimize import least_squares, linear_sum_assignment

    size = cam.size
    rp = rail_px[:: max(1, len(rail_px) // 800)]
    p = cam.params
    n_pairs = 0
    for _ in range(rounds):
        c = Camera.from_params(p, size)
        uv = c.project(horses_w, z=JOCKEY_H)
        if not np.isfinite(uv).all():
            break
        D = np.linalg.norm(uv[:, None] - jockey_px[None], axis=2)
        r, k = linear_sum_assignment(np.where(D < gate, D, 1e6))
        ok = D[r, k] < gate
        r, k = r[ok], k[ok]
        n_pairs = len(r)
        if n_pairs < 4:
            break
        near = _near(rails_w, c.P)
        A, B = horses_w[r], jockey_px[k]

        def res(q):
            cc = Camera.from_params(q, size)
            rr = residuals(cc, near, rp, np.zeros((0, 2)), np.zeros((0, 2)), tau=tau)
            pr = (cc.project(A, z=JOCKEY_H) - B).ravel() / tau
            return np.r_[rr, w_pair * np.nan_to_num(pr, nan=10.0)]

        sol = least_squares(
            res,
            p,
            diff_step=1e-3,
            max_nfev=150,
            loss="soft_l1",
            x_scale=[5, 5, 5, 2, 2, 0.05, 0.05],
        )
        p = sol.x
        gate = max(40.0, gate * 0.6)
    return Camera.from_params(p, size), n_pairs
