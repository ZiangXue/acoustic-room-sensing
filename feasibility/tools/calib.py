"""Calibrate the camera-to-floor mapping from landmarks and measured distances.

Model: pinhole camera over a flat floor with one radial distortion term k1.
Unknowns: focal length f, pitch, roll, camera height, k1, and the floor (X, Y) of every landmark.
Observations: landmark pixel positions, measured distances (inches), and the rectangle constraints
(pairs of edges that must be perpendicular).

Everything room-specific (landmark pixels, measurement files, constraints, the floor frame) comes
from the active preset, see tools/preset.py. Landmarks whose name starts with "S" are the floor
stickers: they are loaded from the preset's sticker pixel file and used as the held-out check in
`report`. The floor frame is defined in the preset (origin, x direction, y toward the interior).

Usage:  calib.py [--preset NAME] report | grid
  report   prediction errors before calibration, residuals after, leave-one-out
  grid     write the calibrated grid drawn on the sample frame to the preset's output/
"""
import csv
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

import preset

CFG = preset.CFG
W, H_IMG = CFG["camera"]["image"]
CX, CY = W / 2, H_IMG / 2
IN = 0.0254
FT = 0.3048

PIX = {k: tuple(v) for k, v in CFG["landmarks_px"].items()}
SIG = CFG["noise"]["landmark_px"]
STICKER_SIG = CFG["noise"]["sticker_px"]
SIG_D = CFG["noise"]["distance_in"] * IN
SIG_ANG = np.radians(CFG["noise"]["angle_deg"])
PERP = [(tuple(a), tuple(b)) for a, b in CFG["rectangle_constraints"]]
ALIAS = CFG["distance_aliases"]
IGNORE = {tuple(p) for p in CFG["ignore_distances"]}
FRAME = CFG["floor_frame"]
AREA = np.array(CFG["walking_area_px"])


def load_points():
    sig = {k: SIG[k[0]] for k in PIX}
    pix = dict(PIX)
    for r in csv.DictReader(open(preset.path(CFG["sticker_pixels_file"]))):
        pix[r["name"]] = (float(r["px"]), float(r["py"]))
        sig[r["name"]] = STICKER_SIG[r["confidence"]]
    return pix, sig


def load_dists():
    out = []
    for f in CFG["distance_files"]:
        for r in csv.DictReader(open(preset.path(f))):
            a, b = ALIAS.get(r["a"], r["a"]), ALIAS.get(r["b"], r["b"])
            if (a, b) in IGNORE or (b, a) in IGNORE:
                continue
            out.append((a, b, float(r["inches"]) * IN))
    return out


def all_names(pix):
    """Fixed landmarks first, then the stickers in name order."""
    return list(PIX) + sorted(n for n in pix if n[0] == "S")


def rot(theta, phi):
    r0 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    c, s = np.cos(theta), np.sin(theta)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(phi), np.sin(phi)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rz @ rx @ r0


def project(p, f, th, ph, h, k1):
    c = (np.c_[p, np.zeros(len(p))] - np.array([0, 0, h])) @ rot(th, ph).T
    xn, yn = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
    s = 1 + k1 * (xn ** 2 + yn ** 2)
    return np.c_[CX + f * xn * s, CY + f * yn * s]


def backproject(px, f, th, ph, h, k1):
    xd, yd = (px[:, 0] - CX) / f, (px[:, 1] - CY) / f
    xn, yn = xd.copy(), yd.copy()
    for _ in range(30):
        s = 1 + k1 * (xn ** 2 + yn ** 2)
        xn, yn = xd / s, yd / s
    d = np.c_[xn, yn, np.ones(len(px))] @ rot(th, ph)
    t = -h / d[:, 2]
    return np.c_[t * d[:, 0], t * d[:, 1]]


class Fit:
    def __init__(self, names, pix, sig, dists, use_k1=True, hfix=None, x0=None, no_pix=()):
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        self.pix = np.array([pix[n] for n in names], float)
        self.sig = np.array([sig[n] for n in names])
        self.use = np.array([n not in no_pix for n in names])
        self.dists = [(self.idx[a], self.idx[b], d) for a, b, d in dists if a in self.idx and b in self.idx]
        self.perp = [(tuple(self.idx[n] for n in a), tuple(self.idx[n] for n in b)) for a, b in PERP]
        self.use_k1, self.hfix = use_k1, hfix
        self.x = self._solve(x0)

    def residuals(self, x):
        f, th, ph, h, k1 = x[:5]
        p = x[5:].reshape(-1, 2)
        r = ((project(p, f, th, ph, h, k1) - self.pix) / self.sig[:, None])[self.use].ravel()
        r = [r, np.array([(np.linalg.norm(p[a] - p[b]) - d) / SIG_D for a, b, d in self.dists])]
        for (a1, a2), (b1, b2) in self.perp:
            va, vb = p[a2] - p[a1], p[b2] - p[b1]
            r.append(np.array([va @ vb / np.linalg.norm(va) / np.linalg.norm(vb) / SIG_ANG]))
        return np.concatenate(r)

    def _solve(self, x0):
        n = len(self.names)
        lo = np.r_[200, 0.1, -0.6, 1.8, -1.0 if self.use_k1 else -1e-9, np.full(2 * n, -20)]
        hi = np.r_[4000, 1.5, 0.6, 4.5, 0.3 if self.use_k1 else 1e-9, np.full(2 * n, 20)]
        if self.hfix:
            lo[3], hi[3] = self.hfix - 1e-3, self.hfix + 1e-3
        starts = [x0] if x0 is not None else []
        if x0 is None:
            for f0 in (500, 900, 1400):
                for th0 in (0.5, 0.9):
                    h0 = float(np.clip(3.0, lo[3], hi[3]))
                    g = [f0, th0, 0.0, h0, 0.0]
                    p0 = backproject(self.pix, *g)
                    starts.append(np.r_[g, p0.ravel()])
        best = None
        for s0 in starts:
            s0 = np.clip(s0, lo + 1e-9, hi - 1e-9)
            s = least_squares(self.residuals, s0, bounds=(lo, hi), x_scale="jac")
            if best is None or s.cost < best.cost:
                best = s
        self.cost = best.cost
        return best.x

    @property
    def plane(self):
        return self.x[5:].reshape(-1, 2)

    def dist(self, a, b):
        p = self.plane
        return np.linalg.norm(p[self.idx[a]] - p[self.idx[b]]) / IN

    def frame(self):
        """Floor frame from the preset: origin, x along x_from -> x_to, y perpendicular toward the interior."""
        p, i = self.plane, self.idx
        o = p[i[FRAME["origin"]]]
        u = p[i[FRAME["x_to"]]] - p[i[FRAME["x_from"]]]
        u /= np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        if np.dot(p[[i[n] for n in FRAME["y_positive_toward"]]].mean(0) - o, v) < 0:
            v = -v
        return o, u, v

    def to_xy(self, px):
        """Pixels -> floor (x, y) in meters in the R2 frame."""
        o, u, v = self.frame()
        q = backproject(np.atleast_2d(px).astype(float), *self.x[:5]) - o
        return np.c_[q @ u, q @ v]

    def to_px(self, xy):
        o, u, v = self.frame()
        return project(o + np.outer(xy[:, 0], u) + np.outer(xy[:, 1], v), *self.x[:5])


def report():
    pix, sig = load_points()
    dists = load_dists()
    sticker_d = [d for d in dists if d[0][0] == "S" or d[1][0] == "S"]
    rug_names = list(PIX)
    names = all_names(pix)

    print("== A. before calibration: fit on the rug only (no k1, h=3.0), predict the sticker distances")
    rug_d = [d for d in dists if d not in sticker_d]
    a = Fit(rug_names, pix, sig, rug_d, use_k1=False, hfix=3.0)
    # place stickers on the floor through the rug-only model
    p_rug = {n: a.plane[a.idx[n]] for n in rug_names}
    sp = backproject(np.array([pix[n] for n in names[len(rug_names):]]), *a.x[:5])
    pos = dict(p_rug); pos.update({n: sp[i] for i, n in enumerate(names[len(rug_names):])})
    err = []
    for x, y, d in sticker_d:
        pr = np.linalg.norm(pos[x] - pos[y]) / IN
        err.append(pr - d / IN)
        print(f"   {x}-{y}: measured {d/IN:6.2f}  predicted {pr:6.2f}  error {pr - d/IN:+5.2f} in")
    print(f"   rms error {np.sqrt(np.mean(np.square(err))):.2f} in, worst {np.max(np.abs(err)):.2f} in")

    print("== B. calibrated on everything (k1 free, height free)")
    b = Fit(names, pix, sig, dists)
    f, th, ph, h, k1 = b.x[:5]
    print(f"   f={f:.0f} pitch={np.degrees(th):.1f} roll={np.degrees(ph):.1f} height={h:.2f} m k1={k1:.3f} cost={b.cost:.1f}")
    errs = []
    for x, y, d in dists:
        pr = b.dist(x, y); errs.append(pr - d / IN)
    print(f"   all {len(dists)} distances: rms {np.sqrt(np.mean(np.square(errs))):.2f} in, worst {np.max(np.abs(errs)):.2f} in")
    pr = project(b.plane, *b.x[:5])
    pe = np.linalg.norm(pr - b.pix, axis=1)
    print("   pixel residuals (px):", {n: round(float(e), 1) for n, e in zip(b.names, pe)})

    print("== C. leave-one-out: drop one sticker distance, refit, predict it (k1 free)")
    errs = []
    for i, (x, y, d) in enumerate(sticker_d):
        rest = [q for q in dists if q != (x, y, d)]
        c = Fit(names, pix, sig, rest, x0=b.x)
        pr = c.dist(x, y); errs.append(pr - d / IN)
        print(f"   {x}-{y}: measured {d/IN:6.2f}  predicted {pr:6.2f}  error {pr - d/IN:+5.2f} in")
    print(f"   rms {np.sqrt(np.mean(np.square(errs))):.2f} in, worst {np.max(np.abs(errs)):.2f} in")
    np.save(preset.out("calib_x.npy"), b.x)
    return b


def draw_grid(b, frame=None, out=None):
    img = cv2.imread(frame or preset.path(CFG["sample_frame"]))
    out = out or preset.out("floor_grid_cal.jpg")
    layer = np.zeros_like(img)
    area = AREA
    mask = np.zeros((H_IMG, W), np.uint8)
    cv2.fillPoly(mask, [area], 255)
    step = 2 * FT
    ts = np.linspace(0, 1, 300)
    # extent of the walking area on the floor; only draw inside it, where the distortion model is valid
    xy = b.to_xy(area.astype(float))
    lo, hi = xy.min(0) - 0.2, xy.max(0) + 0.2
    labels = []

    def line(vals, axis):
        for a in vals:
            pts = (np.c_[np.full(300, a), lo[1] + ts * (hi[1] - lo[1])] if axis == 0
                   else np.c_[lo[0] + ts * (hi[0] - lo[0]), np.full(300, a)])
            px = b.to_px(pts)
            # drop points where the distortion model folds back (round trip no longer returns the same floor point)
            valid = (np.linalg.norm(b.to_xy(px) - pts, axis=1) < 0.02) & (px[:, 0] >= 0) & (px[:, 0] < W) \
                & (px[:, 1] >= 0) & (px[:, 1] < H_IMG)
            for run in np.split(np.arange(300), np.where(np.diff(valid.astype(int)) != 0)[0] + 1):
                if valid[run[0]] and len(run) > 1:
                    cv2.polylines(layer, [px[run].astype(np.int32)], False, (255, 255, 255), 2)
            ix, iy = np.clip(px[:, 0].astype(int), 0, W - 1), np.clip(px[:, 1].astype(int), 0, H_IMG - 1)
            ok = np.where((mask[iy, ix] > 0) & valid)[0]
            if len(ok):
                labels.append((tuple(px[ok[-1]].astype(int)), a / FT, axis))

    line(np.arange(np.ceil(lo[0] / step) * step, hi[0], step), 0)
    line(np.arange(np.ceil(lo[1] / step) * step, hi[1], step), 1)
    ins = mask > 0
    img[ins] = np.where(layer[ins].any(axis=1, keepdims=True), layer[ins], img[ins])
    cv2.polylines(img, [area], True, (0, 255, 0), 2)
    for (x, y), val, ax in labels:
        t = f"{'x' if ax == 0 else 'y'}={round(val) + 0:d}ft"
        pos = (min(max(x - 30, 5), W - 130), min(y + 24, H_IMG - 6) if ax == 0 else max(y - 8, 20))
        cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    o = b.to_px(np.array([[0.0, 0.0]]))[0]
    for d, col, nm in (((1, 0), (0, 0, 255), "x"), ((0, 1), (255, 0, 0), "y")):
        e = b.to_px(np.array([d], float))[0]
        cv2.arrowedLine(img, tuple(o.astype(int)), tuple(e.astype(int)), col, 4, tipLength=0.1)
        cv2.putText(img, nm, tuple((e + 8).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    for n, p in zip(b.names, b.pix):
        cv2.circle(img, tuple(p.astype(int)), 5, (0, 255, 255) if n[0] != "S" else (255, 0, 255), -1)
    cv2.putText(img, f"calibrated: 2 ft cells, origin {FRAME['origin']}, x: {FRAME['x_from']}->{FRAME['x_to']}", (40, 1060), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(out, img)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        report()
    elif cmd == "grid":
        pix, sig = load_points()
        b = Fit(all_names(pix), pix, sig, load_dists())
        print("wrote", draw_grid(b))
        np.save(preset.out("calib_x.npy"), b.x)
