"""AD HOC (first calibration attempt, 2026-10-25). Not maintained; kept as a record.

What it did: fitted a pinhole camera over a flat floor to the rug corners and the tape distances
between them, with no lens distortion. It showed two things that shaped the final method: the rug
alone cannot determine the camera height (heights from 2 m to 4.4 m all fit), and the fit only
became correct once the rug's edges were forced to be perpendicular (it is a rectangle).

Superseded by tools/calib.py, which adds lens distortion, the floor stickers and the held-out
check. It read the old measurements/ path, so it no longer runs as it is.
"""
import csv
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

W, H_IMG = 1920, 1080
CX, CY = W / 2, H_IMG / 2
IN = 0.0254
PX = [(928, 736), (830, 712), (880, 748), (761, 822), (716, 874), (500, 771), (600, 733), (692, 785),
      (688, 623), (962, 735)]  # P1..P8, then the rug corners R1, R2 (picked by eye)
NAMES = [f"P{i}" for i in range(1, 9)] + ["R1", "R2"]
SIGMA = [2.0] * 8 + [4.0, 4.0]
# the rug is a rectangle: R1->R2 is perpendicular to the rug's other edge (P5->P1, P4->P3)
PERP = [((8, 9), (4, 0)), ((8, 9), (3, 2))]
SIG_ANG = np.radians(3.0)
SIG_PX, SIG_D = 2.0, 0.4 * IN


def load(drop=()):
    out = []
    for r in csv.DictReader(open("measurements/rug_distances.csv")):
        if (r["a"], r["b"]) in drop or (r["b"], r["a"]) in drop:
            continue
        out.append((NAMES.index(r["a"]), NAMES.index(r["b"]), float(r["inches"]) * IN))
    return out


def rot(f_theta, phi):
    # world X right, Y forward, Z up; camera x right, y down, z forward
    r0 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    c, s = np.cos(f_theta), np.sin(f_theta)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(phi), np.sin(phi)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rz @ rx @ r0


def project(p, f, theta, phi, h):
    """Floor points (n, 2) -> pixels, for a camera at (0, 0, h)."""
    r = rot(theta, phi)
    w = np.c_[p, np.zeros(len(p))] - np.array([0, 0, h])
    c = w @ r.T
    return np.c_[CX + f * c[:, 0] / c[:, 2], CY + f * c[:, 1] / c[:, 2]]


def backproject(px, f, theta, phi, h):
    r = rot(theta, phi)
    d = np.c_[(px[:, 0] - CX) / f, (px[:, 1] - CY) / f, np.ones(len(px))] @ r  # world ray dirs
    t = -h / d[:, 2]  # h + t*dz = 0
    return np.c_[t * d[:, 0], t * d[:, 1]]


def fit(dists, init, hb=(0.5, 6)):
    px = np.array(PX, float)

    def res(x):
        f, th, ph, h = x[:4]
        p = x[4:].reshape(-1, 2)
        r = [((project(p, f, th, ph, h) - px) / np.array(SIGMA)[:, None]).ravel()]
        r.append(np.array([(np.linalg.norm(p[a] - p[b]) - d) / SIG_D for a, b, d in dists]))
        for (a1, a2), (b1, b2) in PERP:
            va, vb = p[a2] - p[a1], p[b2] - p[b1]
            r.append(np.array([va @ vb / np.linalg.norm(va) / np.linalg.norm(vb) / SIG_ANG]))
        return np.concatenate(r)

    x0 = np.r_[init, backproject(px, *init).ravel()]
    lo = np.r_[200, 0.1, -0.5, hb[0], np.full(2 * len(PX), -20)]
    hi = np.r_[2500, 1.5, 0.5, hb[1], np.full(2 * len(PX), 20)]
    s = least_squares(res, x0, bounds=(lo, hi), x_scale="jac")
    return s, res(s.x)


if __name__ == "__main__":
    drop = set()
    if "--drop" in sys.argv:
        for pair in sys.argv[sys.argv.index("--drop") + 1:]:
            if pair.startswith("--"):
                break
            a, b = pair.split(",")
            drop.add((a, b))
    hb = (0.5, 6)
    if "--h" in sys.argv:
        v = float(sys.argv[sys.argv.index("--h") + 1])
        hb = (v - 0.01, v + 0.01)
    dists = load(drop)
    best = None
    for f0 in (500, 800, 1200):
        for th0 in (0.5, 0.8, 1.1):
            for h0 in (1.5, 2.3, 3.0):
                s, r = fit(dists, [f0, th0, 0.0, float(np.clip(h0, *hb))], hb)
                if best is None or s.cost < best[0].cost:
                    best = (s, r)
    s, r = best
    f, th, ph, h = s.x[:4]
    p = s.x[4:].reshape(-1, 2)
    print(f"f={f:.0f}px pitch={np.degrees(th):.1f}deg roll={np.degrees(ph):.1f}deg height={h:.2f}m cost={s.cost:.1f}")
    pr = project(p, f, th, ph, h)
    print("pixel residuals (px):", np.round(np.linalg.norm(pr - np.array(PX), axis=1), 1))
    pa = [np.degrees(np.arccos(abs((p[a2]-p[a1]) @ (p[b2]-p[b1]) / np.linalg.norm(p[a2]-p[a1]) / np.linalg.norm(p[b2]-p[b1])))) for (a1, a2), (b1, b2) in PERP]
    print("rug edge angles (should be 90):", np.round([90 - x for x in pa], 1) if False else np.round([np.degrees(np.arccos((p[a2]-p[a1]) @ (p[b2]-p[b1]) / np.linalg.norm(p[a2]-p[a1]) / np.linalg.norm(p[b2]-p[b1]))) for (a1, a2), (b1, b2) in PERP], 1))
    print("distance residuals (in):")
    for (a, b, d), rr in zip(dists, r[2 * len(PX):]):
        print(f"  {NAMES[a]}-{NAMES[b]}: measured {d/IN:.2f}  fitted {np.linalg.norm(p[a]-p[b])/IN:.2f}  z={rr:+.1f}")
    np.save("data/floor_fit.npy", s.x)
