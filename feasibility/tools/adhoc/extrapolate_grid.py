"""AD HOC (first calibration attempt, 2026-10-25). Not maintained; kept as a record.

What it did: used the rug-based floor model from fit_floor.py to extend a metric grid over the
whole walking area, drew the 2 ft grid on the camera frame, and refitted at several assumed camera
heights (2.2, 3.0, 4.0 m) to show how far the grid moves between them (about 5-20 cm). That
estimate told us the camera height mattered little, before any sticker measurements existed.

Superseded by tools/calib.py (lens distortion + stickers). It imports fit_floor and reads the old
data/ paths, so it no longer runs as it is.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "tools")
import fit_floor as ff

AREA = np.array([(630, 565), (880, 578), (1030, 612), (1330, 628), (1520, 662), (1600, 760),
                 (1560, 900), (1490, 1050), (1230, 1075), (1000, 1055), (760, 900),
                 (500, 940), (335, 835)])
HEIGHTS = (2.2, 3.0, 4.0)
REF = 3.0
STEP = 0.5  # grid spacing, meters


def fit_height(dists, h):
    hb = (h - 0.01, h + 0.01)
    best = None
    for f0 in (400, 600, 900, 1400):
        for th0 in (0.6, 0.9, 1.2):
            s, _ = ff.fit(dists, [f0, th0, 0.0, h], hb)
            if best is None or s.cost < best.cost:
                best = s
    return best


def rug_frame(p):
    """Floor frame: origin R2, x along R1 -> R2, y perpendicular, positive toward the rug's interior."""
    o = p[9]
    u = p[9] - p[8]
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    if np.dot(p[:8].mean(0) - o, v) < 0:
        v = -v
    return o, u, v


class Model:
    def __init__(self, x):
        self.f, self.th, self.ph, self.h = x[:4]
        self.p = x[4:].reshape(-1, 2)
        o, u, v = rug_frame(self.p)
        self.o, self.u, self.v = o, u, v

    def to_cam(self, uv):
        return self.o + np.outer(uv[:, 0], self.u) + np.outer(uv[:, 1], self.v)

    def from_cam(self, p):
        d = p - self.o
        return np.c_[d @ self.u, d @ self.v]

    def to_px(self, uv):
        return ff.project(self.to_cam(uv), self.f, self.th, self.ph, self.h)

    def to_uv(self, px):
        return self.from_cam(ff.backproject(px, self.f, self.th, self.ph, self.h))


if __name__ == "__main__":
    drop = set()
    if "--drop" in sys.argv:
        a, b = sys.argv[sys.argv.index("--drop") + 1].split(",")
        drop.add((a, b))
    dists = ff.load(drop)
    models = {h: Model(fit_height(dists, h).x) for h in HEIGHTS}
    for h, m in models.items():
        print(f"h={h}: f={m.f:.0f} pitch={np.degrees(m.th):.1f} roll={np.degrees(m.ph):.1f}")

    # walking-area sample pixels
    mask = np.zeros((ff.H_IMG, ff.W), np.uint8)
    cv2.fillPoly(mask, [AREA], 255)
    ys, xs = np.mgrid[560:1080:40, 330:1620:40]
    pts = np.c_[xs.ravel(), ys.ravel()].astype(float)
    pts = pts[mask[pts[:, 1].astype(int), pts[:, 0].astype(int)] > 0]

    ref = models[REF]
    uv_ref = ref.to_uv(pts)
    print(f"{len(pts)} sample pixels in the walking area; floor extent (m): "
          f"u {uv_ref[:,0].min():.1f}..{uv_ref[:,0].max():.1f}, v {uv_ref[:,1].min():.1f}..{uv_ref[:,1].max():.1f}")
    for h, m in models.items():
        if h == REF:
            continue
        d = np.linalg.norm(m.to_uv(pts) - uv_ref, axis=1)
        far = np.linalg.norm(uv_ref - ref.to_uv(np.array(ff.PX, float)).mean(0), axis=1)
        print(f"h={h} vs {REF}: median shift {np.median(d)*100:.0f} cm, 90th pct {np.percentile(d,90)*100:.0f} cm, "
              f"max {d.max()*100:.0f} cm; farthest from rug ({far.max():.1f} m): {d[far.argmax()]*100:.0f} cm")

    img = cv2.imread("data/sample_frame.jpg")
    layer = np.zeros_like(img)
    FT = 0.3048
    step = 2 * FT  # 2 ft cells
    lo = np.floor(uv_ref.min(0) / step) * step - step
    hi = np.ceil(uv_ref.max(0) / step) * step + step
    ts = np.linspace(0, 1, 300)
    labels = []

    def draw(vals, const_axis):
        for a in vals:
            if const_axis == 0:
                line = np.c_[np.full(300, a), lo[1] + ts * (hi[1] - lo[1])]
            else:
                line = np.c_[lo[0] + ts * (hi[0] - lo[0]), np.full(300, a)]
            px = ref.to_px(line)
            cv2.polylines(layer, [px.astype(np.int32)], False, (255, 255, 255), 2)
            ok = np.where(mask[np.clip(px[:, 1].astype(int), 0, ff.H_IMG - 1),
                               np.clip(px[:, 0].astype(int), 0, ff.W - 1)] > 0)[0]
            if len(ok):
                labels.append((tuple(px[ok[-1]].astype(int)), a / FT, const_axis))

    draw(np.arange(np.ceil(lo[0] / step) * step, hi[0], step), 0)
    draw(np.arange(np.ceil(lo[1] / step) * step, hi[1], step), 1)
    inside = mask > 0
    img[inside] = np.where(layer[inside].any(axis=1, keepdims=True), layer[inside], img[inside])
    cv2.polylines(img, [AREA], True, (0, 255, 0), 2)
    for (x, y), val, ax in labels:
        txt = f"{'x' if ax == 0 else 'y'}={round(val) + 0:d}ft"
        pos = (min(max(x - 30, 5), ff.W - 130), min(y + 24, ff.H_IMG - 6) if ax == 0 else max(y - 8, 20))
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    # origin and axes
    o = ref.to_px(np.array([[0.0, 0.0]]))[0]
    for d, col, nm in (((1, 0), (0, 0, 255), "x"), ((0, 1), (255, 0, 0), "y")):
        e = ref.to_px(np.array([[d[0] * 1.0, d[1] * 1.0]]))[0]
        cv2.arrowedLine(img, tuple(o.astype(int)), tuple(e.astype(int)), col, 4, tipLength=0.1)
        cv2.putText(img, nm, tuple((e + 8).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    cv2.putText(img, "R2 = origin", tuple((o + [10, 28]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for px in ff.PX:
        cv2.circle(img, px, 4, (0, 255, 255), -1)
    cv2.putText(img, "2 ft cells, h=3.0 m model. x: R1->R2, y: along the other rug edge", (40, 1060),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite("data/floor_grid.jpg", img)
    np.savez("data/grid_model.npz", params=np.r_[ref.f, ref.th, ref.ph, ref.h], plane=ref.p, o=ref.o, u=ref.u, v=ref.v)
    print("wrote data/floor_grid.jpg and data/grid_model.npz")
