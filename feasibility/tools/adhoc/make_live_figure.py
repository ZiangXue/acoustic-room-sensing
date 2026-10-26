"""AD HOC (project README figure 6, 2026-10-25). Kept because it can be rerun.

What it did: composed the live-tracker figure for the README from a running track_server: one
streamed video frame (with the person pixelated by the server's --blur) next to the top-down map
drawn from /state.json and /calibration.json, like the web page shows. The server in that figure
replayed a recorded still frame, not a live camera.

Run from feasibility/:  python tools/adhoc/make_live_figure.py --url http://localhost:9090
"""
import argparse
import json
import os
import urllib.request

import cv2
import numpy as np

IN, FT = 0.0254, 0.3048
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BG, PANEL, LINE, DIM, OK = (30, 25, 20), (38, 31, 26), (64, 52, 44), (163, 149, 139), (132, 220, 61)


def get(url, n=None):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read(n) if n else r.read()


def video_frame(base):
    buf = get(base + "/video?grid=1", 600_000)
    a = buf.find(b"\xff\xd8")
    b = buf.find(b"\xff\xd9", a)
    return cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)


def draw_map(cal, st, size):
    w, h = size
    img = np.full((h, w, 3), PANEL, np.uint8)
    pts = np.array(cal["area_xy"] + list(cal["landmarks"].values()))
    lo, hi = pts.min(0) - 0.25, pts.max(0) + 0.25
    s = min((w - 20) / (hi[0] - lo[0]), (h - 90) / (hi[1] - lo[1]))
    X = lambda x: int((x - lo[0]) * s + 10)
    Y = lambda y: int((y - lo[1]) * s + 40)
    put = lambda t, o, c=DIM, sc=0.5, th=1: cv2.putText(img, t, o, cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)
    for f in range(int(np.ceil(lo[0] / FT / 2) * 2), int(hi[0] / FT) + 1, 2):
        cv2.line(img, (X(f * FT), Y(lo[1])), (X(f * FT), Y(hi[1])), LINE, 1)
        put(f"{f} ft", (X(f * FT) + 3, Y(lo[1]) + 14))
    for f in range(int(np.ceil(lo[1] / FT / 2) * 2), int(hi[1] / FT) + 1, 2):
        cv2.line(img, (X(lo[0]), Y(f * FT)), (X(hi[0]), Y(f * FT)), LINE, 1)
        put(f"{f} ft", (X(lo[0]) + 3, Y(f * FT) - 4))
    area = np.array([(X(x), Y(y)) for x, y in cal["area_xy"]], np.int32)
    cv2.polylines(img, [area], True, OK, 2, cv2.LINE_AA)
    for n, (x, y) in cal["landmarks"].items():
        c = (255, 100, 194) if n[0] == "S" else (0, 184, 229)
        cv2.circle(img, (X(x), Y(y)), 4, c, -1, cv2.LINE_AA)
        put(n, (X(x) + 6, Y(y) - 4), c)
    if st.get("tracking"):
        cv2.circle(img, (X(st["x"]), Y(st["y"])), 10, OK, -1, cv2.LINE_AA)
        cv2.circle(img, (X(st["x"]), Y(st["y"])), 10, (255, 255, 255), 2, cv2.LINE_AA)
        put(f"x = {st['x'] / IN:.1f} in   y = {st['y'] / IN:.1f} in", (12, 24), (230, 230, 230), 0.7, 2)
        put(f"({st['x']:.2f}, {st['y']:.2f}) m   {st['n_ankles']} ankles, conf {st['conf']:.2f}   {st['fps']} fps",
            (12, h - 12), DIM, 0.55)
    else:
        put("no person", (12, 24), (95, 90, 255), 0.7, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9090")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "img", "6_live_tracker.jpg"))
    a = ap.parse_args()
    cal = json.loads(get(a.url + "/calibration.json"))
    st = json.loads(get(a.url + "/state.json"))
    vid = video_frame(a.url)
    h = 640
    vid = cv2.resize(vid, (int(vid.shape[1] * h / vid.shape[0]), h), interpolation=cv2.INTER_AREA)
    m = draw_map(cal, st, (560, h))
    cv2.imwrite(a.out, np.hstack([vid, m]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", a.out, "| state:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()})


if __name__ == "__main__":
    main()
