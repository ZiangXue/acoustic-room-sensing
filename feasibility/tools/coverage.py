"""How much of the walkable floor do the tracked sessions cover?

Reads one or more *_track.csv files from track_video.py and reports, per session and combined, the
share of 12 in floor cells the person spent time in, the time near each planned stop, and which
parts of the room are thin. Writes a heat map next to the tracks.

Usage: coverage.py [--preset NAME] TRACK.csv [TRACK.csv ...]
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preset

IN = 0.0254
CELL = 12.0          # inches
MIN_S = 1.5          # seconds in a cell to count it as visited
STOP_R = 18.0        # inches around a stop that count as "at the stop"


def load(path):
    t, xy = [], []
    for r in csv.DictReader(open(path)):
        if r["x_m"]:
            t.append(float(r["t_video_s"]))
            xy.append([float(r["x_m"]) / IN, float(r["y_m"]) / IN])
    t, xy = np.array(t), np.array(xy)
    dt = np.minimum(np.diff(t, append=t[-1]), 0.5)      # seconds each sample stands for (gaps count as 0.5 s at most)
    return xy, dt


def main():
    files = sys.argv[1:]
    cal = json.load(open(preset.out("calibration.json")))
    area = (np.array(cal["area_xy"]) / IN).astype(np.float32)
    lo, hi = area.min(0), area.max(0)
    xs, ys = np.arange(lo[0], hi[0], CELL), np.arange(lo[1], hi[1], CELL)
    cells = [(x + CELL / 2, y + CELL / 2) for x in xs for y in ys
             if cv2.pointPolygonTest(area, (float(x + CELL / 2), float(y + CELL / 2)), True) >= 6]
    ci = {c: i for i, c in enumerate(cells)}
    stops = {n: np.array(cal["landmarks"][n]) / IN for n in preset.CFG["session"]["stops"]}
    tot = np.zeros(len(cells))
    per = []
    print(f"{len(cells)} floor cells of {CELL:.0f} in inside the walkable area")
    for f in files:
        xy, dt = load(f)
        sec = np.zeros(len(cells))
        for p, d in zip(xy, dt):
            c = (np.floor((p[0] - lo[0]) / CELL) * CELL + lo[0] + CELL / 2, np.floor((p[1] - lo[1]) / CELL) * CELL + lo[1] + CELL / 2)
            if c in ci:
                sec[ci[c]] += d
        per.append(sec)
        tot += sec
        print(f"  {os.path.basename(f)[:28]:28s} {dt.sum() / 60:5.1f} min tracked, cells visited: {(sec >= MIN_S).mean():.0%}")
    print(f"  combined: cells visited {(tot >= MIN_S).mean():.0%}, visited for 10+ s: {(tot >= 10).mean():.0%}")

    print("\ntime at each planned stop (within 18 in), seconds per session:")
    for n, s in stops.items():
        row = []
        for f in files:
            xy, dt = load(f)
            row.append(dt[np.linalg.norm(xy - s, axis=1) < STOP_R].sum())
        print(f"  {n:3s} " + "  ".join(f"{v:6.1f}" for v in row))

    C = np.array(cells)
    thin = tot < MIN_S
    print("\nparts of the floor with no data (share of cells never visited):")
    for name, m in (("back strip (y < -20 in)", C[:, 1] < -20), ("right side (x > 45 in)", C[:, 0] > 45),
                    ("left side (x < -50 in)", C[:, 0] < -50), ("front (y > 35 in)", C[:, 1] > 35),
                    ("middle (|x| < 45, -20 < y < 35)", (np.abs(C[:, 0]) < 45) & (C[:, 1] > -20) & (C[:, 1] < 35))):
        if m.sum():
            print(f"  {name:34s} {thin[m].mean():4.0%} of {int(m.sum())} cells")

    sc = 5.0
    W, H = int((hi[0] - lo[0]) * sc) + 40, int((hi[1] - lo[1]) * sc) + 50
    img = np.full((H, W, 3), 255, np.uint8)
    for (cx, cy), s in zip(cells, tot):
        v = min(s / 30.0, 1.0)
        col = (int(255 - 200 * v), int(255 - 120 * v), 255) if s >= MIN_S else (235, 235, 235)
        x0, y0 = int((cx - CELL / 2 - lo[0]) * sc) + 20, int((cy - CELL / 2 - lo[1]) * sc) + 36
        cv2.rectangle(img, (x0, y0), (x0 + int(CELL * sc) - 1, y0 + int(CELL * sc) - 1), col, -1)
    cv2.polylines(img, [np.array([((x - lo[0]) * sc + 20, (y - lo[1]) * sc + 36) for x, y in area], np.int32)], True, (60, 170, 60), 2, cv2.LINE_AA)
    for n, s in stops.items():
        c = (int((s[0] - lo[0]) * sc + 20), int((s[1] - lo[1]) * sc + 36))
        cv2.circle(img, c, 7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, n, (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "time spent per 12 in cell (darker = more, grey = never visited)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    out = preset.out("tracks", "coverage.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("\nwrote", out)


if __name__ == "__main__":
    main()
