"""Offline tracker: a recorded camera video -> the person's floor position over time.

Same pose model, calibration and ankle logic as the live tracker, but with the camera's own frame
timestamps, no real-time constraint, and a zero-lag smoother (Savitzky-Golay over each unbroken
stretch of the track) so the positions do not trail behind the person.

Writes, next to the preset's other tracks (output/tracks/):
  <name>_track.csv      one row per processed frame: time, smoothed and raw position, quality
  <name>_overview.jpg   trajectory on the floor plan, and x/y over time with the stationary stretches
and prints a summary, including how the stretches where the person stood still compare with the
planned stop points (a check of the whole chain: calibration, ankle height, tracking).

Usage: track_video.py [--preset NAME] RECORDING.ts [--stride 2] [--limit-s SECONDS] [--start-s SECONDS]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preset
from track_server import Floor, pick_person, ANKLE_CONF

IN = 0.0254
GAP_FILL_S = 0.6        # fill detection gaps up to this long by interpolation
SEG_BREAK_S = 1.0       # a longer gap breaks the track
SMOOTH_S = 1.0          # Savitzky-Golay window
STILL_SPEED = 0.12      # m/s: below this the person counts as standing still
STILL_MIN_S = 10.0


def frame_times(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    t = np.sort(np.array([float(x.strip(",")) for x in out.split() if x.strip(",")]))
    return t - t[0]


def detect(args, floor):
    from ultralytics import YOLO
    model = YOLO(args.model)
    W, H = floor.c["image"]
    accept = np.zeros((H, W), np.uint8)
    cv2.fillPoly(accept, [np.array(floor.c["area_px"], np.int32)], 255)
    accept = cv2.dilate(accept, np.ones((121, 121), np.uint8)) > 0

    times = frame_times(args.video)
    cap = cv2.VideoCapture(args.video)
    n_dec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{len(times)} video packets over {times[-1]:.1f} s, decoder reports {n_dec} frames", flush=True)
    rows, last_feet, last_t, i, t0 = [], None, -9.0, -1, time.time()
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i >= len(times):
            break
        t = float(times[i])
        if t < args.start_s or i % args.stride:
            continue
        if args.limit_s and t > args.start_s + args.limit_s:
            break
        ok, frame = cap.retrieve()
        if not ok:
            continue
        res = model.predict(frame, imgsz=args.imgsz, conf=0.25, device=args.device, verbose=False)[0]
        pick = pick_person(res, last_feet if t - last_t < 2.0 else None, accept)
        if pick is None:
            rows.append(dict(i=i, t=t, ok=0))
        else:
            _, k, pts, feet = pick
            px = np.array([p for p, _ in pts])
            xy = floor.to_xy(px, floor.ankle_h).mean(0)
            rows.append(dict(i=i, t=t, ok=1, x=xy[0], y=xy[1], n=len(pts), conf=float(np.mean([c for _, c in pts])),
                             px=float(feet[0]), py=float(feet[1])))
            last_feet, last_t = feet, t
        if len(rows) % 300 == 0:
            done = t / times[-1]
            el = time.time() - t0
            print(f"  {len(rows)} frames, video {t:5.1f}/{times[-1]:.0f} s, {el:.0f} s elapsed, ~{el / max(done, 1e-3) - el:.0f} s left", flush=True)
    return rows


def smooth(rows, fps):
    n = len(rows)
    t = np.array([r["t"] for r in rows])
    X = np.array([[r.get("x", np.nan), r.get("y", np.nan)] for r in rows])
    status = np.where(np.isnan(X[:, 0]), "lost", "tracked").astype(object)
    # fill short gaps
    good = ~np.isnan(X[:, 0])
    idx = np.where(good)[0]
    Xf = X.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and t[b] - t[a] <= GAP_FILL_S:
            for c in range(2):
                Xf[a + 1:b, c] = np.interp(t[a + 1:b], [t[a], t[b]], [X[a, c], X[b, c]])
            status[a + 1:b] = "interp"
    have = ~np.isnan(Xf[:, 0])
    S = np.full_like(Xf, np.nan)
    win = max(5, int(round(SMOOTH_S * fps)) | 1)
    start = None
    for j in range(n + 1):
        on = j < n and have[j] and (start is None or t[j] - t[j - 1] <= SEG_BREAK_S)
        if on and start is None:
            start = j
        elif not on and start is not None:
            seg = slice(start, j)
            m = j - start
            if m >= 5:
                w = min(win, m if m % 2 else m - 1)
                S[seg] = savgol_filter(Xf[seg], w, 2, axis=0) if w >= 5 else Xf[seg]
            else:
                S[seg] = Xf[seg]
            start = j if (j < n and have[j]) else None
    return t, X, S, status


def stationary(t, S, fps):
    v = np.full(len(t), np.nan)
    k = max(1, int(round(0.5 * fps)))
    for j in range(k, len(t) - k):
        if not np.isnan(S[j - k, 0]) and not np.isnan(S[j + k, 0]) and t[j + k] - t[j - k] < 2.0:
            v[j] = np.linalg.norm(S[j + k] - S[j - k]) / (t[j + k] - t[j - k])
    still = np.nan_to_num(v, nan=9.0) < STILL_SPEED
    segs, start = [], None
    for j in range(len(t) + 1):
        if j < len(t) and still[j] and start is None:
            start = j
        elif (j == len(t) or not still[j]) and start is not None:
            if t[j - 1] - t[start] >= STILL_MIN_S:
                segs.append((start, j))
            start = None
    return v, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default=os.path.join(preset.ROOT, "models", "yolo11m-pose.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame (2 = 10 fps)")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--limit-s", type=float, default=0.0, help="process only this many seconds (0 = all)")
    ap.add_argument("--calibration", default=preset.out("calibration.json"))
    args = ap.parse_args()
    floor = Floor(args.calibration)
    name = os.path.splitext(os.path.basename(args.video))[0]
    rows = detect(args, floor)
    fps = 20.0 / args.stride
    t, X, S, status = smooth(rows, fps)

    path = preset.out("tracks", name + "_track.csv")
    meta = {}
    mp = os.path.splitext(args.video)[0] + ".json"
    if os.path.exists(mp):
        meta = json.load(open(mp))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_video_s", "x_m", "y_m", "x_raw_m", "y_raw_m", "status", "n_ankles", "conf", "feet_px", "feet_py"])
        for r, tt, xr, s, st in zip(rows, t, X, S, status):
            w.writerow([f"{tt:.3f}", *(["", ""] if np.isnan(s[0]) else [f"{s[0]:.4f}", f"{s[1]:.4f}"]),
                        *(["", ""] if np.isnan(xr[0]) else [f"{xr[0]:.4f}", f"{xr[1]:.4f}"]), st,
                        r.get("n", ""), f"{r['conf']:.2f}" if "conf" in r else "", f"{r['px']:.1f}" if "px" in r else "",
                        f"{r['py']:.1f}" if "py" in r else ""])
    print("wrote", path)

    # ---- summary
    tracked = (status != "lost").mean()
    print(f"\nframes processed {len(rows)}; position available {tracked:.0%} "
          f"(detected {(status == 'tracked').mean():.0%}, short gaps filled {(status == 'interp').mean():.0%}, lost {(status == 'lost').mean():.0%})")
    lost_runs, a = [], None
    for j in range(len(t) + 1):
        if j < len(t) and status[j] == "lost" and a is None:
            a = j
        elif (j == len(t) or status[j] != "lost") and a is not None:
            if t[j - 1] - t[a] > 2:
                lost_runs.append((t[a], t[j - 1]))
            a = None
    print("lost for more than 2 s:", [f"{x:.0f}-{y:.0f}s" for x, y in lost_runs] or "none")
    v, segs = stationary(t, S, fps)
    land = floor.c["landmarks"]
    stops = {n: np.array(land[n]) for n in preset.CFG.get("session", {}).get("stops", []) if n in land}
    moving = ~np.isnan(v) & (v >= STILL_SPEED)
    print(f"moving speed: median {np.nanmedian(v[moving]):.2f} m/s, 90th percentile {np.nanpercentile(v[moving], 90):.2f} m/s")
    print(f"\nstanding-still stretches (>= {STILL_MIN_S:.0f} s) and the nearest planned stop:")
    matched = []
    for a, b in segs:
        p = np.nanmedian(S[a:b], axis=0)
        near = min(stops, key=lambda n: np.linalg.norm(stops[n] - p)) if stops else "-"
        d = np.linalg.norm(stops[near] - p) / IN if stops else float("nan")
        matched.append((a, b, near, d))
        print(f"  {t[a]:6.1f}-{t[b - 1]:6.1f} s ({t[b - 1] - t[a]:4.1f} s)  at x={p[0] / IN:6.1f} y={p[1] / IN:6.1f} in  -> {near:3s}  off by {d:5.1f} in")
    if matched:
        d = np.array([m[3] for m in matched])
        print(f"  {len(matched)} stretches: median offset {np.median(d):.1f} in, worst {d.max():.1f} in")

    # ---- overview figure
    cal = floor.c
    P = np.array(cal["area_xy"]) / IN
    allp = np.vstack([P] + [np.array(v_) / IN for v_ in cal["landmarks"].values()])
    lo, hi = allp.min(0) - 12, allp.max(0) + 12
    mw, mh = 800, 640
    sc = min((mw - 20) / (hi[0] - lo[0]), (mh - 50) / (hi[1] - lo[1]))
    Xp = lambda x: int((x - lo[0]) * sc + 10)
    Yp = lambda y: int((y - lo[1]) * sc + 36)
    mp_img = np.full((mh, mw, 3), 255, np.uint8)
    cv2.polylines(mp_img, [np.array([(Xp(x), Yp(y)) for x, y in P], np.int32)], True, (60, 170, 60), 2, cv2.LINE_AA)
    for n, p in stops.items():
        c = (Xp(p[0] / IN), Yp(p[1] / IN))
        cv2.circle(mp_img, c, 9, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(mp_img, n, (c[0] + 10, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    SS = S / IN
    for j in range(1, len(t)):
        if not np.isnan(SS[j, 0]) and not np.isnan(SS[j - 1, 0]) and t[j] - t[j - 1] < SEG_BREAK_S:
            hue = int(150 * t[j] / max(t[-1], 1))
            col = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])
            cv2.line(mp_img, (Xp(SS[j - 1, 0]), Yp(SS[j - 1, 1])), (Xp(SS[j, 0]), Yp(SS[j, 1])), col, 2, cv2.LINE_AA)
    cv2.putText(mp_img, "path, colored by time (blue = start, red = end)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
    tw, th = 1400, 640
    ts_img = np.full((th, tw, 3), 255, np.uint8)
    t_max = max(t[-1], 1)
    Tx = lambda s: int(50 + s / t_max * (tw - 70))
    for k, (name_, col, rng) in enumerate((("x (in)", (200, 90, 0), (-110, 100)), ("y (in)", (0, 120, 220), (-70, 90)))):
        y0, y1 = 40 + k * 300, 40 + k * 300 + 250
        Yv = lambda v_, y0=y0, y1=y1, rng=rng: int(y1 - (v_ - rng[0]) / (rng[1] - rng[0]) * (y1 - y0))
        cv2.rectangle(ts_img, (50, y0), (tw - 20, y1), (230, 230, 230), 1)
        for a, b, near, d in matched:
            cv2.rectangle(ts_img, (Tx(t[a]), y0), (Tx(t[b - 1]), y1), (215, 240, 215), -1)
            if k == 0:
                cv2.putText(ts_img, near, (Tx(t[a]), y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 110, 0), 1, cv2.LINE_AA)
        for j in range(1, len(t)):
            if status[j] == "lost":
                cv2.line(ts_img, (Tx(t[j]), y0), (Tx(t[j]), y1), (200, 200, 255), 1)
            elif not np.isnan(SS[j, k]) and not np.isnan(SS[j - 1, k]) and t[j] - t[j - 1] < SEG_BREAK_S:
                cv2.line(ts_img, (Tx(t[j - 1]), Yv(SS[j - 1, k])), (Tx(t[j]), Yv(SS[j, k])), col, 2, cv2.LINE_AA)
        cv2.putText(ts_img, name_, (6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    for s_ in range(0, int(t_max), 60):
        cv2.putText(ts_img, f"{s_ // 60}:00", (Tx(s_), th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(ts_img, "green = standing still, red ticks = person not found", (tw - 460, th - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    out = preset.out("tracks", name + "_overview.jpg")
    top = np.full((mh, tw, 3), 255, np.uint8)
    top[:, :mw] = mp_img
    cv2.imwrite(out, np.vstack([top, ts_img]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == "__main__":
    main()
