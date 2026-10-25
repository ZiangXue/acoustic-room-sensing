"""Live person tracking server: camera RTSP -> YOLO pose -> floor (x, y) in the R2 frame.

Serves on 0.0.0.0:9090:
  /              web page: annotated video + top-down map + readout
  /video         MJPEG stream of the annotated camera frame   (?grid=1 draws the calibrated 2 ft grid)
  /events        server-sent events, one JSON per processed frame
  /state.json    latest state
  /calibration.json  landmarks and walking area for the map

Position = midpoint of the two ankles. The pose model finds ankles from the body skeleton,
so footwear (slippers, bare feet) does not matter. An ankle is a few cm above the floor, so its
ray is intersected with the plane ANKLE_H above the floor, which puts the point under the ankle.

Usage: track_server.py [--preset NAME] [--model yolo11m-pose.pt] [--imgsz 1280] [--port 9090] [--log NAME.csv|auto]
The camera URL (RTSP_URL) and the calibration come from the preset (presets/NAME/camera.env and
output/calibration.json, the latter written by tools/calib_export.py). --log writes timestamped
positions to the preset's output/tracks/ ("auto" names the file by the start time).
"""
import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import calib
import preset

ANKLE_H = 0.08      # m, ankle keypoint height above the floor (default; the preset can override)
ANKLE_CONF = 0.35   # minimum keypoint confidence to use an ankle
GATE_PX = 350       # max jump (px) of the person's feet between processed frames to keep the same track
LOST_AFTER = 1.0    # s without a valid detection before the state reports tracking=false
LA, RA = 15, 16     # COCO ankle keypoint indices
SKELETON = [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 7), (7, 9), (6, 8), (8, 10)]


class OneEuro:
    """Adaptive low-pass: smooth when still, responsive when moving."""

    def __init__(self, mincutoff=0.8, beta=0.6, dcutoff=1.0):
        self.mc, self.beta, self.dc = mincutoff, beta, dcutoff
        self.t = self.x = self.dx = None

    @staticmethod
    def _a(cut, dt):
        tau = 1 / (2 * math.pi * cut)
        return 1 / (1 + tau / dt)

    def __call__(self, t, x):
        x = np.asarray(x, float)
        if self.t is None:
            self.t, self.x, self.dx = t, x, np.zeros_like(x)
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx = self.dx + self._a(self.dc, dt) * (dx - self.dx)
        cut = self.mc + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._a(cut, dt) * (x - self.x)
        self.t = t
        return self.x


class Floor:
    """Pixel <-> floor (x, y) in the preset's floor frame, from its output/calibration.json."""

    def __init__(self, path):
        self.c = json.load(open(path))
        self.ankle_h = self.c.get("ankle_height_m", ANKLE_H)
        self.f, self.th, self.ph, self.h, self.k1 = self.c["params"]
        self.o, self.u, self.v = (np.array(self.c[k]) for k in ("o", "u", "v"))

    def to_xy(self, px, above_floor=0.0):
        q = calib.backproject(np.atleast_2d(px).astype(float), self.f, self.th, self.ph, self.h - above_floor, self.k1) - self.o
        return np.c_[q @ self.u, q @ self.v]

    def to_px(self, xy):
        xy = np.atleast_2d(xy)
        p = self.o + np.outer(xy[:, 0], self.u) + np.outer(xy[:, 1], self.v)
        return calib.project(p, self.f, self.th, self.ph, self.h, self.k1)


def grid_mask(floor, size):
    """Boolean image mask of the 2 ft grid lines inside the walking area, where the lens model is valid."""
    W, H = size
    lay = np.zeros((H, W), np.uint8)
    area = np.array(floor.c["area_px"], np.int32)
    inside = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inside, [area], 255)
    xy = np.array(floor.c["area_xy"])
    lo, hi = xy.min(0) - 0.2, xy.max(0) + 0.2
    step = 2 * calib.FT
    ts = np.linspace(0, 1, 300)
    for axis in (0, 1):
        for a in np.arange(np.ceil(lo[axis] / step) * step, hi[axis], step):
            pts = (np.c_[np.full(300, a), lo[1] + ts * (hi[1] - lo[1])] if axis == 0
                   else np.c_[lo[0] + ts * (hi[0] - lo[0]), np.full(300, a)])
            px = floor.to_px(pts)
            ok = (np.linalg.norm(floor.to_xy(px) - pts, axis=1) < 0.02) & (px[:, 0] >= 0) & (px[:, 0] < W) \
                & (px[:, 1] >= 0) & (px[:, 1] < H)
            for run in np.split(np.arange(300), np.where(np.diff(ok.astype(int)) != 0)[0] + 1):
                if ok[run[0]] and len(run) > 1:
                    cv2.polylines(lay, [px[run].astype(np.int32)], False, 255, 2)
    return (lay > 0) & (inside > 0)


class Shared:
    def __init__(self):
        self.cv = threading.Condition()
        self.frame = None        # latest raw frame
        self.frame_id = 0
        self.frame_t = 0.0
        self.jpeg = {}           # (grid, width) -> bytes, for the latest annotated frame
        self.annotated = None
        self.ann_id = 0
        self.state = {"tracking": False, "t": 0}
        self.state_id = 0
        self.fps = 0.0


def capture_loop(sh, url):
    while True:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("capture: cannot open stream, retrying", flush=True)
            time.sleep(2)
            continue
        print("capture: connected", flush=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with sh.cv:
                sh.frame, sh.frame_t = frame, time.time()
                sh.frame_id += 1
                sh.cv.notify_all()
        cap.release()
        print("capture: stream lost, reconnecting", flush=True)
        time.sleep(1)


def pick_person(res, last_feet, accept_mask):
    """Choose the tracked person among detections; returns (index, ankle_px list, ankle_conf list) or None."""
    if res.keypoints is None or len(res.boxes) == 0:
        return None
    kxy = res.keypoints.xy.cpu().numpy()
    kc = res.keypoints.conf.cpu().numpy()
    bconf = res.boxes.conf.cpu().numpy()
    box = res.boxes.xyxy.cpu().numpy()
    best = None
    for i in range(len(kxy)):
        pts = [(kxy[i][j], kc[i][j]) for j in (LA, RA) if kc[i][j] >= ANKLE_CONF]
        if not pts:
            continue
        feet = np.mean([p for p, _ in pts], axis=0)
        x, y = int(np.clip(feet[0], 0, accept_mask.shape[1] - 1)), int(np.clip(feet[1], 0, accept_mask.shape[0] - 1))
        if last_feet is not None:
            d = np.linalg.norm(feet - last_feet)
            if d > GATE_PX:
                continue
            score = -d
        else:
            if not accept_mask[y, x] or bconf[i] < 0.5:
                continue
            score = bconf[i] * (box[i][2] - box[i][0]) * (box[i][3] - box[i][1])
        if best is None or score > best[0]:
            best = (score, i, pts, feet)
    return best


def inference_loop(sh, args, floor):
    from ultralytics import YOLO
    model = YOLO(args.model)
    W, H = floor.c["image"]
    accept = np.zeros((H, W), np.uint8)
    cv2.fillPoly(accept, [np.array(floor.c["area_px"], np.int32)], 255)
    accept = cv2.dilate(accept, np.ones((121, 121), np.uint8)) > 0   # walking area plus a margin
    gm = grid_mask(floor, (W, H))
    smoother = OneEuro()
    last_feet, last_valid_t, last_id, ema_dt = None, 0.0, 0, None
    logf = None
    if args.log:
        name = time.strftime("run_%Y%m%d_%H%M%S.csv") if args.log == "auto" else args.log
        log_path = name if os.path.dirname(name) else preset.out("tracks", name)
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        print(f"logging positions to {log_path}", flush=True)
        logf = csv.writer(open(log_path, "w", newline=""))
        logf.writerow(["t_wall", "x_m", "y_m", "x_raw_m", "y_raw_m", "feet_px", "feet_py", "n_ankles", "conf"])
    while True:
        with sh.cv:
            sh.cv.wait_for(lambda: sh.frame_id != last_id)
            frame, t_cap, last_id = sh.frame, sh.frame_t, sh.frame_id
        t0 = time.time()
        res = model.predict(frame, imgsz=args.imgsz, conf=0.25, device=args.device, verbose=False)[0]
        pick = pick_person(res, last_feet if t0 - last_valid_t < 2.0 else None, accept)
        state = {"t": t_cap, "tracking": False, "lag_ms": 0}
        vis = frame.copy()
        if pick is not None:
            _, i, pts, feet = pick
            kxy = res.keypoints.xy.cpu().numpy()[i]
            kc = res.keypoints.conf.cpu().numpy()[i]
            for a, b in SKELETON:
                if kc[a] > 0.3 and kc[b] > 0.3:
                    cv2.line(vis, tuple(kxy[a].astype(int)), tuple(kxy[b].astype(int)), (0, 200, 255), 2)
            ank_px = np.array([p for p, _ in pts])
            ank_xy = floor.to_xy(ank_px, floor.ankle_h)
            xy_raw = ank_xy.mean(0)
            xy = smoother(t_cap, xy_raw)
            last_feet, last_valid_t = feet, t0
            for p in ank_px:
                cv2.circle(vis, tuple(p.astype(int)), 9, (0, 0, 255), -1)
            mid_px = floor.to_px(xy)[0]
            cv2.drawMarker(vis, tuple(mid_px.astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 34, 3)
            conf = float(np.mean([c for _, c in pts]))
            state.update(tracking=True, x=float(xy[0]), y=float(xy[1]), x_raw=float(xy_raw[0]), y_raw=float(xy_raw[1]),
                         px=float(feet[0]), py=float(feet[1]), n_ankles=len(pts), conf=conf)
            label = f"x={xy[0] / calib.IN:6.1f} in  y={xy[1] / calib.IN:6.1f} in  ({xy[0]:.2f}, {xy[1]:.2f}) m"
            cv2.rectangle(vis, (10, 10), (760, 56), (0, 0, 0), -1)
            cv2.putText(vis, label, (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
            if logf:
                logf.writerow([f"{t_cap:.3f}", f"{xy[0]:.4f}", f"{xy[1]:.4f}", f"{xy_raw[0]:.4f}", f"{xy_raw[1]:.4f}",
                               f"{feet[0]:.1f}", f"{feet[1]:.1f}", len(pts), f"{conf:.2f}"])
        else:
            cv2.rectangle(vis, (10, 10), (330, 56), (0, 0, 0), -1)
            cv2.putText(vis, "no person", (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
            if t0 - last_valid_t > LOST_AFTER:
                last_feet = None
        dt = time.time() - t0
        ema_dt = dt if ema_dt is None else 0.9 * ema_dt + 0.1 * dt
        state["lag_ms"] = int((time.time() - t_cap) * 1000)
        encoded = {}
        for grid in (0, 1):
            img = vis
            if grid:
                img = vis.copy()
                img[gm] = (0.5 * img[gm] + 0.5 * 255).astype(np.uint8)
            small = cv2.resize(img, (1280, int(1280 * H / W)), interpolation=cv2.INTER_AREA)
            encoded[grid] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])[1].tobytes()
        with sh.cv:
            sh.jpeg = encoded
            sh.ann_id += 1
            sh.fps = 1 / ema_dt
            state["fps"] = round(sh.fps, 1)
            sh.state, sh.state_id = state, sh.state_id + 1
            sh.cv.notify_all()


def make_handler(sh, floor):
    page = open(os.path.join(HERE, "track_ui.html"), "rb").read()
    calib_json = json.dumps(floor.c).encode()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, _, q = self.path.partition("?")
            if path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif path == "/calibration.json":
                self._send(calib_json, "application/json")
            elif path == "/state.json":
                self._send(json.dumps(sh.state).encode(), "application/json")
            elif path == "/video":
                grid = 1 if "grid=1" in q else 0
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with sh.cv:
                            sh.cv.wait_for(lambda: sh.ann_id != seen, timeout=5)
                            seen, jpg = sh.ann_id, sh.jpeg.get(grid)
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with sh.cv:
                            got = sh.cv.wait_for(lambda: sh.state_id != seen, timeout=10)
                            seen, st = sh.state_id, dict(sh.state)
                        self.wfile.write((f"data: {json.dumps(st)}\n\n" if got else ": keepalive\n\n").encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(b"not found", "text/plain", 404)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "yolo11m-pose.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--log", default="")
    ap.add_argument("--calibration", default=preset.out("calibration.json"))
    ap.add_argument("--url", default="")
    args = ap.parse_args()
    url = args.url or preset.camera_env()["RTSP_URL"]
    floor = Floor(args.calibration)
    sh = Shared()
    threading.Thread(target=capture_loop, args=(sh, url), daemon=True).start()
    threading.Thread(target=inference_loop, args=(sh, args, floor), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(sh, floor))
    srv.daemon_threads = True
    print(f"serving on http://0.0.0.0:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
