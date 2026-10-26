"""AD HOC (project README figures, 2026-10-25). Kept because it can be rerun.

What it did: regenerated the five calibration figures in docs/img/ (camera view, landmarks, rug
distances, stickers found, calibrated grid) from the active preset, after pixelating every person
the pose model could find so no figure shows a recognizable face. Blurred copies of the frames were
cached in the preset's output/blurred/. The frame names it uses are hard-coded below.

Run from feasibility/:  python tools/adhoc/make_figures.py [--preset NAME]
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import calib
import preset
from track_server import blur_boxes

DOCS = os.path.join(preset.ROOT, "docs", "img")
RAW = "frames/sample_frame_v1_nostickers.jpg"          # camera view without stickers
STICKERS = "frames/sample_frame_stickers2.jpg"         # camera view with stickers placed
YELLOW, MAGENTA, GREEN, ORANGE = (0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 128, 255)


def text(img, s, org, col=YELLOW, scale=0.9, th=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th)


def blurred(model, rel):
    """Blurred copy of a preset frame (cached); returns (image, path)."""
    out = preset.out("blurred", os.path.basename(rel))
    img = cv2.imread(preset.path(rel))
    r = model.predict(img, imgsz=1280, conf=0.25, device="mps", verbose=False)[0]
    blur_boxes(img, r.boxes.xyxy.cpu().numpy())
    cv2.imwrite(out, img)
    print(f"  {rel}: blurred {len(r.boxes)} person box(es)")
    return img, out


def save(name, img, quality=88):
    os.makedirs(DOCS, exist_ok=True)
    cv2.imwrite(os.path.join(DOCS, name), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print("  wrote docs/img/" + name)


def main():
    from ultralytics import YOLO
    model = YOLO(os.path.join(preset.ROOT, "models", "yolo11m-pose.pt"))
    pix, sig = calib.load_points()
    area = calib.AREA.astype(np.int32)

    raw, _ = blurred(model, RAW)
    stk, stk_path = blurred(model, STICKERS)
    save("1_camera_view.jpg", raw)

    # 2: landmarks and walking area on the empty room
    img = raw.copy()
    ov = img.copy()
    cv2.fillPoly(ov, [area], (0, 200, 0))
    img = cv2.addWeighted(ov, 0.22, img, 0.78, 0)
    cv2.polylines(img, [area], True, GREEN, 3)
    for n, (x, y) in calib.PIX.items():
        cv2.circle(img, (int(x), int(y)), 7, YELLOW, -1)
        text(img, n, (int(x) + 10, int(y) - 8), scale=0.8)
    text(img, "walking area", (1180, 1040), GREEN)
    save("2_landmarks.jpg", img)

    # 3: zoom on the rug with the measured distances
    pts = {n: p for n, p in calib.PIX.items()}
    x0, y0, x1, y1, z = 470, 600, 990, 900, 3.0
    zoom = cv2.resize(raw[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    to = lambda p: (int((p[0] - x0) * z), int((p[1] - y0) * z))
    for f in calib.CFG["distance_files"]:
        for r in csv.DictReader(open(preset.path(f))):
            a, b = calib.ALIAS.get(r["a"], r["a"]), calib.ALIAS.get(r["b"], r["b"])
            if a in pts and b in pts and (a, b) not in calib.IGNORE and (b, a) not in calib.IGNORE:
                pa, pb = to(pts[a]), to(pts[b])
                cv2.line(zoom, pa, pb, ORANGE, 2)
                text(zoom, r["inches"] + '"', ((pa[0] + pb[0]) // 2 + 6, (pa[1] + pb[1]) // 2 - 6), ORANGE, 0.7)
    for n, p in pts.items():
        cv2.drawMarker(zoom, to(p), YELLOW, cv2.MARKER_CROSS, 24, 2)
        text(zoom, n, (to(p)[0] + 10, to(p)[1] - 10), scale=0.8)
    save("3_rug_distances.jpg", zoom)

    # 4: stickers found in the frame
    img = stk.copy()
    for r in csv.DictReader(open(preset.path(calib.CFG["sticker_pixels_file"]))):
        c = (int(float(r["px"])), int(float(r["py"])))
        cv2.circle(img, c, 22, (0, 0, 255), 3)
        text(img, r["name"], (c[0] + 26, c[1] - 14), scale=1.0)
    save("4_stickers.jpg", img)

    # 5: calibrated grid
    b = calib.Fit(calib.all_names(pix), pix, sig, calib.load_dists())
    tmp = preset.out("blurred", "grid.jpg")
    calib.draw_grid(b, frame=stk_path, out=tmp)
    save("5_calibrated_grid.jpg", cv2.imread(tmp))


if __name__ == "__main__":
    main()
