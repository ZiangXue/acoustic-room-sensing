"""Where should the next recording session go? A PDF that shows, on the camera snapshot and on a
floor plan, the phone positions, what the finished sessions already cover, and the zone to hit next.

Phone positions come from the preset ("session" -> "phones_in", floor inches as measured). Cells
(12 in) inside or near the phone triangle that the tracked sessions never visited are the target.
Spots to stand at are picked from the target cells, far apart from each other.

Usage: next_session.py [--preset NAME] [--out NAME.pdf] TRACK.csv [TRACK.csv ...]
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preset
from track_server import Floor

IN = 0.0254
CELL = 12.0
MIN_S, WELL_S = 1.5, 10.0
NEAR_IN = 24.0
SPOT_MIN_GAP = 28.0
PHONE_CLEAR = 20.0
N_SPOTS = 6

GREEN, YELLOW, RED, MAG = (80, 190, 80), (60, 200, 245), (60, 60, 235), (230, 40, 230)


def font(sz, bold=False):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz, index=1 if bold and p.endswith(".ttc") else 0)
    return ImageFont.load_default()


def load_track(path):
    a = np.array([(float(r["x_m"]) / IN, float(r["y_m"]) / IN, float(r["t_video_s"]))
                  for r in csv.DictReader(open(path)) if r["x_m"]])
    return a[:, :2], np.minimum(np.diff(a[:, 2], append=a[-1, 2]), 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--out", default="next_session.pdf")
    args = ap.parse_args()
    cfg = preset.CFG["session"]
    cal = json.load(open(preset.out("calibration.json")))
    floor = Floor(preset.out("calibration.json"))
    area = (np.array(cal["area_xy"]) / IN).astype(np.float32)
    phones = np.array(cfg["phones_in"], np.float32)
    tri = phones
    lo, hi = area.min(0), area.max(0)

    # --- cells, time per cell
    cells = []
    for x in np.arange(lo[0], hi[0], CELL):
        for y in np.arange(lo[1], hi[1], CELL):
            c = (x + CELL / 2, y + CELL / 2)
            if cv2.pointPolygonTest(area, c, True) >= 4:
                cells.append(c)
    C = np.array(cells)
    sec = np.zeros(len(cells))
    idx = {tuple(c): i for i, c in enumerate(cells)}
    for f in args.tracks:
        xy, dt = load_track(f)
        for p, d in zip(xy, dt):
            c = (np.floor((p[0] - lo[0]) / CELL) * CELL + lo[0] + CELL / 2, np.floor((p[1] - lo[1]) / CELL) * CELL + lo[1] + CELL / 2)
            if c in idx:
                sec[idx[c]] += d
    dist_tri = np.array([cv2.pointPolygonTest(tri, (float(c[0]), float(c[1])), True) for c in cells])
    near_phone = np.array([np.linalg.norm(phones - c, axis=1).min() < PHONE_CLEAR for c in C])
    in_zone = (dist_tri > -NEAR_IN) & ~near_phone
    target = in_zone & (sec < MIN_S)
    thin = in_zone & (sec >= MIN_S) & (sec < WELL_S)
    well = in_zone & (sec >= WELL_S)
    print(f"cells in or near the triangle: {in_zone.sum()}; well covered {well.sum()}, thin {thin.sum()}, never visited {target.sum()}")

    # --- stand spots: far-apart unvisited cells, then a tour
    spots = []
    edge = np.array([cv2.pointPolygonTest(area, (float(c[0]), float(c[1])), True) for c in cells])
    for mask in (target & (edge >= 16), thin & (edge >= 16)):   # unvisited cells first, then thin ones; not at the edge
        cand = C[mask]
        order = np.argsort(-np.minimum(dist_tri[mask], 40))        # prefer cells deeper inside the triangle
        for i in order:
            if len(spots) < N_SPOTS and all(np.linalg.norm(cand[i] - s) >= SPOT_MIN_GAP for s in spots):
                spots.append(cand[i])
    tour = [spots.pop(int(np.argmin([s[0] for s in spots])))]      # start at the left-most spot
    while spots:
        j = int(np.argmin([np.linalg.norm(s - tour[-1]) for s in spots]))
        tour.append(spots.pop(j))
    print("stand spots (in):", [tuple(int(round(v)) for v in s) for s in tour])

    # --- camera view
    img = cv2.imread(preset.out("blurred", os.path.basename(preset.CFG["sample_frame"])))
    ov = img.copy()
    H, W = img.shape[:2]

    def cell_poly(c):
        h = CELL / 2
        return floor.to_px(np.array([[c[0] - h, c[1] - h], [c[0] + h, c[1] - h], [c[0] + h, c[1] + h], [c[0] - h, c[1] + h]]) * IN)

    def valid(c):
        pts = np.array([[c[0] + dx, c[1] + dy] for dx in (-6, 6) for dy in (-6, 6)]) * IN
        px = floor.to_px(pts)
        return bool((np.linalg.norm(floor.to_xy(px) - pts, axis=1) < 0.02).all() and (px >= 0).all() and (px[:, 0] < W).all() and (px[:, 1] < H).all())

    for k, c in enumerate(cells):
        if not in_zone[k] or not valid(c):
            continue
        col = RED if target[k] else (YELLOW if thin[k] else GREEN)
        cv2.fillConvexPoly(ov, cell_poly(c).astype(np.int32), col)
    img = cv2.addWeighted(ov, 0.42, img, 0.58, 0)
    tp = floor.to_px(np.vstack([tri, tri[:1]]) * IN)
    cv2.polylines(img, [tp.astype(np.int32)], False, MAG, 3, cv2.LINE_AA)

    def label(im, t, o, col, sc=0.9, th=2):
        cv2.putText(im, t, o, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(im, t, o, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

    for q, p in zip("ABC", phones):
        c = floor.to_px(p[None] * IN)[0].astype(int)
        cv2.rectangle(img, (c[0] - 13, c[1] - 13), (c[0] + 13, c[1] + 13), MAG, -1)
        label(img, f"phone {q}", (c[0] - 45, c[1] - 24), MAG)
    for n, s in enumerate(tour, 1):
        c = floor.to_px(s[None] * IN)[0].astype(int)
        cv2.circle(img, tuple(c), 17, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(c), 17, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, str(n), (c[0] - 7, c[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cam = img

    # --- floor plan
    sc = 6.0
    pw, ph_ = int((hi[0] - lo[0]) * sc) + 60, int((hi[1] - lo[1]) * sc) + 60
    plan = np.full((ph_, pw, 3), 255, np.uint8)
    X = lambda x: int((x - lo[0]) * sc + 30)
    Y = lambda y: int((y - lo[1]) * sc + 30)
    for f in range(int(np.ceil(lo[0] / 12) * 12), int(hi[0]) + 1, 12):
        cv2.line(plan, (X(f), Y(lo[1])), (X(f), Y(hi[1])), (238, 238, 238), 1)
    for f in range(int(np.ceil(lo[1] / 12) * 12), int(hi[1]) + 1, 12):
        cv2.line(plan, (X(lo[0]), Y(f)), (X(hi[0]), Y(f)), (238, 238, 238), 1)
    for k, c in enumerate(cells):
        col = (60, 60, 235) if target[k] else ((60, 200, 245) if thin[k] else ((80, 190, 80) if well[k] else (240, 240, 240)))
        if in_zone[k] or True:
            a = 0.55 if in_zone[k] else 0.25
            x0, y0 = X(c[0] - CELL / 2), Y(c[1] - CELL / 2)
            sub = plan[y0:Y(c[1] + CELL / 2), x0:X(c[0] + CELL / 2)]
            sub[:] = (a * np.array(col) + (1 - a) * sub).astype(np.uint8)
    cv2.polylines(plan, [np.array([(X(x), Y(y)) for x, y in area], np.int32)], True, (90, 150, 90), 2, cv2.LINE_AA)
    cv2.polylines(plan, [np.array([(X(x), Y(y)) for x, y in np.vstack([tri, tri[:1]])], np.int32)], False, MAG[::-1] if False else (200, 0, 200), 3, cv2.LINE_AA)
    land = cal["landmarks"]
    for n in cfg["stops"]:
        p = np.array(land[n]) / IN
        cv2.circle(plan, (X(p[0]), Y(p[1])), 5, (110, 110, 110), -1, cv2.LINE_AA)
        cv2.putText(plan, n, (X(p[0]) + 7, Y(p[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    for q, p in zip("ABC", phones):
        cv2.rectangle(plan, (X(p[0]) - 9, Y(p[1]) - 9), (X(p[0]) + 9, Y(p[1]) + 9), (200, 0, 200), -1)
        cv2.putText(plan, f"{q} ({p[0]:g}, {p[1]:g})", (X(p[0]) + 13, Y(p[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 0, 160), 2, cv2.LINE_AA)
    for n, s in enumerate(tour, 1):
        cv2.circle(plan, (X(s[0]), Y(s[1])), 15, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(plan, (X(s[0]), Y(s[1])), 15, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(plan, str(n), (X(s[0]) - 6, Y(s[1]) + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    for a, b in zip(tour, tour[1:]):
        cv2.arrowedLine(plan, (X(a[0]), Y(a[1])), (X(b[0]), Y(b[1])), (60, 60, 60), 1, cv2.LINE_AA, tipLength=0.04)

    # --- PDF pages
    def page(title, sub, im_bgr, w=1920):
        im = Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
        im = im.resize((w, int(im.height * w / im.width)))
        pg = Image.new("RGB", (w, im.height + 150), "white")
        d = ImageDraw.Draw(pg)
        d.text((30, 18), title, fill=(20, 20, 20), font=font(44, True))
        d.text((30, 80), sub, fill=(90, 90, 90), font=font(26))
        pg.paste(im, (0, 150))
        return pg, d

    def legend(pg, y):
        d = ImageDraw.Draw(pg)
        x = 30
        for col, txt in (((235, 60, 60), "never visited: go here"), ((245, 200, 60), "thin (< 10 s)"), ((80, 190, 80), "well covered"),
                         ((200, 0, 200), "phones and their triangle")):
            d.rectangle((x, y, x + 34, y + 26), fill=col)
            d.text((x + 44, y - 2), txt, fill=(30, 30, 30), font=font(26))
            x += 44 + int(d.textlength(txt, font=font(26))) + 40

    n_t, n_th, n_w = int(target.sum()), int(thin.sum()), int(well.sum())
    p1, _ = page("Session 3: where to go", f"Camera snapshot. Red = {n_t} floor cells (12 in) inside or near the phone triangle that sessions 1 and 2 never visited. Numbered circles = stand spots.", cam)
    legend(p1, p1.height - 52 if False else 112)
    p2, _ = page("Floor plan (inches from R2)",
                 f"x toward the right wall, y toward the desk. Phones A, B, C at the measured positions. Well covered {n_w}, thin {n_th}, never visited {n_t} cells in or near the triangle.", plan, w=1500)
    legend(p2, 112)

    # page 3: spot table and how to run it
    p3 = Image.new("RGB", (1500, 1000), "white")
    d = ImageDraw.Draw(p3)
    d.text((30, 18), "Stand spots for session 3", fill=(20, 20, 20), font=font(44, True))
    d.text((30, 80), "Visit them in this order, twice around if time allows. Coordinates are floor inches from R2.", fill=(90, 90, 90), font=font(26))
    y = 150
    d.text((30, y), "Spot", font=font(28, True), fill=(0, 0, 0)); d.text((180, y), "x (in)", font=font(28, True), fill=(0, 0, 0))
    d.text((340, y), "y (in)", font=font(28, True), fill=(0, 0, 0)); d.text((500, y), "Near", font=font(28, True), fill=(0, 0, 0))
    names = {n: np.array(land[n]) / IN for n in land}
    for n, s in enumerate(tour, 1):
        y += 56
        near = min(names, key=lambda k: np.linalg.norm(names[k] - s))
        dd = np.linalg.norm(names[near] - s)
        d.text((30, y), str(n), font=font(28), fill=(0, 0, 0)); d.text((180, y), f"{s[0]:.0f}", font=font(28), fill=(0, 0, 0))
        d.text((340, y), f"{s[1]:.0f}", font=font(28), fill=(0, 0, 0)); d.text((500, y), f"{dd:.0f} in from {near}", font=font(28), fill=(0, 0, 0))
    y += 90
    for line in ("How to run it (about 8 to 9 minutes, same claps and settings as the other sessions):",
                 "  - Claps: 3 at the start and 3 at the end, at the center sticker S2, facing the camera.",
                 "  - 8 cycles: sweep the red zone for about 30 s, then stand still at the next spot for 25 s.",
                 "  - Keep the sweep inside the red and yellow areas. Skip the middle band, it is already well covered.",
                 "  - Face a different direction at each visit: the door, the desk, the right wall, the sofa.",
                 "  - Afterwards, 2 to 3 minutes of silence: nobody speaking or moving.",
                 "Do not move the phones. Phone positions are the measured ones above."):
        d.text((30, y), line, font=font(27), fill=(30, 30, 30))
        y += 46
    out = os.path.join(preset.path("output"), args.out)
    for k, pg in enumerate((p1, p2, p3), 1):
        pg.save(preset.out(f"next_session_page{k}.jpg"), quality=85)
    p1.save(out, "PDF", resolution=150, save_all=True, append_images=[p2, p3])
    cv2.imwrite(preset.out("next_session_plan.jpg"), plan)
    print("wrote", out)


if __name__ == "__main__":
    main()
