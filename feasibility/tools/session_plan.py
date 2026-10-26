"""Plan recording sessions: stop points, a walking ring, suggested phone spots, and a timed script.

Inputs: the preset's output/calibration.json and its "session" block. Outputs, in the preset:
  output/session_plan_camera.jpg   stops, ring and phones drawn on the (blurred) camera frame
  output/session_plan_maps.jpg     top-down maps with the stop order of each session
  session_plan.md                  stop coordinates, phone spots and the timed script of every session

Each session = a head (3 claps at the center stop), then n cycles of [walk one ring, stand at a
stop], then a tail (3 claps). The stop order is chosen so successive stops are far apart, and the
facing direction rotates so each stop is visited facing different ways across sessions.

Usage: session_plan.py [--preset NAME]
"""
import itertools
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preset
from track_server import Floor

IN = 0.0254
CFG = preset.CFG["session"]
fl = Floor(preset.out("calibration.json"))
AREA = np.array(fl.c["area_xy"]) / IN
LM = {k: np.array(v) / IN for k, v in fl.c["landmarks"].items()}
STOPS = {n: LM[n] for n in CFG["stops"]}
RUG = fl.to_xy(np.array(CFG["rug_corners_px"], float)) / IN
SPEED = CFG.get("walk_speed_mps", 0.5)
CENTER = "S2"      # claps happen here: central, in view, away from walls


def sd(poly, p):
    """Signed distance (inches) from p to polygon: positive inside."""
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), True)


def tri_inside(P, a, b, c):
    d1 = (P[:, 0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (P[:, 1] - b[1])
    d2 = (P[:, 0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (P[:, 1] - c[1])
    d3 = (P[:, 0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (P[:, 1] - a[1])
    return ((d1 < 0) & (d2 < 0) & (d3 < 0)) | ((d1 > 0) & (d2 > 0) & (d3 > 0))


def pick_phones(on_rug=None):
    """Three phone spots: most stops inside their triangle, plus the share of the walkable floor it covers."""
    on_rug = CFG.get("phones_on_rug", False) if on_rug is None else on_rug
    cand = []
    for x in np.arange(-110, 100, 6.0):
        for y in np.arange(-70, 90, 6.0):
            p = np.array([x, y])
            if sd(AREA, p) < 5 or (not on_rug and sd(RUG, p) > -6):
                continue
            if min(np.linalg.norm(p - s) for s in STOPS.values()) < 18:
                continue
            cand.append(p)
    cand = np.array(cand)
    S = np.array(list(STOPS.values()))
    floor_pts = np.array([[x, y] for x in np.arange(-110, 100, 10.0) for y in np.arange(-70, 90, 10.0)
                          if sd(AREA, np.array([x, y])) > 0])
    best = None
    for i, j, k in itertools.combinations(range(len(cand)), 3):
        a, b, c = cand[i], cand[j], cand[k]
        sides = [np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(a - c)]
        if min(sides) < 75:
            continue
        la, lb, lc = sides[1], sides[2], sides[0]
        angs = [np.degrees(np.arccos(np.clip((x * x + y * y - z * z) / (2 * x * y), -1, 1)))
                for x, y, z in ((lb, lc, la), (la, lc, lb), (la, lb, lc))]
        if min(angs) < 28:
            continue
        cover = float(tri_inside(floor_pts, a, b, c).mean())
        inside = int(tri_inside(S, a, b, c).sum())
        key = (round(inside + 5 * cover, 2), round(cover, 2), inside)
        if best is None or key > best[0]:
            best = (key, [a, b, c])
    return best[1], best[0][2], best[0][1]


def ring_points(tri, near, margin):
    """Closed walking loop: the walkable area, limited to the phone triangle plus `near` inches, shrunk by `margin`."""
    ox, oy = -120, -80
    W, H = 230, 180
    area = np.zeros((H, W), np.uint8)
    cv2.fillPoly(area, [np.round(AREA - [ox, oy]).astype(np.int32)], 255)
    t = np.zeros((H, W), np.uint8)
    cv2.fillPoly(t, [np.round(tri - [ox, oy]).astype(np.int32)], 255)
    k = int(near)
    t = cv2.dilate(t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    for q in tri:   # keep the ring clear of the phones
        cv2.circle(t, (int(round(q[0] - ox)), int(round(q[1] - oy))), int(CFG["phone_clear_in"] + margin), 0, -1)
    m = cv2.erode(area & t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(margin) + 1, 2 * int(margin) + 1)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, 5, True)[:, 0, :].astype(float)
    return ap + [ox, oy]


def order_stops(names, start, rng):
    left, order, cur = [n for n in names if n != start], [start], start
    while left:
        d = np.array([np.linalg.norm(STOPS[n] - STOPS[cur]) for n in left]) + rng.uniform(0, 25, len(left))
        cur = left.pop(int(d.argmax()))
        order.append(cur)
    return order


def make_sessions():
    names = CFG["stops"]
    n_s, per = CFG["n_sessions"], CFG["stops_per_session"]
    rng = np.random.default_rng(7)
    omit_n = len(names) - per
    visits = {n: 0 for n in names}
    sessions = []
    for s in range(n_s):
        omit = [names[(s * omit_n + q + 3) % len(names)] for q in range(omit_n)]
        used = [n for n in names if n not in omit]
        order = order_stops(used, used[rng.integers(len(used))], rng)
        cues = []
        for n in order:
            f = CFG["facings"][(visits[n] + names.index(n)) % len(CFG["facings"])]
            visits[n] += 1
            cues.append((n, f))
        sessions.append(cues)
    return sessions


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def to_px(pts_in):
    return fl.to_px(np.atleast_2d(pts_in) * IN)


def main():
    phones, inside, cover = pick_phones()
    if "track_in" in CFG:      # a hand-drawn loop (floor inches), checked below
        ring = np.array(CFG["track_in"], float)
    else:
        ring = ring_points(np.array(phones), CFG["ring_near_in"], CFG["ring_margin_in"])
    wall = min(sd(AREA, q) for q in ring)
    clear = min(np.linalg.norm(q - p) for q in ring for p in phones)
    print(f"track: closest to a wall {wall:.0f} in, closest to a phone corner {clear:.0f} in (ring corners only)")
    ring_len = sum(np.linalg.norm(ring[i] - ring[(i + 1) % len(ring)]) for i in range(len(ring))) * IN
    sessions = make_sessions()
    names = CFG["stops"]
    tri = np.array(phones)
    print(f"phones: {[tuple(int(v) for v in np.round(p)) for p in phones]}  floor covered: {cover:.0%}  stops inside: {inside}/{len(names)}")
    print(f"ring: {len(ring)} corners, {ring_len:.1f} m per lap ({ring_len / SPEED:.0f} s at {SPEED} m/s)")
    sides = [np.linalg.norm(tri[i] - tri[(i + 1) % 3]) * IN for i in range(3)]
    print("phone spacing (m):", np.round(sides, 2))

    # ---- camera view
    img = cv2.imread(preset.out("blurred", os.path.basename(preset.CFG["sample_frame"])))
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(im, t, o, col, sc=0.8, th=2):
        cv2.putText(im, t, o, font, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(im, t, o, font, sc, col, th, cv2.LINE_AA)

    rp = to_px(np.vstack([ring, ring[:1]]))
    cv2.polylines(img, [rp.astype(np.int32)], False, (255, 200, 0), 3, cv2.LINE_AA)
    tp = to_px(np.vstack([tri, tri[:1]]))
    cv2.polylines(img, [tp.astype(np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
    for n in names:
        c = tuple(to_px(STOPS[n])[0].astype(int))
        cv2.circle(img, c, 14, (0, 255, 0), 3, cv2.LINE_AA)
        label(img, n, (c[0] + 16, c[1] - 12), (0, 255, 0))
    for q, p in zip("ABC", phones):
        c = tuple(to_px(p)[0].astype(int))
        cv2.rectangle(img, (c[0] - 12, c[1] - 12), (c[0] + 12, c[1] + 12), (255, 0, 255), -1)
        label(img, "phone " + q, (c[0] - 40, c[1] - 22), (255, 0, 255))
    label(img, "green: stops   cyan: walking ring   magenta: suggested phone spots",
          (30, 1060), (0, 255, 255), 0.75)
    cv2.imwrite(preset.out("session_plan_camera.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ---- top-down maps
    allp = np.vstack([AREA, tri])
    lo, hi = allp.min(0) - 12, allp.max(0) + 12
    pw, ph = 620, 470
    sc = min((pw - 20) / (hi[0] - lo[0]), (ph - 50) / (hi[1] - lo[1]))
    X = lambda x: int((x - lo[0]) * sc + 10)
    Y = lambda y: int((y - lo[1]) * sc + 36)
    panels = []
    for si, cues in enumerate(sessions):
        pn = np.full((ph, pw, 3), 255, np.uint8)
        for f in range(-96, 97, 24):
            cv2.line(pn, (X(f), Y(lo[1])), (X(f), Y(hi[1])), (225, 225, 225), 1)
        for f in range(-72, 97, 24):
            cv2.line(pn, (X(lo[0]), Y(f)), (X(hi[0]), Y(f)), (225, 225, 225), 1)
        cv2.polylines(pn, [np.array([(X(x), Y(y)) for x, y in AREA], np.int32)], True, (60, 170, 60), 2, cv2.LINE_AA)
        cv2.fillPoly(pn, [np.array([(X(x), Y(y)) for x, y in RUG], np.int32)], (225, 215, 235))
        cv2.polylines(pn, [np.array([(X(x), Y(y)) for x, y in np.vstack([ring, ring[:1]])], np.int32)], False, (200, 150, 0), 2, cv2.LINE_AA)
        cv2.polylines(pn, [np.array([(X(x), Y(y)) for x, y in np.vstack([tri, tri[:1]])], np.int32)], False, (200, 0, 200), 2, cv2.LINE_AA)
        for q, p in zip("ABC", phones):
            cv2.rectangle(pn, (X(p[0]) - 7, Y(p[1]) - 7), (X(p[0]) + 7, Y(p[1]) + 7), (200, 0, 200), -1)
            cv2.putText(pn, q, (X(p[0]) + 10, Y(p[1]) + 5), font, 0.6, (200, 0, 200), 2, cv2.LINE_AA)
        seq = [n for n, _ in cues]
        for a, b in zip(seq, seq[1:]):
            cv2.arrowedLine(pn, (X(STOPS[a][0]), Y(STOPS[a][1])), (X(STOPS[b][0]), Y(STOPS[b][1])), (140, 140, 140), 1, cv2.LINE_AA, tipLength=0.05)
        for n in names:
            on = n in seq
            c = (X(STOPS[n][0]), Y(STOPS[n][1]))
            cv2.circle(pn, c, 11, (0, 150, 0) if on else (190, 190, 190), -1 if on else 2, cv2.LINE_AA)
            if on:
                cv2.putText(pn, str(seq.index(n) + 1), (c[0] - 5 - 5 * (seq.index(n) >= 9), c[1] + 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(pn, n, (c[0] + 13, c[1] - 9), font, 0.5, (0, 90, 0) if on else (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(pn, f"Session {si + 1}: stop order 1-{len(seq)} (grey = unused)", (10, 22), font, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
        panels.append(pn)
    sheet = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:4])]) if len(panels) >= 4 else np.hstack(panels)
    cv2.imwrite(preset.out("session_plan_maps.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # ---- markdown
    head, stand, walk, tail = CFG["head_s"], CFG["stand_s"], CFG["walk_s"], CFG["tail_s"]
    total = head + len(sessions[0]) * (stand + walk) + tail
    md = [f"# Recording session plan ({preset.NAME})", "",
          f"{len(sessions)} sessions of {mmss(total)}. Coordinates are in inches in the floor frame "
          f"(origin R2, x along R1->R2 toward the right wall, y toward the camera / desk).", "",
          "## Stop points", "", "| Stop | Where | x (in) | y (in) | vs phone triangle |", "|---|---|---|---|---|"]
    for n in names:
        where = "floor sticker" if n[0] == "S" else "rug: put a sticker at this block corner (see the rug figure)"
        d = cv2.pointPolygonTest(np.array(phones, np.float32), (float(STOPS[n][0]), float(STOPS[n][1])), True)
        status = "inside the triangle" if d > 0 else ("just outside" if d > -16 else "outside (extrapolation test)")
        md.append(f"| {n} | {where} | {STOPS[n][0]:.0f} | {STOPS[n][1]:.0f} | {status} |")
    md += ["", "## Suggested phone spots", "",
           f"Triangle with sides {sides[0]:.1f}, {sides[1]:.1f}, {sides[2]:.1f} m; it covers {cover:.0%} of the walkable floor and {inside} of {len(names)} stops lie inside it. "
           "Lay the phones face up, microphone end pointing the same way, on bare floor, and do not move them. "
           "An exact spot is not needed: we measure where they landed with the camera.", "",
           "| Phone | x (in) | y (in) |", "|---|---|---|"]
    for q, p in zip("ABC", phones):
        md.append(f"| {q} | {p[0]:.0f} | {p[1]:.0f} |")
    md += ["", "## Walking ring", "",
           f"One lap is about {ring_len:.1f} m, about {ring_len / SPEED:.0f} s at a slow {SPEED} m/s, so a {walk} s walk is about "
           f"{walk * SPEED / ring_len:.1f} laps. Alternate clockwise and counter-clockwise (as seen from above, with the back wall at the top). "
           "Cut a corner now and then, and vary your pace a little.", ""]
    laps = walk * SPEED / ring_len
    for si, cues in enumerate(sessions):
        md += [f"## Session {si + 1}", "",
               f"| Time | Do |", "|---|---|",
               f"| 0:00 | Recording running on all 3 phones, tracker logging. At {CENTER}, face the camera, hands visible: clap 3 times, 1 s apart |",
               f"| 0:10 | Start reading. Settle |"]
        t = head
        for ci, (n, f) in enumerate(cues):
            d = "clockwise" if ci % 2 == 0 else "counter-clockwise"
            md.append(f"| {mmss(t)} | Walk the ring {d}, about {laps:.1f} laps; use the last few seconds to head to **{n}** |")
            md.append(f"| {mmss(t + walk)} | Stand still at **{n}** for {stand} s, facing **{f}** |")
            t += walk + stand
        md.append(f"| {mmss(t)} | Go to {CENTER}, face the camera: clap 3 times, 1 s apart |")
        md.append(f"| {mmss(total)} | Stop reading; stop recording a few seconds later |")
        md.append("")
    md += ["After the 4th session, record 2-3 minutes of silence (nobody speaking or moving).", ""]
    open(preset.path("session_plan.md"), "w").write("\n".join(md))
    print("wrote", preset.path("session_plan.md"))
    print("walk/stand shares:", f"walk {len(sessions[0]) * walk}s, stand {len(sessions[0]) * stand}s of {total}s")


if __name__ == "__main__":
    main()
