"""Find the three claps in the camera video from the person's hands, frame by frame.

Tracks both wrists with the pose model in a time window around where the claps are. A clap is the
moment the wrists meet: the distance between them, in units of the shoulder width, drops from
about one to near zero. The three dips whose spacing best matches the spacing of the claps heard
by the phones (from find_claps.py) are taken as the claps. Also reports where the person stood.

Writes output/sync/sessionN_visual_claps.json and a picture of the hands at each clap, and prints
a first estimate of each phone's clock offset to the video clock.

Usage: find_visual_claps.py [--preset NAME] SESSION --from SECONDS --to SECONDS
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preset
from track_server import Floor
from track_video import frame_times

IN = 0.0254
C_SOUND = 343.0
HAND_Z = 1.2          # m, hands during a clap (assumed)
LW, RW, LS, RS = 9, 10, 5, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    ap.add_argument("--from", dest="t0", type=float, required=True)
    ap.add_argument("--to", dest="t1", type=float, required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()
    n = args.session
    rec = preset.path("output", "recordings")
    video = glob.glob(os.path.join(rec, f"session{n}_*.ts"))[0]
    sync = json.load(open(preset.path("output", "sync", f"session{n}_claps.json")))
    audio = {k: [t for t, _, _ in v["start"]["claps"]] for k, v in sync.items() if k.startswith("Phone") and v["start"]["claps"]}
    ref = audio["Phone A"]
    want_gaps = np.diff(ref)
    print(f"session {n}: audio clap gaps {np.round(want_gaps, 3)} s; searching {args.t0:.0f}-{args.t1:.0f} s of the video")

    from ultralytics import YOLO
    model = YOLO(os.path.join(preset.ROOT, "models", "yolo11m-pose.pt"))
    floor = Floor(preset.out("calibration.json"))
    times = frame_times(video)
    cap = cv2.VideoCapture(video)
    rows, i = [], -1            # decode from the start: seeking inside a .ts is not frame-exact
    frames = {}
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i >= len(times) or times[i] > args.t1:
            break
        if times[i] < args.t0:
            continue
        ok, frame = cap.retrieve()
        res = model.predict(frame, imgsz=args.imgsz, conf=0.25, device="mps", verbose=False)[0]
        best = None
        if res.keypoints is not None and len(res.boxes):
            kxy, kc = res.keypoints.xy.cpu().numpy(), res.keypoints.conf.cpu().numpy()
            area = (res.boxes.xyxy[:, 2] - res.boxes.xyxy[:, 0]).cpu().numpy() * (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1]).cpu().numpy()
            for p in np.argsort(-area):
                if min(kc[p][LW], kc[p][RW], kc[p][LS], kc[p][RS]) >= 0.3:
                    best = p
                    break
        if best is None:
            rows.append(dict(i=i, t=float(times[i]), nd=np.nan))
            continue
        k, c = kxy[best], kc[best]
        sw = np.linalg.norm(k[LS] - k[RS])
        d = np.linalg.norm(k[LW] - k[RW])
        ank = [k[j] for j in (15, 16) if c[j] >= 0.35]
        rows.append(dict(i=i, t=float(times[i]), nd=float(d / max(sw, 1)), d_px=float(d), mid=((k[LW] + k[RW]) / 2).tolist(),
                         feet=(np.mean(ank, axis=0).tolist() if ank else None), sw=float(sw)))
        frames[i] = frame
    t = np.array([r["t"] for r in rows])
    nd = np.array([r["nd"] for r in rows])
    print(f"{len(rows)} frames processed, person with both wrists found in {np.mean(~np.isnan(nd)):.0%}")

    # dips: local minima with a clear approach and retreat
    cand = []
    for j in range(2, len(nd) - 2):
        if np.isnan(nd[j]) or nd[j] > 0.55:
            continue
        w = nd[max(0, j - 8):j + 9]
        if nd[j] <= np.nanmin(w) + 1e-9:
            before = np.nanmax(nd[max(0, j - 10):j]) if j > 0 else 0
            after = np.nanmax(nd[j + 1:j + 11]) if j + 1 < len(nd) else 0
            if min(before, after) - nd[j] >= 0.35:
                cand.append(j)
    print("hand-closing candidates at (s):", [f"{t[j]:.2f}" for j in cand])
    best = None
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            for c_ in range(b + 1, len(cand)):
                tt = t[[cand[a], cand[b], cand[c_]]]
                err = np.abs(np.diff(tt) - want_gaps).sum()
                if err < 0.9 and (best is None or err < best[0]):
                    best = (err, [cand[a], cand[b], cand[c_]])
    if not best:
        print("no triple of hand closings matches the clap rhythm; look at the candidates above")
        return
    err, js = best
    ct = t[js]
    print(f"claps in the video at {np.round(ct, 3)} s (frames {[rows[j]['i'] for j in js]}); gaps {np.round(np.diff(ct), 3)} vs audio {np.round(want_gaps, 3)}; rhythm error {err:.2f} s")

    # where the person stood at the claps, and the travel time to each phone
    phones = np.array(preset.CFG["session"]["phones_in"]) * IN
    feet = [rows[j].get("feet") for j in js]
    feet = [f for f in feet if f]
    out = {"session": n, "frames": [rows[j]["i"] for j in js], "t_video_s": [float(x) for x in ct],
           "wrist_gap_shoulder_widths": [float(nd[j]) for j in js], "audio_gaps_s": [float(x) for x in want_gaps]}
    if feet:
        xy = floor.to_xy(np.array(feet), floor.ankle_h).mean(0)
        print(f"person stood at x={xy[0] / IN:.1f} in, y={xy[1] / IN:.1f} in (floor, from the ankles)")
        d3 = np.sqrt(((phones - xy) ** 2).sum(1) + HAND_Z ** 2)
        travel = d3 / C_SOUND
        out["stood_xy_in"] = [float(xy[0] / IN), float(xy[1] / IN)]
        out["travel_ms"] = {p: float(tr * 1000) for p, tr in zip("ABC", travel)}
        print("sound travel time to the phones (A, B, C), ms:", np.round(travel * 1000, 1))
        # clock offset of each phone to the video: phone_time = video_time + offset
        offs = {}
        for p, tr in zip("ABC", travel):
            T = np.array(audio[f"Phone {p}"])
            o = T - tr - ct
            offs[p] = float(o.mean())
            print(f"  Phone {p}: offset to video clock {o.mean():9.3f} s   (spread across the 3 claps {np.ptp(o) * 1000:5.0f} ms)")
        out["phone_minus_video_s"] = offs
    json.dump(out, open(preset.path("output", "sync", f"session{n}_visual_claps.json"), "w"), indent=1)

    # picture: the frame before, at, and after each clap
    tiles = []
    for j in js:
        row = []
        for dj in (-1, 0, 1):
            fr = frames.get(rows[j + dj]["i"])
            m = rows[j + dj].get("mid")
            if fr is None or m is None:
                row.append(np.zeros((300, 300, 3), np.uint8))
                continue
            x0, y0 = int(max(m[0] - 150, 0)), int(max(m[1] - 150, 0))
            crop = cv2.resize(fr[y0:y0 + 300, x0:x0 + 300], (300, 300))
            cv2.putText(crop, f"{rows[j + dj]['t']:.2f}s f{rows[j + dj]['i']}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            row.append(crop)
        tiles.append(np.hstack(row))
    cv2.imwrite(preset.path("output", "sync", f"session{n}_visual_claps.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", preset.path("output", "sync", f"session{n}_visual_claps.jpg"))


if __name__ == "__main__":
    main()
