"""Build the synced dataset of one session: three clock-aligned phone channels plus the person's position.

Inputs: the three phone recordings, the clock model (sync_session.py) and the tracked positions
(track_video.py). Outputs, in the preset's dataset/sessionN/:
  phone_A/B/C.m4a   the recordings as they came off the phones, clock_model.json, claps.json, visual_claps.json
  audio_abc.wav     (make_wav.py, not committed) 3 channels (A, B, C), 48 kHz, 16 bit. Phone A is exactly as recorded; B and C are
                    moved by their clock offset (spline interpolation), so that a sound reaches the
                    channels at the delays set by the room (distances / 343 m/s), not by the clocks.
  positions.csv     the person's floor position in the same time base (t_audio_s = seconds from the
                    first audio sample), at the tracker's 10 Hz.
  meta.json         phone positions, clock model and its accuracy, clap times, coverage, caveats.

The audio starts where the three phones and the video overlap. After building, the claps are timed
again in the output as a check: each channel's clap arrival, minus the travel time from where the
person stood, should coincide across channels.

Usage: build_dataset.py [--preset NAME] SESSION
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preset
from find_claps import load_audio
from make_wav import render
from sync_session import SR, IN, first_arrival, phones_m, travel

def build(session):
    rec = preset.path("output", "recordings")
    sync_dir = preset.path("output", "sync")
    model = json.load(open(os.path.join(sync_dir, f"session{session}_clockmodel.json")))
    claps = json.load(open(os.path.join(sync_dir, f"session{session}_claps.json")))
    vis = json.load(open(os.path.join(sync_dir, f"session{session}_visual_claps.json")))
    track_path = glob.glob(preset.path("output", "tracks", f"session{session}_*_track.csv"))[0]
    rows = list(csv.DictReader(open(track_path)))
    video_len = float(rows[-1]["t_video_s"])
    off_A = model["off_A_phone_minus_video_s"]
    D, b, u_anchor = model["D_s"], model["b_ppm"], model["u_anchor_s"]
    xs = {p: load_audio(os.path.join(rec, f"Session{session} Phone{p}.m4a"), SR) for p in "ABC"}
    n_src = {p: len(xs[p]) for p in "ABC"}

    # the span where all three phones and the video exist (phone A time u)
    margin = 0.05
    lo = [margin, off_A]
    hi = [n_src["A"] / SR - margin, off_A + video_len]
    for p in "BC":
        lo.append(-D[p] + margin)
        hi.append((n_src[p] / SR - margin) - D[p])
    u_lo, u_hi = max(lo), min(hi)
    k0 = int(np.ceil(u_lo * SR))
    N = int((u_hi - k0 / SR) * SR)
    v0 = k0 / SR - off_A                                    # video time of the first audio sample
    print(f"  audio from video time {v0:.2f} s, {N / SR:.1f} s long ({N} samples)")

    d = preset.path("dataset", f"session{session}")
    os.makedirs(d, exist_ok=True)
    for p in "ABC":                                         # the recordings as they came off the phones, plus what make_wav.py needs
        shutil.copyfile(os.path.join(rec, f"Session{session} Phone{p}.m4a"), os.path.join(d, f"phone_{p}.m4a"))
    for src, dst in ((f"session{session}_clockmodel.json", "clock_model.json"), (f"session{session}_claps.json", "claps.json"),
                     (f"session{session}_visual_claps.json", "visual_claps.json")):
        shutil.copyfile(os.path.join(sync_dir, src), os.path.join(d, dst))
    out = render(d, k0, N, model)

    # check: time the claps again in the output
    ca = [t - k0 / SR for t in [c[0] for c in claps["Phone A"]["start"]["claps"]]]
    tau = travel(np.array(vis["stood_xy_in"]) * IN)
    resid = []
    for t in ca:
        # look for each channel's clap where the geometry says it should be (the window is only 12 ms wide)
        fa = [first_arrival(out[:, c].astype(np.float32) / 32768.0, t + (tau[c] - tau[0])) for c in range(3)]
        resid.append([(fa[1] - fa[0]) - (tau[1] - tau[0]), (fa[2] - fa[0]) - (tau[2] - tau[0])])
    resid = np.array(resid) * 1000
    print(f"  clap check in the output: B-A residual {np.round(resid[:, 0], 2)} ms, C-A residual {np.round(resid[:, 1], 2)} ms (0 = perfect)")

    # outputs
    n_rows = 0
    status = {"tracked": 0, "interp": 0, "lost": 0}
    with open(os.path.join(d, "positions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_audio_s", "t_video_s", "x_in", "y_in", "x_m", "y_m", "x_raw_m", "y_raw_m", "status", "n_ankles", "conf"])
        for r in rows:
            tv = float(r["t_video_s"])
            if not (v0 <= tv <= v0 + N / SR):
                continue
            n_rows += 1
            status[r["status"]] += 1
            ok = r["x_m"] != ""
            w.writerow([f"{tv - v0:.3f}", f"{tv:.3f}", f"{float(r['x_m']) / IN:.2f}" if ok else "", f"{float(r['y_m']) / IN:.2f}" if ok else "",
                        r["x_m"], r["y_m"], r["x_raw_m"], r["y_raw_m"], r["status"], r["n_ankles"], r["conf"]])
    ph = preset.CFG["session"]["phones_in"]
    last_clap = max(ca)
    meta = {
        "session": session, "sample_rate": SR, "channels": ["A", "B", "C"], "bit_depth": 16, "n_samples": N,
        "duration_s": N / SR, "audio_starts_at_video_time_s": v0,
        "floor_frame": "origin at rug corner R2, x along R1->R2 (toward the right wall), y perpendicular (toward the desk); inches and meters",
        "phones_xy_in": {p: ph[i] for i, p in enumerate("ABC")},
        "phones_xy_m": {p: [ph[i][0] * IN, ph[i][1] * IN] for i, p in enumerate("ABC")},
        "clock_offsets_s": D, "drift_ppm": b, "drift_status": model["drift"],
        "clap_agreement_ms": model["clap_agreement_ms"],
        "claps_t_audio_s": ca, "clap_xy_in": vis["stood_xy_in"], "clap_check_residual_ms": {"B-A": resid[:, 0].tolist(), "C-A": resid[:, 1].tolist()},
        "claps_t_video_s": vis["t_video_s"],
        "usable_from_s": last_clap + 3.0,
        "positions": {"rows": n_rows, "rate_hz": 10, "tracked": status["tracked"], "interp": status["interp"], "lost": status["lost"],
                      "source": os.path.basename(track_path)},
        "accuracy": {
            "phone_to_phone_offset_at_the_claps_ms": "about +-0.5 (the 3 claps and 3 timing methods agree to this)",
            "drift_ms_at_the_end": "UNMEASURED. There are no claps at the end, so each phone's clock rate against A is unknown. "
                                   "A drift of d ppm adds d * seconds_since_the_claps / 1000 ms of offset: 5 ppm is 2.7 ms (0.9 m) after 9 minutes. Trust the start more than the end.",
            "audio_to_position_ms": "about +-40 (one video frame is 50 ms; from the hand claps in the video)",
            "position": "feet midpoint from the offline tracker; about 1.6 cm calibration accuracy, plus detection error",
        },
    }
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"  wrote {d}  (audio {os.path.getsize(os.path.join(d, 'audio_abc.wav')) / 1e6:.0f} MB, {n_rows} position rows, "
          f"tracked {status['tracked'] / max(n_rows, 1):.0%}, usable from {last_clap + 3:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    build(ap.parse_args().session)


if __name__ == "__main__":
    main()
