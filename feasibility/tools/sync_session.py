"""Work out how the three phone clocks relate to each other and to the camera video, for one session.

Needs, from the preset's output/sync/: the claps heard by the phones (find_claps.py) and the claps
seen in the video (find_visual_claps.py).

Offsets. At each clap the arrival time at every phone is measured three ways (onset, first arrival
of the sound, short cross-correlation) and averaged. The sound's travel time from where the person
stood (from the video) to each phone is then subtracted, leaving the phones' clock offsets and not
the room's geometry. Across the three claps and three methods the result agrees to about 0.5 ms.

Drift. A clock that runs slightly fast or slow makes the offset grow over the session (1 ms is
34 cm of path difference). Drift can only be measured from a second clap pattern at the end of the
recording: give --end-xy (where the person stood for those claps) and it is computed. Without end
claps the drift is NOT estimated and is set to zero, and the model says so.

Result: output/sync/sessionN_clockmodel.json, with, for phone p and phone A's file time u:
    phone p file time = u + D_p + b_p * (u - u_anchor)        (b_p = 0 when drift is unknown)
    phone A file time = video time + off_A                      (known to about +-40 ms, from the video claps)

Usage: sync_session.py [--preset NAME] SESSION [--end-xy X_IN Y_IN]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preset
from find_claps import load_audio

SR = 48000
IN = 0.0254
C_SOUND = 343.0
CLAP_Z = 1.2            # m, height of the hands above the phones during a clap
_HP = butter(4, 2000, "highpass", fs=SR, output="sos")


def phones_m():
    return np.array(preset.CFG["session"]["phones_in"], float) * IN


def travel(xy_m, z=CLAP_Z):
    """Seconds for sound from a point at height z over floor position xy_m to each phone."""
    return np.sqrt(((phones_m() - np.asarray(xy_m)) ** 2).sum(1) + z ** 2) / C_SOUND


def first_arrival(x, t, frac=0.2):
    """Time (s) when the high-passed signal first reaches `frac` of the clap's peak, near the coarse time t."""
    i0 = int((t - 0.004) * SR)
    i1 = int((t + 0.008) * SR)
    env = np.abs(sosfiltfilt(_HP, x[i0 - 2000:i1 + 2000].astype(np.float64))[2000:-2000])
    pk = env.max()
    j = int(np.argmax(env > frac * pk))
    if j > 0:
        j = j - 1 + (frac * pk - env[j - 1]) / (env[j] - env[j - 1] + 1e-12)
    return (i0 + j) / SR


def xcorr_time(a, b, fa, fb, half=0.0025):
    """Time in b (s) that best matches the clap at time fa in a, by correlating 12 ms of each."""
    n = int(0.012 * SR)
    A = sosfiltfilt(_HP, a[int((fa - 0.003) * SR):int((fa + 0.009) * SR)].astype(np.float64)) * np.hanning(n)
    out = []
    for sft in range(-int(half * SR), int(half * SR) + 1):
        s = int((fb - 0.003) * SR) + sft
        B = sosfiltfilt(_HP, b[s - 1000:s + n + 1000].astype(np.float64))[1000:-1000] * np.hanning(n)
        out.append(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))
    out = np.array(out)
    k = int(np.argmax(out))
    if 0 < k < len(out) - 1:
        k = k + 0.5 * (out[k - 1] - out[k + 1]) / (out[k - 1] - 2 * out[k] + out[k + 1])
    return fb + (k - int(half * SR)) / SR


def offsets_at(xs, coarse, xy_in):
    """Clock offset of phones B and C against A at a clap pattern, and how well the claps agree (ms)."""
    tau = travel(np.array(xy_in) * IN)
    D, spread, per = {"A": 0.0}, {}, {}
    for k, p in ((1, "B"), (2, "C")):
        est = []
        for c in range(3):
            fa = first_arrival(xs["A"], coarse["A"][c])
            fp = first_arrival(xs[p], coarse[p][c])
            est.append([coarse[p][c] - coarse["A"][c], fp - fa, xcorr_time(xs["A"], xs[p], fa, fp) - fa])
        est = np.array(est) - (tau[k] - tau[0])           # remove the travel-time difference
        per[p] = est.tolist()
        D[p] = float(est.mean())
        spread[p] = float(np.ptp(est.mean(axis=1)) * 1000)   # claps against each other, ms
    return D, spread, per


def estimate(session, end_xy=None):
    sync = json.load(open(preset.path("output", "sync", f"session{session}_claps.json")))
    vis = json.load(open(preset.path("output", "sync", f"session{session}_visual_claps.json")))
    rec = preset.path("output", "recordings")
    xs = {p: load_audio(os.path.join(rec, f"Session{session} Phone{p}.m4a"), SR) for p in "ABC"}
    start = {p: [t for t, _, _ in sync[f"Phone {p}"]["start"]["claps"]] for p in "ABC"}
    D, spread, per = offsets_at(xs, start, vis["stood_xy_in"])
    u0 = float(np.mean(start["A"]))
    model = {"session": session, "sr": SR, "off_A_phone_minus_video_s": vis["phone_minus_video_s"]["A"],
             "u_anchor_s": u0, "D_s": D, "b_ppm": {"A": 0.0, "B": 0.0, "C": 0.0}, "clap_xy_in": vis["stood_xy_in"],
             "clap_agreement_ms": spread, "per_clap_offsets_s": per, "drift": "not estimated (no end claps)"}
    for p in "BC":
        print(f"  phone {p} clock offset against A: {D[p]:+.5f} s   (the 3 claps agree to {spread[p]:.2f} ms)")
    ends = {p: [t for t, _, _ in (sync[f"Phone {p}"]["end"]["claps"] or [])] for p in "ABC"}
    if end_xy and all(len(ends[p]) == 3 for p in "ABC"):
        De, sp_e, _ = offsets_at(xs, ends, end_xy)
        u1 = float(np.mean(ends["A"]))
        for p in "BC":
            model["b_ppm"][p] = float((De[p] - D[p]) / (u1 - u0) * 1e6)
            print(f"  phone {p} drift against A: {model['b_ppm'][p]:+.2f} ppm (offset changed {1000 * (De[p] - D[p]):+.2f} ms over {u1 - u0:.0f} s)")
        model["drift"] = "measured from the end claps"
        model["end_clap_xy_in"] = list(end_xy)
    else:
        print("  drift: not estimated, there are no end claps (or --end-xy was not given)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    ap.add_argument("--end-xy", type=float, nargs=2, default=None)
    a = ap.parse_args()
    print(f"session {a.session}")
    m = estimate(a.session, a.end_xy)
    path = preset.path("output", "sync", f"session{a.session}_clockmodel.json")
    json.dump(m, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
