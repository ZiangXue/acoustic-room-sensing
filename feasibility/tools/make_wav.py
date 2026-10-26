"""Make the clock-synced 3-channel WAV of a session from the original phone recordings in its dataset folder.

Reads, from dataset/sessionN/: phone_A.m4a, phone_B.m4a, phone_C.m4a (the recordings as they came off
the phones), clock_model.json (phone clock offsets, from sync_session.py) and meta.json (where the
audio starts, how many samples). Writes audio_abc.wav: 3 channels (A, B, C), 48 kHz, 16 bit. Phone A is
exactly as recorded; B and C are moved by their clock offset (cubic spline), so that a sound reaches the
channels at the delays set by the room (distances / 343 m/s), not by the clocks.

The WAV is about 2.3x the size of the three recordings, so it is not committed: run this to get it.

Usage: make_wav.py DATASET_DIR [DATASET_DIR ...]      e.g. make_wav.py presets/apartment/dataset/session1
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import map_coordinates, spline_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_claps import load_audio

SR = 48000
CHUNK = 1 << 21


def to_int16(x):
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)


def shifted(x, idx):
    """Samples of x at fractional positions idx (cubic spline), as float in [-1, 1]."""
    coef = spline_filter1d(x.astype(np.float64), order=3)
    out = np.empty(len(idx), np.float64)
    for s in range(0, len(idx), CHUNK):
        out[s:s + CHUNK] = map_coordinates(coef, [idx[s:s + CHUNK]], order=3, mode="nearest", prefilter=False)
    return out


def render(d, k0, N, model):
    """Write d/audio_abc.wav: N samples starting at sample k0 of phone A's file."""
    xs = {p: load_audio(os.path.join(d, f"phone_{p}.m4a"), SR) for p in "ABC"}
    D, b, u_anchor = model["D_s"], model["b_ppm"], model["u_anchor_s"]
    u = (k0 + np.arange(N)) / SR
    out = np.zeros((N, 3), np.int16)
    out[:, 0] = to_int16(xs["A"][k0:k0 + N])
    for c, p in ((1, "B"), (2, "C")):
        T = u + D[p] + b[p] * 1e-6 * (u - u_anchor)       # phone p's file time at phone A's file time u
        out[:, c] = to_int16(shifted(xs[p], T * SR))
    wavfile.write(os.path.join(d, "audio_abc.wav"), SR, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    for d in ap.parse_args().dirs:
        model = json.load(open(os.path.join(d, "clock_model.json")))
        meta = json.load(open(os.path.join(d, "meta.json")))
        k0 = int(round((meta["audio_starts_at_video_time_s"] + model["off_A_phone_minus_video_s"]) * SR))
        render(d, k0, meta["n_samples"], model)
        print("wrote", os.path.join(d, "audio_abc.wav"))


if __name__ == "__main__":
    main()
