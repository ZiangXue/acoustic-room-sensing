"""Find the 3-clap pattern at the start and the end of every audio recording of a session.

Looks at the phone files (SessionN PhoneX.m4a) and the camera's audio track (sessionN_*.ts) in the
preset's output/recordings/. A clap is a sudden burst of high-frequency energy (rises by 15+ dB in
a few ms); speech is not. Within each window the three loudest, sharpest events that follow each
other within a few seconds are taken as the clap pattern.

Prints the clap times per device and writes a figure, output/sync/sessionN_claps.jpg, with the
waveforms at the start and the end, the chosen claps in red and the other candidates in grey.
Times are in seconds from the first audio sample of each file. Nothing here corrects for the
sound's travel time from the clap to each device; that comes in the next step.

Usage: find_claps.py [--preset NAME] SESSION_NUMBER [--window 150]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preset

HP_HZ = 1500.0
RISE_DB = 14.0          # a clap rises at least this much above the sound just before it
MAX_SPAN_S = 8.0        # the three claps happen within this time


def load_audio(path, sr):
    # the camera drops ~10% of its audio packets: place samples by their timestamps (silence fills the holes)
    af = ["-af", "aresample=async=1:first_pts=0"] if path.endswith(".ts") else []
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-map", "0:a:0", *af, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.float32)


def stream_start(path, kind):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", kind, "-show_entries", "stream=start_time",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.split()
    return float(out[0].strip(",")) if out else 0.0


def candidates(x, sr):
    """Onset times (s) with rise (dB) and level (dB above the median), sharpest first."""
    sos = butter(4, HP_HZ, "highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, x.astype(np.float64))
    p = y * y
    w = max(int(0.002 * sr), 2)
    c = np.concatenate([[0], np.cumsum(p)])
    after = (c[w:] - c[:-w]) / w                        # mean power in [i, i+2 ms)
    pre_n = int(0.035 * sr)
    gap = int(0.004 * sr)
    n = len(after)
    before = np.full(n, np.nan)
    idx = np.arange(pre_n + gap, n)
    before[idx] = (c[idx - gap] - c[idx - gap - pre_n]) / pre_n   # mean power in the 35 ms before
    eps = 1e-14
    rise = 10 * np.log10((after + eps) / (np.nan_to_num(before, nan=1.0) + eps))
    level = 10 * np.log10(after + eps) - 10 * np.log10(np.median(after) + eps)
    ok = (rise > RISE_DB) & (level > 25)
    cand = []
    i = 0
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j] and j - i < int(0.05 * sr):
                j += 1
            k = i + int(np.argmax(rise[i:j]))
            cand.append((k / sr, float(rise[k]), float(level[k])))
            i = j + int(0.12 * sr)                      # at least 120 ms between events
        else:
            i += 1
    return sorted(cand, key=lambda c: -(c[1] + c[2]))


def pick_triple(cand):
    """The three events within MAX_SPAN_S that are loudest and sharpest together."""
    best = None
    ts = sorted(cand[:12], key=lambda c: c[0])
    for a in range(len(ts)):
        for b in range(a + 1, len(ts)):
            for c in range(b + 1, len(ts)):
                t0, t1, t2 = ts[a][0], ts[b][0], ts[c][0]
                if t2 - t0 > MAX_SPAN_S or min(t1 - t0, t2 - t1) < 0.25:
                    continue
                score = sum(e[1] + e[2] for e in (ts[a], ts[b], ts[c]))
                # the three should be similar in strength: penalize a spread
                spread = max(e[2] for e in (ts[a], ts[b], ts[c])) - min(e[2] for e in (ts[a], ts[b], ts[c]))
                score -= 2 * spread
                if best is None or score > best[0]:
                    best = (score, [ts[a], ts[b], ts[c]])
    return best[1] if best else None


def figure(session, devices, result, window, out):
    rows = len(devices)
    W, H = 1800, 150 * rows + 70
    img = np.full((H, W), 255, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colw = (W - 60) // 2
    for r, (name, x, sr) in enumerate(devices):
        for c, which in enumerate(("start", "end")):
            seg_n = int(window * sr)
            seg = x[:seg_n] if which == "start" else x[-seg_n:]
            t_off = 0.0 if which == "start" else len(x) / sr - len(seg) / sr
            x0, y0 = 40 + c * (colw + 20), 50 + r * 150
            cv2.rectangle(img, (x0, y0), (x0 + colw, y0 + 130), (235, 235, 235), 1)
            env = np.abs(seg)
            nb = colw
            blocks = env[: len(env) // nb * nb].reshape(nb, -1).max(axis=1)
            blocks = blocks / (blocks.max() + 1e-9)
            for i, v in enumerate(blocks):
                cv2.line(img, (x0 + i, y0 + 65), (x0 + i, y0 + 65 - int(60 * v)), (140, 110, 90), 1)
                cv2.line(img, (x0 + i, y0 + 65), (x0 + i, y0 + 65 + int(60 * v)), (140, 110, 90), 1)
            res = result[name][which]
            for t, rise, lev in res["others"]:
                if 0 <= t - t_off < window:
                    X = x0 + int((t - t_off) / window * colw)
                    cv2.line(img, (x0 + int((t - t_off) / window * colw), y0), (X, y0 + 130), (170, 170, 170), 1)
            for t, rise, lev in res["claps"] or []:
                X = x0 + int((t - t_off) / window * colw)
                cv2.line(img, (X, y0), (X, y0 + 130), (40, 40, 230), 2)
            lab = f"{name}  {which}: " + ("claps at " + ", ".join(f"{t:.2f}" for t, _, _ in res["claps"]) + " s" if res["claps"] else "NO CLAP PATTERN FOUND")
            cv2.putText(img, lab, (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20) if res["claps"] else (0, 0, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"Session {session}: first and last {window:.0f} s of each recording. red = chosen claps, grey = other candidates",
                (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    ap.add_argument("--window", type=float, default=150.0)
    args = ap.parse_args()
    rec = preset.path("output", "recordings")
    files = {f"Phone {p}": glob.glob(os.path.join(rec, f"Session{args.session} Phone{p}.m4a"))[0] for p in "ABC"}
    cams = glob.glob(os.path.join(rec, f"session{args.session}_*.ts"))
    if cams:
        files["Camera"] = cams[0]
    devices, result = [], {}
    for name, path in files.items():
        sr = 16000 if name == "Camera" else 48000
        x = load_audio(path, sr)
        devices.append((name, x, sr))
        result[name] = {"duration_s": len(x) / sr, "sr": sr}
        for which in ("start", "end"):
            n = int(args.window * sr)
            seg = x[:n] if which == "start" else x[-n:]
            off = 0.0 if which == "start" else (len(x) - len(seg)) / sr
            cand = [(t + off, r, l) for t, r, l in candidates(seg, sr)]
            tri = pick_triple(cand)
            others = [c for c in cand if not tri or c not in tri]
            result[name][which] = {"claps": sorted(tri) if tri else None, "others": others[:8]}
        if name == "Camera":
            result[name]["audio_start_minus_video_start_s"] = stream_start(path, "a") - stream_start(path, "v")
    print(f"Session {args.session}")
    for name in files:
        d = result[name]
        row = []
        for which in ("start", "end"):
            cl = d[which]["claps"]
            row.append(f"{which}: " + (", ".join(f"{t:8.3f}" for t, _, _ in cl) + "  (gaps " + " / ".join(f"{b[0] - a[0]:.3f}" for a, b in zip(cl, cl[1:])) + " s)" if cl else "not found"))
        print(f"  {name:8s} {d['duration_s']:7.1f} s   " + "    ".join(row))
    os.makedirs(preset.path("output", "sync"), exist_ok=True)
    out = preset.path("output", "sync", f"session{args.session}_claps.jpg")
    figure(args.session, devices, result, args.window, out)
    json.dump(result, open(preset.path("output", "sync", f"session{args.session}_claps.json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
