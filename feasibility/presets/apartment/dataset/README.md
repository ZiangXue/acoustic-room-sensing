# Synced dataset: three phones + position, 3 sessions

One folder per session: `session1/`, `session2/`, `session3/`. Each has

| File | What |
|---|---|
| `phone_A.m4a`, `phone_B.m4a`, `phone_C.m4a` | the original recordings, as they came off the phones (ALAC, lossless) |
| `clock_model.json`, `claps.json`, `visual_claps.json` | clock offsets between the phones, and the clap detections they came from |
| `audio_abc.wav` | **not committed, rebuild it** (below). 3 channels, 48 kHz, 16 bit: channel 0 = phone A, 1 = phone B, 2 = phone C, clock-aligned |
| `positions.csv` | the person's floor position, 10 rows per second, on the same clock as the audio |
| `meta.json` | phone positions, clock model and accuracy, clap times, coverage, caveats |

## Time
`t_audio_s` in `positions.csv` is seconds from the first audio sample (sample index / 48000). So the
position at audio sample `n` is the row with `t_audio_s` nearest to `n / 48000`. `t_video_s` is the same
moment on the camera video's clock.

## Space
Floor coordinates in the frame of the rug: origin at corner R2, x along R1 to R2 (toward the right wall),
y perpendicular (toward the desk). Columns `x_in, y_in` (inches) and `x_m, y_m` (meters, smoothed with a
zero-lag filter); `x_raw_m, y_raw_m` are unsmoothed. The point is the midpoint of the two feet.
`status`: `tracked` (detected), `interp` (short gap filled), `lost` (no position, columns empty).
Phone positions (measured) are in `meta.json`: A (-82, -23.5), B (68.4, -26.5), C (-33.8, 63.8) inches.

## What the audio alignment means
Phone A is exactly as recorded. B and C are shifted by their clock offset (found from the three start
claps, with the sound's travel time from the clapper to each phone removed), so a sound reaches the three
channels at the delays the room produces (distance / 343 m/s), and not at delays caused by the phones' clocks.
The claps themselves are in the first seconds (`claps_t_audio_s`); `usable_from_s` is a few seconds after the last one.

## How far to trust it
* Phone to phone, at the claps: about +-0.5 ms (1 ms is 34 cm of path difference).
* **Drift is not corrected, because it could not be measured.** There are no claps at the end of any
  session, and speech and mains hum turned out too noisy to measure each phone's clock rate. A phone
  that runs 5 ppm fast or slow is 2.7 ms (0.9 m) off after 9 minutes. The start of each session is the
  most reliable part; the error grows with the time since the claps. `meta.json` lists the sources.
* Audio to position: about +-40 ms (one video frame is 50 ms), which at walking speed is under 1 cm.
* Position: calibrated to about 1.6 cm on known distances; the real error is larger near the frame edges
  and where the feet are partly hidden. Only part of the room is covered (see the project README).

## Get the WAV
`python tools/make_wav.py presets/apartment/dataset/session1 ...` (from `feasibility/`, venv + ffmpeg) writes
`audio_abc.wav` from the original recordings and `clock_model.json`. About 150 MB per session.

## Load it
```python
import numpy as np, pandas as pd
from scipy.io import wavfile
sr, audio = wavfile.read("session1/audio_abc.wav")        # (n_samples, 3) int16
pos = pd.read_csv("session1/positions.csv")
t = np.arange(len(audio)) / sr
x = np.interp(t, pos.t_audio_s, pos.x_m); y = np.interp(t, pos.t_audio_s, pos.y_m)   # NaN where lost
```

Rebuild with `python tools/build_dataset.py N` (from `feasibility/`) after `tools/sync_session.py N`.
The WAVs are not committed (large; rebuilt from the recordings). The recordings contain the person's voice.
