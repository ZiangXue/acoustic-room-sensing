# Feasibility test: floor localization

Plan and scope: [feasibility_test.md](feasibility_test.md).

## Layout

| Path | What |
|---|---|
| `tools/` | generic code, no room-specific numbers |
| `presets/<name>/` | everything for one room and camera: config, measurements, frames, outputs (see `presets/_template/README.md`) |
| `presets/apartment/` | the living room setup used for the experiments |
| `models/` | YOLO pose weights (downloaded on first use, not committed) |

## Commands

Run from this directory with the venv's Python (`/Users/ziangxue/.venvs/chatter-feas/bin/python`).
Every tool takes `--preset NAME` (default `apartment`, or set `CHATTER_PRESET`).

```
tools/snap_frame.sh [--preset NAME] [file.jpg]        capture one frame
python tools/calib.py report                          calibration accuracy (before, after, leave-one-out)
python tools/calib.py grid                            draw the calibrated grid on the sample frame
python tools/calib_export.py                          write output/calibration.json for the tracker
python tools/track_server.py [--log auto]             live tracker on http://<this Mac's IP>:9090
```
