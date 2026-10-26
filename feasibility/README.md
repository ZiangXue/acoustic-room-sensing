# Feasibility test: floor localization

Plan and scope: [feasibility_test.md](feasibility_test.md).

## Layout

| Path | What |
|---|---|
| `tools/` | generic, reusable code, no room-specific numbers |
| `tools/adhoc/` | one-off scripts from specific tasks, kept as a record (each docstring says what it did) |
| `presets/<name>/` | everything for one room and camera: config, measurements, frames, outputs, dataset (see `presets/_template/README.md`) |
| `presets/apartment/dataset/` | the synced audio + position dataset of the three sessions (see its README) |
| `docs/img/` | figures used in the top-level README (people blurred) |
| `models/` | YOLO pose weights (downloaded on first use, not committed) |
| `trash/` | files no longer useful, waiting for review (see `trash/README.txt`) |

## Workflow and commands

Run from this directory with the venv's Python (`/Users/ziangxue/.venvs/chatter-feas/bin/python`).
Every tool takes `--preset NAME` (default `apartment`, or set `CHATTER_PRESET`).

```
# set up a room
tools/snap_frame.sh [--preset NAME] [file.jpg]        capture one frame
python tools/calib.py report | grid                   calibration accuracy / calibrated grid on the sample frame
python tools/calib_export.py                          write output/calibration.json for the tracker

# plan and record a session
python tools/session_plan.py                          stops, walking loop and timed script
python tools/next_session.py --out X.pdf TRACK.csv..  PDF of where the next session should go
python tools/track_server.py --no-track               Record tab only, no model: use this during sessions
python tools/track_server.py [--log auto] [--blur]    live tracker + Record tab on http://<this Mac's IP>:9090

# process a session N
python tools/track_video.py RECORDING.ts              recording -> positions (about 5 min per 9 min of video)
python tools/coverage.py TRACK.csv [TRACK.csv ...]    how much of the floor the sessions cover
python tools/find_claps.py N                          3-clap pattern at the start/end of each phone and camera audio
python tools/find_visual_claps.py N --from S --to S   the same claps in the video, from the hands (frame numbers)
python tools/sync_session.py N [--end-xy X Y]         clock offsets (and drift, if there are end claps)
python tools/build_dataset.py N                       synced 3-channel audio + positions in presets/<name>/dataset/
```
