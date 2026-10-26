# Preset template

A preset holds everything specific to one room and camera. To set up a new room:

1. `cp -r presets/_template presets/NAME`, edit `camera.env` with the RTSP URL.
2. `tools/snap_frame.sh --preset NAME` saves a sample frame into `frames/`.
3. Fill `preset.json`: the frame size, the walkable area (pixel polygon), fixed landmarks (pixels of
   distinctive floor points, e.g. rug corners), the floor frame, and any rectangle constraints
   (pairs of edges known to be perpendicular).
4. Put floor stickers down, record their pixel positions in `measurements/stickers_pixels.csv`
   (`name,px,py,confidence` with confidence high/medium/low) and tape-measured distances in the
   `*_distances.csv` files (`a,b,inches`). Sticker names start with `S`; fixed landmarks must not.
5. `python tools/calib.py --preset NAME report`, then `tools/calib_export.py --preset NAME`.

Files in this directory:

| Path | What | Committed |
|---|---|---|
| `preset.json` | landmarks, constraints, floor frame, noise levels | yes |
| `measurements/` | pixel positions and tape measurements | yes |
| `camera.env` | camera URL and credentials | no (gitignored) |
| `frames/` | captured camera frames (they show the room and people) | no |
| `output/` | calibration, grid images, tracking logs | no |
| `legacy/` | superseded scripts kept for reference | yes |
