# Ad hoc scripts

One-off scripts written for a single task during the feasibility test. They are kept as a record of
what was done and how, not as maintained tools. Each file's docstring says what it did.

| Script | What it did |
|---|---|
| `annotate_sample.py` | marked landmarks and sticker spots on a camera frame so the person knew what to measure |
| `rug_corners.py` | zoomed rug view marking the block corners P1-P8 to measure |
| `fit_floor.py` | first camera + floor fit from the rug (no lens distortion); showed the camera height is not determined by the rug |
| `extrapolate_grid.py` | first extension of the floor grid to the whole room, refit at several camera heights |
| `make_figures.py` | regenerated the README calibration figures with people pixelated (rerunnable) |
| `make_live_figure.py` | composed the README live-tracker figure from a running server (rerunnable) |

The first four read the project's old layout (`data/`, `measurements/`) and do not run as they are.
The calibration they led to lives in `../calib.py`.
