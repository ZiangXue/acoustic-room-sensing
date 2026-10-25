"""Fit the calibration (tools/calib.py) and save what the tracking server needs to the preset's
output/calibration.json.

Usage: calib_export.py [--preset NAME]
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import calib
import preset

pix, sig = calib.load_points()
names = calib.all_names(pix)
dists = calib.load_dists()
b = calib.Fit(names, pix, sig, dists)
o, u, v = b.frame()
errs = [b.dist(x, y) - d / calib.IN for x, y, d in dists]
xy = b.to_xy(np.array(b.pix))
out = {
    "params": [float(q) for q in b.x[:5]],  # f, pitch, roll, camera height (m), k1
    "o": o.tolist(), "u": u.tolist(), "v": v.tolist(),
    "image": [calib.W, calib.H_IMG],
    "landmarks": {n: [float(p[0]), float(p[1])] for n, p in zip(b.names, xy)},
    "area_px": calib.AREA.tolist(),
    "area_xy": b.to_xy(calib.AREA.astype(float)).tolist(),
    "fit_rms_in": float(np.sqrt(np.mean(np.square(errs)))),
    "ankle_height_m": preset.CFG["tracking"]["ankle_height_m"],
    "preset": preset.NAME,
}
path = preset.out("calibration.json")
json.dump(out, open(path, "w"), indent=1)
print(f"wrote {path}  (distance rms {out['fit_rms_in']:.2f} in, {len(dists)} distances, {len(names)} landmarks)")
