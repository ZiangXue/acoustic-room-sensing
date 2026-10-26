"""Locate and load the active preset: everything specific to one room and camera.

The preset is chosen with `--preset NAME` on any tool's command line (the flag is removed from
sys.argv here, so other argument parsing is unaffected), else the CHATTER_PRESET environment
variable, else "apartment". It lives in presets/NAME/ (see presets/_template/README.md).
"""
import json
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)


def _pick():
    if "--preset" in sys.argv:
        i = sys.argv.index("--preset")
        name = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return name
    return os.environ.get("CHATTER_PRESET", "apartment")


NAME = _pick()
DIR = os.path.join(ROOT, "presets", NAME)
if not os.path.isfile(os.path.join(DIR, "preset.json")):
    sys.exit(f"preset '{NAME}' not found: expected {DIR}/preset.json (copy presets/_template)")
CFG = json.load(open(os.path.join(DIR, "preset.json")))


def path(*parts):
    """A path inside the preset directory."""
    return os.path.join(DIR, *parts)


def out(*parts):
    """A path inside the preset's output directory (created if needed)."""
    p = path("output", *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def camera_env():
    """Variables from the preset's camera env file (RTSP_URL=...)."""
    env = {}
    for line in open(path(CFG["camera"]["env_file"])):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            env[k] = v
    return env
