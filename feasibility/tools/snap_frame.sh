#!/bin/bash
# Grab one frame from a preset's camera (RTSP) and print the stream info.
# Usage: tools/snap_frame.sh [--preset NAME] [out.jpg]
#   a bare file name goes to presets/NAME/frames/, default sample_frame.jpg
set -euo pipefail
cd "$(dirname "$0")/.."
PRESET="${CHATTER_PRESET:-apartment}"
if [ "${1:-}" = "--preset" ]; then PRESET="$2"; shift 2; fi
DIR="presets/$PRESET"
[ -f "$DIR/preset.json" ] || { echo "preset '$PRESET' not found in presets/" >&2; exit 1; }
ENVFILE="$DIR/$(python3 -c "import json,sys; print(json.load(open('$DIR/preset.json'))['camera']['env_file'])")"
source "$ENVFILE"

OUT="${1:-sample_frame.jpg}"
case "$OUT" in */*) ;; *) OUT="$DIR/frames/$OUT" ;; esac
mkdir -p "$(dirname "$OUT")"

ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate \
  -of default=nw=1 "$RTSP_URL" || true

ffmpeg -y -v error -rtsp_transport tcp -i "$RTSP_URL" -frames:v 1 -q:v 1 "$OUT"
echo "saved $OUT"
