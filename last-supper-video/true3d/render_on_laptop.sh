#!/usr/bin/env bash
# One-shot GPU render of the 3D sample on your own machine.
# Needs: Blender 4.2+ (set BLENDER=/path/to/blender if it isn't on PATH), curl, ffmpeg.
# usage: ./render_on_laptop.sh            -> renders frames/ at 1080x1920 on your GPU
set -euo pipefail
cd "$(dirname "$0")"
B=${BLENDER:-blender}
./fetch_assets.sh
"$B" -b -P build_set.py -- assets set.blend
"$B" -b -P sample_shot.py -- set.blend frames
ffmpeg -y -framerate 24 -i frames/f%04d.png -c:v libx264 -crf 14 -pix_fmt yuv420p flight_raw.mp4
echo "done: frames/ and flight_raw.mp4 — send me the frames (or the mp4) for the grade + painting dissolve + score"
