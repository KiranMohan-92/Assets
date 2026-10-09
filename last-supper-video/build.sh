#!/usr/bin/env bash
# One-shot build: painting image -> 3D layers -> soundtrack -> rendered MP4.
# usage: ./build.sh path/to/last_supper.jpg [out.mp4] [width height]
set -euo pipefail
cd "$(dirname "$0")"
IMG=${1:-source/last_supper.jpg}
OUT=${2:-last_supper.mp4}
W=${3:-1920}
H=${4:-1080}
mkdir -p build
[ -d node_modules/three ] || npm i --silent
python3 enhance.py "$IMG" build/enhanced.jpg
python3 prep.py build/enhanced.jpg calib.json build ${DEPTH:-source/last_supper_depth16.png}
python3 music.py build/score.wav
node render.mjs --assets build --w "$W" --h "$H" --fps 30 --audio build/score.wav --out "$OUT"
