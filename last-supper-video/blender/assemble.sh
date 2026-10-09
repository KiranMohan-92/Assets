#!/usr/bin/env bash
# Turn rendered Blender frames into the final video.
# usage: assemble.sh WORK_DIR FRAMES_DIR OUT.mp4 [SHARE.mp4]
set -euo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "usage: $0 WORK_DIR FRAMES_DIR OUT.mp4 [SHARE.mp4]" >&2
  exit 1
fi
WORK="$1"; FRAMES="$2"; OUT="$3"; SHARE="${4:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
POST="$FRAMES/post"
FPS=24

echo "[1/4] post-processing frames -> $POST"
python3 "$HERE/post.py" "$FRAMES" "$FRAMES/timeline.json" "$POST"

echo "[2/4] soundtrack"
if [ ! -f "$WORK/score.wav" ]; then
  echo "      $WORK/score.wav missing, synthesizing with music.py"
  python3 "$HERE/../music.py" "$WORK/score.wav"
else
  echo "      using $WORK/score.wav"
fi

echo "[3/4] encoding $OUT (1920x1080, crf 18)"
ffmpeg -y -hide_banner -loglevel warning -stats \
  -framerate "$FPS" -i "$POST/f_%04d.jpg" -i "$WORK/score.wav" \
  -vf "scale=1920:1080:flags=lanczos" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 256k -shortest -movflags +faststart \
  "$OUT"

if [ -n "$SHARE" ]; then
  echo "[4/4] two-pass share copy $SHARE (6400k video, 192k audio)"
  LOG="$FRAMES/share_passlog"
  VF="scale=1920:1080:flags=lanczos"
  ffmpeg -y -hide_banner -loglevel warning -stats \
    -framerate "$FPS" -i "$POST/f_%04d.jpg" \
    -vf "$VF" -c:v libx264 -preset slow -b:v 6400k -pix_fmt yuv420p \
    -pass 1 -passlogfile "$LOG" -an -f mp4 /dev/null
  ffmpeg -y -hide_banner -loglevel warning -stats \
    -framerate "$FPS" -i "$POST/f_%04d.jpg" -i "$WORK/score.wav" \
    -vf "$VF" -c:v libx264 -preset slow -b:v 6400k -pix_fmt yuv420p \
    -pass 2 -passlogfile "$LOG" \
    -c:a aac -b:a 192k -shortest -movflags +faststart "$SHARE"
  rm -f "$LOG"*.log "$LOG"*.mbtree
else
  echo "[4/4] no share copy requested"
fi
echo "done: $OUT${SHARE:+ $SHARE}"
