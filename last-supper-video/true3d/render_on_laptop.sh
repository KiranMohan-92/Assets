#!/usr/bin/env bash
# One command, start to finish, on your own machine's GPU:
# download the 3D figures -> build the set -> render the 7 s flight -> painting dissolve,
# grade, score -> sample_3d.mp4
# Needs: Blender 4.2+ (set BLENDER=/path/to/blender if it isn't on PATH), curl, ffmpeg,
#        python3 with numpy, scipy and pillow (pip install numpy scipy pillow).
set -euo pipefail
cd "$(dirname "$0")"
B=${BLENDER:-blender}
./fetch_assets.sh
[ -f set.blend ] || "$B" -b -P build_set.py -- assets set.blend
"$B" -b -P sample_shot.py -- set.blend frames          # resumable: re-run to continue
python3 make_painting_crop.py ../source/last_supper.jpg painting_crop.png
[ -f score.wav ] || python3 ../ai_film/epic_score.py score.wav
./finish_sample.sh .
echo "done -> $(pwd)/sample_3d.mp4"
