#!/usr/bin/env bash
# Finish the 3D sample: the real painting dissolves into the 3D set, then the rendered
# flight; cinematic grade (bloom, warm haze, vignette, grain), 1080x1920, score cut so its
# drop lands as Christ is revealed.
# usage: finish_sample.sh DIR   (DIR has frames/f####.png, painting_crop.png, score.wav)
set -euo pipefail
cd "${1:?dir}"
FPS=24
HOLD=1.2          # seconds on the flat painting
XF=1.0            # dissolve painting -> 3D

ffmpeg -y -loglevel error -framerate $FPS -i frames/f%04d.png \
  -vf "scale=1080:1920:flags=lanczos,format=yuv420p" -c:v libx264 -crf 12 -preset fast flight.mp4
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 flight.mp4)

# the painting, with a slow push so the motion carries straight into the 3D move
ffmpeg -y -loglevel error -loop 1 -framerate $FPS -i painting_crop.png -t $(python3 -c "print($HOLD+$XF)") \
  -vf "scale=2160:3840,zoompan=z='1+0.0009*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps=$FPS,format=yuv420p" \
  -c:v libx264 -crf 12 -preset fast paint.mp4

ffmpeg -y -loglevel error -i paint.mp4 -i flight.mp4 -filter_complex \
  "[0:v][1:v]xfade=transition=fade:duration=$XF:offset=$HOLD,format=yuv420p[v]" -map "[v]" -c:v libx264 -crf 12 -preset fast joined.mp4
TOTAL=$(ffprobe -v error -show_entries format=duration -of csv=p=0 joined.mp4)

# score: start so the 24.0 s drop hits ~0.6 s before the end of the move (Christ found)
START=$(python3 -c "print(round(24.0-($TOTAL-1.6),2))")
FO=$(python3 -c "print(round($TOTAL-1.4,2))")
ffmpeg -y -loglevel error -ss $START -t $TOTAL -i score.wav -af "afade=t=in:d=0.4,afade=t=out:st=$FO:d=1.4" score_cut.wav

# grade: warm bloom + haze lift, contrast, vignette, grain
ffmpeg -y -loglevel error -i joined.mp4 -i score_cut.wav -filter_complex "\
[0:v]split[a][b];\
[b]gblur=sigma=28,eq=brightness=0.02:saturation=1.2,colorbalance=rs=0.10:gs=0.04:bs=-0.06[glow];\
[a][glow]blend=all_mode=screen:all_opacity=0.28,\
eq=contrast=1.10:saturation=1.06:gamma=0.97,\
colorbalance=rs=0.02:bs=0.03:rh=0.05:gh=0.02:bh=-0.05,\
vignette=angle=PI/4.2,noise=alls=6:allf=t,\
fade=t=in:st=0:d=0.6,fade=t=out:st=$FO:d=1.4,format=yuv420p[v]" \
  -map "[v]" -map 1:a -c:v libx264 -b:v 14M -maxrate 18M -bufsize 28M -preset slow -r $FPS \
  -c:a aac -b:a 256k -movflags +faststart sample_3d.mp4
ffprobe -v error -show_entries format=duration,size -of compact sample_3d.mp4
