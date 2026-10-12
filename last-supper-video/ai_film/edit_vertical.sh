#!/usr/bin/env bash
# "Into the Painting" — vertical 9:16 reel, one continuous camera flight.
# Kling segments are chained frame-to-frame (each one's end keyframe is the next
# one's start keyframe), so stitched with a short blend they read as a single
# unbroken shot: hall -> through the fresco -> wine -> money pouch -> Judas ->
# Thomas -> land on Christ's face -> slow pull-back.  The figures never move.
# usage: edit_vertical.sh DIR   (DIR holds v1..v5.mp4, vz.mp4, score.wav)
set -euo pipefail
D=${1:?dir}
cd "$D"
FPS=24
W=1080; H=1920
X=0.25          # blend at each seam (s): hides the micro-jump between segments
norm="scale=$W:$H:force_original_aspect_ratio=increase,crop=$W:$H,fps=$FPS,setsar=1,format=yuv420p"

# flight: segments v1..v5, trimmed of Kling's hesitant first/last frames
SEGS=(v1.mp4 v2.mp4 v3.mp4 v4.mp4 v5.mp4 vz.mp4)
for i in "${!SEGS[@]}"; do
  ffmpeg -y -loglevel error -i "${SEGS[$i]}" -vf "$norm" -an -c:v libx264 -crf 12 -preset fast "n$i.mp4"
done

# chain with short crossfades
python3 - "$X" "${#SEGS[@]}" <<'PY'
import subprocess, sys
x = float(sys.argv[1]); n = int(sys.argv[2])
dur = [float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'csv=p=0', f'n{i}.mp4']).decode()) for i in range(n)]
inputs = sum((['-i', f'n{i}.mp4'] for i in range(n)), [])
fc, prev, off = [], '[0:v]', 0.0
for i in range(1, n):
    off += dur[i - 1] - x
    fc.append(f'{prev}[{i}:v]xfade=transition=fade:duration={x}:offset={off:.3f}[x{i}]')
    prev = f'[x{i}]'
subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', *inputs, '-filter_complex', ';'.join(fc),
                       '-map', prev, '-c:v', 'libx264', '-crf', '12', '-preset', 'fast', 'flight.mp4'])
open('seams.txt', 'w').write(' '.join(f'{sum(dur[:i]) - i * x:.2f}' for i in range(1, n)) + '\n')
PY
LEN=$(ffprobe -v error -show_entries format=duration -of csv=p=0 flight.mp4)
LAND=$(awk '{print $5}' seams.txt)      # moment the camera lands on Christ's face
echo "flight ${LEN}s, landing at ${LAND}s"

# score: the 40 s "Hour Before" cue re-cut so its drop hits the landing on Christ
python3 - "$LAND" "$LEN" <<'PY'
import sys, numpy as np
from scipy.io import wavfile
land, total = float(sys.argv[1]), float(sys.argv[2])
sr, a = wavfile.read('score.wav'); a = a.astype(np.float32)
S = lambda t0, t1: a[int(t0 * sr):int(t1 * sr)]
def xcat(parts, xf=0.6):
    out = parts[0]
    for p in parts[1:]:
        k = int(xf * sr); r = np.linspace(0, 1, k)[:, None]
        out = np.concatenate([out[:-k], out[-k:] * (1 - r) + p[:k] * r, p[k:]])
    return out
# pre-drop: piano/strings/choir intro, then the ostinato + Judas hit + taiko build
build = S(13.4, 24.0)                       # ends on the breath before the drop
intro_len = land - len(build) / sr + 0.6
intro = S(0.0, max(intro_len, 3.0))
pre = xcat([intro, build])
pre = pre[-int(land * sr):] if len(pre) > land * sr else np.concatenate([np.zeros((int(land * sr) - len(pre), a.shape[1]), np.float32), pre])
post = xcat([S(24.0, 28.6), S(33.5, 40.0)], 0.8)  # Christ drop -> D-major resolution
out = np.concatenate([pre, post])[:int(total * sr)]
n = int(2.5 * sr); out[-n:] *= np.linspace(1, 0, n)[:, None]
wavfile.write('score_v.wav', sr, out.astype(np.int16) if a.dtype != np.float32 else out)
PY

# title (vertical layout, lower third)
python3 - <<'PY'
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import glob
W, H = 1080, 1920
def font(names, size):
    for n in names:
        for p in glob.glob(f'/usr/share/fonts/**/{n}', recursive=True):
            try: return ImageFont.truetype(p, size)
            except Exception: pass
    return ImageFont.load_default()
serif = font(['DejaVuSerif.ttf', 'LiberationSerif-Regular.ttf'], 74)
sans = font(['Montserrat-Medium.ttf', 'DejaVuSans.ttf'], 24)
img = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
def spaced(text, f, sp, y, alpha):
    w = sum(d.textlength(c, font=f) for c in text) + sp * (len(text) - 1); x = (W - w) / 2
    for c in text:
        d.text((x, y), c, font=f, fill=(244, 234, 214, alpha)); x += d.textlength(c, font=f) + sp
y = H * 0.74
spaced('THE HOUR', serif, 12, y, 255); spaced('BEFORE', serif, 12, y + 92, 255)
d.rectangle([(W - 120) / 2, y + 205, (W + 120) / 2, y + 206], fill=(244, 234, 214, 180))
spaced('AFTER LEONARDO DA VINCI', sans, 8, y + 228, 215)
spaced('THE LAST SUPPER  ·  1498', sans, 8, y + 264, 215)
glow = img.split()[3].filter(ImageFilter.GaussianBlur(10))
sh = Image.new('RGBA', (W, H), (0, 0, 0, 0)); sh.putalpha(glow.point(lambda a: int(a * 0.8)))
Image.alpha_composite(sh, img).save('title_v.png')
PY

TIN=$(python3 -c "print(round($LEN-6.0,2))"); TOUT=$(python3 -c "print(round($LEN-1.6,2))")
FOUT=$(python3 -c "print(round($LEN-1.4,2))")
ffmpeg -y -loglevel error -i flight.mp4 -loop 1 -i title_v.png -i score_v.wav -filter_complex "\
[0:v]eq=contrast=1.06:saturation=1.10:gamma=0.98,\
colorbalance=rs=-0.03:bs=0.04:rh=0.04:gh=0.01:bh=-0.04,\
vignette=angle=PI/5,noise=alls=6:allf=t,fade=t=in:st=0:d=1.0[v];\
[1:v]format=rgba,fade=t=in:st=$TIN:d=1.4:alpha=1,fade=t=out:st=$TOUT:d=1.2:alpha=1[t];\
[v][t]overlay=0:0:shortest=1,fade=t=out:st=$FOUT:d=1.4[vo]" \
  -map "[vo]" -map 2:a -t "$LEN" -c:v libx264 -b:v 12M -maxrate 16M -bufsize 24M -preset slow \
  -pix_fmt yuv420p -r $FPS -c:a aac -b:a 256k -movflags +faststart into_the_painting_9x16.mp4
ffprobe -v error -show_entries format=duration,size -of compact into_the_painting_9x16.mp4
