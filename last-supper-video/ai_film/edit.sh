#!/usr/bin/env bash
# Assemble "The Hour Before" from the eight AI shots + the original score.
# usage: edit.sh DIR   (DIR holds s0.mp4..s7.mp4 and score.wav; writes DIR/the_hour_before.mp4)
set -euo pipefail
D=${1:?dir}
cd "$D"
FPS=24
# shot: file  in-point  length  (cut points land on the score's cues)
SHOTS=(
  "s0.mp4 0.0 4.6"     # 0.0  wine macro
  "s1.mp4 0.0 4.6"     # 4.6  through the window
  "s2.mp4 0.3 4.2"     # 9.2  high crane
  "s3.mp4 0.8 3.4"     # 13.4 hands / pouch
  "s4.mp4 0.8 3.4"     # 16.8 Judas (brass hit)
  "s5.mp4 0.6 3.3"     # 20.2 Thomas / James (taiko build)
)
i=0
for s in "${SHOTS[@]}"; do
  set -- $s
  ffmpeg -y -loglevel error -ss "$2" -t "$3" -i "$1" -vf "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=$FPS,setsar=1" -an -c:v libx264 -crf 14 -preset fast "p$i.mp4"
  i=$((i+1))
done
# 23.5 - 24.0 breath of black
ffmpeg -y -loglevel error -f lavfi -i "color=black:s=1920x1080:r=$FPS:d=0.5" -c:v libx264 -crf 14 "p$i.mp4"; i=$((i+1))
# 24.0 Christ lifts his eyes (drop)
ffmpeg -y -loglevel error -ss 0.2 -t 4.6 -i s6.mp4 -vf "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=$FPS,setsar=1" -an -c:v libx264 -crf 14 -preset fast "p$i.mp4"; i=$((i+1))
# 28.6 crane pull-back, eased slower, then hold on the fresco to the end (40 s)
ffmpeg -y -loglevel error -i s7.mp4 -vf "setpts=1.15*PTS,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=$FPS,setsar=1,tpad=stop_mode=clone:stop_duration=3,trim=duration=11.4" -an -c:v libx264 -crf 14 -preset fast "p$i.mp4"; i=$((i+1))
: > list.txt
for k in $(seq 0 $((i-1))); do echo "file 'p$k.mp4'" >> list.txt; done
ffmpeg -y -loglevel error -f concat -safe 0 -i list.txt -c copy cut.mp4

# title card (transparent PNG)
python3 - <<'PY'
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import glob
W, H = 1920, 1080
def font(names, size):
    for n in names:
        for p in glob.glob(f'/usr/share/fonts/**/{n}', recursive=True) + glob.glob(f'/home/user/**/{n}', recursive=True):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()
serif = font(['DejaVuSerif.ttf', 'LiberationSerif-Regular.ttf'], 92)
sans = font(['Montserrat-Medium.ttf', 'Metropolis-Medium.otf', 'Montserrat-Regular.ttf', 'DejaVuSans.ttf'], 26)
img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
def spaced(text, f, sp, y, alpha):
    w = sum(d.textlength(c, font=f) for c in text) + sp * (len(text) - 1)
    x = (W - w) / 2
    for c in text:
        d.text((x, y), c, font=f, fill=(244, 234, 214, alpha))
        x += d.textlength(c, font=f) + sp
spaced('THE HOUR BEFORE', serif, 14, H * 0.40, 255)
d.rectangle([(W - 140) / 2, H * 0.40 + 122, (W + 140) / 2, H * 0.40 + 123], fill=(244, 234, 214, 180))
spaced('AFTER LEONARDO DA VINCI  ·  THE LAST SUPPER  ·  1498', sans, 9, H * 0.40 + 150, 215)
glow = img.split()[3].filter(ImageFilter.GaussianBlur(10))
shadow = Image.new('RGBA', (W, H), (0, 0, 0, 0)); shadow.putalpha(glow.point(lambda a: int(a * 0.75)))
Image.alpha_composite(shadow, img).save('title.png')
PY

# grade + letterbox + grain + title + score
ffmpeg -y -loglevel error -i cut.mp4 -loop 1 -i title.png -i score.wav -filter_complex "\
[0:v]eq=contrast=1.07:saturation=1.10:gamma=0.98,\
colorbalance=rs=-0.03:gs=0.0:bs=0.04:rh=0.04:gh=0.01:bh=-0.04,\
vignette=angle=PI/5,noise=alls=7:allf=t,\
drawbox=x=0:y=0:w=iw:h=78:color=black:t=fill,drawbox=x=0:y=ih-78:w=iw:h=78:color=black:t=fill,\
fade=t=in:st=0:d=1.2[v];\
[1:v]format=rgba,fade=t=in:st=33.6:d=1.4:alpha=1,fade=t=out:st=38.4:d=1.4:alpha=1[t];\
[v][t]overlay=0:0:shortest=1,fade=t=out:st=38.6:d=1.4[vo]" \
  -map "[vo]" -map 2:a -t 40 -c:v libx264 -crf 17 -preset slow -pix_fmt yuv420p -r $FPS -c:a aac -b:a 256k -movflags +faststart the_hour_before.mp4
ffprobe -v error -show_entries format=duration,size -of compact the_hour_before.mp4
