"""Vertical crop of the fresco that lines up with the first frame of the 3D flight
(John's and Christ's heads, the table edge and the windows land in the same places),
so the painting can dissolve straight into the 3D set.
usage: python3 make_painting_crop.py ../source/last_supper.jpg painting_crop.png"""
import sys
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGB'); W, H = im.size
# heads in the fresco (normalised, see blender/apostles.py) and in the 3D frame 1
JOHN_P, CHRIST_P = (0.365, 0.516), (0.511, 0.480)
JOHN_R, CHRIST_R = (0.328, 0.453), (0.810, 0.460)
cw = (CHRIST_P[0] - JOHN_P[0]) * W / (CHRIST_R[0] - JOHN_R[0])
ch = cw * 16 / 9
left = JOHN_P[0] * W - JOHN_R[0] * cw
top = JOHN_P[1] * H - JOHN_R[1] * ch
canvas = Image.new('RGB', (round(cw), round(ch)), (28, 22, 18))   # the wall below the fresco
canvas.paste(im.crop((round(left), round(top), round(left + cw), round(min(top + ch, H)))), (0, 0))
canvas.resize((1080, 1920), Image.LANCZOS).save(out)
print('crop', round(left), round(top), round(cw), round(ch))
