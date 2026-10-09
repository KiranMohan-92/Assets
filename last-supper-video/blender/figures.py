"""Split the painted group into 13 individual apostles and give each a metric depth.

usage: python3 figures.py ../source out_dir

Writes out_dir/figures.npz with, on a GW x GH grid over the painting:
  labels   int16   0 = none, 1..13 = apostle index (order of APOSTLES)
  hidden   uint8   1 where an apostle's extended silhouette is covered by a nearer one
  zfront   float32 metric z-distance of the visible front surface
  plus a debug image figures_debug.jpg
and out_dir/palette_XX.jpg: each apostle's colours with everything else inpainted,
heavily blurred (used for the sides and back that the painting never shows).
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy import ndimage

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
G = json.load(open(os.path.join(os.path.dirname(__file__), 'geometry.json')))

# head centres in normalized image coords, read off the painting
APOSTLES = [
    ('bartholomew', 0.157, 0.483), ('jamesMinor', 0.190, 0.480), ('andrew', 0.231, 0.490),
    ('judas', 0.305, 0.530), ('peter', 0.328, 0.508), ('john', 0.365, 0.516),
    ('christ', 0.511, 0.480),
    ('thomas', 0.589, 0.473), ('jamesMajor', 0.609, 0.497), ('philip', 0.646, 0.460),
    ('matthew', 0.797, 0.477), ('thaddeus', 0.862, 0.483), ('simon', 0.903, 0.493),
]
TABLE_EDGE_V = 0.688   # front edge of the table top in the image
FIG_BOTTOM_V = 0.664   # figures stop here; the tabletop band below holds plates and bread

img = cv2.imread(f'{SRC}/last_supper_enhanced.jpg')
H0, W0 = img.shape[:2]
GW = 1600
GH = int(round(GW * H0 / W0))
im = cv2.resize(img, (GW, GH), interpolation=cv2.INTER_AREA)
fg = cv2.resize(cv2.imread(f'{SRC}/last_supper_fgmask.png', 0), (GW, GH), interpolation=cv2.INTER_AREA) > 110
disp = cv2.resize(cv2.imread(f'{SRC}/last_supper_depth16.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535, (GW, GH), interpolation=cv2.INTER_AREA)
vv = (np.arange(GH)[:, None] + 0.5) / GH * np.ones((1, GW))

# ---- metric depth from the depth model --------------------------------------
# Depth Anything gives affine-invariant disparity: fit 1/z = a*d + b on two
# surfaces of known depth: the table's front edge and the back wall.
Zt, Zb = G['table']['Zt'], -G['room']['zb']
row = int(0.700 * GH)
d_table = np.median(disp[row - 2:row + 2, int(0.2 * GW):int(0.8 * GW)])
d_back = np.median(disp[int(0.25 * GH):int(0.37 * GH), int(0.40 * GW):int(0.61 * GW)])
a = (1 / Zt - 1 / Zb) / (d_table - d_back)
b = 1 / Zt - a * d_table
zmap = 1.0 / np.clip(a * disp + b, 1 / 40, 1 / 3)
print('depth fit: table d=%.3f back d=%.3f -> figures median z=%.2f' % (d_table, d_back, np.median(zmap[fg & (vv < 0.62)])))

# ---- watershed split ----------------------------------------------------------
markers = np.zeros((GH, GW), np.int32)
bgmark = ~cv2.dilate(fg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
markers[bgmark] = 100
markers[vv > TABLE_EDGE_V + 0.01] = 100
r = int(0.009 * GW)
for i, (_, u, v) in enumerate(APOSTLES, 1):
    x, y = int(u * GW), int(v * GH)
    cv2.circle(markers, (x, y), r, i, -1)
    # a short spine down from the head so robes join their owner
    cv2.line(markers, (x, y), (x, int(min(v + 0.09, 0.64) * GH)), i, max(2, r // 3))
# feed watershed an image where depth edges dominate colour edges
dimg = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ws_in = cv2.merge([dimg, dimg, (gray * 0.5).astype(np.uint8)])
ws = cv2.watershed(ws_in, markers.copy())
labels = np.where((ws >= 1) & (ws <= 13) & fg & (vv < FIG_BOTTOM_V), ws, 0).astype(np.int16)
# drop specks and keep each apostle's largest piece
for i in range(1, 14):
    m = (labels == i).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 2:
        keep = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        labels[(lab != keep) & (m > 0)] = 0
    labels[(labels == i) & ~ndimage.binary_opening(labels == i, iterations=1)] = 0
# hand any unclaimed figure pixels (robes, backs, elbows) to the nearest apostle
unl = fg & (vv < FIG_BOTTOM_V) & (labels == 0)
_, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
labels[unl] = labels[iy[unl], ix[unl]]

# ---- extend silhouettes behind nearer neighbours ----------------------------------
# a figure partly hidden behind another would otherwise have a hole in it once the
# camera moves; close each silhouette and keep the closed part where a nearer
# apostle covered it
zfig = np.array([np.median(zmap[labels == i]) if (labels == i).any() else 99 for i in range(14)])
ext = np.zeros((GH, GW), np.int16)
hidden = np.zeros((GH, GW), np.uint8)
order = sorted(range(1, 14), key=lambda i: -zfig[i])  # far to near
for i in order:
    m = labels == i
    closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((int(0.03 * GW),) * 2, np.uint8)) > 0
    nearer = np.isin(labels, [j for j in range(1, 14) if zfig[j] < zfig[i]])
    add = closed & nearer & ~m
    ext[m | add] = i
    hidden[add] = 1
    # nearer ones overwrite later (painter's order) so record per figure in a stack
np.savez_compressed(f'{OUT}/figures_masks.npz', labels=labels, zmap=zmap.astype(np.float32), zfig=zfig.astype(np.float32))
# per-figure extended masks are rebuilt cheaply in the Blender script; store the
# extension for each figure separately
exts = np.zeros((14, GH, GW), np.uint8)
for i in range(1, 14):
    m = labels == i
    closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((int(0.03 * GW),) * 2, np.uint8)) > 0
    nearer = np.isin(labels, [j for j in range(1, 14) if zfig[j] < zfig[i]])
    exts[i] = (m * 1 + ((closed & nearer & ~m) * 2)).astype(np.uint8)  # 1 visible, 2 hidden
np.savez_compressed(f'{OUT}/figures.npz', exts=exts, zmap=zmap.astype(np.float32), zfig=zfig.astype(np.float32),
                    names=np.array([n for n, _, _ in APOSTLES]), grid=np.array([GW, GH]))

# ---- palettes for unseen sides ----------------------------------------------------
small = 4
for i in range(1, 14):
    m = (labels == i).astype(np.uint8)
    ims = cv2.resize(im, (GW // small, GH // small), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m, (GW // small, GH // small), interpolation=cv2.INTER_NEAREST)
    pal = cv2.inpaint(ims, (1 - ms).astype(np.uint8) * 255, 12, cv2.INPAINT_TELEA)
    pal = cv2.GaussianBlur(pal, (0, 0), 4)
    cv2.imwrite(f'{OUT}/palette_{i:02d}.jpg', cv2.resize(pal, (GW // 2, GH // 2)), [cv2.IMWRITE_JPEG_QUALITY, 90])

# ---- debug -------------------------------------------------------------------------
rng = np.random.default_rng(4)
cols = rng.integers(60, 255, (14, 3)).astype(np.uint8)
cols[0] = 0
dbg = (im * 0.45 + cols[labels] * 0.55).astype(np.uint8)
dbg[hidden > 0] = (dbg[hidden > 0] * 0.5 + 128).astype(np.uint8)
for i, (n, u, v) in enumerate(APOSTLES, 1):
    cv2.putText(dbg, f'{n} {zfig[i]:.1f}', (int(u * GW) - 40, int(v * GH) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.imwrite(f'{OUT}/figures_debug.jpg', dbg[int(0.38 * GH):int(0.75 * GH)])
print('apostle depths:', ', '.join(f'{n}={zfig[i]:.2f}' for i, (n, _, _) in enumerate(APOSTLES, 1)))
