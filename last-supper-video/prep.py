"""Turn the flat painting into a two-layer 3D scene.

usage: python3 prep.py painting.jpg calib.json out_dir

Writes to out_dir:
  bg.jpg       painting with the figures/table inpainted away (room layer)
  fg.png       RGBA cut-out of figures + table (foreground layer)
  fgdepth.bin  float32 grid of foreground z-depths (metres), row-major
  scene.json   geometry the renderer needs (camera origin, room box, grids)
"""
import json
import sys

import cv2
import numpy as np
from scipy import ndimage

src, calib_path, out = sys.argv[1:4]
C = json.load(open(calib_path))
img = cv2.imread(src, cv2.IMREAD_COLOR)
H0, W0 = img.shape[:2]
# cap texture size for WebGL (8192 max, keep it lighter)
scale = min(1.0, 6000 / W0)
if scale < 1:
    img = cv2.resize(img, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)
H, W = img.shape[:2]
aspect = W / H

PW = C['paintingWidthM']
PH = PW / aspect
Dp = C['viewDistM']


def img_xy(u, v):
    """Normalized image coords -> point on the picture plane (z = -Dp)."""
    return (u - 0.5) * PW, (0.5 - v) * PH


cx, cy = img_xy(*C['vp'])  # camera sits on the axis through the vanishing point

# ---- room box from the back-wall rectangle -------------------------------
u0, v0, u1, v1 = C['back']
bx0, by0 = img_xy(u0, v0)
bx1, by1 = img_xy(u1, v1)
# side walls meet the picture plane at its left/right edges
xl, xr = -PW / 2, PW / 2
Zb = Dp * 0.5 * ((xl - cx) / (bx0 - cx) + (xr - cx) / (bx1 - cx))
yt = cy + (by0 - cy) * Zb / Dp  # ceiling
yb = cy + (by1 - cy) * Zb / Dp  # floor
xl = cx + (bx0 - cx) * Zb / Dp
xr = cx + (bx1 - cx) * Zb / Dp


def ray_depth_plane_y(v, yplane):
    _, py = img_xy(0.5, v)
    d = py - cy
    with np.errstate(divide='ignore'):
        return np.where(np.abs(d) > 1e-6, Dp * (yplane - cy) / d, np.inf)


# ---- foreground mask -----------------------------------------------------
tableTop, tableBottom = C['tableTop'], C['tableBottom']
poly = [(u * W, v * H) for u, v in C['figuresTop']]
poly += [(C['figuresTop'][-1][0] * W, tableBottom * H), (C['figuresTop'][0][0] * W, tableBottom * H)]
mask = np.zeros((H, W), np.uint8)
cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
# snap mask edges to image content: grow/shrink along strong edges a little
mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, W / 2500))

# ---- background: inpaint the hole ---------------------------------------
hole = (mask > 8).astype(np.uint8) * 255
hole = cv2.dilate(hole, np.ones((5, 5), np.uint8), iterations=max(1, W // 800))
small = 4
img_s = cv2.resize(img, (W // small, H // small), interpolation=cv2.INTER_AREA)
hole_s = cv2.resize(hole, (W // small, H // small), interpolation=cv2.INTER_NEAREST)
inp = cv2.inpaint(img_s, hole_s, 9, cv2.INPAINT_TELEA)
inp = cv2.GaussianBlur(inp, (0, 0), 2.5)
inp = cv2.resize(inp, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
# re-add fine grain so the filled area doesn't look like plastic
grain = np.random.default_rng(1).normal(0, 6, (H, W)).astype(np.float32)
inp += cv2.GaussianBlur(grain, (0, 0), 1.2)[..., None]
hm = cv2.GaussianBlur(hole.astype(np.float32) / 255, (0, 0), 3)[..., None]
bg = img.astype(np.float32) * (1 - hm) + inp * hm
cv2.imwrite(f'{out}/bg.jpg', np.clip(bg, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])

fg = np.dstack([img, mask])
cv2.imwrite(f'{out}/fg.png', fg)

# ---- foreground depth grid ----------------------------------------------
GW = 768
GH = int(round(GW / aspect))
uu, vv = np.meshgrid((np.arange(GW) + 0.5) / GW, (np.arange(GH) + 0.5) / GH)
behind = C.get('figureBehindTableM', 0.85)
Zt = float(ray_depth_plane_y(np.array(tableBottom), yb))  # tablecloth touches the floor
# Leonardo's table is wider than the room allows at that depth: pull the group forward
# until its outermost figures sit inside the side walls.
ul, ur = C['figuresTop'][0][0], C['figuresTop'][-1][0]
zmax = min(Dp * (xl - cx) / (img_xy(ul, 0)[0] - cx), Dp * (xr - cx) / (img_xy(ur, 0)[0] - cx)) - 0.05
Zt = min(Zt, zmax - behind - 0.45)
if 'tableFrontM' in C:
    Zt = C['tableFrontM']
_, py_top = img_xy(0.5, tableTop)
ytable = cy + (py_top - cy) * Zt / Dp  # height of the table top
Zf = Zt + behind

mgrid = cv2.resize(mask, (GW, GH), interpolation=cv2.INTER_AREA) > 127
dist = ndimage.distance_transform_edt(mgrid)
bulge = 0.45 * np.clip(dist / (0.035 * GW), 0, 1) ** 0.6  # rounded bodies
fig_depth = Zf - bulge
table_plane = ray_depth_plane_y(vv, ytable)
depth = np.where(vv >= tableTop, Zt, np.minimum(np.where(table_plane > 0, table_plane, np.inf), fig_depth))
depth = np.minimum(depth, Zf)
# outside the mask, copy the nearest foreground depth so edge triangles don't stretch
_, idx = ndimage.distance_transform_edt(~mgrid, return_indices=True)
depth = depth[idx[0], idx[1]]
depth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 1.2)
depth.astype('<f4').tofile(f'{out}/fgdepth.bin')

scene = dict(
    aspect=aspect, PW=PW, PH=PH, Dp=Dp, origin=[cx, cy, 0.0],
    room=dict(xl=xl, xr=xr, yb=yb, yt=yt, zb=-Zb),
    table=dict(Zt=Zt, Zf=Zf, y=ytable),
    fgGrid=[GW, GH], texSize=[W, H],
    subjects=C['subjects'],
)
json.dump(scene, open(f'{out}/scene.json', 'w'), indent=1)
print(json.dumps({k: scene[k] for k in ('room', 'table', 'origin')}, indent=1))
