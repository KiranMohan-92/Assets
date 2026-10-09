"""Turn the flat painting into a two-layer 3D scene.

usage: python3 prep.py painting.jpg calib.json out_dir [disparity.png]

disparity.png (optional) is a monocular depth estimate (brighter = nearer, e.g.
Depth Anything V2) used to sculpt real relief into the figures.

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
disp_path = sys.argv[4] if len(sys.argv) > 4 else None
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


def grabcut_mask(mask):
    """Snap the hand-drawn outline to real edges with GrabCut (half res)."""
    gs = 2
    m_s = cv2.resize(mask, (W // gs, H // gs), interpolation=cv2.INTER_NEAREST)
    band = max(3, W // gs // 150)
    gc = np.full(m_s.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(m_s, np.ones((3, 3), np.uint8), iterations=band) > 0] = cv2.GC_PR_BGD
    gc[m_s > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(m_s, np.ones((3, 3), np.uint8), iterations=band) > 0] = cv2.GC_FGD
    bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.resize(img, (W // gs, H // gs), interpolation=cv2.INTER_AREA), gc, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
    m_s = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cv2.resize(m_s, (W, H), interpolation=cv2.INTER_LINEAR)


def depth_mask(mask):
    """Inside the (generously grown) outline, keep pixels clearly nearer than the
    wall just above the heads in the same column."""
    dfull = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    dfull = cv2.resize(dfull[..., 0] if dfull.ndim == 3 else dfull, (W, H), interpolation=cv2.INTER_CUBIC) / 65535.0
    poly_m = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=max(2, W // 120)) > 0
    tops = np.argmax(poly_m, axis=0)
    ref = np.ones(W, np.float32)
    off0, off1 = int(0.07 * H), int(0.015 * H)
    for x in range(W):
        if poly_m[:, x].any():
            ref[x] = np.median(dfull[max(0, tops[x] - off0):max(1, tops[x] - off1), x])
    ref = ndimage.uniform_filter1d(ndimage.maximum_filter1d(ref, W // 60), W // 60)
    return ((dfull > ref[None, :] + C.get('depthCutMargin', 0.04)) & poly_m).astype(np.uint8) * 255


m = grabcut_mask(mask)
if disp_path:
    # the depth model separates figures from the back wall superbly, but it reads the
    # side walls as nearer than the outer apostles, so only trust it in the centre
    lo, hi = C.get('depthCutSpan', [0.25, 0.75])
    w = np.clip(np.minimum((np.arange(W) / W - lo), (hi - np.arange(W) / W)) / 0.03, 0, 1)[None, :]
    m = (depth_mask(mask) * w + m * (1 - w)).astype(np.uint8)
# the table itself is certain foreground
t0, t1 = int(tableTop * H), int(tableBottom * H)
m[t0:t1] = np.maximum(m[t0:t1], mask[t0:t1])
m = np.where(m > 127, 255, 0).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
if n > 1:
    m = np.where(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
mask = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, W / 2500))
cv2.imwrite(f'{out}/mask_debug.jpg', cv2.resize(np.dstack([img[..., 0] // 2, img[..., 1] // 2, np.maximum(img[..., 2] // 2, mask)]), (W // 4, H // 4)))

# ---- background: inpaint the hole ---------------------------------------
hole = (mask > 8).astype(np.uint8) * 255
hole = cv2.dilate(hole, np.ones((5, 5), np.uint8), iterations=max(1, W // 800))
for a, b, c, d in C.get('inpaintExtra', []):
    hole[int(b * H):int(d * H), int(a * W):int(c * W)] = 255
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
GW = 1024
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
bulge = 0.25 * np.clip(dist / (0.035 * GW), 0, 1) ** 0.6  # rounded bodies
relief = np.zeros((GH, GW), np.float32)
if disp_path:
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if disp.ndim == 3:
        disp = disp[..., 0]
    disp = cv2.resize(disp, (GW, GH), interpolation=cv2.INTER_AREA)
    fm = mgrid.astype(np.float32)
    lo, hi = np.percentile(disp[mgrid], [3, 97])
    dn = np.clip((disp - lo) / (hi - lo + 1e-6), 0, 1)

    def masked_blur(x, sig):
        return cv2.GaussianBlur(x * fm, (0, 0), sig) / (cv2.GaussianBlur(fm, (0, 0), sig) + 1e-6)

    # band-pass: keep faces, arms and figure-vs-figure ordering, drop the global tilt
    rel = dn - masked_blur(dn, 0.07 * GW)
    rel = cv2.GaussianBlur(rel, (0, 0), 0.8)
    rel /= np.percentile(np.abs(rel[mgrid]), 97) + 1e-6
    relief = np.clip(rel, -1.2, 1.2) * C.get('reliefM', 0.5)
    relief = np.where(vv >= tableTop, relief * 0.15, relief)
fig_depth = Zf - bulge - relief
table_plane = ray_depth_plane_y(vv, ytable)
depth = np.where(vv >= tableTop, Zt - relief, np.minimum(np.where(table_plane > 0, table_plane, np.inf) - relief * 0.3, fig_depth))
depth = np.minimum(depth, Zf + 0.3)
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
