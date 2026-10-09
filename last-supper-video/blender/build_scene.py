"""Build the Last Supper refectory as a real 3D Blender scene.

usage: python build_scene.py SOURCE_DIR WORK_DIR out.blend
  (run with the Python that has the `bpy` module, after figures.py has written
   WORK_DIR/figures.npz and palette_XX.jpg)

Coordinates while building ("painting space"): X right, Y up, Z = distance into
the room from the eye plane, eye at (O0, O1, 0) looking down +Z. The painting is
the picture plane Z = Dp.  Converted to Blender space as (X, Z, Y).

Everything visible in the painting is textured by projecting the painting from
Leonardo's own viewpoint, so the hero camera sees exactly the fresco; faces the
painting never shows (backs of apostles, beam sides, the near ceiling) take each
object's own palette instead.
"""
import json
import math
import os
import sys

import bpy
import cv2
import numpy as np
from mathutils import Vector
from scipy import ndimage

SRC, WORK, OUT = [os.path.abspath(p) for p in sys.argv[-3:]]
HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, 'geometry.json')))
PW, PH, Dp = G['PW'], G['PH'], G['Dp']
O0, O1 = G['origin'][0], G['origin'][1]
R = G['room']
XL, XR, YB, YT, ZB = R['xl'], R['xr'], R['yb'], R['yt'], -R['zb']
ZT, YTAB = G['table']['Zt'], G['table']['y']
ZTB = ZT + 0.85               # back edge of the table
FIG_BASE, FIG_SPREAD, FIG_RELIEF = 7.35, 0.28, 0.35


# ------------------------------------------------------------------ projection
def W(u, v, z):
    """normalized image coords + depth -> painting-space point"""
    k = z / Dp
    px, py = (u - 0.5) * PW, (0.5 - v) * PH
    return np.array([O0 + (px - O0) * k, O1 + (py - O1) * k, z])


def proj(P):
    """painting-space points (N,3) -> normalized image coords (N,2)"""
    P = np.asarray(P, np.float64).reshape(-1, 3)
    k = np.maximum(P[:, 2], 1e-3) / Dp
    px = O0 + (P[:, 0] - O0) / k
    py = O1 + (P[:, 1] - O1) / k
    return np.stack([px / PW + 0.5, 0.5 - py / PH], 1)


def to_bl(P):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    return np.stack([P[:, 0], P[:, 2], P[:, 1]], 1)


# ------------------------------------------------------------------ mesh helpers
def make_mesh(name, verts, faces, mat=None, uv_proj=True, attrs=None, smooth=False):
    """verts in painting space; UVs projected from the hero viewpoint per corner"""
    me = bpy.data.meshes.new(name)
    vb = to_bl(verts)
    me.from_pydata(vb.tolist(), [], [list(map(int, f)) for f in faces])
    me.update()
    if uv_proj:
        uv = me.uv_layers.new(name='proj')
        uvs = proj(verts)
        loops_v = np.zeros(len(me.loops), np.int64)
        me.loops.foreach_get('vertex_index', loops_v)
        luv = uvs[loops_v]
        luv[:, 1] = 1 - luv[:, 1]
        uv.data.foreach_set('uv', luv.astype(np.float32).ravel())
    if attrs:
        for an, vals in attrs.items():
            a = me.attributes.new(an, 'FLOAT', 'POINT')
            a.data.foreach_set('value', np.asarray(vals, np.float32))
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if mat:
        ob.data.materials.append(mat)
    return ob


def grid_quad(p00, p10, p01, p11, nu, nv):
    """bilinear grid between 4 painting-space corners (dense, so projected UVs stay exact)"""
    us, vs = np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1)
    U, V = np.meshgrid(us, vs)
    P = (np.array(p00)[None, None] * ((1 - U) * (1 - V))[..., None] + np.array(p10)[None, None] * (U * (1 - V))[..., None]
         + np.array(p01)[None, None] * ((1 - U) * V)[..., None] + np.array(p11)[None, None] * (U * V)[..., None])
    verts = P.reshape(-1, 3)
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nv + 1, nu + 1)
    faces = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    return verts, faces


def box(c0, c1, sub=(1, 1, 1)):
    """axis-aligned box in painting space as 6 dense quads -> (verts, faces)"""
    (x0, y0, z0), (x1, y1, z1) = c0, c1
    V, F = [], []
    sides = [  # (p00, p10, p01, p11, nu, nv)  outward facing
        ((x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0), sub[0], sub[1]),  # front (z0)
        ((x1, y0, z1), (x0, y0, z1), (x1, y1, z1), (x0, y1, z1), sub[0], sub[1]),  # back
        ((x0, y0, z1), (x0, y0, z0), (x0, y1, z1), (x0, y1, z0), sub[2], sub[1]),  # left
        ((x1, y0, z0), (x1, y0, z1), (x1, y1, z0), (x1, y1, z1), sub[2], sub[1]),  # right
        ((x0, y1, z0), (x1, y1, z0), (x0, y1, z1), (x1, y1, z1), sub[0], sub[2]),  # top
        ((x0, y0, z1), (x1, y0, z1), (x0, y0, z0), (x1, y0, z0), sub[0], sub[2]),  # bottom
    ]
    for p00, p10, p01, p11, nu, nv in sides:
        v, f = grid_quad(p00, p10, p01, p11, nu, nv)
        F.append(f + sum(len(x) for x in V))
        V.append(v)
    return np.concatenate(V), np.concatenate(F)


def merge(parts):
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.concatenate(V), np.concatenate(F)


# ------------------------------------------------------------------ materials
def load_img(path, name):
    im = bpy.data.images.load(path)
    im.name = name
    im.colorspace_settings.name = 'sRGB'
    return im


def proj_material(name, img, fallback_rgb, emission=0.45, rough=0.85, palette_img=None, vis_attr=None, facing_blend=False):
    """painting projected through the 'proj' UVs; outside the frame -> fallback colour.
    With palette_img + vis_attr: where vis_attr==0 use the palette image instead."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = rough
    uv = nt.nodes.new('ShaderNodeUVMap')
    uv.uv_map = 'proj'
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.extension = 'EXTEND'
    tex.interpolation = 'Cubic'
    nt.links.new(uv.outputs['UV'], tex.inputs['Vector'])
    col = tex.outputs['Color']
    # in-frame factor: 1 inside the painting, fading to 0 just outside it
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(uv.outputs['UV'], sep.inputs['Vector'])

    def edge(sock):
        a = nt.nodes.new('ShaderNodeMapRange')
        a.inputs['From Min'].default_value, a.inputs['From Max'].default_value = -0.04, 0.0
        nt.links.new(sock, a.inputs['Value'])
        b = nt.nodes.new('ShaderNodeMapRange')
        b.inputs['From Min'].default_value, b.inputs['From Max'].default_value = 1.04, 1.0
        nt.links.new(sock, b.inputs['Value'])
        mm = nt.nodes.new('ShaderNodeMath')
        mm.operation = 'MINIMUM'
        nt.links.new(a.outputs['Result'], mm.inputs[0])
        nt.links.new(b.outputs['Result'], mm.inputs[1])
        return mm.outputs['Value']
    fx, fy = edge(sep.outputs['X']), edge(sep.outputs['Y'])
    inframe = nt.nodes.new('ShaderNodeMath')
    inframe.operation = 'MINIMUM'
    nt.links.new(fx, inframe.inputs[0])
    nt.links.new(fy, inframe.inputs[1])
    fb = nt.nodes.new('ShaderNodeRGB')
    fb.outputs[0].default_value = (*fallback_rgb, 1)
    # plaster/fabric grain for the fallback so it is not flat
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 40
    nmix = nt.nodes.new('ShaderNodeMix')
    nmix.data_type = 'RGBA'
    nmix.blend_type = 'MULTIPLY'
    nmix.inputs['Factor'].default_value = 0.25
    nt.links.new(fb.outputs[0], nmix.inputs['A'])
    nt.links.new(noise.outputs['Color'], nmix.inputs['B'])
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    nt.links.new(inframe.outputs['Value'], mix.inputs['Factor'])
    nt.links.new(nmix.outputs['Result'], mix.inputs['A'])
    nt.links.new(col, mix.inputs['B'])
    col = mix.outputs['Result']
    # how squarely the surface faces the hero eye: grazing faces get the fallback/palette
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    eye = nt.nodes.new('ShaderNodeCombineXYZ')
    eye.inputs['X'].default_value, eye.inputs['Y'].default_value, eye.inputs['Z'].default_value = O0, 0.0, O1
    dv = nt.nodes.new('ShaderNodeVectorMath')
    dv.operation = 'SUBTRACT'
    nt.links.new(eye.outputs['Vector'], dv.inputs[0])
    nt.links.new(geo.outputs['Position'], dv.inputs[1])
    dn = nt.nodes.new('ShaderNodeVectorMath')
    dn.operation = 'NORMALIZE'
    nt.links.new(dv.outputs['Vector'], dn.inputs[0])
    dot = nt.nodes.new('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    nt.links.new(dn.outputs['Vector'], dot.inputs[0])
    nt.links.new(geo.outputs['Normal'], dot.inputs[1])
    facing = nt.nodes.new('ShaderNodeMapRange')
    facing.inputs['From Min'].default_value, facing.inputs['From Max'].default_value = 0.12, 0.38
    nt.links.new(dot.outputs['Value'], facing.inputs['Value'])
    if palette_img is None and facing_blend:
        fm = nt.nodes.new('ShaderNodeMix')
        fm.data_type = 'RGBA'
        nt.links.new(facing.outputs['Result'], fm.inputs['Factor'])
        nt.links.new(nmix.outputs['Result'], fm.inputs['A'])
        nt.links.new(col, fm.inputs['B'])
        col = fm.outputs['Result']
    if palette_img is None and vis_attr:
        at = nt.nodes.new('ShaderNodeAttribute')
        at.attribute_name = vis_attr
        hm = nt.nodes.new('ShaderNodeMix')
        hm.data_type = 'RGBA'
        nt.links.new(at.outputs['Fac'], hm.inputs['Factor'])
        nt.links.new(nmix.outputs['Result'], hm.inputs['A'])
        nt.links.new(col, hm.inputs['B'])
        col = hm.outputs['Result']
    if palette_img is not None:
        ptex = nt.nodes.new('ShaderNodeTexImage')
        ptex.image = palette_img
        ptex.extension = 'EXTEND'
        nt.links.new(uv.outputs['UV'], ptex.inputs['Vector'])
        at = nt.nodes.new('ShaderNodeAttribute')
        at.attribute_name = vis_attr
        vf = nt.nodes.new('ShaderNodeMath')
        vf.operation = 'MULTIPLY'
        nt.links.new(at.outputs['Fac'], vf.inputs[0])
        nt.links.new(facing.outputs['Result'], vf.inputs[1])
        pm = nt.nodes.new('ShaderNodeMix')
        pm.data_type = 'RGBA'
        nt.links.new(vf.outputs['Value'], pm.inputs['Factor'])
        nt.links.new(ptex.outputs['Color'], pm.inputs['A'])
        nt.links.new(col, pm.inputs['B'])
        col = pm.outputs['Result']
    nt.links.new(col, bsdf.inputs['Base Color'])
    nt.links.new(col, bsdf.inputs['Emission Color'])
    bsdf.inputs['Emission Strength'].default_value = emission
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return m


def flat_material(name, rgb, rough=0.6, metallic=0.0, emission=0.0, transmission=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metallic
    if transmission:
        b.inputs['Transmission Weight'].default_value = transmission
        b.inputs['IOR'].default_value = 1.45
    if emission:
        b.inputs['Emission Color'].default_value = (*rgb, 1)
        b.inputs['Emission Strength'].default_value = emission
    return m


def srgb2lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


# ------------------------------------------------------------------ scene setup
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
paint_cv = cv2.imread(f'{SRC}/last_supper_enhanced.jpg')
IH, IW = paint_cv.shape[:2]


def sample(u, v, r=6):
    x, y = int(u * IW), int(v * IH)
    p = paint_cv[max(0, y - r):y + r, max(0, x - r):x + r].reshape(-1, 3).mean(0)
    return p[::-1]  # BGR -> RGB


img_paint = load_img(f'{SRC}/last_supper_enhanced.jpg', 'painting')
img_bg = load_img(f'{SRC}/last_supper_bg_enhanced.jpg', 'painting_bg')
plaster = srgb2lin(sample(0.80, 0.30, 20))
ceiling_c = srgb2lin(sample(0.50, 0.08, 20))
tapestry_c = srgb2lin(sample(0.07, 0.30, 30))
floor_c = srgb2lin(sample(0.30, 0.93, 30))
cloth_c = srgb2lin(np.minimum(255, sample(0.30, 0.75, 30) * 1.18))

M_room = proj_material('room', img_bg, plaster, emission=0.32)
M_ceiling = proj_material('ceiling', img_bg, ceiling_c, emission=0.32)
M_tap = proj_material('tapestry', img_bg, tapestry_c, emission=0.32)
M_clothfront = proj_material('cloth_front', img_paint, cloth_c, emission=0.3)
M_cloth = flat_material('cloth', cloth_c, rough=0.9)

# ---- room shell -------------------------------------------------------------
ZN = 0.3   # the room extends a little in front of the eye plane
# floor: the strip behind the table is hidden from Leonardo's eye, so it gets the
# plain floor colour instead of whatever the table projects onto it
M_floor = proj_material('floor', img_bg, floor_c, emission=0.32, vis_attr='vis')
fv, ff = grid_quad((XL, YB, ZN), (XR, YB, ZN), (XL, YB, ZB), (XR, YB, ZB), 60, 160)
fuv = proj(fv)
hidden = (fv[:, 2] > ZT + 0.02) & (fuv[:, 0] > 0.06) & (fuv[:, 0] < 0.95) & (fuv[:, 1] < 0.83)
make_mesh('floor', fv, ff, M_floor, attrs={'vis': (~hidden).astype(np.float32)})
make_mesh('ceiling', *grid_quad((XL, YT, ZB), (XR, YT, ZB), (XL, YT, ZN), (XR, YT, ZN), 40, 120), M_ceiling)
make_mesh('wall_left', *grid_quad((XL, YB, ZB), (XL, YB, ZN), (XL, YT, ZB), (XL, YT, ZN), 120, 30), M_room)
make_mesh('wall_right', *grid_quad((XR, YB, ZN), (XR, YB, ZB), (XR, YT, ZN), (XR, YT, ZB), 120, 30), M_room)

# coffered ceiling: beams hanging under the ceiling plane
beam_d, beam_w = 0.32, 0.16
beams = []
for z in np.linspace(ZB, 11.0, 7):
    beams.append(box((XL, YT - beam_d, z - beam_w / 2), (XR, YT, z + beam_w / 2), (40, 2, 2)))
for x in np.linspace(XL + 0.9, XR - 0.9, 6):
    beams.append(box((x - beam_w / 2, YT - beam_d, ZN), (x + beam_w / 2, YT, ZB), (2, 2, 120)))
M_beams = proj_material('beams', img_bg, ceiling_c, emission=0.32, facing_blend=True)
make_mesh('ceiling_beams', *merge(beams), M_beams)
# a cornice running along both walls under the ceiling
make_mesh('cornice', *merge([box((XL, YT - 0.55, ZN), (XL + 0.22, YT - 0.32, ZB), (2, 2, 120)),
                             box((XR - 0.22, YT - 0.55, ZN), (XR, YT - 0.32, ZB), (2, 2, 120))]), M_room)


# tapestries: raised panels on both walls, edges measured off the painting
def wall_z(u, side):
    px = (u - 0.5) * PW
    xw = XL if side < 0 else XR
    return Dp * (xw - O0) / (px - O0)


taps = []
for side, spans in ((-1, [(0.025, 0.137), (0.172, 0.206), (0.256, 0.297), (0.307, 0.330)]),
                    (1, [(0.866, 0.981), (0.770, 0.806), (0.706, 0.744), (0.667, 0.684)])):
    xw = XL if side < 0 else XR
    for u0, u1 in spans:
        z0, z1 = sorted((wall_z(u0, side), wall_z(u1, side)))
        x0, x1 = (xw, xw + 0.07) if side < 0 else (xw - 0.07, xw)
        taps.append(box((x0, -0.9, z0), (x1, 2.15, z1), (2, 30, 30)))
make_mesh('tapestries', *merge(taps), M_tap)

# back wall with three window openings
WINS = [(0.378, 0.409), (0.466, 0.537), (0.578, 0.609)]
WIN_V0, WIN_V1 = 0.396, 0.56
wx = [(W(u0, 0, ZB)[0], W(u1, 0, ZB)[0]) for u0, u1 in WINS]
wy1, wy0 = W(0.5, WIN_V0, ZB)[1], W(0.5, WIN_V1, ZB)[1]
xs = sorted({XL, XR, *[a for p in wx for a in p]})
ys = sorted({YB, YT, wy0, wy1})
bw = []
for i in range(len(xs) - 1):
    for j in range(len(ys) - 1):
        cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
        if any(a < cx < b for a, b in wx) and wy0 < cy < wy1:
            continue  # window hole
        nu = max(2, int((xs[i + 1] - xs[i]) * 12))
        nv = max(2, int((ys[j + 1] - ys[j]) * 12))
        bw.append(grid_quad((xs[i], ys[j], ZB), (xs[i + 1], ys[j], ZB), (xs[i], ys[j + 1], ZB), (xs[i + 1], ys[j + 1], ZB), nu, nv))
make_mesh('back_wall', *merge(bw), M_room)
rev = []
T = 0.45
for a, b in wx:
    rev += [box((a - 0.02, wy0 - 0.02, ZB), (b + 0.02, wy0, ZB + T), (6, 1, 3)),     # sill
            box((a - 0.02, wy1, ZB), (b + 0.02, wy1 + 0.02, ZB + T), (6, 1, 3)),     # head
            box((a - 0.02, wy0, ZB), (a, wy1, ZB + T), (1, 6, 3)),                   # jambs
            box((b, wy0, ZB), (b + 0.02, wy1, ZB + T), (1, 6, 3))]
make_mesh('window_reveals', *merge(rev), M_room)
# arched pediment over the middle window
a, b = wx[1]
cx, r = (a + b) / 2, (b - a) / 2 + 0.25
arc_v, arc_f = [], []
n = 24
for k in range(n + 1):
    t = math.pi * k / n
    for rr, dz in ((r, -0.12), (r - 0.18, -0.12), (r, 0.0), (r - 0.18, 0.0)):
        arc_v.append((cx + rr * math.cos(t), wy1 + 0.35 + rr * 0.55 * math.sin(t), ZB + dz))
for k in range(n):
    q = 4 * k
    for s0, s1 in ((0, 1), (2, 0), (1, 3)):
        arc_f.append((q + s0, q + s1, q + 4 + s1, q + 4 + s0))
make_mesh('pediment', np.array(arc_v), np.array(arc_f), M_room)

# landscape seen through the windows: painted procedurally in the fresco's own
# sky and hill colours (Leonardo's distant Lombard hills)
rng_l = np.random.default_rng(5)
LW, LH = 4096, 1024
sky_top = np.array([138., 168., 204.])  # the fresco's window blues, de-faded
sky_low = np.array([214., 222., 226.])
hill_far = np.array([104., 132., 172.])
yy = np.linspace(0, 1, LH)[:, None, None]
land = (sky_top[None, None, :] * (1 - yy) + sky_low[None, None, :] * yy) * np.ones((1, LW, 1))
clouds = cv2.GaussianBlur(rng_l.random((LH // 16, LW // 16)).astype(np.float32), (0, 0), 3)
clouds = cv2.resize(clouds, (LW, LH))
land += ((clouds - clouds.mean()) * 90 * (1 - yy[..., 0]) ** 2)[..., None]


def ridge(base, amp, rough):
    walk = np.cumsum(rng_l.normal(0, 1, LW))
    walk = cv2.GaussianBlur(walk.reshape(1, -1).astype(np.float32), (0, 0), rough).ravel()
    walk = (walk - walk.min()) / (np.ptp(walk) + 1e-6)
    return (base - amp * walk) * LH


for base, amp, rough, col, haze in ((0.62, 0.10, 90, hill_far, 0.55), (0.72, 0.12, 60, np.array([92., 122., 140.]), 0.3),
                                    (0.86, 0.10, 40, np.array([88., 112., 92.]), 0.12)):
    r = ridge(base, amp, rough)
    mask = (np.arange(LH)[:, None] > r[None, :])[..., None]
    c = col * (1 - haze) + sky_low * haze
    land = np.where(mask, c[None, None, :] * (0.92 + 0.08 * rng_l.random((LH, LW, 1))), land)
land = cv2.GaussianBlur(np.clip(land, 0, 255).astype(np.uint8)[..., ::-1], (0, 0), 1.0)
cv2.imwrite(f'{WORK}/landscape.jpg', land)
img_land = load_img(f'{WORK}/landscape.jpg', 'landscape')
M_land = bpy.data.materials.new('landscape')
M_land.use_nodes = True
nt = M_land.node_tree
for nd in list(nt.nodes):
    nt.nodes.remove(nd)
o, e, t = nt.nodes.new('ShaderNodeOutputMaterial'), nt.nodes.new('ShaderNodeEmission'), nt.nodes.new('ShaderNodeTexImage')
t.image = img_land
e.inputs['Strength'].default_value = 1.1
nt.links.new(t.outputs['Color'], e.inputs['Color'])
nt.links.new(e.outputs['Emission'], o.inputs['Surface'])
LZ = ZB + 7
lv, lf = grid_quad((XL - 6, wy0 - 3.5, LZ), (XR + 6, wy0 - 3.5, LZ), (XL - 6, wy1 + 3.0, LZ), (XR + 6, wy1 + 3.0, LZ), 1, 1)
land_ob = make_mesh('landscape', lv, lf, M_land, uv_proj=False)
uvl = land_ob.data.uv_layers.new(name='UVMap')
uvl.data.foreach_set('uv', np.array([[0, 0], [1, 0], [1, 1], [0, 1]] * 1, np.float32).ravel())

# ---- table ------------------------------------------------------------------
TU0, TU1, TV_BOT = 0.07, 0.94, 0.82
tx0, tx1 = W(TU0, 0, ZT)[0], W(TU1, 0, ZT)[0]
ycloth = W(0.5, TV_BOT, ZT)[1]
make_mesh('cloth_front', *grid_quad((tx0, ycloth, ZT), (tx1, ycloth, ZT), (tx0, YTAB, ZT), (tx1, YTAB, ZT), 200, 30), M_clothfront)
cloth = [box((tx0, YTAB - 0.03, ZT), (tx1, YTAB, ZTB), (60, 1, 10)),
         box((tx0 - 0.005, ycloth, ZT), (tx0, YTAB, ZTB), (1, 10, 10)),
         box((tx1, ycloth, ZT), (tx1 + 0.005, YTAB, ZTB), (1, 10, 10))]
make_mesh('cloth', *merge(cloth), M_cloth, uv_proj=True)
M_wood = flat_material('wood', srgb2lin((70, 48, 32)), rough=0.7)
legs = []
for u in (0.13, 0.205, 0.80, 0.87):
    x = W(u, 0, ZT + 0.3)[0]
    legs.append(box((x - 0.05, YB, ZT + 0.15), (x + 0.05, ycloth, ZTB - 0.15), (1, 1, 1)))
make_mesh('trestles', *merge(legs), M_wood, uv_proj=False)

# ---- tabletop items -----------------------------------------------------------
pewter = flat_material('pewter', srgb2lin(sample(0.505, 0.683, 8)), rough=0.35, metallic=0.7)
M_glass = flat_material('glass', (0.75, 0.78, 0.8), rough=0.12, metallic=0.6)
items = []
plates = [0.11, 0.165, 0.22, 0.265, 0.315, 0.395, 0.505, 0.60, 0.66, 0.725, 0.775, 0.86, 0.92]
for u in plates:
    rad = 0.21 if abs(u - 0.505) < 0.01 else 0.14
    c = W(u, 0, ZT + 0.32)
    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=rad, depth=0.018, location=to_bl([c[0], YTAB + 0.009, c[2]])[0])
    ob = bpy.context.active_object
    ob.data.materials.append(pewter)
    bev = ob.modifiers.new('bev', 'BEVEL')
    bev.width, bev.segments = 0.008, 2
    ob.data.shade_smooth()
rng = np.random.default_rng(11)
for u in plates[::2]:
    c = W(u + 0.018, 0, ZT + 0.5)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.035, depth=0.12, location=to_bl([c[0], YTAB + 0.06, c[2]])[0])
    bpy.context.active_object.data.materials.append(M_glass)

# bread rolls: find the orange-brown blobs on the tablecloth band and drop rolls there
band0, band1 = int(0.655 * IH), int(0.705 * IH)
hsv = cv2.cvtColor(paint_cv[band0:band1], cv2.COLOR_BGR2HSV)
bread = ((hsv[..., 0] > 4) & (hsv[..., 0] < 24) & (hsv[..., 1] > 80) & (hsv[..., 2] > 90)).astype(np.uint8)
bread = cv2.morphologyEx(bread, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(bread)
M_bread = flat_material('bread', srgb2lin((176, 112, 62)), rough=0.75)
rolls = 0
for i in range(1, n):
    if st[i, cv2.CC_STAT_AREA] < 140 or st[i, cv2.CC_STAT_WIDTH] > 160:
        continue
    u = cen[i][0] / IW
    if not (TU0 + 0.01 < u < TU1 - 0.01):
        continue
    vrow = (cen[i][1] + band0) / IH
    # depth along the tabletop plane for this image row
    py = (0.5 - vrow) * PH
    z = Dp * (YTAB - O1) / (py - O1) if py < O1 else ZT + 0.2
    z = float(np.clip(z, ZT + 0.08, ZTB - 0.1))
    c = W(u, vrow, z)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=0.05, location=to_bl([c[0], YTAB + 0.03, c[2]])[0])
    ob = bpy.context.active_object
    ob.scale = (1.35, 1.0, 0.75)
    ob.rotation_euler[2] = rng.uniform(0, math.pi)
    ob.data.materials.append(M_bread)
    ob.data.shade_smooth()
    rolls += 1
print('bread rolls placed:', rolls)

# ---- the thirteen apostles: full 3D people ----------------------------------------
# MakeHuman (CC0) bodies, posed to the fresco's gestures, scaled to Leonardo's
# larger-than-life proportions and placed so each face lands exactly where it is
# painted.  Robes, cloaks, hair and skin take their colours from the painting.
sys.path.insert(0, HERE)
import mh_human as mh                       # noqa: E402
from apostles import APOSTLES               # noqa: E402

SCALE = (YTAB - YB) / 0.76                  # Leonardo's table is ~1.5x a real one
FLOOR = YB


def bl(p):
    return Vector(to_bl([p])[0])


def table_point(u, v):
    py = (0.5 - v) * PH
    z = Dp * (YTAB - O1) / (py - O1) if py < O1 - 1e-6 else ZT + 0.3
    z = float(np.clip(z, ZT + 0.06, ZTB - 0.08))
    p = W(u, v, z)
    p[1] = YTAB + 0.035 * SCALE
    return p


heads = {a['name']: W(*a['head'], a['z']) for a in APOSTLES}


def target(a, spec):
    if spec is None:
        return None
    if spec[0] == 'table':
        return bl(table_point(spec[1], spec[2]))
    return bl(W(spec[1], spec[2], spec[3]))


def look_point(a):
    if a['look'] == 'table':
        # gaze down the table toward the viewer rather than straight into the lap
        p = W(a['head'][0], 0.66, ZT - 1.2)
        p[1] = YTAB
        return bl(p)
    return bl(heads[a['look']])


def head_world(body):
    """midpoint between the eyes, posed or baked"""
    bpy.context.view_layer.update()
    arm = body.parent if body.parent and body.parent.type == 'ARMATURE' else None
    if arm is None:
        arm = next((m.object for m in body.modifiers if m.type == 'ARMATURE' and m.object), None)
    if arm is not None and 'eye.L' in arm.pose.bones:
        return (arm.matrix_world @ arm.pose.bones['eye.L'].head + arm.matrix_world @ arm.pose.bones['eye.R'].head) / 2
    dg = bpy.context.evaluated_depsgraph_get()
    eyes = [c for c in body.children_recursive if 'eye' in c.name.lower()]
    pts = [e.evaluated_get(dg).matrix_world @ Vector(np.mean([v.co for v in e.data.vertices], 0)) for e in eyes if e.type == 'MESH']
    return sum(pts, Vector()) / len(pts)


def delete_tree(ob):
    for c in list(ob.children_recursive) + [ob]:
        bpy.data.objects.remove(c, do_unlink=True)


def make_person(a, i, height, loc, apply):
    skin = srgb2lin(sample(*a['skin'], 5))
    arm = mh.build_human(f"apostle_{a['name']}", gender=1.0, age=a['age'], muscle=0.5,
                         weight=0.45 if a['age'] < 0.5 else 0.55,
                         ethnic=(0.35, 0.0, 0.65) if a.get('skin_dark') else (0.05, 0.0, 0.95),
                         height_m=height, face_seed=i * 7 + 3, skin_rgb=skin)
    mh.place_human(arm, location=loc, rot_z_deg=a['face'])
    body = mh.pose_human(arm, sit=a['sit'], lean_fwd=a['lean_fwd'], lean_side=a['lean_side'], twist=a['twist'],
                         head_target=look_point(a), head_tilt=a.get('tilt', 0),
                         l_hand=target(a, a['l_hand']), r_hand=target(a, a['r_hand']), apply=apply)
    return body


people = 0
for i, a in enumerate(APOSTLES):
    goal = bl(heads[a['name']])
    height = 1.80 * SCALE * (0.97 + 0.06 * ((i * 37) % 7) / 6)
    # torso sits roughly under the head, a little further back from the table
    loc = Vector((goal.x, goal.y + 0.10 * SCALE, FLOOR))
    for it in range(3):
        body = make_person(a, i, height, loc, apply=False)
        h = head_world(body)
        arm = body.parent if body.parent and body.parent.type == 'ARMATURE' else next(
            (m.object for m in body.modifiers if m.type == 'ARMATURE' and m.object), None)
        delete_tree(body)
        if arm is not None and arm.name in bpy.data.objects:
            delete_tree(arm)
        # one consistent build for everyone; the bench / stance height absorbs the
        # vertical difference (legs are hidden behind the tablecloth)
        loc.z += goal.z - h.z
        loc.x += goal.x - h.x
        loc.y += goal.y - h.y
    body = make_person(a, i, height, loc, apply=True)
    tunic = srgb2lin(sample(*a['tunic'], 10))
    mantle = srgb2lin(sample(*a['mantle'], 10)) if a.get('mantle') else None
    mh.add_robe(body, tunic_rgb=tunic, mantle_rgb=mantle, mantle_over=a.get('mantle_over') or 'L')
    mh.add_hair(body, hair_rgb=srgb2lin(sample(*a['hair_c'], 5)), style=a['hair'], beard=a['beard'])
    h = head_world(body)
    print(f"{a['name']:12s} height {height:.2f} m  floor offset {loc.z - FLOOR:+.2f} m  head error {(h - goal).length * 100:.1f} cm", flush=True)
    people += 1
print('apostles built:', people)

# ---- fresco detail on the people -------------------------------------------------------
# Each person's surfaces that Leonardo's eye can see AND that fall inside that
# apostle's own painted silhouette take the fresco's pixels (faces, beards, folds);
# everything else keeps the 3D person's own palette colours.
from mathutils.bvhtree import BVHTree   # noqa: E402

FIGS = np.load(f'{WORK}/figures.npz')
own_masks = {}
for i, a in enumerate(APOSTLES, 1):
    m = (FIGS['exts'][i] == 1).astype(np.float32)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    own_masks[a['name']] = cv2.GaussianBlur(m, (0, 0), 2.0)
MGH, MGW = own_masks['christ'].shape
STRENGTH = {'skin': 0.0, 'hair': 0.45, 'beard': 0.45, 'brow': 0.0, 'robe': 0.55, 'mantle': 0.55, 'tunic': 0.55}
eye_bl = Vector((O0, 0.0, O1))
person_objs = {}
for a in APOSTLES:
    root = bpy.data.objects.get(f"apostle_{a['name']}_body")
    if root is None:
        continue
    person_objs[a['name']] = [o for o in [root] + list(root.children_recursive)
                              if o.type == 'MESH' and 'eye' not in o.name.lower()]
dg = bpy.context.evaluated_depsgraph_get()
occluders = [o for o in bpy.data.objects if o.type == 'MESH' and (o.name.startswith('apostle_') or o.parent)
             or o.name in ('cloth', 'cloth_front')]
all_v, all_f = [], []
for o in set(occluders) | {o for objs in person_objs.values() for o in objs}:
    me = o.evaluated_get(dg).to_mesh()
    vs = np.array([o.matrix_world @ v.co for v in me.vertices])
    off = sum(len(x) for x in all_v)
    all_f += [[off + i for i in p.vertices] for p in me.polygons]
    all_v.append(vs)
    o.evaluated_get(dg).to_mesh_clear()
tree = BVHTree.FromPolygons([tuple(v) for v in np.concatenate(all_v)], all_f, epsilon=0.0)


def fresco_blend(mat, strength):
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        return
    inp = bsdf.inputs['Base Color']
    src = inp.links[0].from_socket if inp.links else None
    uv = nt.nodes.new('ShaderNodeUVMap')
    uv.uv_map = 'proj'
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image, tex.extension, tex.interpolation = img_paint, 'EXTEND', 'Cubic'
    nt.links.new(uv.outputs['UV'], tex.inputs['Vector'])
    at = nt.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'own'
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    eye = nt.nodes.new('ShaderNodeCombineXYZ')
    eye.inputs['X'].default_value, eye.inputs['Y'].default_value, eye.inputs['Z'].default_value = O0, 0.0, O1
    dv = nt.nodes.new('ShaderNodeVectorMath')
    dv.operation = 'SUBTRACT'
    nt.links.new(eye.outputs['Vector'], dv.inputs[0])
    nt.links.new(geo.outputs['Position'], dv.inputs[1])
    dn = nt.nodes.new('ShaderNodeVectorMath')
    dn.operation = 'NORMALIZE'
    nt.links.new(dv.outputs['Vector'], dn.inputs[0])
    dot = nt.nodes.new('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    nt.links.new(dn.outputs['Vector'], dot.inputs[0])
    nt.links.new(geo.outputs['Normal'], dot.inputs[1])
    facing = nt.nodes.new('ShaderNodeMapRange')
    facing.inputs['From Min'].default_value, facing.inputs['From Max'].default_value = 0.15, 0.45
    nt.links.new(dot.outputs['Value'], facing.inputs['Value'])
    f1 = nt.nodes.new('ShaderNodeMath')
    f1.operation = 'MULTIPLY'
    nt.links.new(at.outputs['Fac'], f1.inputs[0])
    nt.links.new(facing.outputs['Result'], f1.inputs[1])
    f2 = nt.nodes.new('ShaderNodeMath')
    f2.operation = 'MULTIPLY'
    f2.inputs[1].default_value = strength
    nt.links.new(f1.outputs['Value'], f2.inputs[0])
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    nt.links.new(f2.outputs['Value'], mix.inputs['Factor'])
    if src is not None:
        nt.links.new(src, mix.inputs['A'])
    else:
        mix.inputs['A'].default_value = inp.default_value
    nt.links.new(tex.outputs['Color'], mix.inputs['B'])
    nt.links.new(mix.outputs['Result'], inp)


done_mats = {}
for name, objs in person_objs.items():
    mask = own_masks[name]
    for o in objs:
        me = o.data
        wv = np.array([o.matrix_world @ v.co for v in me.vertices])
        P = np.stack([wv[:, 0], wv[:, 2], wv[:, 1]], 1)        # back to painting space
        uvs = proj(P)
        # visible from Leonardo's eye?
        vis = np.zeros(len(wv), np.float32)
        for k, w in enumerate(wv):
            d = Vector(w) - eye_bl
            L = d.length
            hit = tree.ray_cast(eye_bl, d / L, L - 0.03)
            vis[k] = 1.0 if hit[0] is None else 0.0
        mx = np.clip((uvs[:, 0] * MGW).astype(int), 0, MGW - 1)
        my = np.clip((uvs[:, 1] * MGH).astype(int), 0, MGH - 1)
        own = mask[my, mx] * vis
        a = me.attributes.get('own') or me.attributes.new('own', 'FLOAT', 'POINT')
        a.data.foreach_set('value', own.astype(np.float32))
        uv = me.uv_layers.get('proj') or me.uv_layers.new(name='proj')
        lv = np.zeros(len(me.loops), np.int64)
        me.loops.foreach_get('vertex_index', lv)
        luv = uvs[lv].copy()
        luv[:, 1] = 1 - luv[:, 1]
        uv.data.foreach_set('uv', luv.astype(np.float32).ravel())
        for slot in o.material_slots:
            m = slot.material
            if m is None or m.name in done_mats:
                continue
            key = next((k for k in STRENGTH if k in m.name.lower()), None)
            if STRENGTH.get(key, 0.5) > 0:
                fresco_blend(m, STRENGTH.get(key, 0.5))
            done_mats[m.name] = True
    print('fresco detail:', name, flush=True)

# ---- lighting -----------------------------------------------------------------------
world = bpy.data.worlds.new('world')
scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.06, 0.05, 0.04, 1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.35
sun = bpy.data.lights.new('key', 'SUN')
sun.energy, sun.angle = 1.3, math.radians(10)
sun.color = (1.0, 0.92, 0.8)
so = bpy.data.objects.new('key', sun)
scene.collection.objects.link(so)
# light from the upper left, like the refectory's real windows
so.rotation_euler = (math.radians(55), math.radians(-35), math.radians(-20))
for a, b in wx:
    L = bpy.data.lights.new('window', 'AREA')
    L.energy, L.size, L.color = 120, max(0.3, b - a), (0.85, 0.9, 1.0)
    lo = bpy.data.objects.new('window', L)
    lo.location = to_bl([((a + b) / 2, (wy0 + wy1) / 2, ZB + 0.3)])[0]
    lo.rotation_euler = (math.radians(-90), 0, 0)   # face into the room (-Y in Blender)
    scene.collection.objects.link(lo)
fill = bpy.data.lights.new('fill', 'AREA')
fill.energy, fill.size, fill.color = 350, 6.0, (1.0, 0.9, 0.78)
fo = bpy.data.objects.new('fill', fill)
fo.location = to_bl([(O0, O1 + 1.5, 1.5)])[0]
fo.rotation_euler = (math.radians(90), 0, 0)        # face +Y into the room
scene.collection.objects.link(fo)

# ---- camera + render settings ----------------------------------------------------------
cam = bpy.data.cameras.new('cam')
co = bpy.data.objects.new('cam', cam)
scene.collection.objects.link(co)
scene.camera = co
cam.sensor_fit = 'HORIZONTAL'
cam.sensor_width = 36
cam.clip_start, cam.clip_end = 0.03, 200
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 10
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.03
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 3
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 1
scene.cycles.transmission_bounces = 4
scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
scene.render.use_persistent_data = True
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
scene.render.image_settings.file_format = 'PNG'
json.dump({'Dp': Dp, 'PW': PW, 'PH': PH, 'O': [O0, O1], 'room': R, 'ZT': ZT, 'ZTB': ZTB, 'YTAB': YTAB,
           'windows': wx, 'win_y': [wy0, wy1], 'heads': {a['name']: [*a['head'], a['z']] for a in APOSTLES}}, open(f'{WORK}/scene_info.json', 'w'), indent=1)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(OUT))
print('saved', OUT, 'objects:', len(bpy.data.objects))
