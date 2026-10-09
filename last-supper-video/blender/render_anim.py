"""Animate the camera through the Blender refectory and render the frames.

usage: python render_anim.py WORK_DIR FRAMES_DIR [--res 1280x720] [--fps 24] [--samples 6]
                             [--from N --to M] [--stills t1,t2,...] [--timeline-only]

Writes FRAMES_DIR/f_XXXX.png and FRAMES_DIR/timeline.json (per-frame FX values and
the screen position of the central window for post.py's light rays).
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camtools import set_camera, to_bl  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
WORK, FR = argv[0], argv[1]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


RES = tuple(map(int, opt('--res', '1280x720').split('x')))
FPS = int(opt('--fps', 24))
DUR = 32.0
NF = int(DUR * FPS)
os.makedirs(FR, exist_ok=True)
I = json.load(open(f'{WORK}/scene_info.json'))
Dp, PW, PH = I['Dp'], I['PW'], I['PH']
O0, O1 = I['O']
ZT, ZB = I['ZT'], -I['room']['zb']
names = I['names']
zfig = dict(zip(names, I['zfig'][1:]))


# ------------------------------------------------------------------ geometry helpers
def V(*a):
    return np.array(a, np.float64)


def W(u, v, z):
    k = z / Dp
    px, py = (u - 0.5) * PW, (0.5 - v) * PH
    return V(O0 + (px - O0) * k, O1 + (py - O1) * k, z)


HEADS = {'bartholomew': (0.157, 0.483), 'jamesMinor': (0.190, 0.480), 'andrew': (0.231, 0.490),
         'judas': (0.305, 0.530), 'peter': (0.328, 0.508), 'john': (0.365, 0.516), 'christ': (0.511, 0.480),
         'thomas': (0.589, 0.473), 'jamesMajor': (0.609, 0.497), 'philip': (0.646, 0.460),
         'matthew': (0.797, 0.477), 'thaddeus': (0.862, 0.483), 'simon': (0.903, 0.493)}


def figz(name):
    return I['fig_base'] + I['fig_spread'] * (zfig[name] - I['zmed']) - 0.12


def subj(name, dv=0.0, dz=0.0):
    u, v = HEADS[name]
    return W(u, v + dv, figz(name) + dz)


EXTRA = {'christHands': (0.566, 0.655, 'christ', -0.35), 'thomasFinger': (0.566, 0.455, 'thomas', -0.1),
         'bread': (0.48, 0.675, None, 0)}


def point(name):
    u, v, owner, dz = EXTRA[name]
    z = figz(owner) + dz if owner else ZT + 0.35
    return W(u, v, z)


clamp = lambda x, a=0.0, b=1.0: min(b, max(a, x))
lerp = lambda a, b, t: a + (b - a) * t
ease = {
    'io': lambda t: t * t * (3 - 2 * t),
    'io3': lambda t: 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2,
    'outExpo': lambda t: 1.0 if t >= 1 else 1 - 2 ** (-10 * t),
    'out3': lambda t: 1 - (1 - t) ** 3,
    'in3': lambda t: t ** 3,
}


def catmull(pts, t):
    pts = [np.asarray(p, np.float64) for p in pts]
    if len(pts) == 1:
        return pts[0]
    n = len(pts) - 1
    f = clamp(t) * n
    i = min(int(f), n - 1)
    s = f - i
    p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n)]
    s2, s3 = s * s, s * s * s
    return (p0 * (-0.5 * s3 + s2 - 0.5 * s) + p1 * (1.5 * s3 - 2.5 * s2 + 1)
            + p2 * (-1.5 * s3 + 2 * s2 + 0.5 * s) + p3 * (0.5 * s3 - 0.5 * s2))


def orbit(pivot, a, r, h=0.0):
    """a = 0 straight in front of the pivot (toward the eye), + swings to the right"""
    return pivot + V(math.sin(a) * r, h, -math.cos(a) * r)


def noise1(t, seed):
    return math.sin(t * 13.1 + seed) * 0.5 + math.sin(t * 7.3 + seed * 2.1) * 0.3 + math.sin(t * 23.7 + seed * 0.7) * 0.2


# ------------------------------------------------------------------ shot list
O = V(O0, O1, 0.0)
center = W(0.5, 0.5, Dp)
face = subj('christ', dv=0.012)
hero_vfov = math.degrees(2 * math.atan(PW / 2 / Dp / (16 / 9)))
hero_shift = -O1 / Dp
win = W(0.5015, 0.47, ZB)

SHOTS = [
    # 1. the fresco itself, exactly as Leonardo framed it; then it starts to breathe
    (0, 4, lambda p: dict(
        pos=catmull([O, O + V(0, 0.05, 0.9), O + V(0, 0.1, 2.6)], ease['io'](p)),
        look=catmull([O + V(0, 0, 1), center], ease['io'](clamp(p * 1.3))),
        shift=hero_shift * (1 - ease['io'](clamp(p * 1.3))), fov=lerp(hero_vfov, 38, ease['io'](p)), roll=0,
        focus='look', ap=0.4 * p, fade=1 - ease['out3'](clamp(p * 2.2)), rays=0.5 * p)),
    # 2. crane up and over the table
    (4, 8, lambda p: dict(
        pos=catmull([O + V(0, 0.1, 2.6), W(0.5, 0.25, Dp + 0.2), W(0.5, 0.33, ZT - 1.25)], ease['in3'](p) * 0.6 + ease['io'](p) * 0.4),
        look=catmull([center, W(0.5, 0.6, ZT + 0.4), point('bread')], ease['io'](p)),
        fov=40 - 6 * p, roll=3 * math.sin(p * math.pi), focus='look', ap=1.0, rays=0.6 + 0.4 * p,
        shake=0.002 + 0.01 * ease['in3'](p))),
    # 3. DROP: crash zoom into Christ's face
    (8, 10, lambda p: dict(
        pos=catmull([face + V(0.4, 0.3, -6.5), face + V(0.05, 0.06, -1.3)], ease['outExpo'](p)),
        look=face, fov=50 - 24 * ease['outExpo'](p), roll=4 * (1 - ease['outExpo'](p)), focus=face, ap=2.4,
        flash=max(0, 1 - p * 6), shake=0.05 * max(0, 1 - p * 3) + 0.004, ca=1 - ease['outExpo'](p), rays=1)),
    # 4. rotate across the left trio: Bartholomew -> James the Less -> Andrew
    (10, 12, lambda p: dict(
        pos=orbit(subj('jamesMinor', 0.03), lerp(-0.5, 0.45, ease['io'](p)), 2.3, 0.15),
        look=catmull([subj('bartholomew', 0.03), subj('jamesMinor', 0.03), subj('andrew', 0.03)], ease['io'](p)),
        fov=36, roll=lerp(-4, 3, p), focus='look', ap=2.0, shake=0.004, ca=0.3 * (1 - p),
        flash=max(0, 0.5 - p * 4), rays=0.7)),
    # 5. low-angle push on Judas, Peter and John
    (12, 14, lambda p: dict(
        pos=catmull([W(0.31, 0.71, ZT - 2.4), W(0.325, 0.6, ZT - 0.6)], ease['out3'](p)),
        look=0.7 * subj('judas') + 0.3 * subj('peter'), fov=38 - 6 * p, roll=lerp(5, 1, p), focus=subj('judas'), ap=2.4,
        flash=max(0, 0.6 - p * 5), shake=0.006, ca=0.6 * (1 - ease['out3'](p)), rays=0.8)),
    # 6. swing between Thomas, James the Greater and Philip
    (14, 16, lambda p: dict(
        pos=orbit(subj('jamesMajor', 0.03), lerp(0.6, -0.5, ease['io'](p)), 2.4, 0.2 + 0.15 * p),
        look=catmull([subj('thomas', 0.03), subj('jamesMajor', 0.03), subj('philip', 0.03)], ease['io'](p)),
        fov=35, roll=lerp(-3, 3, p), focus='look', ap=2.2, flash=max(0, 0.5 - p * 4), shake=0.004, rays=0.8)),
    # 7. tilted pull-out rotating across Simon, Thaddeus and Matthew
    (16, 18, lambda p: dict(
        pos=orbit(subj('thaddeus', 0.03), lerp(0.15, -0.5, ease['out3'](p)), lerp(1.5, 3.1, ease['out3'](p)), 0.12),
        look=catmull([subj('simon', 0.03), subj('thaddeus', 0.03), subj('matthew', 0.03)], ease['out3'](p)),
        fov=30 + 12 * ease['out3'](p), roll=lerp(9, -3, ease['out3'](p)), focus='look', ap=2.0,
        flash=max(0, 0.5 - p * 4), shake=0.005, ca=0.4 * (1 - p), rays=0.8)),
    # 8. over Christ's head and out through the central window
    (18, 20, lambda p: dict(
        pos=catmull([W(0.505, 0.6, ZT - 1.4), W(0.505, 0.36, figz('christ') + 0.2), W(0.5015, 0.47, ZB - 1.0), W(0.5015, 0.47, ZB + 1.2)],
                    ease['in3'](p) * 0.7 + p * 0.3),
        look=catmull([face, win, win + V(0, 0, 6)], ease['io'](clamp(p * 1.4))),
        fov=38 + 14 * (ease['in3'](p) * 0.7 + p * 0.3), roll=6 * math.sin(p * math.pi), focus='look', ap=1.0,
        shake=0.004 + 0.012 * p, exposure=1 + 0.8 * ease['in3'](p), flash=ease['in3'](clamp((p - 0.85) / 0.15)),
        rays=1 + p, ca=p * 0.6)),
    # 9. rip back in through the window and out to the whole room
    (20, 22, lambda p: dict(
        pos=catmull([W(0.5015, 0.47, ZB + 0.8), W(0.5015, 0.45, ZB - 2.0), W(0.505, 0.33, figz('christ') + 0.4), O + V(0, 0.15, 1.2)],
                    ease['outExpo'](p)),
        look=catmull([win, W(0.505, 0.52, figz('christ'))], ease['outExpo'](p)), fov=60 - 18 * ease['outExpo'](p),
        roll=lerp(-10, 0, ease['outExpo'](p)), focus='look', ap=0.9, flash=max(0, 1 - p * 5),
        shake=0.03 * (1 - ease['outExpo'](p)) + 0.003, ca=1 - ease['outExpo'](p), exposure=1.3 - 0.3 * ease['outExpo'](p), rays=1.4)),
]
# 10. snap zooms on the beat
for i, name in enumerate(['christHands', 'judas', 'john', 'thomasFinger']):
    def mk(name=name, i=i):
        def f(p):
            tgt = point(name) if name in EXTRA else subj(name)
            e = ease['outExpo'](clamp(p * 1.4))
            side = 1 if i % 2 else -1
            return dict(pos=orbit(tgt, side * lerp(0.45, 0.3, e), lerp(2.6, 1.3, e) + 0.12 * p, 0.2), look=tgt,
                        fov=lerp(44, 30, e), roll=side * lerp(7, 3, e), focus=tgt, ap=2.2, flash=max(0, 0.7 - p * 4),
                        shake=0.02 * (1 - e) + 0.003, ca=0.8 * (1 - e), rays=1)
        return f
    SHOTS.append((22 + i * 0.5, 22.5 + i * 0.5, mk()))
SHOTS += [
    # 11. half a second of black silence
    (24, 24.5, lambda p: dict(pos=O, look=center, fov=40, fade=1)),
    # 12. the big rotation around Christ, then rise back to the whole room
    (24.5, 29, lambda p: (lambda k, back: dict(
        pos=(1 - back) * orbit(face + V(0, -0.25, 0), lerp(-0.75, 0.7, k), lerp(2.4, 3.0, k), lerp(0.05, 0.45, k)) + back * (O + V(0, 0.1, 2.4)),
        look=(1 - back) * catmull([subj('john'), face, subj('thomas')], k) + back * center,
        fov=lerp(36, 40, back), roll=lerp(-4, 3, k) * (1 - back), focus='look', ap=lerp(2.0, 0.5, back),
        flash=max(0, 1 - p * 10), shake=0.03 * max(0, 1 - p * 6) + 0.002, ca=max(0, 1 - p * 5), rays=1.5 - 0.5 * back))(
        ease['io3'](clamp(p / 0.78)), ease['io3'](clamp((p - 0.62) / 0.38)))),
    # 13. settle exactly back into the fresco; title
    (29, 32.01, lambda p: (lambda e: dict(
        pos=catmull([O + V(0, 0.1, 2.4), O], e), look=catmull([center, O + V(0, 0, 1)], e), shift=hero_shift * e,
        fov=lerp(40, hero_vfov, e), roll=0, focus=Dp, ap=0.3 * (1 - e), lb=1 - 0.6 * e,
        exposure=1 - 0.35 * ease['io'](clamp((p - 0.3) / 0.4)), fade=ease['in3'](clamp((p - 0.8) / 0.2)), rays=1))(ease['io3'](clamp(p * 2.0)))),
]
DEFAULTS = dict(fov=40, roll=0, shift=0, focus='look', ap=1, fade=0, flash=0, shake=0, ca=0, exposure=1, rays=0.6, lb=1)


def state(t):
    shot = SHOTS[-1]
    for s in SHOTS:
        if s[0] <= t < s[1]:
            shot = s
            break
    p = clamp((t - shot[0]) / (shot[1] - shot[0]))
    st = dict(DEFAULTS)
    st.update(shot[2](p))
    st['pos'], st['look'] = np.asarray(st['pos'], float), np.asarray(st['look'], float)
    k = st['shake']
    if k:
        st['pos'] = st['pos'] + V(noise1(t * 3, 1) * k, noise1(t * 3, 2) * k, noise1(t * 3, 3) * k * 0.5)
        st['look'] = st['look'] + V(noise1(t * 2.3, 4) * k * 2, noise1(t * 2.3, 5) * k * 2, 0)
    f = st['focus']
    st['focus_d'] = float(np.linalg.norm(st['look'] - st['pos'])) if isinstance(f, str) else (
        float(f) if np.isscalar(f) else float(np.linalg.norm(np.asarray(f) - st['pos'])))
    return st


# ------------------------------------------------------------------ render
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(f'{WORK}/lastsupper.blend'))
sc = bpy.context.scene
cam = sc.camera
sc.render.resolution_x, sc.render.resolution_y = RES
sc.render.fps = FPS
sc.cycles.samples = int(opt('--samples', 6))
sc.cycles.max_bounces, sc.cycles.diffuse_bounces = 1, 1
sc.render.use_persistent_data = True
sc.render.use_motion_blur = True
sc.render.motion_blur_shutter = 0.5
sc.frame_start, sc.frame_end = 0, NF - 1


def apply(st):
    fstop = clamp(5.6 / max(st['ap'], 0.05), 1.2, 22)
    set_camera(cam, st['pos'], st['look'], st['fov'], st['roll'], st['shift'], focus=st['focus_d'],
               fstop=fstop if st['ap'] > 0.15 else None)


# keyframe every frame so Cycles can motion-blur the camera
timeline = []
for fi in range(NF):
    t = fi / FPS
    st = state(t)
    apply(st)
    cam.keyframe_insert('location', frame=fi)
    cam.keyframe_insert('rotation_quaternion', frame=fi)
    cam.data.keyframe_insert('lens', frame=fi)
    cam.data.keyframe_insert('shift_y', frame=fi)
    cam.data.dof.keyframe_insert('focus_distance', frame=fi)
    cam.data.dof.keyframe_insert('aperture_fstop', frame=fi)
    sc.frame_set(fi)
    sx, sy, sz = world_to_camera_view(sc, cam, to_bl(win))
    sun = [float(sx), float(1 - sy)] if sz > 0 and -0.2 < sx < 1.2 and -0.2 < sy < 1.2 else None
    timeline.append(dict(t=round(t, 4), exposure=st['exposure'], flash=clamp(st['flash']), fade=clamp(st['fade']),
                         ca=st['ca'], lb=st['lb'], rays=st['rays'], sun=sun))
for fc in (cam.animation_data.action.fcurves if cam.animation_data and hasattr(cam.animation_data.action, 'fcurves') else []):
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'
json.dump(timeline, open(f'{FR}/timeline.json', 'w'))
if '--timeline-only' in argv:
    sys.exit(0)

if opt('--stills'):
    frames = [int(round(float(x) * FPS)) for x in opt('--stills').split(',')]
else:
    frames = range(int(opt('--from', 0)), int(opt('--to', NF - 1)) + 1)
t0 = time.time()
for n, fi in enumerate(frames):
    out = f'{FR}/f_{fi:04d}.png'
    if os.path.exists(out) and not opt('--stills'):
        continue
    sc.frame_set(fi)
    if timeline[fi]['fade'] >= 0.999:
        # pure black frame: skip the render
        import numpy as _np
        img = bpy.data.images.new('black', RES[0], RES[1])
        img.pixels.foreach_set(_np.tile([0, 0, 0, 1], RES[0] * RES[1]).astype(_np.float32))
        img.filepath_raw, img.file_format = out, 'PNG'
        img.save()
        bpy.data.images.remove(img)
        continue
    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)
    print(f'frame {fi} done  {(time.time() - t0) / (n + 1):.1f}s/f', flush=True)
