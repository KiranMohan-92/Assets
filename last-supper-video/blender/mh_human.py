"""
mh_human.py - realistic, posed, clothed 3D people from the CC0 MakeHuman assets.

Reusable Blender (bpy) module.  Import it from any bpy script:

    import sys; sys.path.insert(0, '/home/user/Assets/last-supper-video/blender')
    import mh_human as mh
    arm = mh.build_human('Peter', gender=1, age=0.6, face_seed=3)
    mh.place_human(arm, (0.4, 0.0, 0.0), rot_z_deg=10)
    body = mh.pose_human(arm, sit=True, r_hand=(0.2, -0.3, 1.0))
    mh.add_robe(body, tunic_rgb=(0.3, 0.1, 0.08))
    mh.add_hair(body, hair_rgb=(0.1, 0.05, 0.03), style='long', beard='full')

Conventions
  * Metres, Blender Z-up.  Feet on local z = 0, the person faces -Y (a camera at
    negative Y sees the front).  The person's LEFT side is +X (in local space).
  * The body mesh keeps the MakeHuman skinning weights as vertex groups (named
    after the MakeHuman bones) after baking; a few per-vertex attributes are
    stored on the mesh (mh_rest, reg_*) and landmarks in body['mh_landmarks'].
"""
import json
import math
import os
import random

import numpy as np
import bpy
import bmesh
from mathutils import Matrix, Vector

__all__ = ['build_human', 'place_human', 'pose_human', 'add_robe', 'add_hair', 'MH_DATA']

MH_DATA = '/home/user/makehumancommunity/makehuman/makehuman/data'

# How far the "old" macro target is blended when age == 1 (1.0 = MakeHuman's
# full 90 year old; 0.8 gives roughly a 70 year old).
AGE_OLD_MAX = 0.85

_C = {}   # parsed-data cache


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _base():
    """Parse base.obj body group.  Returns dict with full verts, body face
    quads (in body-vertex numbering) and the full->body index map."""
    if 'base' in _C:
        return _C['base']
    verts = []
    faces = []
    cur = None
    for line in open(os.path.join(MH_DATA, '3dobjs', 'base.obj')):
        if line.startswith('v '):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('g '):
            cur = line.split()[1]
        elif line.startswith('f ') and cur == 'body':
            faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:]])
    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    used = np.unique(faces)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    d = dict(verts_full=verts, used=used, remap=remap, faces=remap[faces])
    _C['base'] = d
    return d


def _skeleton():
    if 'skel' in _C:
        return _C['skel']
    s = json.load(open(os.path.join(MH_DATA, 'rigs', 'default.mhskel')))
    w = json.load(open(os.path.join(MH_DATA, 'rigs', 'default_weights.mhw')))['weights']
    _C['skel'] = (s, w)
    return _C['skel']


def _target(relpath):
    """Return (FULL base vertex idx, delta[k,3]) of a .target file (decimetres).
    Helper vertices (joint cubes) are morphed too, which is how MakeHuman
    places its skeleton joints."""
    key = 't:' + relpath
    if key in _C:
        return _C[key]
    path = os.path.join(MH_DATA, 'targets', relpath)
    toks = []
    with open(path) as f:
        for line in f:
            if line[0] == '#' or not line.strip():
                continue
            toks.append(line)
    arr = np.array(''.join(toks).split(), dtype=np.float64).reshape(-1, 4)
    r = (arr[:, 0].astype(np.int64), arr[:, 1:4])
    _C[key] = r
    return r


def _mh2bl(a):
    """MakeHuman (x right-hand-side, y up, z front; dm) -> Blender metres
    (x, -z, y) / 10.  The figure then faces -Y and its left side is +X."""
    a = np.asarray(a)
    out = np.empty_like(a)
    out[..., 0] = a[..., 0]
    out[..., 1] = -a[..., 2]
    out[..., 2] = a[..., 1]
    return out * 0.1


# ----------------------------------------------------------------------------
# Macro / detail targets
# ----------------------------------------------------------------------------
def _tri(v):
    """MakeHuman style 3-way split of a 0..1 slider centred on 0.5:
    returns (min, average, max) weights."""
    v = min(max(v, 0.0), 1.0)
    mx = max(0.0, (v - 0.5) * 2.0)
    mn = max(0.0, (0.5 - v) * 2.0)
    return mn, 1.0 - mx - mn, mx


def _macro_weights(gender, age, muscle, weight, ethnic):
    g = min(max(gender, 0.0), 1.0)
    genders = {'male': g, 'female': 1.0 - g}
    old = min(max(age, 0.0), 1.0) * AGE_OLD_MAX
    ages = {'young': 1.0 - old, 'old': old}
    mn, av, mx = _tri(muscle)
    muscles = {'minmuscle': mn, 'averagemuscle': av, 'maxmuscle': mx}
    mn, av, mx = _tri(weight)
    weights = {'minweight': mn, 'averageweight': av, 'maxweight': mx}
    eth = np.maximum(np.array(ethnic, dtype=float), 0)
    eth = eth / eth.sum() if eth.sum() > 0 else np.array([0, 0, 1.0])
    eths = dict(zip(['african', 'asian', 'caucasian'], eth))
    out = {}
    for gn, gw in genders.items():
        for an, aw in ages.items():
            for en, ew in eths.items():
                w = gw * aw * ew
                if w > 1e-4:
                    out['macrodetails/%s-%s-%s.target' % (en, gn, an)] = w
            for mn_, mw in muscles.items():
                for wn, ww in weights.items():
                    w = gw * aw * mw * ww
                    if w > 1e-4:
                        out['macrodetails/universal-%s-%s-%s-%s.target' % (gn, an, mn_, wn)] = w
    return out


# pools of face-detail targets (paired l-/r- files are applied together)
_FACE_POOL = [
    'nose/nose-scale-horiz-incr', 'nose/nose-scale-horiz-decr', 'nose/nose-scale-vert-incr',
    'nose/nose-scale-vert-decr', 'nose/nose-scale-depth-incr', 'nose/nose-hump-incr',
    'nose/nose-hump-decr', 'nose/nose-greek-incr', 'nose/nose-curve-convex',
    'nose/nose-curve-concave', 'nose/nose-flaring-incr', 'nose/nose-point-down',
    'nose/nose-point-up', 'nose/nose-nostrils-width-incr', 'nose/nose-base-up',
    'chin/chin-prominent-incr', 'chin/chin-prominent-decr', 'chin/chin-width-incr',
    'chin/chin-width-decr', 'chin/chin-height-incr', 'chin/chin-height-decr',
    'chin/chin-jaw-drop-incr', 'chin/chin-bones-incr', 'chin/chin-prognathism-incr',
    'chin/chin-cleft-incr', 'forehead/forehead-scale-vert-incr', 'forehead/forehead-scale-vert-decr',
    'forehead/forehead-nubian-incr', 'forehead/forehead-temple-incr', 'forehead/forehead-trans-forward',
    'eyebrows/eyebrows-trans-up', 'eyebrows/eyebrows-trans-down', 'eyebrows/eyebrows-angle-up',
    'eyebrows/eyebrows-angle-down', 'eyebrows/eyebrows-trans-forward',
    'head/head-oval', 'head/head-round', 'head/head-square', 'head/head-rectangular',
    'head/head-triangular', 'head/head-scale-horiz-incr', 'head/head-scale-horiz-decr',
    'head/head-scale-vert-incr', 'head/head-scale-depth-incr', 'head/head-back-scale-depth-incr',
    'neck/neck-scale-horiz-incr', 'neck/neck-scale-vert-incr', 'neck/neck-double-incr',
    'mouth/mouth-lowerlip-height-incr', 'mouth/mouth-lowerlip-height-decr',
    'mouth/mouth-angles-down', 'mouth/mouth-cupidsbow-incr',
    'cheek/{s}-cheek-volume-incr', 'cheek/{s}-cheek-volume-decr', 'cheek/{s}-cheek-bones-incr',
    'cheek/{s}-cheek-bones-decr', 'cheek/{s}-cheek-inner-incr', 'cheek/{s}-cheek-trans-down',
    'ears/{s}-ear-scale-incr', 'ears/{s}-ear-scale-decr', 'ears/{s}-ear-wing-incr',
    'ears/{s}-ear-lobe-incr', 'ears/{s}-ear-flap-incr', 'ears/{s}-ear-trans-down',
    'eyes/{s}-eye-height1-incr', 'eyes/{s}-eye-height1-decr', 'eyes/{s}-eye-scale-incr',
    'eyes/{s}-eye-bag-incr', 'eyes/{s}-eye-trans-down', 'eyes/{s}-eye-eyefold-angle-down',
    'eyes/{s}-eye-corner1-down',
]


def _face_detail_weights(seed, age):
    rng = random.Random(1000 + int(seed))
    out = {}
    picks = rng.sample(_FACE_POOL, 9)
    for p in picks:
        w = rng.uniform(0.25, 0.85)
        if '{s}' in p:
            for s in 'lr':
                out[p.format(s=s) + '.target'] = w * rng.uniform(0.85, 1.0)
        else:
            out[p + '.target'] = w
    # ageing helpers
    if age > 0.2:
        out['head/head-age-incr.target'] = out.get('head/head-age-incr.target', 0) + 0.8 * age
        for s in 'lr':
            out['eyes/%s-eye-bag-incr.target' % s] = out.get('eyes/%s-eye-bag-incr.target' % s, 0) + 0.5 * age
    return out


# ----------------------------------------------------------------------------
# Small numpy helpers
# ----------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rot_axis(axis, ang):
    """Rodrigues rotation matrix (radians)."""
    a = _unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = a
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def _rot_between(a, b, frac=1.0):
    a = _unit(a)
    b = _unit(b)
    ax = np.cross(a, b)
    s = np.linalg.norm(ax)
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees: pick any perpendicular axis
        p = np.cross(a, [1, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1, 0])
        return _rot_axis(p, math.pi * frac)
    return _rot_axis(ax, math.atan2(s, c) * frac)


def _signed_angle(a, b, axis):
    """Signed angle from a to b about axis (both projected on the plane)."""
    n = _unit(axis)
    a = a - n * np.dot(a, n)
    b = b - n * np.dot(b, n)
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return 0.0
    return math.atan2(np.dot(n, np.cross(a, b)), np.dot(a, b))


def _np(m):
    return np.array([list(r) for r in m], dtype=np.float64)


def _vnormals(P, F):
    """Area weighted vertex normals for quad (or tri) face array F."""
    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    if F.shape[1] == 4:
        p3 = P[F[:, 3]]
        fn = np.cross(p2 - p0, p3 - p1)
    else:
        fn = np.cross(p1 - p0, p2 - p0)
    N = np.zeros_like(P)
    for k in range(F.shape[1]):
        np.add.at(N, F[:, k], fn)
    l = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(l, 1e-12)


def _adjacency(n, F, row_normalize=True):
    import scipy.sparse as sp
    rows, cols = [], []
    k = F.shape[1]
    for i in range(k):
        j = (i + 1) % k
        rows += [F[:, i], F[:, j]]
        cols += [F[:, j], F[:, i]]
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    if row_normalize:
        d = np.asarray(A.sum(axis=1)).ravel()
        d[d == 0] = 1
        A = sp.diags(1.0 / d) @ A
    return A


def _taubin(P, A, iters=10, lam=0.5, mu=-0.53, pinned=None):
    P = P.copy()
    for _ in range(iters):
        d = A @ P - P
        if pinned is not None:
            d[pinned] = 0
        P += lam * d
        d = A @ P - P
        if pinned is not None:
            d[pinned] = 0
        P += mu * d
    return P


def _noise3(P, seed, freq=(1.0, 1.0, 1.0), octaves=3, waves=5):
    """Cheap smooth 3D value-ish noise (sum of random plane waves), ~[-1,1]."""
    rng = np.random.RandomState(seed & 0x7fffffff)
    Q = np.asarray(P) * np.asarray(freq)
    out = np.zeros(len(Q))
    amp, f, tot = 1.0, 1.0, 0.0
    for _ in range(octaves):
        for _k in range(waves):
            d = rng.randn(3)
            d /= np.linalg.norm(d)
            out += amp * np.sin((Q @ d) * f * 2 * math.pi + rng.uniform(0, 6.283))
            tot += amp * amp
        f *= 2.0
        amp *= 0.5
    return out / math.sqrt(tot * 0.5) * 0.7


def _collection():
    c = bpy.context.collection
    return c if c is not None else bpy.context.scene.collection


def _name_seed(name):
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) & 0xffff


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------
def _pbsdf(nt):
    for n in nt.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    return None


def _set(node, name, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def _noise_bump(nt, bsdf, scale, strength, detail=8.0, stretch=(1, 1, 1), rough=0.6,
                coord=None, distance=0.002, name='Bump'):
    """Add Noise -> Bump -> Principled Normal chain (object coordinates)."""
    tc = coord or nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = stretch
    nz = nt.nodes.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = scale
    nz.inputs['Detail'].default_value = detail
    nz.inputs['Roughness'].default_value = rough
    bp = nt.nodes.new('ShaderNodeBump')
    bp.inputs['Strength'].default_value = strength
    bp.inputs['Distance'].default_value = distance
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], nz.inputs['Vector'])
    nt.links.new(nz.outputs['Fac'], bp.inputs['Height'])
    return bp, tc


def _skin_material(name, rgb):
    mat = bpy.data.materials.new(name + '_skin')
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _pbsdf(nt)
    rgb = tuple(rgb)
    # slight dark/red mottling so the skin is not one flat colour
    tc = nt.nodes.new('ShaderNodeTexCoord')
    nz = nt.nodes.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = 9.0
    nz.inputs['Detail'].default_value = 4.0
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.inputs['Factor'].default_value = 0.0
    nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeMapRange')
    ramp.inputs['From Min'].default_value = 0.35
    ramp.inputs['From Max'].default_value = 0.65
    ramp.inputs['To Min'].default_value = 0.0
    ramp.inputs['To Max'].default_value = 0.35
    nt.links.new(nz.outputs['Fac'], ramp.inputs['Value'])
    nt.links.new(ramp.outputs['Result'], mix.inputs['Factor'])
    mix.inputs['A'].default_value = rgb + (1.0,)
    ruddy = (min(rgb[0] * 1.08, 1.0), rgb[1] * 0.82, rgb[2] * 0.78, 1.0)
    mix.inputs['B'].default_value = ruddy
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    bk = nt.nodes.new('ShaderNodeMix')
    bk.data_type = 'RGBA'
    bk.inputs['B'].default_value = (0.02, 0.008, 0.006, 1.0)     # socket interior reads dark
    nt.links.new(geo.outputs['Backfacing'], bk.inputs['Factor'])
    nt.links.new(mix.outputs['Result'], bk.inputs['A'])
    nt.links.new(bk.outputs['Result'], bsdf.inputs['Base Color'])
    _set(bsdf, 'Roughness', 0.5)
    _set(bsdf, 'Subsurface Weight', 0.15)
    _set(bsdf, 'Subsurface Radius', (1.0, 0.35, 0.2))
    _set(bsdf, 'Subsurface Scale', 0.012)
    _set(bsdf, 'Specular IOR Level', 0.45)
    try:
        bsdf.subsurface_method = 'RANDOM_WALK'
    except Exception:
        pass
    # pores: fine noise bump, plus a larger softer one
    bp1, _ = _noise_bump(nt, bsdf, 420.0, 0.25, detail=2.0, coord=tc, distance=0.0004)
    bp2, _ = _noise_bump(nt, bsdf, 38.0, 0.12, detail=3.0, coord=tc, distance=0.001)
    nt.links.new(bp1.outputs['Normal'], bp2.inputs['Normal'])
    nt.links.new(bp2.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def _eye_material(name, iris_rgb):
    mat = bpy.data.materials.new(name + '_eye')
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _pbsdf(nt)
    tc = nt.nodes.new('ShaderNodeAttribute')     # 'ec' = rest-space direction from eye centre
    tc.attribute_name = 'ec'
    tc.attribute_type = 'GEOMETRY'
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nrm = nt.nodes.new('ShaderNodeVectorMath')
    nrm.operation = 'NORMALIZE'
    nt.links.new(tc.outputs['Vector'], nrm.inputs[0])
    nt.links.new(nrm.outputs['Vector'], sep.inputs['Vector'])
    # forward axis = -Y -> dot = -y
    neg = nt.nodes.new('ShaderNodeMath')
    neg.operation = 'MULTIPLY'
    neg.inputs[1].default_value = -1.0
    nt.links.new(sep.outputs['Y'], neg.inputs[0])
    # radial streaks in the iris
    ang = nt.nodes.new('ShaderNodeMath')
    ang.operation = 'ARCTAN2'
    nt.links.new(sep.outputs['X'], ang.inputs[0])
    nt.links.new(sep.outputs['Z'], ang.inputs[1])
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    cr = ramp.color_ramp
    cr.interpolation = 'LINEAR'
    sclera = (0.80, 0.76, 0.70, 1)
    iris_o = tuple(c * 0.55 for c in iris_rgb) + (1,)
    iris_i = tuple(min(c * 1.35, 1.0) for c in iris_rgb) + (1,)
    dark = (0.004, 0.003, 0.003, 1)
    while len(cr.elements) < 6:
        cr.elements.new(0.5)
    stops = [(0.0, sclera), (0.872, sclera), (0.886, iris_o), (0.935, iris_i),
             (0.968, iris_o), (0.975, dark)]
    for el, (p, c) in zip(cr.elements, stops):
        el.position = p
        el.color = c
    nt.links.new(neg.outputs[0], ramp.inputs['Fac'])
    # slight darker limbal ring + streak variation
    sk = nt.nodes.new('ShaderNodeTexNoise')
    sk.inputs['Scale'].default_value = 25.0
    sk.inputs['Detail'].default_value = 3.0
    nt.links.new(tc.outputs['Vector'], sk.inputs['Vector'])
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.inputs['Factor'].default_value = 0.35
    nt.links.new(ramp.outputs['Color'], mix.inputs['A'])
    nt.links.new(sk.outputs['Color'], mix.inputs['B'])
    nt.links.new(mix.outputs['Result'], bsdf.inputs['Base Color'])
    _set(bsdf, 'Roughness', 0.12)
    _set(bsdf, 'Coat Weight', 1.0)
    _set(bsdf, 'Coat Roughness', 0.02)
    _set(bsdf, 'Specular IOR Level', 0.6)
    return mat


# ----------------------------------------------------------------------------
# build_human
# ----------------------------------------------------------------------------
_HEAD_PREFIX = ('head', 'jaw', 'special', 'levator', 'temporalis', 'oculi', 'orbicularis',
                'risorius', 'oris', 'tongue', 'eye')
_NECK = ('neck01', 'neck02', 'neck03')


def _bone_category(b):
    if b in _NECK:
        return 'neck'
    if b.startswith(('wrist', 'metacarpal', 'finger')):
        return 'hand'
    if b.startswith(('foot', 'toe')):
        return 'foot'
    if b.startswith(_HEAD_PREFIX):
        return 'head'
    return 'body'


def _set_float_attr(mesh, name, arr, domain='POINT'):
    if name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[name])
    a = mesh.attributes.new(name, 'FLOAT', domain)
    a.data.foreach_set('value', np.asarray(arr, dtype=np.float32))


def _set_vec_attr(mesh, name, arr):
    if name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[name])
    a = mesh.attributes.new(name, 'FLOAT_VECTOR', 'POINT')
    a.data.foreach_set('vector', np.asarray(arr, dtype=np.float32).ravel())


def _get_float_attr(mesh, name):
    a = mesh.attributes[name]
    out = np.empty(len(a.data), dtype=np.float32)
    a.data.foreach_get('value', out)
    return out.astype(np.float64)


def _get_vec_attr(mesh, name):
    a = mesh.attributes[name]
    out = np.empty(len(a.data) * 3, dtype=np.float32)
    a.data.foreach_get('vector', out)
    return out.reshape(-1, 3).astype(np.float64)


def _unique_name(name):
    if name not in bpy.data.objects:
        return name
    i = 1
    while '%s.%03d' % (name, i) in bpy.data.objects:
        i += 1
    return '%s.%03d' % (name, i)


def build_human(name, *, gender=1.0, age=0.7, muscle=0.5, weight=0.5,
                ethnic=(0.0, 0.0, 1.0), height_m=1.75, face_seed=0,
                skin_rgb=(0.80, 0.62, 0.50)):
    """Create a skinned MakeHuman body.  Returns the ARMATURE object (the body
    mesh is its child, with an Armature modifier, plus two eye objects).

    age     0 = young adult (~25) ... 1 = old (~70)
    gender  0 = female ... 1 = male
    ethnic  (african, asian, caucasian) weights
    The figure stands with its feet on z=0, facing -Y (left side = +X).
    """
    base = _base()
    skel, wts = _skeleton()
    Vfull = base['verts_full'].copy()                      # decimetres, MH axes
    remap = base['remap']

    # ---- shape targets -------------------------------------------------
    tw = _macro_weights(gender, age, muscle, weight, ethnic)
    tw.update(_face_detail_weights(face_seed, age))
    for rel, w in tw.items():
        try:
            idx, d = _target(rel)
        except FileNotFoundError:
            continue
        Vfull[idx] += d * w
    V = Vfull[base['used']]

    # ---- joints = mean of their (morphed) helper vertices, Blender metres --
    Vb = _mh2bl(V)
    Jb = {k: _mh2bl(Vfull[idxs].mean(axis=0)) for k, idxs in skel['joints'].items()}

    # ---- uniform scale to the requested height; feet on z=0 --------------
    zmin, zmax = Vb[:, 2].min(), Vb[:, 2].max()
    s = height_m / (zmax - zmin)
    ank = 0.5 * (Jb['foot.L____head'] + Jb['foot.R____head'])
    off = np.array([0.0, ank[1], zmin])
    Vb = (Vb - off) * s
    Jb = {k: (v - off) * s for k, v in Jb.items()}

    faces = base['faces']
    n_v = len(Vb)

    # ---- blender mesh ---------------------------------------------------
    oname = _unique_name(name)
    me = bpy.data.meshes.new(oname + '_body')
    me.from_pydata(Vb.tolist(), [], faces.tolist())
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    me.update()
    body = bpy.data.objects.new(oname + '_body', me)
    _collection().objects.link(body)

    # ---- skinning weights (normalised) -----------------------------------
    wsum = np.zeros(n_v)
    wl = {}
    for bone, lst in wts.items():
        a = np.array(lst, dtype=np.float64)
        idx = remap[a[:, 0].astype(np.int64)]
        ok = idx >= 0
        wl[bone] = (idx[ok], a[ok, 1])
        np.add.at(wsum, idx[ok], a[ok, 1])
    wsum[wsum == 0] = 1.0
    cat_w = {c: np.zeros(n_v) for c in ('head', 'neck', 'hand', 'foot', 'body',
                                         'arm', 'uarm', 'leg', 'torso', 'jaw')}
    for bone, (idx, w) in wl.items():
        wn = w / wsum[idx]
        wl[bone] = (idx, wn)
        np.add.at(cat_w[_bone_category(bone)], idx, wn)
        extra = None
        if bone.startswith(('clavicle', 'shoulder', 'upperarm', 'lowerarm')):
            extra = 'arm'
        elif bone.startswith(('upperleg', 'lowerleg', 'pelvis')):
            extra = 'leg'
        elif bone.startswith(('spine', 'root', 'breast')):
            extra = 'torso'
        if extra:
            np.add.at(cat_w[extra], idx, wn)
        if bone.startswith(('upperarm', 'shoulder')):
            np.add.at(cat_w['uarm'], idx, wn)
        if bone in ('jaw', 'special04') or bone.startswith(('oris',)):
            np.add.at(cat_w['jaw'], idx, wn)
    for bone, (idx, w) in wl.items():
        vg = body.vertex_groups.new(name=bone)
        for wv in np.unique(np.round(w, 3)):
            sel = idx[np.round(w, 3) == wv]
            vg.add(sel.tolist(), float(wv), 'REPLACE')
    for c, arr in cat_w.items():
        _set_float_attr(me, 'reg_' + c, arr)
    _set_vec_attr(me, 'mh_rest', Vb)

    # landmarks (rest space, metres)
    head_mask = cat_w['head'] > 0.5
    hv = Vb[head_mask]
    lm = dict(
        eye_l=Jb['eye.L____head'].tolist(),
        eye_r=Jb['eye.R____head'].tolist(),
        mouth=Jb['jaw____head'].tolist(),
        head_head=Jb['head____head'].tolist(),
        head_top=float(hv[:, 2].max()),
        height=float(height_m),
        scale=float(s),
        z_neck=float(Jb['neck01____head'][2]),
        z_pelvis=float(Jb['spine05____head'][2]),
        z_knee=float(Jb['lowerleg01.L____head'][2]),
        z_elbow=float(Jb['lowerarm01.L____head'][2]),
        z_sh=float(Jb['upperarm01.L____head'][2]),
        x_sh=float(Jb['upperarm01.L____head'][0]),
        z_head=float(Jb['head____head'][2]),
        jaw_z=float(Jb['jaw____head'][2]),
    )
    # better mouth estimate: joint named "...mouth" if present
    for k in Jb:
        if k.startswith('oris01____head') or k.startswith('oris06____head'):
            lm['mouth'] = Jb[k].tolist()
            break
    body['mh_landmarks'] = json.dumps(lm)

    # ---- armature ----------------------------------------------------------
    arm_data = bpy.data.armatures.new(oname + '_rig')
    arm = bpy.data.objects.new(oname, arm_data)
    _collection().objects.link(arm)
    prev_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = {}
    for bn, bd in skel['bones'].items():
        eb = arm_data.edit_bones.new(bn)
        h = Jb[bd['head']]
        t = Jb[bd['tail']]
        if np.linalg.norm(t - h) < 2e-4:
            t = h + np.array([0, 0, 2e-3])
        eb.head = Vector(h.tolist())
        eb.tail = Vector(t.tolist())
        ebs[bn] = eb
    for bn, bd in skel['bones'].items():
        if bd['parent']:
            ebs[bn].parent = ebs[bd['parent']]
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.select_set(False)
    if prev_active is not None:
        bpy.context.view_layer.objects.active = prev_active
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    arm.show_in_front = True
    arm_data.display_type = 'STICK'

    body.parent = arm
    mod = body.modifiers.new('Armature', 'ARMATURE')
    mod.object = arm
    me.materials.append(_skin_material(oname, skin_rgb))

    # ---- eyes (skinned to the eye bones) -------------------------------------
    rng = random.Random(77 + face_seed)
    iris = rng.choice([(0.20, 0.09, 0.035), (0.12, 0.06, 0.03), (0.28, 0.16, 0.06),
                       (0.10, 0.12, 0.10), (0.16, 0.10, 0.05)])
    emat = _eye_material(oname, iris)
    eyes = []
    r_eye = 0.0132 * (height_m / 1.75)
    for side, bn in (('L', 'eye.L'), ('R', 'eye.R')):
        c = Jb[bn + '____head']
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=24, radius=r_eye)
        for v in bm.verts:                      # pole +Z -> front (-Y)
            x, y, z = v.co
            v.co = Vector((x, -z, y))
        dirs = []
        for v in bm.verts:
            d = v.co.normalized()
            dirs.append((d.x, d.y, d.z))
            if v.co.y < 0:                      # slight cornea bulge
                k = max(0.0, -v.co.y / r_eye - 0.82) / 0.18
                v.co *= 1.0 + 0.05 * k * k
            v.co += Vector(c.tolist())
        eme = bpy.data.meshes.new(oname + '_eye' + side)
        bm.to_mesh(eme)
        bm.free()
        _set_vec_attr(eme, 'ec', np.array(dirs))
        for p in eme.polygons:
            p.use_smooth = True
        eo = bpy.data.objects.new(oname + '_eye' + side, eme)
        _collection().objects.link(eo)
        eo.data.materials.append(emat)
        vg = eo.vertex_groups.new(name=bn)
        vg.add(list(range(len(eme.vertices))), 1.0, 'REPLACE')
        eo.parent = arm
        m = eo.modifiers.new('Armature', 'ARMATURE')
        m.object = arm
        eyes.append(eo)
    arm['mh_body'] = body.name
    arm['mh_eyes'] = json.dumps([e.name for e in eyes])
    arm['mh_height'] = float(height_m)
    return arm


# ----------------------------------------------------------------------------
# Posing
# ----------------------------------------------------------------------------
def place_human(arm, location=(0.0, 0.0, 0.0), rot_z_deg=0.0):
    """Move/rotate the (unposed) armature in the world.  Call BEFORE pose_human so
    that world-space hand / look targets make sense.  The figure's feet are on
    local z=0 so location.z is the floor level; rot_z_deg turns the figure about
    Z (0 = facing -Y, positive = counter-clockwise seen from above)."""
    arm.location = Vector(tuple(location))
    arm.rotation_mode = 'XYZ'
    arm.rotation_euler = (0.0, 0.0, math.radians(rot_z_deg))
    bpy.context.view_layer.update()
    return arm


class _FK:
    """Tiny forward-kinematics solver over an armature's rest data.  Rotations
    are given in ARMATURE-space axes about each bone's head."""

    def __init__(self, arm):
        bones = list(arm.data.bones)
        # topological order (parents first)
        order, seen = [], set()
        for b in bones:
            chain = []
            c = b
            while c is not None and c.name not in seen:
                chain.append(c)
                c = c.parent
            for c in reversed(chain):
                seen.add(c.name)
                order.append(c)
        self.names = [b.name for b in order]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.parent = [self.idx[b.parent.name] if b.parent else -1 for b in order]
        self.rest = np.array([_np(b.matrix_local) for b in order])
        self.rest_inv = np.linalg.inv(self.rest)
        self.length = np.array([b.length for b in order])
        self.rot = np.tile(np.eye(3), (len(order), 1, 1))
        self.trans = np.zeros(3)
        self.M = None
        self.solve()

    def reset(self):
        self.rot[:] = np.eye(3)
        self.trans[:] = 0
        self.solve()

    def solve(self):
        n = len(self.names)
        M = np.empty((n, 4, 4))
        for i in range(n):
            p = self.parent[i]
            if p < 0:
                M0 = self.rest[i].copy()
                M0[:3, 3] += self.trans
            else:
                M0 = M[p] @ self.rest_inv[p] @ self.rest[i]
            M0[:3, :3] = self.rot[i] @ M0[:3, :3]
            M[i] = M0
        self.M = M

    # --- queries (armature space)
    def head(self, n):
        return self.M[self.idx[n]][:3, 3].copy()

    def tail(self, n):
        i = self.idx[n]
        return (self.M[i] @ np.array([0, self.length[i], 0, 1.0]))[:3]

    def ydir(self, n):
        i = self.idx[n]
        return _unit(self.M[i][:3, 1])

    def local_to_arm(self, n, v_rest):
        """Map a rest-space direction attached to bone n to its posed direction."""
        i = self.idx[n]
        return self.M[i][:3, :3] @ (self.rest_inv[i][:3, :3] @ v_rest)

    # --- edits
    def rotate(self, n, R):
        i = self.idx[n]
        self.rot[i] = R @ self.rot[i]

    def aim(self, n, wanted, frac=1.0, from_dir=None):
        cur = self.ydir(n) if from_dir is None else from_dir
        self.rotate(n, _rot_between(cur, wanted, frac))

    def apply_to_pose(self, arm):
        M = self.M
        for i, n in enumerate(self.names):
            p = self.parent[i]
            if p < 0:
                B = self.rest_inv[i] @ M[i]
            else:
                B = self.rest_inv[i] @ self.rest[p] @ np.linalg.inv(M[p]) @ M[i]
            arm.pose.bones[n].matrix_basis = Matrix(B.tolist())


def _fx(deg):
    return _rot_axis([1, 0, 0], math.radians(deg))


def _fy(deg):
    return _rot_axis([0, 1, 0], math.radians(deg))


def _fz(deg):
    return _rot_axis([0, 0, 1], math.radians(deg))


_SPINE = ['spine05', 'spine04', 'spine03', 'spine02', 'spine01']
_FWD_W = [0.24, 0.22, 0.20, 0.18, 0.16]
_SIDE_W = [0.16, 0.20, 0.22, 0.22, 0.20]
_TWIST_W = [0.06, 0.12, 0.22, 0.28, 0.32]
_HEADCHAIN = ['neck01', 'neck02', 'neck03', 'head']
_NECK_W = [0.2, 0.2, 0.2, 0.4]


def _palm_normal_rest(fk, s):
    """Rest-space palm normal for hand side s ('L'/'R'): points to where the
    palm faces (down in the MakeHuman rest pose)."""
    w = fk.head('wrist.' + s)
    u = fk.tail('finger3-1.' + s) - w
    side = fk.head('metacarpal4.' + s) - fk.head('metacarpal1.' + s)
    n = _unit(np.cross(u, side))
    return n if n[2] < 0 else -n


_PALMS = {
    'down': np.array([0, 0, -1.0]),
    'up': np.array([0, 0, 1.0]),
    'forward': np.array([0, -1.0, 0]),
    'back': np.array([0, 1.0, 0]),
}


def _palm_dir(spec, s):
    medial = np.array([-1.0 if s == 'L' else 1.0, 0, 0])
    if isinstance(spec, str):
        if spec == 'in':
            return medial
        if spec == 'out':
            return -medial
        if spec in _PALMS:
            return _PALMS[spec]
        if spec == 'forward_in':
            return _unit(_PALMS['forward'] + medial)
        raise ValueError('palm: %r' % spec)
    return _unit(np.asarray(spec, dtype=float))


def _solve_arm(fk, s, target, pole_dir, palm, curl, flat):
    """Two-bone IK for arm side s toward armature-space point target."""
    sh = fk.head('upperarm01.' + s)
    el0 = fk.head('lowerarm01.' + s)
    wr0 = fk.head('wrist.' + s)
    L1 = np.linalg.norm(el0 - sh)
    L2 = np.linalg.norm(wr0 - el0)
    # shoulder girdle assists the reach (shrug / protraction), up to ~30 degrees
    if np.linalg.norm(target - sh) > 0.9 * (L1 + L2):
        ch = fk.head('clavicle.' + s)
        v = sh - ch
        want = target - ch
        ang = math.acos(max(-1.0, min(1.0, float(np.dot(_unit(v), _unit(want))))))
        frac = min(1.0, math.radians(30) / max(ang, 1e-6))
        fk.rotate('clavicle.' + s, _rot_between(v, want, frac))
        fk.solve()
        sh = fk.head('upperarm01.' + s)
    d_vec = target - sh
    d = np.linalg.norm(d_vec)
    dmax = (L1 + L2) * 0.995
    dmin = abs(L1 - L2) * 1.05 + 0.02
    dc = min(max(d, dmin), dmax)
    u = _unit(d_vec)
    a = (L1 * L1 - L2 * L2 + dc * dc) / (2 * dc)
    h = math.sqrt(max(L1 * L1 - a * a, 1e-8))
    p = pole_dir - u * np.dot(pole_dir, u)
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(u, [1, 0, 0])
    p = _unit(p)
    E = sh + u * a + p * h
    W = sh + u * dc
    # upper arm (carries the twist bones), then forearm
    fk.aim('upperarm01.' + s, E - sh, from_dir=_unit(fk.head('lowerarm01.' + s) - fk.head('upperarm01.' + s)))
    fk.solve()
    fk.aim('lowerarm01.' + s, W - fk.head('lowerarm01.' + s),
           from_dir=_unit(fk.head('wrist.' + s) - fk.head('lowerarm01.' + s)))
    fk.solve()
    # --- palm orientation: twist the forearm about its axis
    f = _unit(fk.head('wrist.' + s) - fk.head('lowerarm01.' + s))
    n_rest = _palm_normal_rest(fk_rest(fk), s)
    n_cur = fk.local_to_arm('wrist.' + s, n_rest)
    n_t = _palm_dir(palm, s)
    phi = _signed_angle(n_cur, n_t, f)
    fk.rotate('lowerarm02.' + s, _rot_axis(f, phi * 0.45))
    fk.rotate('wrist.' + s, _rot_axis(f, phi * 0.55))
    fk.solve()
    # --- wrist flexion so the hand lies flat when the palm faces down
    n_cur = fk.local_to_arm('wrist.' + s, n_rest)
    hand = _unit(fk.tail('finger3-3.' + s) - fk.head('wrist.' + s))
    f = _unit(fk.head('wrist.' + s) - fk.head('lowerarm01.' + s))
    if flat > 0:
        t = f.copy()
        t[2] *= (1.0 - flat)
        t = _unit(t)
        ax = _unit(np.cross(n_cur, hand))
        beta = _signed_angle(hand, t, ax)
        fk.rotate('wrist.' + s, _rot_axis(ax, beta))
        fk.solve()
    # --- relaxed fingers: close the spread of the rest pose, then curl
    n_cur = fk.local_to_arm('wrist.' + s, n_rest)
    d_mid = fk.ydir('finger3-1.' + s)
    for f_i, k in ((2, 0.75), (4, 0.75), (5, 0.8)):
        bn = 'finger%d-1.%s' % (f_i, s)
        phi = _signed_angle(fk.ydir(bn), d_mid, n_cur)
        fk.rotate(bn, _rot_axis(n_cur, phi * k))
        fk.solve()
    for fi, (fa, fb, fc) in enumerate([(14, 26, 18), (16, 30, 22), (18, 34, 24), (22, 36, 26)]):
        f_i = fi + 2
        for seg, ang in zip((1, 2, 3), (fa, fb, fc)):
            bn = 'finger%d-%d.%s' % (f_i, seg, s)
            fd = fk.ydir(bn)
            axis = np.cross(fd, n_cur)
            fk.rotate(bn, _rot_axis(axis, math.radians(ang * curl)))
            fk.solve()
    for seg, ang in zip((1, 2, 3), (8, 14, 14)):
        bn = 'finger1-%d.%s' % (seg, s)
        fd = fk.ydir(bn)
        axis = np.cross(fd, n_cur)
        fk.rotate(bn, _rot_axis(axis, math.radians(ang * curl)))
        fk.solve()
    return bool(dc < d - 0.02)    # True if target was out of reach


_REST_FK = {}


def fk_rest(fk):
    """An FK object frozen in rest pose (cached per instance)."""
    key = id(fk)
    if key not in _REST_FK:
        r = _FK.__new__(_FK)
        r.__dict__.update(fk.__dict__)
        r.rot = np.tile(np.eye(3), (len(fk.names), 1, 1))
        r.trans = np.zeros(3)
        r.solve()
        _REST_FK.clear()
        _REST_FK[key] = r
    return _REST_FK[key]


def pose_human(arm, *, sit=True, lean_fwd=0, lean_side=0, twist=0,
               head_target=None, head_tilt=0,
               l_hand=None, r_hand=None, l_elbow_pole=None, r_elbow_pole=None,
               apply=True, l_palm=None, r_palm=None, finger_curl=0.35,
               knee_splay=0.10, ground=True):
    """Pose the figure.  All targets are WORLD-space points (call place_human
    first).  Angles in degrees; lean_fwd>0 leans toward the face direction,
    lean_side>0 toward the person's LEFT, twist>0 turns the torso toward the
    person's left, head_tilt>0 drops the left ear toward the left shoulder.

    sit        thighs horizontal, shins vertical (as on a bench).  With
               ground=True the figure is lowered so its soles are on local z=0.
    l_hand/r_hand  world point for the wrist (two-bone IK); None = relaxed
                   (hands rest on the thighs when sitting, hang when standing)
    l_palm/r_palm  'down','up','forward','back','in','out' or a world vector.
    apply      bake into a static mesh, delete the armature and return the
               body mesh (vertex groups kept).  Otherwise returns the body
               mesh still bound to the posed armature.
    """
    bpy.context.view_layer.update()
    fk = _FK(arm)
    fkr = fk_rest(fk)
    body = bpy.data.objects[arm['mh_body']]
    inv_w = np.linalg.inv(_np(arm.matrix_world))

    def to_local(p):
        return (inv_w @ np.array([p[0], p[1], p[2], 1.0]))[:3]

    R_w = _np(arm.matrix_world)[:3, :3]
    inv_R = np.linalg.inv(R_w)

    # ---------------- legs / floor ------------------------------------------
    rest_ankle_z = [fkr.head('foot.L')[2], fkr.head('foot.R')[2]]
    for s, sg in (('L', 1.0), ('R', -1.0)):
        up_hip = fkr.head('upperleg01.' + s)
        knee = fkr.head('lowerleg01.' + s)
        ankle = fkr.head('foot.' + s)
        foot_dir = fkr.ydir('foot.' + s)
        if sit:
            thigh_t = _unit([sg * knee_splay, -1.0, 0.0])
            shin_t = _unit([sg * 0.05, 0.10, -1.0])
        else:
            thigh_t = _unit([sg * 0.06, 0.0, -1.0])
            shin_t = _unit([sg * 0.03, 0.0, -1.0])
        fk.aim('upperleg01.' + s, thigh_t, from_dir=_unit(fk.head('lowerleg01.' + s) - fk.head('upperleg01.' + s)))
        fk.solve()
        fk.aim('lowerleg01.' + s, shin_t, from_dir=_unit(fk.head('foot.' + s) - fk.head('lowerleg01.' + s)))
        fk.solve()
        fk.aim('foot.' + s, foot_dir)          # keep the sole flat on the floor
        fk.solve()
    if ground:
        dz = min(fk.head('foot.L')[2] - rest_ankle_z[0], fk.head('foot.R')[2] - rest_ankle_z[1])
        fk.trans[:] = [0, 0, -dz]
        fk.solve()

    # ---------------- spine --------------------------------------------------
    for k, bn in enumerate(_SPINE):
        R = (_fz(twist * _TWIST_W[k]) @ _fy(lean_side * _SIDE_W[k]) @ _fx(lean_fwd * _FWD_W[k]))
        fk.rotate(bn, R)
        fk.solve()

    # ---------------- head ---------------------------------------------------
    eyes_mid = 0.5 * (fk.head('eye.L') + fk.head('eye.R'))
    i_head = fk.idx['head']
    f_local = fkr.M[i_head][:3, :3].T @ np.array([0, -1.0, 0])      # facing in head space

    def head_facing():
        return _unit(fk.M[i_head][:3, :3] @ f_local)

    def head_up():
        up_local = fkr.M[i_head][:3, :3].T @ np.array([0, 0, 1.0])
        return _unit(fk.M[i_head][:3, :3] @ up_local)

    def distribute(R_axis, ang, weights=None):
        weights = weights or _NECK_W
        for bn, w in zip(_HEADCHAIN, weights):
            fk.rotate(bn, _rot_axis(R_axis, ang * w))
            fk.solve()

    for _pass in range(2):
        f_cur = head_facing()
        if head_target is not None:
            d = _unit(to_local(head_target) - eyes_mid)
            ang_lim = math.radians(80)
        else:
            # relaxed: keep looking roughly ahead (compensate part of the lean)
            fh = _unit([f_cur[0], f_cur[1], 0.0]) if np.linalg.norm(f_cur[:2]) > 1e-6 else np.array([0, -1.0, 0])
            pitch = math.asin(max(-1, min(1, f_cur[2])))
            ph = pitch * 0.35
            d = _unit(np.array([fh[0] * math.cos(ph), fh[1] * math.cos(ph), math.sin(ph)]))
            ang_lim = math.radians(90)
        ax = np.cross(f_cur, d)
        sn = np.linalg.norm(ax)
        ang = math.atan2(sn, np.dot(f_cur, d))
        ang = min(ang, ang_lim)
        if sn > 1e-8 and ang > 1e-5:
            distribute(ax, ang)
    # level the head (keep ears horizontal) then apply the requested tilt
    f_cur = head_facing()
    up_cur = head_up()
    up_want = np.array([0, 0, 1.0])
    roll = _signed_angle(up_cur, up_want, f_cur)
    tilt = math.radians(head_tilt)
    # positive tilt: top of head goes toward +X(left) of the person's frame
    lateral = np.cross(f_cur, up_want)              # person's left? (-Y facing => +X)
    if lateral[0] < 0:
        lateral = -lateral
    roll_total = roll * 0.85 + (-tilt if True else 0)
    distribute(f_cur, roll_total)

    # ---------------- arms ----------------------------------------------------
    unreachable = {}
    for s, tgt_w, pole_w, palm in (('L', l_hand, l_elbow_pole, l_palm), ('R', r_hand, r_elbow_pole, r_palm)):
        sg = 1.0 if s == 'L' else -1.0
        sh = fk.head('upperarm01.' + s)
        L = np.linalg.norm(fkr.head('lowerarm01.' + s) - fkr.head('upperarm01.' + s)) + \
            np.linalg.norm(fkr.head('wrist.' + s) - fkr.head('lowerarm01.' + s))
        if tgt_w is not None:
            tgt = to_local(tgt_w)
            auto_palm = 'down' if tgt[2] < sh[2] - 0.30 else 'forward_in'
        else:
            if sit:
                hip = fk.head('upperleg01.' + s)
                knee = fk.head('lowerleg01.' + s)
                tgt = hip + (knee - hip) * 0.62 + np.array([-sg * 0.035, 0, 0.115])
                auto_palm = 'down'
            else:
                tgt = sh + np.array([sg * 0.05, 0.03, -L * 0.955])
                auto_palm = 'in'
        if pole_w is not None:
            pole = to_local(pole_w) - sh
        else:
            pole = np.array([sg * 0.55, 0.9, -0.55])
            if tgt_w is not None and tgt[2] > sh[2] - 0.05:
                pole = np.array([sg * 0.5, 0.35, -0.9])      # raised hand: elbow drops
        pal = palm if palm is not None else auto_palm
        flat = 0.95 if (isinstance(pal, str) and pal == 'down') else 0.1
        curl = finger_curl * (0.35 if flat > 0.5 else 1.0)
        unreachable[s] = _solve_arm(fk, s, tgt, pole, pal, curl, flat)

    # ---------------- apply -----------------------------------------------------
    fk.apply_to_pose(arm)
    bpy.context.view_layer.update()
    arm['mh_unreachable'] = json.dumps(unreachable)
    for sd_, bad in unreachable.items():
        if bad:
            print('[mh_human] %s: %s hand target out of reach (wrist stops short)' % (arm.name, sd_))
    if not apply:
        return body
    return _bake(arm)


def _eval_coords(ob, dg):
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    ev.to_mesh_clear()
    return co


def _bake(arm):
    """Apply the armature deformation to the body and eyes, detach, delete the rig."""
    body = bpy.data.objects[arm['mh_body']]
    eyes = [bpy.data.objects[n] for n in json.loads(arm['mh_eyes'])]
    dg = bpy.context.evaluated_depsgraph_get()
    mw = arm.matrix_world.copy()
    for ob in [body] + eyes:
        co = _eval_coords(ob, dg)
        ob.data.vertices.foreach_set('co', co)
        for m in list(ob.modifiers):
            if m.type == 'ARMATURE':
                ob.modifiers.remove(m)
        ob.data.update()
        ob.parent = None
        ob.matrix_world = mw
    for ob in [body] + eyes:
        if ob.vertex_groups and ob is not body:
            for vg in list(ob.vertex_groups):
                ob.vertex_groups.remove(vg)
    for e in eyes:
        e.parent = body
        e.matrix_parent_inverse = body.matrix_world.inverted()
    adata = arm.data
    bpy.data.objects.remove(arm)
    if adata.users == 0:
        bpy.data.armatures.remove(adata)
    bpy.context.view_layer.update()
    return body


# ----------------------------------------------------------------------------
# Cloth / hair shell helpers
# ----------------------------------------------------------------------------
class _Body:
    """Numpy view of a (posed) body mesh built by build_human."""

    def __init__(self, body):
        self.ob = body
        me = body.data
        n = len(me.vertices)
        if any(m.type == 'ARMATURE' for m in body.modifiers):
            dg = bpy.context.evaluated_depsgraph_get()
            co = _eval_coords(body, dg)
        else:
            co = np.empty(n * 3)
            me.vertices.foreach_get('co', co)
        self.P = co.reshape(-1, 3)
        F = np.empty(len(me.polygons) * 4, dtype=np.int64)
        me.polygons.foreach_get('vertices', F)
        self.F = F.reshape(-1, 4)
        self.rest = _get_vec_attr(me, 'mh_rest')
        self.reg = {k[4:]: _get_float_attr(me, k) for k in me.attributes.keys() if k.startswith('reg_')}
        self.lm = json.loads(body['mh_landmarks'])
        self.N = _vnormals(self.P, self.F)
        self.Nr = _vnormals(self.rest, self.F)
        self.seed = _name_seed(body.name)
        self._smooth = {}

    def smooth_ref(self, iters):
        """Volume-preserving smoothed copy of the whole body (positions, normals)."""
        if iters not in self._smooth:
            A = _adjacency(len(self.P), self.F)
            Ps = _taubin(self.P, A, iters=iters, lam=0.5)
            self._smooth[iters] = (Ps, _vnormals(Ps, self.F))
        return self._smooth[iters]


def _stored_mesh_object(name, V, F, body, extra_attrs=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(V, dtype=float).tolist(), [], np.asarray(F).tolist())
    if smooth:
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    me.update()
    for k, v in (extra_attrs or {}).items():
        _set_float_attr(me, k, v)
    ob = bpy.data.objects.new(name, me)
    _collection().objects.link(ob)
    ob.parent = body
    # shell coordinates are body-local: identity parent inverse keeps them in the body's frame
    ob.matrix_parent_inverse = Matrix.Identity(4)
    return ob


def _fabric_material(name, rgb, fold_scale=14.0, weave_scale=700.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _pbsdf(nt)
    rgb = tuple(rgb)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    # large soft noise darkens/lightens the cloth like fold shading
    nz = nt.nodes.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = fold_scale
    nz.inputs['Detail'].default_value = 4.0
    nz.inputs['Roughness'].default_value = 0.55
    nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['From Min'].default_value = 0.3
    mr.inputs['From Max'].default_value = 0.7
    mr.inputs['To Min'].default_value = 0.78
    mr.inputs['To Max'].default_value = 1.12
    nt.links.new(nz.outputs['Fac'], mr.inputs['Value'])
    mul = nt.nodes.new('ShaderNodeMix')
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.inputs['Factor'].default_value = 1.0
    mul.inputs['A'].default_value = rgb + (1.0,)
    nt.links.new(mr.outputs['Result'], mul.inputs['B'])
    nt.links.new(mul.outputs['Result'], bsdf.inputs['Base Color'])
    _set(bsdf, 'Roughness', 0.85)
    _set(bsdf, 'Sheen Weight', 0.3)
    _set(bsdf, 'Sheen Roughness', 0.5)
    _set(bsdf, 'Specular IOR Level', 0.25)
    # weave (fine, slightly stretched) + folds (large)
    bp1, _ = _noise_bump(nt, bsdf, weave_scale, 0.35, detail=1.0, coord=tc,
                         stretch=(1.0, 1.0, 1.6), distance=0.0006)
    bp2, _ = _noise_bump(nt, bsdf, fold_scale * 0.9, 0.25, detail=3.0, coord=tc, distance=0.004)
    nt.links.new(bp1.outputs['Normal'], bp2.inputs['Normal'])
    nt.links.new(bp2.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def _push_out(Ps, P, N, want, tree, k=10, ub=0.12):
    """Push shell points out of every nearby body surface patch until they are
    `want` metres above it (max over the k nearest surface vertices)."""
    n = len(Ps)
    d, i = tree.query(Ps, k=k, distance_upper_bound=ub)
    valid = np.isfinite(d)
    i2 = np.where(valid, i, 0)
    sd = np.einsum('nkj,nkj->nk', Ps[:, None, :] - P[i2], N[i2])
    deficit = want[:, None] - sd
    deficit[~valid] = 0.0
    deficit[sd < -0.09] = 0.0
    j = deficit.argmax(axis=1)
    rows = np.arange(n)
    push = np.maximum(deficit[rows, j], 0.0)
    return Ps + N[i2[rows, j]] * push[:, None]


def _shell(B, face_mask, offset, *, iters=14, lam=0.5, clearance=0.6, noise=0.0,
           noise_freq=(7, 7, 3), seed=0, ref=None, ref_clear=0.0, vert_scale=None,
           lap_iters=0, presmooth=0, pin_rim=False):
    """Build an offset, smoothed shell from selected body faces.

    offset      per-vertex (full body vertex array) or scalar offset in metres
    ref         optional (P, N) arrays of an inner shell to keep clear of
    Returns (verts, faces, vmap) with vmap = body vertex index per shell vertex.
    """
    P, N, F = (ref[0], ref[1], B.F) if ref is not None else (B.P, B.N, B.F)
    P_raw, N_raw = P, N
    if presmooth and ref is None:
        P, N = B.smooth_ref(presmooth)
    off = np.broadcast_to(np.asarray(offset, dtype=float), (len(P),)).copy()
    sel = F[face_mask]
    vsel = np.unique(sel)
    remap = -np.ones(len(P), dtype=np.int64)
    remap[vsel] = np.arange(len(vsel))
    Fs = remap[sel]
    Ps = P[vsel] + N[vsel] * off[vsel, None]
    A = _adjacency(len(vsel), Fs)
    pin = _boundary_verts(len(vsel), Fs) if pin_rim else None
    from scipy.spatial import cKDTree
    tree_raw = cKDTree(P_raw)
    if ref is not None and ref_clear:
        want = np.full(len(vsel), ref_clear)
    else:
        want = np.maximum(off[vsel] * clearance, 0.004)
    if lap_iters:
        # plain (shrinking) Laplacian wipes out anatomy; interleaved push-outs keep
        # the cloth draped over convex corners (knees, elbows, shoulders)
        rounds = max(1, lap_iters // 6)
        for _ in range(rounds):
            for _k in range(6):
                d_ = A @ Ps - Ps
                if pin is not None:
                    d_[pin] = 0
                Ps = Ps + 0.5 * d_
            Ps = _push_out(Ps, P_raw, N_raw, want, tree_raw)
    Ps = _taubin(Ps, A, iters=iters, lam=lam, pinned=pin)
    for _ in range(4):
        Ps = _push_out(Ps, P_raw, N_raw, want, tree_raw)
        Ps = _taubin(Ps, A, iters=2, lam=0.4, pinned=pin)
    Ps = _push_out(Ps, P_raw, N_raw, want * 0.9, tree_raw)
    if pin_rim:
        Ps = _smooth_boundary(Ps, Fs, 10)
    if noise > 0:
        nzv = _noise3(Ps, seed, freq=noise_freq, octaves=3)
        Nn = _vnormals(Ps, Fs)
        amp = noise * (1.0 if vert_scale is None else (vert_scale[vsel] if np.ndim(vert_scale) else vert_scale))
        Ps = Ps + Nn * (nzv * amp)[:, None]
        Ps = _taubin(Ps, A, iters=1, lam=0.3)
    return Ps, Fs, vsel


def _add_modifiers_cloth(ob, thickness=0.006, subsurf=1):
    sol = ob.modifiers.new('Thickness', 'SOLIDIFY')
    sol.thickness = thickness
    sol.offset = -1.0
    sol.use_rim = True
    sol.use_quality_normals = True
    if subsurf:
        ss = ob.modifiers.new('Subsurf', 'SUBSURF')
        ss.levels = subsurf
        ss.render_levels = subsurf


def add_robe(body, *, tunic_rgb, mantle_rgb=None, mantle_over='L', loose=0.035):
    """Long robe / tunic (+ optional mantle) as offset, smoothed shells following
    the posed body.  Returns the list of created objects (children of body)."""
    B = _Body(body)
    reg = B.reg
    lm = B.lm
    keep = (reg['body'] > 0.5)
    fm = keep[B.F].all(axis=1)
    rest = B.rest
    z = rest[:, 2]
    H = lm['height']

    # per-vertex offset: sleeves flare toward the cuff, hem flares toward the ankle
    off = np.full(len(B.P), loose)
    arm_t = np.clip((lm['z_sh'] - np.maximum(z, 0) * 0 - z) * 0, 0, 1)
    wrist_t = np.clip((np.abs(rest[:, 0]) - 0.30) / 0.28, 0, 1) * (reg['arm'] > 0.5)
    off += 0.022 * wrist_t ** 1.3 + 0.15 * loose * (reg['arm'] > 0.5)
    leg_t = np.clip((lm['z_knee'] * 1.15 - z) / (lm['z_knee'] * 1.05), 0, 1) * (reg['leg'] > 0.5)
    off += 0.045 * leg_t ** 1.4
    # looser around the belly
    belly = np.exp(-((z - lm['z_pelvis'] - 0.15) / 0.18) ** 2) * (reg['torso'] > 0.5)
    off += 0.012 * belly

    fold = 0.0045 + 0.004 * leg_t
    Ps, Fs, vmap = _shell(B, fm, off, iters=10, lap_iters=90, clearance=0.5, presmooth=140, pin_rim=True, noise=1.0,
                          vert_scale=fold, noise_freq=(9, 9, 4), seed=B.seed)
    out = []
    ob = _stored_mesh_object(body.name + '_robe', Ps, Fs, body)
    ob.data.materials.append(_fabric_material(body.name + '_robe_mat', tunic_rgb))
    _add_modifiers_cloth(ob, 0.006, 1)
    out.append(ob)

    if mantle_rgb is not None:
        # inner reference = the robe shell scattered back to full-length arrays
        RP = B.P.copy()
        RP[vmap] = Ps
        RF = B.F
        RN = _vnormals(RP, RF[fm]) if False else None
        # normals of the robe over its own faces
        Nfull = np.zeros_like(RP)
        Nr = _vnormals(Ps, Fs)
        Nfull[vmap] = Nr
        sgn = 1.0 if mantle_over == 'L' else -1.0
        u = rest[:, 0] * sgn
        y = rest[:, 1]
        back = B.Nr[:, 1] > -0.05
        zn, zp = lm['z_neck'], lm['z_pelvis']
        zhem = lm['z_knee'] * 0.92
        u0 = -0.20 + 0.27 * np.clip((z - zp) / max(zn - zp, 1e-3), 0, 1)
        front_band = (u > u0) & (~back)
        back_part = back & (z < zn + 0.01) & (np.abs(rest[:, 0]) < lm['x_sh'] * 1.12) & (reg['arm'] < 0.5)
        sleeve = (reg['uarm'] > 0.5) & (u > 0) & (z > lm['z_elbow'] - 0.02)
        mask = (front_band | back_part | sleeve) & (z > zhem + 0.05 * np.sin(rest[:, 0] * 23.0)) & keep
        # remove the far-side arm entirely and keep the forearms free
        mask &= ~((reg['arm'] > 0.5) & (u < 0))
        mask &= ~((reg['arm'] > 0.5) & (reg['uarm'] < 0.5))
        # blur the mask over the surface so it has no isolated holes / spikes
        Ab = _adjacency(len(B.P), B.F)
        mm = mask.astype(float)
        for _ in range(5):
            mm = Ab @ mm
        mask = (mm > 0.5) & keep & ~((reg['arm'] > 0.5) & (u < 0)) & ~((reg['arm'] > 0.5) & (reg['uarm'] < 0.5))
        mfm = mask[B.F].all(axis=1)
        gap = 0.020 + loose * 0.4
        Pm, Fm, vm = _shell(B, mfm, gap, iters=18, lap_iters=14, noise=1.0, vert_scale=0.008,
                            noise_freq=(6, 6, 3), seed=B.seed + 5, ref=(RP, Nfull),
                            ref_clear=0.014)
        mo = _stored_mesh_object(body.name + '_mantle', Pm, Fm, body)
        mo.data.materials.append(_fabric_material(body.name + '_mantle_mat', mantle_rgb,
                                                  fold_scale=9.0, weave_scale=500.0))
        _add_modifiers_cloth(mo, 0.011, 1)
        out.append(mo)
    return out


# ----------------------------------------------------------------------------
# Hair, beard
# ----------------------------------------------------------------------------
def _boundary_verts(n, F):
    e = np.concatenate([np.stack([F[:, i], F[:, (i + 1) % F.shape[1]]], 1) for i in range(F.shape[1])])
    e.sort(axis=1)
    key = e[:, 0] * n + e[:, 1]
    u, cnt = np.unique(key, return_counts=True)
    bnd = u[cnt == 1]
    return np.unique(np.concatenate([bnd // n, bnd % n]))


def _smooth_boundary(P, F, iters=8):
    """Smooth the open-boundary rings of a patch along themselves (removes the
    stair-stepping left by cutting a mesh at a weight threshold)."""
    import scipy.sparse as sp
    n = len(P)
    e = np.concatenate([np.stack([F[:, i], F[:, (i + 1) % F.shape[1]]], 1) for i in range(F.shape[1])])
    es = np.sort(e, axis=1)
    key = es[:, 0] * n + es[:, 1]
    u, cnt = np.unique(key, return_counts=True)
    bk = u[cnt == 1]
    a, b = bk // n, bk % n
    if len(a) == 0:
        return P
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    isb = deg > 0
    d = np.where(isb, deg, 1.0)
    A = sp.diags(1.0 / d) @ A
    P = P.copy()
    for _ in range(iters):
        mv = A @ P - P
        P[isb] += 0.5 * mv[isb]
    return P


def _boundary_fade(n, F, steps=3.0):
    """0 on the open boundary of the patch rising to 1 `steps` edges inside."""
    from scipy.sparse.csgraph import dijkstra
    e = np.concatenate([np.stack([F[:, i], F[:, (i + 1) % F.shape[1]]], 1) for i in range(F.shape[1])])
    e.sort(axis=1)
    key = e[:, 0] * n + e[:, 1]
    u, cnt = np.unique(key, return_counts=True)
    bnd = u[cnt == 1]
    bverts = np.unique(np.concatenate([bnd // n, bnd % n]))
    if len(bverts) == 0:
        return np.ones(n)
    A = _adjacency(n, F, row_normalize=False)
    d = dijkstra(A, directed=False, indices=bverts, min_only=True)
    d[~np.isfinite(d)] = steps
    return np.clip(d / steps, 0, 1)


def _hair_material(name, rgb, strand_scale=420.0, stretch_z=0.06, gray=0.0, opacity=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _pbsdf(nt)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (1.0, 1.0, stretch_z)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    st = nt.nodes.new('ShaderNodeTexNoise')       # fine strands (stretched along Z)
    st.inputs['Scale'].default_value = strand_scale
    st.inputs['Detail'].default_value = 2.0
    st.inputs['Roughness'].default_value = 0.7
    nt.links.new(mp.outputs['Vector'], st.inputs['Vector'])
    cl = nt.nodes.new('ShaderNodeTexNoise')       # clumps
    cl.inputs['Scale'].default_value = 60.0
    cl.inputs['Detail'].default_value = 3.0
    nt.links.new(tc.outputs['Object'], cl.inputs['Vector'])
    # colour variation
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['From Min'].default_value = 0.35
    mr.inputs['From Max'].default_value = 0.65
    mr.inputs['To Min'].default_value = 0.45
    mr.inputs['To Max'].default_value = 1.15
    nt.links.new(st.outputs['Fac'], mr.inputs['Value'])
    mul = nt.nodes.new('ShaderNodeMix')
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.inputs['Factor'].default_value = 1.0
    base = tuple(rgb) + (1.0,)
    mul.inputs['A'].default_value = base
    nt.links.new(mr.outputs['Result'], mul.inputs['B'])
    nt.links.new(mul.outputs['Result'], bsdf.inputs['Base Color'])
    _set(bsdf, 'Roughness', 0.42)
    _set(bsdf, 'Specular IOR Level', 0.6)
    _set(bsdf, 'Coat Weight', 0.15)
    _set(bsdf, 'Coat Roughness', 0.3)
    _set(bsdf, 'Anisotropic', 0.4)
    bp = nt.nodes.new('ShaderNodeBump')
    bp.inputs['Strength'].default_value = 0.9
    bp.inputs['Distance'].default_value = 0.003
    nt.links.new(st.outputs['Fac'], bp.inputs['Height'])
    bp2 = nt.nodes.new('ShaderNodeBump')
    bp2.inputs['Strength'].default_value = 0.5
    bp2.inputs['Distance'].default_value = 0.006
    nt.links.new(cl.outputs['Fac'], bp2.inputs['Height'])
    nt.links.new(bp.outputs['Normal'], bp2.inputs['Normal'])
    nt.links.new(bp2.outputs['Normal'], bsdf.inputs['Normal'])
    # feathered, strand-dissolving boundary: alpha from the 'fade' point attribute
    at = nt.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'fade'
    at.attribute_type = 'GEOMETRY'
    m1 = nt.nodes.new('ShaderNodeMath')
    m1.operation = 'MULTIPLY'
    m1.inputs[1].default_value = 2.2
    nt.links.new(at.outputs['Fac'], m1.inputs[0])
    m2 = nt.nodes.new('ShaderNodeMath')
    m2.operation = 'SUBTRACT'
    m2.inputs[1].default_value = 0.5
    nt.links.new(st.outputs['Fac'], m2.inputs[0])
    m3 = nt.nodes.new('ShaderNodeMath')
    m3.operation = 'MULTIPLY'
    m3.inputs[1].default_value = 1.8
    nt.links.new(m2.outputs[0], m3.inputs[0])
    m4 = nt.nodes.new('ShaderNodeMath')
    m4.operation = 'ADD'
    nt.links.new(m1.outputs[0], m4.inputs[0])
    nt.links.new(m3.outputs[0], m4.inputs[1])
    m5 = nt.nodes.new('ShaderNodeMath')
    m5.operation = 'SUBTRACT'
    m5.inputs[1].default_value = 0.35
    m5.use_clamp = True
    nt.links.new(m4.outputs[0], m5.inputs[0])
    if opacity < 1.0:
        m6 = nt.nodes.new('ShaderNodeMath')
        m6.operation = 'MULTIPLY'
        m6.inputs[1].default_value = opacity
        nt.links.new(m5.outputs[0], m6.inputs[0])
        nt.links.new(m6.outputs[0], bsdf.inputs['Alpha'])
    else:
        nt.links.new(m5.outputs[0], bsdf.inputs['Alpha'])
    return mat


def _head_frame(B):
    """Rest-space frame data for head-region logic."""
    lm = B.lm
    reg_h = B.reg['head'] > 0.5
    R = B.rest
    el = np.array(lm['eye_l'])
    er = np.array(lm['eye_r'])
    Ze = 0.5 * (el[2] + er[2])
    top = lm['head_top']
    cx = 0.5 * (el[0] + er[0])
    cy = 0.5 * (el[1] + er[1]) + 0.085 * lm['height'] / 1.75
    ang_x = R[:, 0] - cx
    ang_y = -(R[:, 1] - cy)           # forward positive
    ang = np.arctan2(ang_x, ang_y)    # 0 = front, +-pi = back
    f = np.cos(ang)
    zrel = (R[:, 2] - Ze) / (top - Ze)
    hw = np.percentile(np.abs(ang_x[reg_h & (zrel > 0.55)]), 92)
    mouth = np.array(lm['mouth'])
    return dict(Ze=Ze, top=top, cx=cx, cy=cy, ang=ang, f=f, zrel=zrel, hw=hw,
                reg_h=reg_h, Zm=mouth[2], D=Ze - mouth[2], q=np.abs(ang_x) / hw)


def _add_shell_object(B, name, fmask, offset, mat, *, iters=6, noise=0.0, noise_freq=(40, 40, 40),
                      fade_steps=3.0, thickness=0.0, subsurf=0, seed=0, extra_move=None,
                      lam=0.5):
    Ps, Fs, vmap = _shell(B, fmask, offset, iters=iters, lam=lam, clearance=0.5, noise=noise,
                          noise_freq=noise_freq, seed=seed)
    if extra_move is not None:
        Ps = Ps + extra_move[vmap]
    fade = _boundary_fade(len(Ps), Fs, fade_steps)
    ob = _stored_mesh_object(name, Ps, Fs, B.ob, extra_attrs={'fade': fade})
    ob.data.materials.append(mat)
    if thickness > 0 or subsurf:
        if thickness > 0:
            sol = ob.modifiers.new('Thickness', 'SOLIDIFY')
            sol.thickness = thickness
            sol.offset = -1.0
        if subsurf:
            ss = ob.modifiers.new('Subsurf', 'SUBSURF')
            ss.levels = subsurf
            ss.render_levels = subsurf
    return ob, Ps, Fs, vmap


def _head_facing_frame(B):
    """Posed head centre + forward/back horizontal unit vectors."""
    reg_h = B.reg['head'] > 0.5
    C = B.P[reg_h].mean(axis=0)
    # front-most head vertex in rest = nose tip
    cand = np.where(reg_h & (B.rest[:, 2] > (B.lm['eye_l'][2] - 0.07)))[0]
    nose = cand[np.argmin(B.rest[cand, 1])]
    fwd = B.P[nose] - C
    fwd[2] = 0
    fwd = _unit(fwd)
    return C, fwd


def add_hair(body, *, hair_rgb, style='long', beard='full', brows=True):
    """Hair and beard as thin shells (no particles).  style: 'long'|'short'|
    'bald'; beard: 'none'|'short'|'full'.  Returns the list of created objects."""
    B = _Body(body)
    H = _head_frame(B)
    out = []
    Zm, D, Ze = H['Zm'], H['D'], H['Ze']
    R = B.rest
    z = R[:, 2]
    zrel, f, q = H['zrel'], H['f'], H['q']
    reg_h = H['reg_h']
    hair_rgb = tuple(hair_rgb)

    # ------------------------------------------------------------ hair cap
    if style in ('short', 'long'):
        if style == 'short':
            xs = [-1.0, -0.5, -0.25, 0.0, 0.2, 0.4, 0.6, 1.0]
            ys = [-0.30, -0.15, 0.0, 0.10, 0.18, 0.30, 0.46, 0.62]
        else:
            xs = [-1.0, -0.5, -0.25, 0.0, 0.3, 0.6, 1.0]
            ys = [-0.40, -0.15, 0.0, 0.02, 0.18, 0.40, 0.60]
        hl = np.interp(f, xs, ys)
        sel = reg_h & (zrel > hl)
        # keep the ears free
        ear = (np.abs(R[:, 0] - H['cx']) > H['hw'] * 1.015) & (zrel < 0.45) & (np.abs(f) < 0.5)
        if style == 'short':
            sel &= ~ear
        fm = sel[B.F].all(axis=1)
        mat = _hair_material(body.name + '_hair', hair_rgb)
        thick = 0.010 if style == 'short' else 0.013
        ob, Ps, Fs, vmap = _add_shell_object(
            B, body.name + '_hair', fm, thick, mat, iters=5, noise=0.0035, noise_freq=(45, 45, 45),
            fade_steps=2.5, thickness=0.004, subsurf=1, seed=B.seed + 11)
        out.append(ob)

        if style == 'long':
            out.append(_long_hair_curtain(B, H, body, mat, hair_rgb, (Ps, Fs)))

    # ------------------------------------------------------------ beard
    if beard in ('short', 'full'):
        full = beard == 'full'
        qq = np.array([0.0, 0.40, 0.58, 0.80, 1.0])
        if full:
            up = np.array([0.30, 0.32, 0.62, 0.95, 1.10])
        else:
            up = np.array([0.20, 0.22, 0.38, 0.55, 0.62])
        zup = Zm + D * np.interp(q, qq, up)
        fmin = np.interp(z, [Zm - 1.1 * D, Zm - 0.6 * D, Zm - 0.2 * D, Zm + 0.5 * D],
                         [-0.40, -0.05, 0.25, 0.30])
        sel = reg_h & (z < zup) & (f > fmin) & (z > Zm - 1.6 * D)
        # lips stay visible
        sc_ = lm_scale = B.lm['height'] / 1.75
        lips = (((R[:, 0] - H['cx']) / (0.022 * sc_)) ** 2 + ((z - Zm - 0.0015) / (0.0085 * sc_)) ** 2 < 1.0) & (f > 0.5)
        sel &= ~lips
        # nostrils / nose: nothing above the mustache line near the centre
        fm = sel[B.F].sum(axis=1) >= 3
        mat = _hair_material(body.name + '_beard', hair_rgb, strand_scale=520.0, stretch_z=0.08,
                             opacity=1.0 if full else 0.82)
        off = 0.011 if full else 0.0035
        # beard gets longer toward the chin (full only)
        mv = np.zeros_like(B.P)
        if full:
            chin = np.clip((Zm - 0.012 - z) / 0.04, 0, 1) * np.clip(1.2 - q * 1.3, 0, 1) * (f > 0.3)
            mv[:, 2] = -0.028 * chin
            mv[:, 1] = -0.010 * chin
        ob, Ps, Fs, vmap = _add_shell_object(
            B, body.name + '_beard', fm, off, mat, iters=4, noise=0.004 if full else 0.0012,
            noise_freq=(60, 60, 60), fade_steps=2.0 if full else 3.5, thickness=0.003 if full else 0.0012,
            subsurf=1, seed=B.seed + 21, extra_move=mv)
        out.append(ob)
    if brows:
        _add_brows(body, B, H, hair_rgb)
    return out


def _math(nt, op, a, b=None, clamp=False):
    n = nt.nodes.new('ShaderNodeMath')
    n.operation = op
    n.use_clamp = clamp
    for k, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[k].default_value = float(v)
        else:
            nt.links.new(v, n.inputs[k])
    return n.outputs[0]


def _maprange(nt, v, a0, a1, b0, b1, clamp=True):
    n = nt.nodes.new('ShaderNodeMapRange')
    n.clamp = clamp
    nt.links.new(v, n.inputs['Value'])
    n.inputs['From Min'].default_value = a0
    n.inputs['From Max'].default_value = a1
    n.inputs['To Min'].default_value = b0
    n.inputs['To Max'].default_value = b1
    return n.outputs['Result']


def Zm_lip(B, H):
    return H['Zm'] + 0.0085 * B.lm['height'] / 1.75


def _add_brows(body, B, H, rgb):
    """Eyebrows painted procedurally into the skin shader from rest-space position."""
    if not body.data.materials:
        return
    mat = body.data.materials[0]
    nt = mat.node_tree
    bsdf = _pbsdf(nt)
    if bsdf is None or any(n.name == 'MH_brow_mix' for n in nt.nodes):
        return
    link = [l for l in nt.links if l.to_socket == bsdf.inputs['Base Color']]
    if not link:
        return
    src = link[0].from_socket
    sc_ = B.lm['height'] / 1.75
    ex = abs(0.5 * (B.lm['eye_l'][0] - B.lm['eye_r'][0]))
    ze = H['Ze']
    ey = 0.5 * (B.lm['eye_l'][1] + B.lm['eye_r'][1])
    at = nt.nodes.new('ShaderNodeAttribute')
    at.attribute_name = 'mh_rest'
    at.attribute_type = 'GEOMETRY'
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(at.outputs['Vector'], sep.inputs['Vector'])
    X = _math(nt, 'ABSOLUTE', sep.outputs['X'])
    t = _maprange(nt, X, ex - 0.014 * sc_, ex + 0.031 * sc_, 0.0, 1.0, clamp=False)
    mx1 = _maprange(nt, t, 0.0, 0.14, 0.0, 1.0)
    mx2 = _maprange(nt, t, 0.80, 1.0, 1.0, 0.0)
    mx = _math(nt, 'MULTIPLY', mx1, mx2)
    tc = _math(nt, 'MAXIMUM', _math(nt, 'MINIMUM', t, 1.0), 0.0)
    arch = _math(nt, 'SINE', _math(nt, 'MULTIPLY', tc, 3.14159))
    zc = _math(nt, 'ADD', _math(nt, 'MULTIPLY', arch, 0.0045 * sc_), ze + 0.0105 * sc_)
    dz = _math(nt, 'ABSOLUTE', _math(nt, 'SUBTRACT', sep.outputs['Z'], zc))
    hh = _math(nt, 'SUBTRACT', 0.0068 * sc_, _math(nt, 'MULTIPLY', tc, 0.0032 * sc_))
    rz = _math(nt, 'DIVIDE', dz, hh)
    mz = _maprange(nt, rz, 0.4, 1.0, 1.0, 0.0)
    front = _math(nt, 'LESS_THAN', sep.outputs['Y'], ey + 0.022 * sc_)
    mask = _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY', mx, mz), front)
    # hair-like stipple
    nz = nt.nodes.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = 900.0
    nz.inputs['Detail'].default_value = 1.0
    nt.links.new(at.outputs['Vector'], nz.inputs['Vector'])
    stip = _maprange(nt, nz.outputs['Fac'], 0.2, 0.6, 0.95, 1.6)
    mask = _math(nt, 'MULTIPLY', mask, stip, clamp=True)
    mix = nt.nodes.new('ShaderNodeMix')
    mix.name = 'MH_brow_mix'
    mix.data_type = 'RGBA'
    mix.inputs['B'].default_value = tuple(c * 0.55 for c in rgb) + (1.0,)
    nt.links.new(mask, mix.inputs['Factor'])
    nt.links.new(src, mix.inputs['A'])
    cur = mix.outputs['Result']

    # lips: soft reddish ellipse around the mouth line
    Xc = _math(nt, 'SUBTRACT', sep.outputs['X'], H['cx'])
    lx = _math(nt, 'DIVIDE', Xc, 0.026 * sc_)
    lz = _math(nt, 'DIVIDE', _math(nt, 'SUBTRACT', sep.outputs['Z'], Zm_lip(B, H)), 0.0105 * sc_)
    r2 = _math(nt, 'ADD', _math(nt, 'MULTIPLY', lx, lx), _math(nt, 'MULTIPLY', lz, lz))
    lm_ = _maprange(nt, r2, 0.35, 1.0, 1.0, 0.0)
    lm_ = _math(nt, 'MULTIPLY', lm_, front, clamp=True)
    lip = nt.nodes.new('ShaderNodeMix')
    lip.data_type = 'RGBA'
    lip.inputs['B'].default_value = (0.42, 0.17, 0.15, 1.0)
    nt.links.new(_math(nt, 'MULTIPLY', lm_, 0.75), lip.inputs['Factor'])
    nt.links.new(cur, lip.inputs['A'])
    cur = lip.outputs['Result']
    nt.links.new(cur, bsdf.inputs['Base Color'])


def _long_hair_curtain(B, H, body, mat, rgb, cap):
    """Hair falling from the sides/back of the head onto the shoulders."""
    P = B.P
    C, fwd = _head_facing_frame(B)
    back = -fwd
    reg_h = H['reg_h']
    lm = B.lm
    z_start = P[reg_h][:, 2].min() + 0.0       # chin height as a lower bound for the sides
    Ze_p = B.P[(B.rest[:, 2] > lm['eye_l'][2] - 0.002) & (B.rest[:, 2] < lm['eye_l'][2] + 0.002) & reg_h][:, 2]
    ze = float(np.mean(Ze_p)) if len(Ze_p) else float(C[2])
    top_z = P[reg_h][:, 2].max()
    # shoulders: posed z of the upper arm joints
    z_sh = float(np.max(P[(B.reg['uarm'] > 0.6)][:, 2]))
    z_end = z_sh - 0.07
    n_th, n_t = 41, 18
    th_max = np.radians(96)
    thetas = np.linspace(-th_max, th_max, n_th)
    # angular position of every body vertex around the head axis (0 = straight back)
    # hair must also clear clothes added before it (robe / mantle shells)
    P_hit = [P]
    for ch in body.children:
        if ch.type == 'MESH' and ch.name.startswith(body.name) and ('_robe' in ch.name or '_mantle' in ch.name):
            co = np.empty(len(ch.data.vertices) * 3)
            ch.data.vertices.foreach_get('co', co)
            P_hit.append(co.reshape(-1, 3) + np.array([0, 0, 0.0]))
    P_hit = np.concatenate(P_hit)
    rel = P_hit - C
    ang_all = np.arctan2(rel[:, 0] * back[1] - rel[:, 1] * back[0],
                         rel[:, 0] * back[0] + rel[:, 1] * back[1])
    rad_all = np.hypot(rel[:, 0], rel[:, 1])
    z_all = P_hit[:, 2]
    zs_row = np.linspace(0, 1, n_t)

    # the cap's lower boundary provides the start of every lock
    Pc, Fc = cap
    bv = _boundary_verts(len(Pc), Fc)
    rc = Pc[bv] - C
    th_c = np.arctan2(rc[:, 0] * back[1] - rc[:, 1] * back[0], rc[:, 0] * back[0] + rc[:, 1] * back[1])
    keep_b = np.abs(th_c) < np.radians(112)
    bv, th_c = bv[keep_b], th_c[keep_b]
    order = np.argsort(th_c)
    bv, th_c = bv[order], th_c[order]
    # for each angle take the LOWEST boundary vertex around it (the hair's lower edge)
    z0s = np.empty(n_th)
    r0s = np.empty(n_th)
    for j, th in enumerate(thetas):
        m = np.abs(th_c - th) < np.radians(4.0)
        if not m.any():
            m = np.abs(th_c - th) <= np.abs(th_c - th).min() + 1e-6
        cand = bv[m]
        k = cand[np.argmin(Pc[cand][:, 2])]
        z0s[j] = Pc[k][2] + 0.004
        r0s[j] = np.hypot(Pc[k][0] - C[0], Pc[k][1] - C[1]) - 0.003

    def radius(theta, zz):
        m = (np.abs(ang_all - theta) < np.radians(3.5)) & (np.abs(z_all - zz) < 0.02)
        return float(rad_all[m].max()) if m.any() else 0.0

    rng = np.random.RandomState(B.seed)
    tip_noise = np.convolve(rng.randn(n_th + 8), np.ones(5) / 5.0, mode='same')[4:4 + n_th]
    cols = []
    for j, th in enumerate(thetas):
        w = abs(th) / th_max
        zend = z_end - 0.02 * (1 - w) + 0.045 * tip_noise[j]
        z0 = z0s[j]
        r0 = r0s[j]
        col = []
        r_prev = r0
        for t in zs_row:
            zz = z0 + (zend - z0) * t
            rb = radius(th, zz) + 0.016
            rr = max(r_prev, rb, r0 + 0.006 * t)
            rr += 0.006 * t ** 0.8 + 0.0035 * math.sin(th * 15.0 + t * 2.0 + 1.0)
            col.append((rr, th, zz))
            r_prev = rr - 0.003
        cols.append(col)
    V = np.empty((n_th, n_t, 3))
    for j in range(n_th):
        for i in range(n_t):
            rr, th, zz = cols[j][i]
            d = np.array([back[0] * math.cos(th) - back[1] * math.sin(th),
                          back[0] * math.sin(th) + back[1] * math.cos(th)])
            V[j, i] = (C[0] + d[0] * rr, C[1] + d[1] * rr, zz)
    V = V.reshape(-1, 3)
    F = []
    for j in range(n_th - 1):
        for i in range(n_t - 1):
            a = j * n_t + i
            b = (j + 1) * n_t + i
            F.append([a, b, b + 1, a + 1])
    F = np.array(F)
    A = _adjacency(len(V), F)
    V = _taubin(V, A, iters=6, lam=0.4, pinned=np.arange(n_th) * n_t)
    # fade towards the tips (and the side edges)
    fade = np.ones((n_th, n_t))
    fade[:, -3:] = np.linspace(0.9, 0.0, 3)[None, :]
    fade[0, :] = np.minimum(fade[0, :], 0.35)
    fade[-1, :] = np.minimum(fade[-1, :], 0.35)
    fade[:, 0] = np.minimum(fade[:, 0], 0.6)
    fade = fade.reshape(-1)
    # orientation: make normals point outward (away from head axis)
    Nn = _vnormals(V, F)
    outward = np.stack([V[:, 0] - C[0], V[:, 1] - C[1], np.zeros(len(V))], 1)
    if np.mean(np.einsum('ij,ij->i', Nn, outward)) < 0:
        F = F[:, ::-1]
    ob = _stored_mesh_object(body.name + '_hairlong', V, F, body, extra_attrs={'fade': fade})
    ob.data.materials.append(mat)
    sol = ob.modifiers.new('Thickness', 'SOLIDIFY')
    sol.thickness = 0.014
    sol.offset = -1.0
    ss = ob.modifiers.new('Subsurf', 'SUBSURF')
    ss.levels = 1
    ss.render_levels = 2
    return ob
