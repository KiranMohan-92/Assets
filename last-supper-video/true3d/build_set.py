"""The Last Supper as a real 3D set, for a camera that flies through it.

Room, table and props are built procedurally at metre scale; the people are
image-to-3D meshes (Tripo / SAM 3D) lifted from hyper-real recreations of the
painting's figures.  Layout follows the fresco: the table runs along X, the
viewer stands at -Y, the back wall with its three windows is at +Y.

usage (bpy 4.2):  python3 build_set.py ASSET_DIR OUT.blend
ASSET_DIR holds glb/{jesus,g1,g2,g3,g4}.glb and img/{tapestry,landscape}.png
"""
import math, os, sys
import bpy
from mathutils import Vector

ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]   # works as bpy module or `blender -b -P`
A = os.path.abspath(ARGV[0])
OUT = os.path.abspath(ARGV[1])

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

ROOM_W, ROOM_D, ROOM_H = 11.0, 14.0, 5.6       # x half-span 5.5, back wall y=+6, front y=-8
BACK_Y, FRONT_Y = 6.0, -8.0
TABLE_Y, TABLE_H, TABLE_D, TABLE_L = 0.0, 0.80, 0.95, 9.4
FIG_Y = 0.85                                   # the figures sit just behind the table


# ---------------------------------------------------------------- materials
def mat(name, color=(0.8, 0.8, 0.8), rough=0.6, tex=None, scale=(1, 1), emit=0.0, bump=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Roughness'].default_value = rough
    if tex:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(tex)
        mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (scale[0], scale[1], 1)
        tc = nt.nodes.new('ShaderNodeTexCoord')
        nt.links.new(tc.outputs['UV'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], t.inputs['Vector'])
        nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
        if emit:
            nt.links.new(t.outputs['Color'], b.inputs['Emission Color'])
            b.inputs['Emission Strength'].default_value = emit
        if bump:
            bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = bump
            nt.links.new(t.outputs['Color'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
    else:
        b.inputs['Base Color'].default_value = (*color, 1)
    if not tex and bump:
        n = nt.nodes.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value = 40; n.inputs['Detail'].default_value = 8
        bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = bump
        nt.links.new(n.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
    return m


def linen():
    """White table linen with Leonardo's thin blue woven bands and a soft weave."""
    m = bpy.data.materials.new('linen'); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Roughness'].default_value = 0.85
    b.inputs['Subsurface Weight'].default_value = 0.08
    tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(tc.outputs['Object'], sep.inputs['Vector'])
    wave = nt.nodes.new('ShaderNodeTexWave'); wave.wave_type = 'BANDS'; wave.bands_direction = 'X'
    wave.inputs['Scale'].default_value = 0.55; wave.inputs['Distortion'].default_value = 0
    nt.links.new(tc.outputs['Object'], wave.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeValToRGB'); r = ramp.color_ramp
    r.elements[0].position = 0.0; r.elements[0].color = (0.86, 0.85, 0.80, 1)
    r.elements[1].position = 0.982; r.elements[1].color = (0.86, 0.85, 0.80, 1)
    e = r.elements.new(0.985); e.color = (0.16, 0.25, 0.52, 1)
    e = r.elements.new(0.993); e.color = (0.16, 0.25, 0.52, 1)
    nt.links.new(wave.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    n = nt.nodes.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value = 900; n.inputs['Detail'].default_value = 4
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.12
    nt.links.new(n.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
    return m


M = dict(
    stone=mat('stone', (0.20, 0.16, 0.12), 0.85, bump=0.08),
    floor=mat('floor', (0.16, 0.085, 0.05), 0.35, bump=0.04),
    coffer=mat('coffer', (0.30, 0.22, 0.15), 0.7, bump=0.2),
    wood=mat('wood', (0.20, 0.12, 0.07), 0.55, bump=0.1),
    pewter=mat('pewter', (0.55, 0.55, 0.53), 0.28),
    bread=mat('bread', (0.55, 0.33, 0.14), 0.75, bump=0.6),
    wine=mat('wine', (0.35, 0.02, 0.04), 0.05),
    linen=linen(),
    tapestry=mat('tapestry', tex=f'{A}/img/tapestry.png', rough=0.95, bump=0.4),
    landscape=mat('landscape', tex=f'{A}/img/landscape.png', emit=4.0),
)
M['pewter'].node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.9
g = M['wine'].node_tree.nodes['Principled BSDF']
g.inputs['Transmission Weight'].default_value = 0.9
glass = mat('glass', (0.95, 0.97, 0.95), 0.02)
gb = glass.node_tree.nodes['Principled BSDF']; gb.inputs['Transmission Weight'].default_value = 1.0; gb.inputs['IOR'].default_value = 1.45


def box(name, size, loc, material, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object; o.name = name; o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(material)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.uv.cube_project(cube_size=1.0); bpy.ops.object.mode_set(mode='OBJECT')
    return o


def plane(name, w, h, loc, rot, material):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object; o.name = name; o.scale = (w, h, 1)
    bpy.ops.object.transform_apply(scale=True); o.data.materials.append(material)
    return o


# ---------------------------------------------------------------- room
hx = ROOM_W / 2
box('ENV_floor', (ROOM_W, ROOM_D, 0.1), (0, (BACK_Y + FRONT_Y) / 2, -0.05), M['floor'])
# back wall with three window openings (centre large, two flanking)
wins = [(-1.75, 1.0, 1.55, 3.0), (0.0, 1.55, 1.45, 3.35), (1.75, 1.0, 1.55, 3.0)]   # x, w, sill z, top z
xs = [-hx] + sum([[x - w / 2, x + w / 2] for x, w, _, _ in wins], []) + [hx]
for i in range(0, len(xs), 2):
    a, b = xs[i], xs[i + 1]
    box(f'ENV_back_{i}', (b - a, 0.5, ROOM_H), ((a + b) / 2, BACK_Y + 0.25, ROOM_H / 2), M['stone'])
for k, (x, w, s, t) in enumerate(wins):
    box(f'ENV_sill_{k}', (w, 0.5, s), (x, BACK_Y + 0.25, s / 2), M['stone'])
    box(f'ENV_lintel_{k}', (w, 0.5, ROOM_H - t), (x, BACK_Y + 0.25, (t + ROOM_H) / 2), M['stone'])
    box(f'ENV_frame_{k}', (w + 0.16, 0.06, 0.08), (x, BACK_Y - 0.01, t + 0.04), M['wood'])
# the land beyond the windows
plane('ENV_landscape', 34, 13, (0, 22, 3.0), (math.radians(90), 0, 0), M['landscape'])
# side walls, each with four tapestries between pilasters (as in the fresco)
for side in (-1, 1):
    box(f'ENV_side_{side}', (0.4, ROOM_D, ROOM_H), (side * (hx + 0.2), (BACK_Y + FRONT_Y) / 2, ROOM_H / 2), M['stone'])
    for j in range(4):
        y = BACK_Y - 1.4 - j * 2.6
        t = plane(f'ENV_tapestry_{side}_{j}', 1.9, 3.1, (side * (hx - 0.02), y, 2.25), (math.radians(90), 0, math.radians(90 * side)), M['tapestry'])
        box(f'ENV_pilaster_{side}_{j}', (0.12, 0.35, ROOM_H), (side * (hx - 0.06), y - 1.3, ROOM_H / 2), M['stone'])
box('ENV_front', (ROOM_W, 0.4, ROOM_H), (0, FRONT_Y - 0.2, ROOM_H / 2), M['stone'])
# coffered ceiling
box('ENV_ceiling', (ROOM_W, ROOM_D, 0.2), (0, (BACK_Y + FRONT_Y) / 2, ROOM_H + 0.1), M['coffer'])
for i in range(-5, 6):
    box(f'ENV_beamx_{i}', (0.16, ROOM_D, 0.28), (i * 1.0, (BACK_Y + FRONT_Y) / 2, ROOM_H - 0.14), M['wood'])
for j in range(15):
    box(f'ENV_beamy_{j}', (ROOM_W, 0.16, 0.28), (0, FRONT_Y + j * 1.0, ROOM_H - 0.14), M['wood'])

# ---------------------------------------------------------------- table & props
box('SET_tabletop', (TABLE_L, TABLE_D, 0.05), (0, TABLE_Y, TABLE_H), M['wood'])
cloth = box('SET_cloth', (TABLE_L + 0.1, TABLE_D + 0.1, 0.012), (0, TABLE_Y, TABLE_H + 0.03), M['linen'])
box('SET_cloth_front', (TABLE_L + 0.1, 0.012, 0.62), (0, TABLE_Y - TABLE_D / 2 - 0.05, TABLE_H - 0.28), M['linen'])
for x in (-4.4, -2.2, 0, 2.2, 4.4):
    box(f'SET_trestle_{x}', (0.08, 0.7, TABLE_H), (x, TABLE_Y, TABLE_H / 2), M['wood'])

import random
random.seed(7)
top = TABLE_H + 0.04
for i in range(13):
    x = -4.2 + i * 0.7 + random.uniform(-0.1, 0.1)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=0.012, location=(x, TABLE_Y - 0.12, top + 0.006))
    p = bpy.context.object; p.data.materials.append(M['pewter']); p.name = f'PROP_plate_{i}'
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(x + 0.18, TABLE_Y - 0.25, top + 0.03))
    br = bpy.context.object; br.scale = (1, 1, 0.55); br.data.materials.append(M['bread']); br.name = f'PROP_bread_{i}'
    bpy.ops.object.shade_smooth()
    if i % 2 == 0:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.035, depth=0.11, location=(x - 0.2, TABLE_Y - 0.05, top + 0.055))
        gl = bpy.context.object; gl.data.materials.append(glass); gl.name = f'PROP_glass_{i}'
        bpy.ops.mesh.primitive_cylinder_add(radius=0.031, depth=0.05, location=(x - 0.2, TABLE_Y - 0.05, top + 0.03))
        w = bpy.context.object; w.data.materials.append(M['wine']); w.name = f'PROP_wine_{i}'

# ---------------------------------------------------------------- people
def place_glb(name, path, x, height, y=FIG_Y, yaw=0.0):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(f'FIG_{name}', None); sc.collection.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    bpy.context.view_layer.update()
    meshes = [o for o in new if o.type == 'MESH']
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    s = height / (hi.z - lo.z)
    root.scale = (s, s, s)
    root.location = (x - (lo.x + hi.x) / 2 * s, y - (lo.y + hi.y) / 2 * s, -lo.z * s)
    bpy.context.view_layer.update()
    if yaw:
        piv = bpy.data.objects.new(f'PIV_{name}', None); sc.collection.objects.link(piv)
        piv.location = (x, y, 0); root.parent = piv; root.location -= piv.location
        piv.rotation_euler = (0, 0, math.radians(yaw))
    for o in meshes:
        for slot in o.material_slots:
            if slot.material and slot.material.use_nodes:
                bs = slot.material.node_tree.nodes.get('Principled BSDF')
                if bs:
                    bs.inputs['Roughness'].default_value = max(bs.inputs['Roughness'].default_value, 0.6)
                    bs.inputs['Metallic'].default_value = 0.0
    return root


FACE = -90    # the Tripo meshes face +X; turn them to face the viewer (-Y)
FIGS = dict(   # name: (x centre, height of the whole seated group, yaw)
    g1=(-3.55, 1.85, FACE + 6), g2=(-1.6, 1.75, FACE + 3), jesus=(0.0, 1.66, FACE), g3=(1.65, 1.85, FACE - 3), g4=(3.55, 1.75, FACE - 6))
for n, (x, h, yaw) in FIGS.items():
    p = f'{A}/glb/{n}.glb'
    if os.path.exists(p):
        place_glb(n, p, x, h, yaw=yaw)

# ---------------------------------------------------------------- light
w = bpy.data.worlds.new('world'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.05, 0.045, 0.04, 1)
w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.25
# late-afternoon sun from the upper left (Leonardo's light comes from the refectory's real windows on the left)
sun = bpy.data.objects.new('LGT_key', bpy.data.lights.new('LGT_key', 'SUN')); sc.collection.objects.link(sun)
sun.data.energy = 7.0; sun.data.color = (1.0, 0.86, 0.68); sun.data.angle = math.radians(3)
sun.rotation_euler = Vector((-0.28, -0.88, -0.38)).to_track_quat('-Z', 'Y').to_euler()   # low sun behind the windows -> shafts onto the table
# daylight spilling through the back windows
for k, (x, ww, s, t) in enumerate(wins):
    L = bpy.data.objects.new(f'LGT_win_{k}', bpy.data.lights.new(f'LGT_win_{k}', 'AREA')); sc.collection.objects.link(L)
    L.data.shape = 'RECTANGLE'; L.data.size = ww; L.data.size_y = t - s
    L.data.energy = 520 if k == 1 else 260; L.data.color = (0.82, 0.88, 1.0)
    L.location = (x, BACK_Y + 0.1, (s + t) / 2); L.rotation_euler = (math.radians(-90), 0, 0)
# soft warm fill from the viewer's side
F = bpy.data.objects.new('LGT_fill', bpy.data.lights.new('LGT_fill', 'AREA')); sc.collection.objects.link(F)
F.data.size = 6; F.data.energy = 180; F.data.color = (1.0, 0.78, 0.55)
F.location = (0.5, -5.5, 3.6); F.rotation_euler = (math.radians(62), 0, 0)
# a rim on the figures from behind-left
R = bpy.data.objects.new('LGT_rim', bpy.data.lights.new('LGT_rim', 'SPOT')); sc.collection.objects.link(R)
R.data.energy = 900; R.data.spot_size = math.radians(70); R.data.color = (1.0, 0.9, 0.75)
R.location = (-3.0, 4.5, 4.4); R.rotation_euler = (Vector((0, FIG_Y, 1.2)) - R.location).to_track_quat('-Z', 'Y').to_euler()

# a faint haze in the air so the window light reads as shafts
haze = bpy.data.materials.new('haze'); haze.use_nodes = True
hn = haze.node_tree.nodes; hn.remove(hn['Principled BSDF'])
vs = hn.new('ShaderNodeVolumePrincipled'); vs.inputs['Density'].default_value = 0.006
vs.inputs['Color'].default_value = (0.95, 0.88, 0.78, 1)
haze.node_tree.links.new(vs.outputs['Volume'], hn['Material Output'].inputs['Volume'])
hz = box('ENV_haze', (ROOM_W - 0.1, ROOM_D - 0.6, ROOM_H - 0.4), (0, (BACK_Y + FRONT_Y) / 2, ROOM_H / 2), haze)
hz.visible_shadow = False

# ---------------------------------------------------------------- render defaults
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = 40
sc.cycles.use_denoising = True
sc.cycles.max_bounces = 6
sc.view_settings.view_transform = 'AgX'
sc.view_settings.look = 'AgX - High Contrast'
sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
sc.render.fps = 24
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print('saved', OUT, len(bpy.data.objects), 'objects')
