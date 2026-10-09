"""
Smoke test / demo for mh_human.py.  Run with the bpy venv:

    python test_mh_human.py [out_dir]

Builds three clothed men around a table box and renders (Cycles CPU, 960x540,
24 samples, denoised):
  * test.png        - the group, 3/4 front view
  * face_closeup.png - the bald bearded old man's face
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from mathutils import Vector

import mh_human as mh

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    '/tmp/claude-0/-home-user-Assets/eedd8701-2857-5284-a79c-1d037274266b/scratchpad/mhtest'
os.makedirs(OUT, exist_ok=True)

TABLE_Z = 0.76        # table-top height (about seated elbow height)
BENCH_Z = 0.46


def mat(name, rgb, rough=0.8):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = tuple(rgb) + (1,)
    b.inputs['Roughness'].default_value = rough
    return m


def box(name, loc, size, material):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = size
    o.data.materials.append(material)
    return o


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def area_light(name, loc, target, energy, size):
    d = bpy.data.lights.new(name, 'AREA')
    d.energy = energy
    d.size = size
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    look_at(o, target)
    return o


def setup_scene(res=(960, 540), samples=24):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.exposure = -0.2
    w = bpy.data.worlds.new('World')
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.5, 0.55, 0.6, 1)
    w.node_tree.nodes['Background'].inputs[1].default_value = 0.18
    return sc


def build_scene(sc):
    wall = mat('wall', (0.38, 0.33, 0.27), 0.95)
    wood = mat('wood', (0.18, 0.09, 0.04), 0.6)
    cloth = mat('cloth', (0.45, 0.40, 0.33), 0.95)
    box('floor', (0, 0, -0.05), (6, 5, 0.05), mat('floor', (0.25, 0.2, 0.15), 0.9))
    box('wall', (0, 2.4, 1.6), (6, 0.05, 1.7), wall)
    box('table', (0, -0.05, TABLE_Z - 0.03), (1.75, 0.42, 0.025), wood)
    box('cloth', (0, -0.05, TABLE_Z - 0.35), (1.72, 0.40, 0.33), cloth)
    for x in (-1.05, 0.0):
        box('bench%+.1f' % x, (x, 0.64, BENCH_Z - 0.025), (0.5, 0.2, 0.025), wood)

    people = []
    timings = []

    # ---- 1. old, bald, full beard: seated, right hand raised ---------------
    t = time.time()
    a = mh.build_human('Old', gender=1, age=1.0, muscle=0.4, weight=0.55, face_seed=11,
                       height_m=1.72, skin_rgb=(0.62, 0.43, 0.33))
    mh.place_human(a, (-1.05, 0.64, 0.0), 8)
    body = mh.pose_human(a, sit=True, lean_fwd=6, twist=-6, head_target=(-0.2, -2.0, 1.3),
                         head_tilt=4, r_hand=(-1.35, 0.1, 1.38), r_elbow_pole=(-1.55, 0.7, 1.0),
                         l_hand=(-0.82, 0.05, TABLE_Z + 0.035), r_palm='forward')
    mh.add_robe(body, tunic_rgb=(0.36, 0.22, 0.12), mantle_rgb=(0.10, 0.17, 0.30), mantle_over='L')
    mh.add_hair(body, hair_rgb=(0.52, 0.5, 0.48), style='bald', beard='full')
    timings.append(time.time() - t)
    people.append(body)

    # ---- 2. young, long hair, no beard: seated leaning on the table --------
    t = time.time()
    b = mh.build_human('Young', gender=1, age=0.0, muscle=0.5, weight=0.45, face_seed=5,
                       height_m=1.78, skin_rgb=(0.72, 0.52, 0.40))
    mh.place_human(b, (0.0, 0.64, 0.0), -4)
    body = mh.pose_human(b, sit=True, lean_fwd=22, head_target=(0.5, -2.0, 1.25), head_tilt=-6,
                         l_hand=(0.18, 0.08, TABLE_Z + 0.035), r_hand=(-0.20, 0.10, TABLE_Z + 0.035),
                         l_elbow_pole=(0.7, 0.6, 0.85), r_elbow_pole=(-0.7, 0.6, 0.85))
    mh.add_robe(body, tunic_rgb=(0.55, 0.12, 0.10), mantle_rgb=(0.12, 0.28, 0.46), mantle_over='R')
    mh.add_hair(body, hair_rgb=(0.20, 0.10, 0.05), style='long', beard='none')
    timings.append(time.time() - t)
    people.append(body)

    # ---- 3. middle aged, short hair & short beard: standing, leaning -------
    t = time.time()
    c = mh.build_human('Mid', gender=1, age=0.45, muscle=0.6, weight=0.55, face_seed=23,
                       height_m=1.74, skin_rgb=(0.66, 0.47, 0.36))
    mh.place_human(c, (1.05, 0.80, 0.0), -10)
    body = mh.pose_human(c, sit=False, lean_fwd=52, head_target=(0.2, -2.0, 1.3),
                         l_hand=(1.25, 0.22, TABLE_Z + 0.035), r_hand=(0.88, 0.26, TABLE_Z + 0.035))
    mh.add_robe(body, tunic_rgb=(0.28, 0.33, 0.16))
    mh.add_hair(body, hair_rgb=(0.10, 0.06, 0.04), style='short', beard='short')
    timings.append(time.time() - t)
    people.append(body)
    print('build+pose+clothe per person (s):', ['%.2f' % x for x in timings])
    return people


def main():
    t_all = time.time()
    sc = setup_scene()
    people = build_scene(sc)

    # ---- camera and lights ---------------------------------------------------
    cam_d = bpy.data.cameras.new('cam')
    cam_d.lens = 42
    cam = bpy.data.objects.new('cam', cam_d)
    sc.collection.objects.link(cam)
    cam.location = (1.5, -3.3, 1.5)
    look_at(cam, (0.0, 0.5, 1.12))
    sc.camera = cam
    area_light('key', (-2.5, -3.0, 3.4), (0, 0.4, 1.1), 650, 2.5)
    area_light('fill', (3.5, -2.5, 1.8), (0, 0.4, 1.1), 160, 3.0)
    area_light('rim', (0.5, 3.0, 3.0), (0, 0.4, 1.2), 300, 2.0)

    sc.render.filepath = os.path.join(OUT, 'test.png')
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print('render test.png: %.1fs' % (time.time() - t))

    # ---- face close-up of the old man ------------------------------------------
    sc.render.resolution_x, sc.render.resolution_y = 720, 720
    head = people[0]
    face_c = Vector((-1.05, 0.64, 0.0)) + Vector((0.0, 0.0, 1.30))
    cam.data.lens = 85
    cam.location = (-0.85, -0.45, 1.30)
    look_at(cam, (-1.04, 0.64, 1.19))
    sc.view_settings.exposure = -0.2
    for o in list(bpy.data.objects):
        if o.type == 'LIGHT':
            o.data.energy *= 0.45
    area_light('closekey', (-0.2, -1.6, 2.0), (-1.05, 0.6, 1.25), 90, 1.0)
    sc.render.filepath = os.path.join(OUT, 'face_closeup.png')
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print('render face_closeup.png: %.1fs' % (time.time() - t))
    print('total %.1fs' % (time.time() - t_all))


if __name__ == '__main__':
    main()
