"""Quick look at an image-to-3D GLB: four turntable views in one strip.
usage: python3 preview_glb.py in.glb out.png"""
import sys, math, bpy
from mathutils import Vector
src, out = sys.argv[1], sys.argv[2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr, size = (lo + hi) / 2, max(hi - lo)
print('bbox', tuple(round(v, 3) for v in lo), tuple(round(v, 3) for v in hi))
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.samples = 24; sc.cycles.use_denoising = True
sc.render.resolution_x = sc.render.resolution_y = 360
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[1].default_value = 1.2
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
import numpy as np
tiles = []
for i, a in enumerate([0, 90, 180, 270]):
    r = math.radians(a); d = size * 1.8
    cam.location = ctr + Vector((math.sin(r) * d, -math.cos(r) * d, size * 0.15))
    cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = f'/tmp/pv{i}.png'; bpy.ops.render.render(write_still=True)
from PIL import Image
ims = [Image.open(f'/tmp/pv{i}.png').convert('RGB') for i in range(4)]
strip = Image.new('RGB', (360 * 4, 360)); [strip.paste(im, (360 * i, 0)) for i, im in enumerate(ims)]
strip.save(out, quality=80)
