"""Establishing views of the whole set (sanity check of layout and light)."""
import sys, math, bpy
from mathutils import Vector
bpy.ops.wm.open_mainfile(filepath=sys.argv[1]); sc = bpy.context.scene
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam
sc.render.resolution_x, sc.render.resolution_y = 640, 360; sc.cycles.samples = 12
for i, (pos, tgt, lens) in enumerate([((0, -7.5, 1.7), (0, 0.6, 1.5), 24), ((-4.5, -2.5, 2.6), (0, 1.5, 1.2), 22)]):
    cam.location = pos; cam.data.lens = lens
    cam.rotation_euler = (Vector(tgt) - Vector(pos)).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = f'wide{i}.png'; bpy.ops.render.render(write_still=True)
