"""Camera helpers shared by the test and render scripts (painting-space coords)."""
import math

import bpy
from mathutils import Vector


def to_bl(p):
    return Vector((p[0], p[2], p[1]))


def set_camera(cam_ob, pos, look, vfov_deg, roll_deg=0.0, shift_tan=0.0, focus=None, fstop=None, aspect=16 / 9):
    """pos/look in painting space; vfov in degrees; shift_tan = vertical lens shift in tan units"""
    cam = cam_ob.data
    p, l = to_bl(pos), to_bl(look)
    d = l - p
    q = d.to_track_quat('-Z', 'Y')
    cam_ob.rotation_mode = 'QUATERNION'
    from mathutils import Quaternion
    q = q @ Quaternion((0, 0, 1), math.radians(roll_deg))
    cam_ob.location = p
    cam_ob.rotation_quaternion = q
    tan_v = math.tan(math.radians(vfov_deg) / 2)
    tan_h = tan_v * aspect
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = 36
    cam.lens = 18 / tan_h
    cam.shift_y = shift_tan / (2 * tan_h)
    if focus is not None and fstop is not None:
        cam.dof.use_dof = True
        cam.dof.focus_distance = max(0.05, focus)
        cam.dof.aperture_fstop = max(0.5, fstop)
    else:
        cam.dof.use_dof = False


def hero(info):
    """Leonardo's own viewpoint: camera at the eye, looking straight in, lens-shifted"""
    O0, O1 = info['O']
    Dp, PW = info['Dp'], info['PW']
    vfov = math.degrees(2 * math.atan(PW / 2 / Dp / (16 / 9)))
    return dict(pos=(O0, O1, 0.0), look=(O0, O1, 1.0), vfov_deg=vfov, shift_tan=-O1 / Dp)
