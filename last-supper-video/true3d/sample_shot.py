"""Sample shot: from the painting's own viewpoint the camera pushes in, skims the tabletop, then circles Judas, Peter and John
(~265 deg, passing behind them) and comes round to find Christ beyond them.

usage: python3 sample_shot.py SET.blend OUTDIR [first last] [--still]
Renders OUTDIR/f####.png, skipping frames that already exist (resumable).
"""
import math, os, sys
import bpy
from mathutils import Vector

ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]   # works as bpy module or `blender -b -P`
blend, outdir = ARGV[0], ARGV[1]
args = [a for a in ARGV[2:] if not a.startswith('--')]
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend))
sc = bpy.context.scene
os.makedirs(outdir, exist_ok=True)

N = 168                                    # 7 s at 24 fps
ENTER = 36                                 # 1.5 s push out of the painting's own viewpoint
sc.frame_start, sc.frame_end = 1, N
C = Vector((-1.6, 0.85, 0.0))              # Judas / Peter / John
CHRIST = Vector((0.0, 0.85, 1.45))
FRONT_POS, FRONT_TGT = Vector((-0.8, -3.4, 1.38)), Vector((-0.8, 0.85, 1.3))


def smooth(t):                             # ease-in-out
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


cam = bpy.data.objects.new('CAM_main', bpy.data.cameras.new('CAM_main'))
sc.collection.objects.link(cam); sc.camera = cam
cd = cam.data
cd.sensor_width = 36
cd.dof.use_dof = True
cd.dof.aperture_fstop = 2.0


def orbit(t):
    s = smooth(t)
    th = math.radians(lerp(228, -38, s))                 # front-left -> behind them -> front-right
    r = 3.0 - 0.6 * math.sin(math.pi * s)
    # skim the table, crane up over Bartholomew's group (they sit right on the orbit), drop in behind Judas
    z = 1.05 + 1.45 * math.exp(-((s - 0.2) / 0.13) ** 2) + 0.45 * math.sin(math.pi * s)
    pos = C + Vector((math.cos(th) * r, math.sin(th) * r, z))
    look_grp = C + Vector((0.05, 0, 1.25))
    w = smooth(max(0.0, (t - 0.7) / 0.3))                # last stretch: find Christ beyond them
    return pos, look_grp.lerp(CHRIST, w), lerp(28, 45, s)


for f in range(1, N + 1):
    if f <= ENTER:
        u = smooth((f - 1) / (ENTER - 1))
        p0, t0, l0 = orbit(0.0)
        pos, target, lens = FRONT_POS.lerp(p0, u), FRONT_TGT.lerp(t0, u), lerp(32, l0, u)
        roll = 0.0
    else:
        t = (f - ENTER) / (N - ENTER)
        pos, target, lens = orbit(t)
        roll = 4 * math.sin(2 * math.pi * smooth(t))
    cam.location = pos
    cam.rotation_euler = (target - pos).to_track_quat('-Z', 'Y').to_euler()
    cam.rotation_euler.y += math.radians(roll)          # a breath of roll
    cd.lens = lens
    cd.dof.focus_distance = (target - pos).length
    cam.keyframe_insert('location', frame=f); cam.keyframe_insert('rotation_euler', frame=f)
    cd.keyframe_insert('lens', frame=f); cd.dof.keyframe_insert('focus_distance', frame=f)

sc.render.use_motion_blur = True
sc.render.motion_blur_shutter = 0.5
if '--preview' in ARGV:
    sc.render.resolution_x, sc.render.resolution_y = 360, 640
    sc.cycles.samples = 12
elif '--still' in ARGV:
    sc.render.resolution_x, sc.render.resolution_y = 1080, 1920
    sc.cycles.samples = 96
else:
    sc.render.resolution_x, sc.render.resolution_y = 720, 1280
    sc.cycles.samples = 24
sc.cycles.use_denoising = True
# speed: haze is added in the grade instead of as a volume; fewer bounces; adaptive sampling
if 'ENV_haze' in bpy.data.objects:
    bpy.data.objects['ENV_haze'].hide_render = True
sc.cycles.max_bounces = 4
sc.cycles.use_adaptive_sampling = True
sc.cycles.adaptive_threshold = 0.03
sc.render.image_settings.file_format = 'PNG'


def use_gpu():
    """Render on the GPU when there is one (OptiX/CUDA on NVIDIA, HIP on AMD, Metal on Apple)."""
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
    except KeyError:
        return None
    for kind in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except TypeError:
            continue
        devs = [d for d in prefs.devices if d.type == kind]
        if devs:
            for d in prefs.devices:
                d.use = d.type == kind
            sc.cycles.device = 'GPU'
            return kind
    return None


print('render device:', use_gpu() or 'CPU', flush=True)
if sc.cycles.device == 'GPU' and '--preview' not in ARGV:
    sc.render.resolution_x, sc.render.resolution_y = 1080, 1920   # a GPU can afford full HD
    sc.cycles.samples = 64

pick = [a for a in ARGV if a.startswith('--frames=')]
if pick:
    frames = [int(x) for x in pick[0].split('=')[1].split(',')]
else:
    first, last = (int(args[0]), int(args[1])) if len(args) >= 2 else (1, N)
    frames = range(first, last + 1)
for f in frames:
    p = os.path.join(outdir, f'f{f:04d}.png')
    if os.path.exists(p):
        continue
    sc.frame_set(f)
    sc.render.filepath = p
    bpy.ops.render.render(write_still=True)
    print('done', f, flush=True)
