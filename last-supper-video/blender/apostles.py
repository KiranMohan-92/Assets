"""Staging of the thirteen figures, read off the fresco.

All positions are normalized image coords (u right, v down) of the 4000-px-wide
source; depths are metres from the eye plane (the table front is at Z = 6.3, its
back edge at 7.15).  Hands are either ('table', u, v): resting on the tabletop at
that image point, ('air', u, v, z): free in space, or None: hanging/relaxed.
Colours are image points to sample (tunic, mantle, hair, skin), so the people wear
exactly the fresco's palette.

Angles are degrees, positive toward the viewer's right: `face` turns the whole
body, `twist` the upper body on top of that, `lean_side` bends sideways, `tilt`
rolls the head.  `lean_fwd` > 0 bends toward the table.  `head` is the image
point of the face and `z` its depth; each person is scaled and placed so the face
lands exactly there.
"""

APOSTLES = [
    dict(name='bartholomew', head=(0.157, 0.483), z=7.30, sit=False, lean_fwd=28, lean_side=0, twist=-35, face=60,
         look='christ', l_hand=('table', 0.140, 0.672), r_hand=('table', 0.075, 0.676),
         age=0.45, hair='short', beard='short', mantle_over='L',
         tunic=(0.070, 0.560), mantle=(0.110, 0.530), hair_c=(0.150, 0.465), skin=(0.163, 0.493)),
    dict(name='jamesMinor', head=(0.190, 0.480), z=7.50, sit=True, lean_fwd=10, lean_side=-6, twist=-20, face=55,
         look='christ', l_hand=('air', 0.250, 0.545, 7.30), r_hand=None,
         age=0.35, hair='long', beard='short', mantle_over=None,
         tunic=(0.195, 0.570), mantle=None, hair_c=(0.183, 0.470), skin=(0.199, 0.492)),
    dict(name='andrew', head=(0.231, 0.490), z=7.45, sit=True, lean_fwd=0, lean_side=0, twist=-15, face=40,
         look='christ', l_hand=('air', 0.270, 0.565, 7.15), r_hand=('air', 0.208, 0.560, 7.15),
         age=0.95, hair='bald', beard='full', mantle_over='R',
         tunic=(0.236, 0.585), mantle=(0.258, 0.540), hair_c=(0.245, 0.515), skin=(0.232, 0.488)),
    dict(name='judas', head=(0.305, 0.530), z=7.35, sit=True, lean_fwd=8, lean_side=8, twist=20, face=30,
         look='christ', l_hand=('table', 0.300, 0.655), r_hand=('table', 0.268, 0.640),
         age=0.5, hair='short', beard='full', mantle_over='L', skin_dark=True,
         tunic=(0.290, 0.600), mantle=(0.315, 0.585), hair_c=(0.300, 0.515), skin=(0.308, 0.540)),
    dict(name='peter', head=(0.328, 0.508), z=7.50, sit=True, lean_fwd=22, lean_side=6, twist=10, face=50,
         look='john', l_hand=('air', 0.375, 0.560, 7.40), r_hand=('air', 0.272, 0.600, 7.55),
         age=0.85, hair='short', beard='full', mantle_over='R',
         tunic=(0.268, 0.560), mantle=(0.335, 0.590), hair_c=(0.322, 0.494), skin=(0.336, 0.512)),
    dict(name='john', head=(0.365, 0.516), z=7.45, sit=True, lean_fwd=5, lean_side=-18, twist=-5, face=-30, tilt=-22,
         look='table', l_hand=('table', 0.420, 0.660), r_hand=('table', 0.405, 0.662),
         age=0.05, hair='long', beard='none', mantle_over='L',
         tunic=(0.360, 0.572), mantle=(0.405, 0.580), hair_c=(0.358, 0.500), skin=(0.368, 0.522)),
    dict(name='christ', head=(0.511, 0.480), z=7.55, sit=True, lean_fwd=4, lean_side=-3, twist=0, face=-6, tilt=-8,
         look='table', l_hand=('table', 0.566, 0.660), r_hand=('table', 0.438, 0.664),
         age=0.35, hair='long', beard='short', mantle_over='L',
         tunic=(0.470, 0.575), mantle=(0.540, 0.585), hair_c=(0.492, 0.500), skin=(0.511, 0.490)),
    dict(name='thomas', head=(0.589, 0.473), z=7.95, sit=False, lean_fwd=12, lean_side=-5, twist=-20, face=-45,
         look='christ', l_hand=None, r_hand=('air', 0.566, 0.452, 7.75),
         age=0.4, hair='short', beard='short', mantle_over=None,
         tunic=(0.590, 0.520), mantle=None, hair_c=(0.583, 0.462), skin=(0.590, 0.480)),
    dict(name='jamesMajor', head=(0.609, 0.497), z=7.55, sit=True, lean_fwd=-6, lean_side=0, twist=-5, face=-25, tilt=-6,
         look='table', l_hand=('air', 0.668, 0.640, 7.25), r_hand=('air', 0.553, 0.545, 7.30),
         age=0.5, hair='long', beard='full', mantle_over=None,
         tunic=(0.615, 0.585), mantle=None, hair_c=(0.598, 0.488), skin=(0.611, 0.505)),
    dict(name='philip', head=(0.646, 0.460), z=7.40, sit=False, lean_fwd=24, lean_side=-8, twist=-25, face=-40, tilt=8,
         look='christ', l_hand=('air', 0.660, 0.560, 7.15), r_hand=('air', 0.640, 0.548, 7.12),
         age=0.1, hair='short', beard='none', mantle_over='R',
         tunic=(0.668, 0.540), mantle=(0.690, 0.600), hair_c=(0.640, 0.452), skin=(0.646, 0.472)),
    dict(name='matthew', head=(0.797, 0.477), z=7.40, sit=True, lean_fwd=15, lean_side=-15, twist=-10, face=-20,
         look='thaddeus', l_hand=('air', 0.745, 0.585, 7.10), r_hand=('air', 0.725, 0.560, 7.15),
         age=0.1, hair='short', beard='none', mantle_over=None,
         tunic=(0.765, 0.565), mantle=None, hair_c=(0.790, 0.460), skin=(0.800, 0.484)),
    dict(name='thaddeus', head=(0.862, 0.483), z=7.60, sit=True, lean_fwd=8, lean_side=0, twist=15, face=40,
         look='simon', l_hand=('air', 0.835, 0.575, 7.35), r_hand=('table', 0.845, 0.650),
         age=0.9, hair='short', beard='full', mantle_over=None,
         tunic=(0.840, 0.545), mantle=None, hair_c=(0.858, 0.468), skin=(0.865, 0.490)),
    dict(name='simon', head=(0.903, 0.493), z=7.40, sit=True, lean_fwd=6, lean_side=0, twist=30, face=-70,
         look='thaddeus', l_hand=('air', 0.880, 0.592, 7.20), r_hand=('air', 0.860, 0.590, 7.20),
         age=0.85, hair='bald', beard='full', mantle_over='L',
         tunic=(0.890, 0.590), mantle=(0.945, 0.575), hair_c=(0.880, 0.515), skin=(0.902, 0.495)),
]
