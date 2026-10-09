# The Last Supper: 3D fly-through

A 32-second cinematic 3D camera piece built from Leonardo's *Last Supper*. The flat painting is rebuilt as a real 3D room (back wall, side walls, coffered ceiling, floor) plus a separate cut-out layer for the table and the thirteen figures. A virtual camera then flies through that scene with real parallax, depth of field, motion blur, light rays, dust, bloom and a film grade, cut to a synthesized 120 BPM trailer score.

## Blender version (true 3D reconstruction)

The newer pipeline lives in `blender/`. Instead of cut-out layers it models the refectory as real geometry and renders it with Cycles: a room shell with three windows, a coffered ceiling, the table and tabletop items, and thirteen separate apostles. The earlier Three.js 2.5D version described below still exists and still builds with `build.sh`.

```bash
# Python 3.13 venv with the bpy module
pip install bpy==5.2.2 numpy scipy opencv-python-headless

python3 blender/figures.py source WORK                             # split + depth the apostles, palettes
python blender/build_scene.py source WORK WORK/lastsupper.blend    # build the scene (bpy python)
python blender/render_anim.py WORK FRAMES --res 1280x720 --samples 6
blender/assemble.sh WORK FRAMES out.mp4 share.mp4                  # post, encode, mux the score
```

`render_anim.py` is resumable (it skips frames that already exist; `--from N --to M` limits the range). `--stills 8.5,13` renders single frames at those times for checking. `assemble.sh` post-processes the frames, encodes 1920x1080 H.264 with the score (generated with `music.py` if `WORK/score.wav` is missing), and optionally makes a two-pass copy under 30 MB.

| file | what it does |
|---|---|
| `blender/geometry.json` | the room calibration: Leonardo's eye, picture plane distance, room box, table position |
| `blender/figures.py` | splits the painting into 13 apostles using the depth map and foreground mask, gives each a metric depth, finds the parts hidden by nearer figures, and writes a blurred, inpainted palette texture per apostle |
| `blender/build_scene.py` | builds the room, windows with a painted landscape, table, tabletop items, apostles with relief, lights and Cycles settings, and saves the `.blend` plus `scene_info.json` |
| `blender/camtools.py` | camera helper: position, look-at, field of view, roll, lens shift and depth of field in painting space |
| `blender/render_anim.py` | the shot list (same 32 s timing as below) as keyframed camera moves at 24 fps, writes `f_XXXX.png` and `timeline.json` (per-frame FX values and the window position for light rays) |
| `blender/post.py` | per-frame finish: light rays, bloom, tone, chromatic aberration, vignette, grain, letterbox and titles |
| `blender/assemble.sh` | runs `post.py`, encodes the video with the score, and makes the optional share copy |

### Colour matching

Everything the painting shows is textured by projecting `last_supper_enhanced.jpg` from Leonardo's own viewpoint, so the hero camera at the end sees exactly the fresco. Surfaces the painting never shows (backs and sides of the apostles, beam sides, the near ceiling) take each object's own blurred palette texture instead, with grazing faces blending toward it. Where no texture applies, the fallback colours are sampled from the painting. The landscape through the windows is painted procedurally from the fresco's own sky blues and hill colours.

### Render time

About 13 s per 720p frame on 4 CPU cores, so roughly 3 hours for the 768 frames.

## Source

`source/last_supper.jpg` is Wikimedia Commons' public-domain scan *Última Cena - Da Vinci 5.jpg* (Leonardo, 1495–1498), with the decorative frieze above the scene cropped off and scaled to 4000 px. `source/last_supper_depth16.png` is a 16-bit monocular depth estimate of it from Depth Anything V2 Large (brighter = nearer). It's used to cut the figures cleanly off the back wall and to sculpt real relief (faces, beards, hands, arms) into them, so the camera can orbit between the apostles.

## Build

```bash
./build.sh                                                   # 1920x1080 from source/
./build.sh source/last_supper.jpg preview.mp4 960 540        # quick preview
```

Requires python3 (numpy, scipy, opencv-python-headless), node 18+, Playwright's Chromium and ffmpeg.

## Files

| file | what it does |
|---|---|
| `calib.json` | perspective calibration in normalized image coords: vanishing point, back-wall rectangle, table edges, the figures' silhouette, and subject positions for the close-ups |
| `enhance.py` | restoration-style colour pass: softens flaked plaster and pits, light denoise, local contrast, revives faded pigment (vibrance), gentle sharpen. `source/enhance_comparison.jpg` shows before and after |
| `prep.py` | solves the room box from the vanishing point, cuts out the figures, inpaints the wall behind them and builds the foreground depth map |
| `music.py` | synthesizes the score (drone, choir, braams, drums, risers, the silence before the second drop) |
| `scene.js` / `index.html` | Three.js scene, the shot list, post FX and titles |
| `render.mjs` | headless frame-accurate renderer piped into ffmpeg (`--stills 8.5,13` renders single frames for checking) |
| `standin.py` | draws a crude stand-in painting for testing without the real image |

## Shot list

| time | shot |
|---|---|
| 0–4 s | the painting floats in darkness; slow push in |
| 4–8 s | through the picture plane, crane over the table |
| 8 s | **drop**: crash zoom into Christ's face |
| 10–18 s | orbits that rotate between the apostles: Bartholomew → James the Less → Andrew; a low push on Judas, Peter and John; Thomas → James the Greater → Philip; a tilted pull-out across Simon, Thaddeus and Matthew |
| 18–22 s | fly over Christ's head into the window light, then rip back out to reveal the room |
| 22–24 s | beat-synced snap zooms on Christ's hand, Judas, John and Thomas's raised finger |
| 24 s | half a second of silence |
| 24.5–29 s | second drop: the camera rotates around Christ from John's side to Thomas's side, then rises back to the full room |
| 29–32 s | settles exactly on Leonardo's viewpoint; title |
