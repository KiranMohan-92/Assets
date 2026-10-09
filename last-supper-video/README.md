# The Last Supper: 3D fly-through

A 32-second cinematic 3D camera piece built from Leonardo's *Last Supper*. The flat painting is rebuilt as a real 3D room (back wall, side walls, coffered ceiling, floor) plus a separate cut-out layer for the table and the thirteen figures. A virtual camera then flies through that scene with real parallax, depth of field, motion blur, light rays, dust, bloom and a film grade, cut to a synthesized 120 BPM trailer score.

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
