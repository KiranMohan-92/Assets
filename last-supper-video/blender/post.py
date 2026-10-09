#!/usr/bin/env python3
"""Per-frame finishing compositor for the Last Supper video.

usage: post.py FRAMES_DIR TIMELINE_JSON OUT_DIR [--from N --to M] [--preview f1,f2,...]
Requires only numpy, cv2 (opencv-python-headless) and Pillow.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

FPS = 30
W, H = 1920, 1080

# ---------------------------------------------------------------- fonts
def _first(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

INTER = "/usr/share/fonts/opentype/inter/"
FONT_SANS_MED = _first(INTER + "Inter-Medium.otf", INTER + "Inter-Regular.otf",
                       "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_SANS_XB = _first(INTER + "InterDisplay-ExtraBold.otf", INTER + "Inter-ExtraBold.otf",
                      INTER + "Inter-Bold.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_SERIF = _first("/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
FONT_SERIF_IT = _first("/usr/share/fonts/truetype/crosextra/Caladea-Italic.ttf",
                       "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
                       FONT_SERIF)

_font_cache = {}
def font(path, size):
    key = (path, int(round(size)))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, key[1])
    return _font_cache[key]

# ---------------------------------------------------------------- helpers
def clamp01(x):
    return max(0.0, min(1.0, x))

def ease_out_cubic(x):
    return 1.0 - (1.0 - x) ** 3

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def luma_of(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722

# ---------------------------------------------------------------- effects
def god_rays(img, sun, strength):
    h, w = img.shape[:2]
    hw, hh = w // 2, h // 2
    small = cv2.resize(img, (hw, hh), interpolation=cv2.INTER_AREA)
    m = smoothstep(0.6, 0.9, luma_of(small))[..., None]
    src = small * m
    sx, sy = sun[0] * hw, sun[1] * hh
    n = 48
    acc = np.zeros_like(src)
    wsum = 0.0
    for i in range(n):
        s = 1.0 - 0.45 * i / (n - 1)
        wgt = 0.955 ** i
        # sample position p' = sun + (p - sun) * s  (scale about the sun)
        # warpAffine with WARP_INVERSE_MAP: dst(p) = src(M p)
        M = np.array([[s, 0, sx * (1 - s)], [0, s, sy * (1 - s)]], np.float32)
        warped = cv2.warpAffine(src, M, (hw, hh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        acc += warped * wgt
        wsum += wgt
    acc /= wsum
    acc *= 2.5  # normalised average is faint; gain keeps rays visible
    acc = cv2.GaussianBlur(acc, (0, 0), 2.0)
    acc = cv2.resize(acc, (w, h), interpolation=cv2.INTER_LINEAR)
    tint = np.array([1.0, 0.88, 0.65], np.float32)
    return acc * tint * (0.35 * strength)

def bloom(img):
    h, w = img.shape[:2]
    qw, qh = w // 4, h // 4
    small = cv2.resize(img, (qw, qh), interpolation=cv2.INTER_AREA)
    m = smoothstep(0.7, 1.0, luma_of(small))[..., None]
    b = small * m
    b1 = cv2.GaussianBlur(b, (0, 0), 0.015 * w / 4)
    b2 = cv2.GaussianBlur(b, (0, 0), 0.05 * w / 4)
    out = cv2.resize(b1 + b2, (w, h), interpolation=cv2.INTER_LINEAR)
    return out * 0.35

_maps = {}
def chroma(img, amount):
    h, w = img.shape[:2]
    if "grid" not in _maps:
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        _maps["grid"] = (xs - w / 2, ys - h / 2)
    dx, dy = _maps["grid"]
    maxr = float(np.hypot(w / 2, h / 2))
    s = amount * w / maxr  # fractional radial scale; corner displacement = amount*W px
    out = np.empty_like(img)
    for ch, k in ((0, 1.0 - s), (1, 1.0), (2, 1.0 + s)):
        # R channel sampled slightly inward (appears pushed outward), B outward
        if k == 1.0:
            out[..., ch] = img[..., ch]
            continue
        mx = (dx * k + w / 2).astype(np.float32)
        my = (dy * k + h / 2).astype(np.float32)
        out[..., ch] = cv2.remap(np.ascontiguousarray(img[..., ch]), mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return out

def vignette_map(h, w):
    if "vig" not in _maps:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        u = (xs + 0.5) / w - 0.5
        v = (ys + 0.5) / h - 0.5
        d = np.hypot(u * 1.0, v * 0.85)
        sm = smoothstep(0.95, 0.25, d)
        _maps["vig"] = (0.55 + 0.45 * sm).astype(np.float32)[..., None]
    return _maps["vig"]

# ---------------------------------------------------------------- titles
def spaced_width(text, fnt, spacing_px):
    return sum(fnt.getlength(c) for c in text) + spacing_px * (len(text) - 1)

def draw_spaced(draw, x, baseline, text, fnt, spacing_px, fill):
    for c in text:
        draw.text((x, baseline), c, font=fnt, fill=fill, anchor="ls")
        x += fnt.getlength(c) + spacing_px

def cap_h(fnt):
    b = fnt.getbbox("H", anchor="ls")
    return -b[1]

CAPTIONS = {
    "c1": (1.2, 3.7), "c2": (5.0, 7.6), "c3": (8.25, 9.8),
    "c4": (12.3, 13.8), "c5": (25.2, 28.4), "title": (29.6, 32.5),
}

def draw_caption(name, d, t, fade_in):
    """Draw caption elements into ImageDraw d (mode L). fade_in = eased 0..1 for spacing."""
    ease = ease_out_cubic(fade_in)
    wide = 0.18 + 0.25 * (1 - ease)
    if name in ("c1", "c3", "c5"):
        text = {"c1": "SANTA MARIA DELLE GRAZIE · MILAN · 1498",
                "c3": "THE MOMENT AFTER THE WORDS",
                "c5": "THIRTEEN FIGURES · ONE VANISHING POINT"}[name]
        f = font(FONT_SANS_MED, 0.022 * H)
        sp = wide * f.size
        tw = spaced_width(text, f, sp)
        base = 0.83 * H + cap_h(f) / 2 if name != "c1" else 0.83 * H
        draw_spaced(d, (W - tw) / 2, base, text, f, sp, 255)
    elif name == "c2":
        text = "“One of you will betray me.”"
        f = font(FONT_SERIF_IT, 0.042 * H)
        sp = 0.02 * f.size
        tw = spaced_width(text, f, sp)
        draw_spaced(d, (W - tw) / 2, 0.84 * H, text, f, sp, 255)
    elif name == "title":
        f1 = font(FONT_SERIF, 0.085 * H)
        f2 = font(FONT_SANS_MED, 0.021 * H)
        c1h, c2h = cap_h(f1), cap_h(f2)
        gap_rule, gap_sub = 0.025 * H, 0.02 * H
        total = c1h + gap_rule + 1 + gap_sub + c2h
        top = (H - total) / 2
        t1 = "THE LAST SUPPER"
        sp1 = (0.06 + 0.25 * (1 - ease)) * f1.size
        w1 = spaced_width(t1, f1, sp1)
        draw_spaced(d, (W - w1) / 2, top + c1h, t1, f1, sp1, 255)
        ry = int(round(top + c1h + gap_rule))
        rw = 0.12 * H
        d.rectangle([(W - rw) / 2, ry, (W + rw) / 2, ry], fill=int(255 * 0.7))
        t2 = "LEONARDO DA VINCI · 1495–1498"
        sp2 = 0.45 * f2.size
        w2 = spaced_width(t2, f2, sp2)
        draw_spaced(d, (W - w2) / 2, ry + 1 + gap_sub + c2h, t2, f2, sp2, int(255 * 0.85))

def titles(img, t):
    active = None
    for name, (t0, t1) in CAPTIONS.items():
        if t0 <= t <= t1:
            active = name
            op = min(clamp01((t - t0) / 0.6), clamp01((t1 - t) / 0.5))
            fade_in = clamp01((t - t0) / 0.6)
            break
    if active is None or op <= 0:
        return img
    if active == "c2":
        fade_in = 1.0  # fixed spacing, but blur still follows fade-in
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    if active == "c4":
        draw_c4(d, fade_in)
    else:
        draw_caption(active, d, t, fade_in)
    mask = np.asarray(layer, np.float32) / 255.0
    ease = ease_out_cubic(clamp01((t - CAPTIONS[active][0]) / 0.6))
    blur = (1 - ease) * 12.0
    if blur > 0.3:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    mask *= op
    glow = cv2.GaussianBlur(mask, (0, 0), 0.008 * H)
    glow = np.clip(glow * 1.8, 0, 1)
    out = img * (1 - 0.7 * glow[..., None])
    color = np.array([243, 233, 214], np.float32) / 255.0
    m = mask[..., None]
    return out * (1 - m) + color * m

def draw_c4(d, fade_in):
    x = 0.07 * W
    big = font(FONT_SANS_XB, 0.075 * H)
    small = font(FONT_SANS_MED, 0.02 * H)
    base_big = 0.82 * H
    draw_spaced(d, x, base_big, "JUDAS", big, 0.02 * big.size, 255)
    base_small = base_big - cap_h(big) - 0.022 * H
    draw_spaced(d, x, base_small, "CLUTCHING THE SILVER", small, 0.3 * small.size, 204)

# ---------------------------------------------------------------- pipeline
def process(img, e, frame):
    t = e.get("t", frame / FPS)
    h, w = img.shape[:2]
    sun = e.get("sun")
    rays = float(e.get("rays", 0) or 0)
    if sun is not None and rays > 0:
        img = img + god_rays(img, sun, rays)
    img = img + bloom(img)
    img = img * float(e.get("exposure", 1.0))
    img = img / (1.0 + 0.22 * np.maximum(img - 0.75, 0.0))
    # split tone
    l = np.clip(luma_of(img), 0, 1)[..., None]
    teal = np.array([0.86, 1.0, 1.10], np.float32)
    warm = np.array([1.10, 1.0, 0.86], np.float32)
    img = img * (1 + (teal - 1) * ((1 - l) * 0.30)) * (1 + (warm - 1) * (l * 0.25))
    img = (img - 0.5) * 1.06 + 0.5
    img = np.clip(img, 0, None)
    # chromatic aberration
    amt = 0.0006 + 0.004 * float(e.get("ca", 0))
    img = chroma(img.astype(np.float32), amt)
    img = img * vignette_map(h, w)
    fl = float(e.get("flash", 0))
    if fl > 0:
        img = img * (1 - fl) + np.array([1.0, 0.97, 0.9], np.float32) * fl
    fd = float(e.get("fade", 0))
    if fd > 0:
        img = img * (1 - fd)
    rng = np.random.default_rng(1000003 + frame)
    img = img + rng.standard_normal((h, w, 1), dtype=np.float32) * 0.022
    # letterbox
    half = (0.5 + (0.372 - 0.5) * float(e.get("lb", 0))) * h
    ys = np.abs(np.arange(h) + 0.5 - h / 2)
    img[ys > half] = 0.0
    img = titles(img, t)
    return np.clip(img, 0, 1)

_ctx = {}
def init_worker(frames_dir, timeline, out_dir):
    cv2.setNumThreads(1)
    _ctx.update(frames=frames_dir, tl=timeline, out=out_dir)

def work(frame):
    path = os.path.join(_ctx["frames"], "f_%04d.png" % frame)
    if not os.path.exists(path):
        return frame, False
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tl = _ctx["tl"]
    e = tl[frame] if frame < len(tl) else {"t": frame / FPS}
    out = process(img, e, frame)
    o8 = (out * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(_ctx["out"], "f_%04d.jpg" % frame),
                cv2.cvtColor(o8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return frame, True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("timeline")
    ap.add_argument("out_dir")
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", dest="end", type=int, default=None, help="inclusive")
    ap.add_argument("--preview", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with open(a.timeline) as f:
        tl = json.load(f)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.preview:
        frames = [int(x) for x in a.preview.split(",") if x.strip()]
    else:
        end = a.end if a.end is not None else len(tl) - 1
        frames = list(range(a.start, end + 1))
    done = 0
    missing = []
    with Pool(a.workers, initializer=init_worker, initargs=(a.frames_dir, tl, a.out_dir)) as p:
        for fr, ok in p.imap_unordered(work, frames):
            done += 1
            if not ok:
                missing.append(fr)
            if done % 20 == 0 or done == len(frames):
                print("[post] %d/%d" % (done, len(frames)), flush=True)
    if missing:
        print("[post] missing input frames: %s" % sorted(missing)[:20], file=sys.stderr)

if __name__ == "__main__":
    main()
