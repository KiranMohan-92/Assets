"""Restoration-style colour enhancement of the fresco scan.

usage: python3 enhance.py in.jpg out.jpg [compare.jpg]

1. soften losses: tiny pale/dark flakes are found with morphological top-hats
   and blended toward a local median
2. light denoise of the remaining mottling
3. recover local contrast (CLAHE on lightness)
4. revive faded pigment: vibrance-style chroma boost (weak colours gain most),
   slight warm white balance, gentle S-curve
5. light unsharp mask
"""
import sys

import cv2
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
img = cv2.imread(src, cv2.IMREAD_COLOR)
H, W = img.shape[:2]
s = W / 4000  # tuned at 4000 px wide

# ---- 1. losses ---------------------------------------------------------------
# soften pale flakes of exposed plaster and dark pits by blending toward a local
# median, weighted by how strongly each pixel stands out; real detail (eyes,
# contours) is larger than the kernel and is left alone
L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
ks = max(3, int(5 * s) | 1)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
white = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)
black = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k)
spot = np.maximum(white, black * 0.6)
wgt = np.clip((spot - 10) / 25, 0, 0.85)
wgt = cv2.GaussianBlur(wgt, (0, 0), 0.8)[..., None]
med = cv2.medianBlur(img, ks).astype(np.float32)
ret = (img.astype(np.float32) * (1 - wgt) + med * wgt).astype(np.uint8)

# ---- 2. denoise (light) --------------------------------------------------------
den = cv2.fastNlMeansDenoisingColored(ret, None, 3, 6, 5, 15)

# ---- 3 + 4. tone and colour ----------------------------------------------------
lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB).astype(np.float32)
Lc = lab[..., 0].astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
Lc = clahe.apply(Lc).astype(np.float32)
Lc = lab[..., 0] * 0.5 + Lc * 0.5
x = Lc / 255.0
x = x + 0.12 * (x - x * x) * (2 * x - 1) * -1  # gentle S-curve around mid grey
x = np.clip((x - 0.03) / 0.95, 0, 1)            # open up blacks/whites slightly
a, b = lab[..., 1] - 128, lab[..., 2] - 128
chroma = np.sqrt(a * a + b * b)
gain = 1.0 + 0.75 * np.exp(-chroma / 22.0) + 0.25  # vibrance: dull colours gain most
a, b = a * gain, b * gain
b += 1.5   # warm the grey plaster a touch
a += 1.0
out = np.dstack([x * 255, np.clip(a + 128, 0, 255), np.clip(b + 128, 0, 255)]).astype(np.uint8)
out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

# ---- 5. sharpen ----------------------------------------------------------------
blur = cv2.GaussianBlur(out, (0, 0), 1.6 * s)
out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)

cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
print('softened', int((wgt > 0.2).mean() * 1000) / 10, '% of pixels as flakes')

if len(sys.argv) > 3:
    def crop(im, x, y, w, h):
        return cv2.resize(im[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)], (800, int(800 * h * H / (w * W))))
    rows = []
    for (x, y, w, h) in [(0.455, 0.40, 0.11, 0.16), (0.25, 0.43, 0.16, 0.2), (0.0, 0.0, 1.0, 1.0)]:
        rows.append(np.hstack([crop(img, x, y, w, h), crop(out, x, y, w, h)]))
    cv2.imwrite(sys.argv[3], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
