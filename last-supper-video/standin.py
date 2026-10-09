"""Draw a crude stand-in 'painting' from calib.json, for testing the pipeline
before the real image is available.  usage: python3 standin.py calib.json out.jpg"""
import json
import sys

import cv2
import numpy as np

C = json.load(open(sys.argv[1]))
W, H = 3840, 2000
img = np.zeros((H, W, 3), np.uint8)
P = lambda u, v: (int(u * W), int(v * H))
u0, v0, u1, v1 = C['back']
cv2.fillPoly(img, [np.array([P(0, 0), P(1, 0), P(u1, v0), P(u0, v0)])], (60, 80, 110))      # ceiling
cv2.fillPoly(img, [np.array([P(0, 1), P(1, 1), P(u1, v1), P(u0, v1)])], (70, 100, 130))     # floor
cv2.fillPoly(img, [np.array([P(0, 0), P(u0, v0), P(u0, v1), P(0, 1)])], (40, 60, 80))       # left
cv2.fillPoly(img, [np.array([P(1, 0), P(u1, v0), P(u1, v1), P(1, 1)])], (45, 65, 85))       # right
cv2.rectangle(img, P(u0, v0), P(u1, v1), (90, 120, 150), -1)                                # back
for a, b in ((0.37, 0.42), (0.46, 0.54), (0.58, 0.63)):
    cv2.rectangle(img, P(a, 0.25), P(b, 0.5), (230, 210, 170), -1)                          # windows
for k in range(1, 8):  # coffers / tapestries for parallax cues
    f = k / 8
    cv2.line(img, P(u0 * f, v0 * f), P(u0 * f, 1 - (1 - v1) * f), (20, 30, 40), 6)
    cv2.line(img, P(1 - (1 - u1) * f, v0 * f), P(1 - (1 - u1) * f, 1 - (1 - v1) * f), (20, 30, 40), 6)
poly = [P(u, v) for u, v in C['figuresTop']] + [P(C['figuresTop'][-1][0], C['tableBottom']), P(C['figuresTop'][0][0], C['tableBottom'])]
cv2.fillPoly(img, [np.array(poly)], (50, 70, 140))
rng = np.random.default_rng(3)
for i, (u, v) in enumerate(C['figuresTop'][1:-1]):
    col = tuple(int(c) for c in rng.integers(60, 220, 3))
    cv2.circle(img, P(u, v + 0.05), int(0.025 * W), (120, 160, 210), -1)
    cv2.ellipse(img, P(u, v + 0.2), (int(0.02 * W), int(0.12 * H)), 0, 0, 360, col, -1)
cv2.rectangle(img, P(0.04, C['tableTop']), P(0.96, C['tableBottom']), (225, 225, 220), -1)
for k in range(40):
    cv2.line(img, P(0.04 + k * 0.023, C['tableTop']), P(0.04 + k * 0.023, C['tableBottom']), (180, 180, 175), 3)
img = cv2.GaussianBlur(img, (0, 0), 2)
cv2.imwrite(sys.argv[2], img)
