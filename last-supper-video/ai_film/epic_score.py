"""Original 40 s orchestral score for "The Hour Before" (AI film cut of the Last Supper).

usage: python3 epic_score.py out.wav

Synthesized from scratch (no samples): piano, string ensemble, choir, horns,
low brass, timpani/taiko, sub booms, cymbal swells and a long hall reverb.
D minor, resolving to D major at the end. Cue points match the edit:
  0.0  wine macro            solo piano over a warm string pad
  4.6  through the window    strings swell
  9.2  the high crane        choir enters
 13.4  hands / money pouch   low string ostinato starts (90 bpm 8ths)
 16.8  Judas                 dark brass + boom
 20.2  Thomas / James        taiko build, riser
 23.5  breath of silence
 24.0  Christ                full drop: choir, horns, timpani
 28.6  crane pull-back       theme climax
 33.5  title                 resolution to D major, long tail
"""
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 44100
DUR = 40.0
N = int(SR * DUR)
rng = np.random.default_rng(1498)


def t_(d):
    return np.arange(int(SR * d)) / SR


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lp(x, fc, order=2):
    b, a = signal.butter(order, min(fc, SR / 2 - 200) / (SR / 2))
    return signal.lfilter(b, a, x)


def hp(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), 'high')
    return signal.lfilter(b, a, x)


def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], 'band')
    return signal.lfilter(b, a, x)


def env(n, a, r, sus=1.0):
    a, r = max(1, int(a * SR)), max(1, int(r * SR))
    e = np.ones(n) * sus
    e[:min(a, n)] = np.linspace(0, sus, a)[:min(a, n)]
    if r < n:
        e[-r:] *= np.linspace(1, 0, r) ** 1.5
    return e


class Mix:
    def __init__(self):
        self.L = np.zeros(N)
        self.R = np.zeros(N)

    def add(self, x, at, gain=1.0, pan=0.0):
        """x mono or (2, n); pan -1..1"""
        i = int(at * SR)
        if i >= N:
            return
        if x.ndim == 1:
            gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            xl, xr = x * gl * 1.414, x * gr * 1.414
        else:
            xl, xr = x
        n = min(len(xl), N - i)
        self.L[i:i + n] += xl[:n] * gain
        self.R[i:i + n] += xr[:n] * gain


# ------------------------------------------------------------------ instruments
def saw_bank(f, d, voices=5, spread=0.012, vib=0.0035, seed=0):
    """detuned saw ensemble, returns stereo"""
    t = t_(d)
    r = np.random.default_rng(seed)
    out = np.zeros((2, len(t)))
    for v in range(voices):
        det = 1 + spread * (v - (voices - 1) / 2) / voices * 2 + r.normal(0, spread / 6)
        vb = 1 + vib * np.sin(2 * np.pi * (4.6 + r.random()) * t + r.random() * 6)
        ph = np.cumsum(f * det * vb) / SR + r.random()
        s = 2 * (ph % 1.0) - 1
        p = (v / max(1, voices - 1)) * 2 - 1
        out[0] += s * np.cos((p + 1) * np.pi / 4)
        out[1] += s * np.sin((p + 1) * np.pi / 4)
    return out / voices


def strings(notes, d, att=1.2, rel=1.5, bright=2500, seed=0):
    out = sum(saw_bank(hz(m), d, 6, 0.010, 0.004, seed + i) for i, m in enumerate(notes))
    e = env(out.shape[1], att, rel)
    out = np.stack([lp(hp(c, 90), bright, 2) for c in out]) * e
    return out / max(1, len(notes)) * 0.9


def choir(notes, d, att=1.0, rel=1.8, vowel='ah', seed=10):
    F = {'ah': [(700, 1.0), (1220, 0.5), (2600, 0.25)], 'oh': [(450, 1.0), (800, 0.5), (2830, 0.15)]}[vowel]
    out = np.zeros((2, int(SR * d)))
    for i, m in enumerate(notes):
        src = saw_bank(hz(m), d, 4, 0.006, 0.006, seed + i)
        for c in range(2):
            v = sum(bp(src[c], f * 0.85, f * 1.15) * g for f, g in F) + lp(src[c], 400) * 0.2
            out[c] += v
    e = env(out.shape[1], att, rel)
    return out * e / max(1, len(notes)) * 1.6


def piano(m, d=3.5, vel=1.0):
    t = t_(d)
    f = hz(m)
    x = np.zeros_like(t)
    for k in range(1, 9):
        fk = f * k * np.sqrt(1 + 0.0004 * k * k)
        x += np.sin(2 * np.pi * fk * t + rng.random()) * (0.6 ** (k - 1)) * np.exp(-t * (0.9 + 0.55 * k))
    x += hp(rng.normal(0, 1, len(t)), 2000) * np.exp(-t * 90) * 0.05
    x *= np.minimum(1, t / 0.004)
    return x * vel * 0.35


def horns(notes, d, att=0.25, rel=1.2, power=1.0, seed=20):
    out = sum(saw_bank(hz(m), d, 3, 0.006, 0.002, seed + i) for i, m in enumerate(notes))
    n = out.shape[1]
    tt = np.arange(n) / SR
    # brassy filter opening
    o = np.zeros_like(out)
    seg = 1024
    for i in range(0, n, seg):
        fc = 500 + 2200 * power * min(1, tt[i] / max(att, 0.05)) * np.exp(-tt[i] * 0.25)
        for c in range(2):
            o[c, i:i + seg] = lp(out[c, max(0, i - 2048):i + seg], fc)[-len(out[c, i:i + seg]):]
    e = env(n, att, rel)
    return np.tanh(o * e * 1.6 / max(1, len(notes)) ** 0.5) * 0.55


def timpani(m, d=2.5, vel=1.0):
    t = t_(d)
    f = hz(m) * (1 + 0.08 * np.exp(-t * 18))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    x += np.sin(2 * np.pi * np.cumsum(f * 1.5) / SR) * np.exp(-t * 4) * 0.3
    x += lp(rng.normal(0, 1, len(t)), 900) * np.exp(-t * 25) * 0.6
    return np.tanh(x * 1.5 * vel) * 0.8


def taiko(vel=1.0, d=1.6):
    t = t_(d)
    f = 62 * (1 + 1.2 * np.exp(-t * 30))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    x += lp(rng.normal(0, 1, len(t)), 1500) * np.exp(-t * 35) * 0.8
    return np.tanh(x * 2.2) * vel * 0.8


def boom(d=4.0, vel=1.0):
    t = t_(d)
    f = 32 + 60 * np.exp(-t * 6)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 0.9)
    x += lp(rng.normal(0, 1, len(t)), 400) * np.exp(-t * 8) * 0.4
    return np.tanh(x * 1.8) * vel


def swell(d=2.0, rev=True):
    t = t_(d)
    n = hp(rng.normal(0, 1, (2, len(t))), 3000)
    e = (t / d) ** 3 if rev else np.exp(-t * 2.5)
    return n * e * 0.25


def riser(d=3.0):
    t = t_(d)
    out = np.zeros((2, len(t)))
    for c in range(2):
        nz = rng.normal(0, 1, len(t))
        o = np.zeros_like(nz)
        seg = 2048
        for i in range(0, len(t), seg):
            p = i / len(t)
            ctr = 300 + 7000 * p ** 2
            o[i:i + seg] = bp(nz[max(0, i - 4096):i + seg], ctr * 0.7, min(ctr * 1.3, 18000))[-len(nz[i:i + seg]):]
        out[c] = o
    f = 220 * 2 ** (t / d * 2)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.15
    return (out * 0.5 + tone) * (t / d) ** 2.5


# ------------------------------------------------------------------ score
mx_dry = Mix()   # drums/booms (less reverb)
mx = Mix()       # orchestra (hall)

Dm, Bb, F, C, Gm, A, D = [38, 50, 53, 57], [34, 46, 50, 53], [41, 48, 53, 57], [36, 48, 52, 55], [43, 50, 55, 58], [45, 49, 52, 57], [38, 50, 54, 57]
Asus = [45, 50, 52, 57]

# 0 - 9.2: piano motif over warm pad
motif = [(0.4, 62, 0.9), (1.15, 65, 0.7), (1.9, 69, 0.75), (2.65, 67, 0.6), (3.4, 65, 0.65), (4.15, 64, 0.55),
         (5.0, 62, 0.8), (5.75, 65, 0.6), (6.5, 70, 0.75), (7.25, 69, 0.65), (8.0, 67, 0.6), (8.6, 65, 0.5)]
for t0, m, v in motif:
    mx.add(piano(m, 4.0, v), t0, 0.9, pan=0.15)
    mx.add(piano(m - 12, 4.0, v * 0.5), t0, 0.5, pan=-0.1)
mx.add(strings([50, 57, 62], 4.9, 2.2, 1.6, 1400, 1), 0.0, 0.55)
mx.add(strings([46, 53, 58, 62], 4.9, 1.6, 1.8, 1900, 2), 4.6, 0.8)
mx.add(swell(1.5), 7.7, 0.6)

# 9.2 - 13.4: choir enters
mx.add(choir([53, 57, 62, 65], 4.6, 1.2, 1.6, 'oh'), 9.2, 1.6)
mx.add(strings([41, 48, 53, 57, 60], 4.6, 0.8, 1.6, 2600, 3), 9.2, 1.3)
mx.add(piano(57, 5.0, 0.6), 9.2, 0.7)
mx.add(piano(65, 5.0, 0.5), 9.9, 0.6)
mx_dry.add(boom(4, 0.5), 9.2, 0.6)

# 13.4 - 20.2: ostinato (90 bpm 8ths), harmony C -> Dm -> Gm
beat = 60 / 90
step = beat / 2
t0 = 13.4
roots = [(13.4, 36), (15.1, 38), (16.8, 43), (18.5, 45)]
k = 0
while t0 < 23.5:
    r = [m for (ts, m) in roots if ts <= t0][-1]
    acc = 1.0 if k % 2 == 0 else 0.6
    note = saw_bank(hz(r + 12), step * 0.9, 3, 0.004, 0, 40 + k)
    note = np.stack([lp(c, 900 + 900 * (t0 > 20.2)) for c in note]) * env(note.shape[1], 0.005, 0.08)
    mx.add(note, t0, 0.55 * acc * (1 + 0.4 * (t0 > 20.2)))
    t0 += step
    k += 1
mx.add(strings([48, 52, 55, 60], 3.6, 0.6, 1.0, 2000, 5), 13.4, 0.6)
mx.add(strings([38, 50, 53, 57], 3.6, 0.4, 1.4, 2200, 6), 15.1, 0.6)

# 16.8: Judas — dark brass + boom
mx.add(horns([31, 38, 43, 46], 3.6, 0.08, 1.6, 1.0), 16.8, 1.0)
mx_dry.add(boom(4.5, 1.0), 16.8, 0.9)
mx_dry.add(timpani(31, 3.0, 1.0), 16.8, 0.8)
mx.add(choir([55, 58, 62], 3.4, 0.5, 1.2, 'ah'), 16.9, 0.55)

# 20.2 - 23.5: Thomas / James — taiko build + riser
for i, tt in enumerate(np.arange(20.2, 23.5, beat / 2)):
    v = 0.5 + 0.5 * (tt - 20.2) / 3.3
    mx_dry.add(taiko(v), tt, 0.75, pan=0.3 if i % 2 else -0.3)
for tt in np.arange(22.2, 23.45, beat / 4):
    mx_dry.add(taiko(0.5 + 0.4 * (tt - 22.2)), tt, 0.5)
mx.add(strings(Asus, 3.3, 0.4, 0.3, 3200, 7), 20.2, 0.8)
mx.add(choir([57, 62, 64, 69], 3.3, 0.6, 0.3, 'ah'), 20.2, 0.6)
mx.add(riser(3.3), 20.2, 0.9)
mx.add(swell(1.2), 22.3, 0.8)

# 24.0: DROP — Christ
mx_dry.add(boom(5.0, 1.2), 24.0, 1.0)
mx_dry.add(timpani(38, 3.5, 1.2), 24.0, 1.0)
mx_dry.add(timpani(26, 3.5, 1.0), 24.0, 0.6)
mx.add(horns([38, 45, 50, 53, 57], 4.8, 0.1, 1.4, 1.0), 24.0, 1.0)
mx.add(choir([62, 65, 69, 74], 4.8, 0.15, 1.4, 'ah'), 24.0, 1.1)
mx.add(strings([38, 50, 57, 62, 65], 4.8, 0.2, 1.4, 4000, 8), 24.0, 0.9)
# theme on high strings/choir
theme = [(24.0, 74, 1.35), (25.35, 77, 0.65), (26.0, 76, 0.65), (26.65, 74, 0.65), (27.3, 72, 1.3)]
for t0, m, d in theme:
    mx.add(strings([m], d + 0.4, 0.08, 0.5, 5000, 30 + m), t0, 0.7)
for tt in (25.35, 26.65, 27.3, 27.95):
    mx_dry.add(taiko(0.7), tt, 0.6)

# 28.6 - 33.5: climax on the pull-back  Bb -> C -> (Asus) -> A
mx.add(horns([34, 46, 53, 58, 62], 2.5, 0.12, 0.9, 1.0), 28.6, 1.0)
mx.add(choir([62, 65, 70, 74], 2.5, 0.2, 0.9, 'ah'), 28.6, 1.0)
mx.add(strings([46, 53, 58, 62, 70], 2.5, 0.2, 0.9, 4500, 11), 28.6, 0.9)
mx.add(horns([36, 48, 55, 60, 64], 2.4, 0.12, 0.9, 1.0), 31.0, 1.0)
mx.add(choir([64, 67, 72, 76], 2.4, 0.2, 0.9, 'ah'), 31.0, 1.0)
mx.add(strings([48, 55, 60, 64, 72], 2.4, 0.2, 0.9, 4500, 12), 31.0, 0.9)
for t0, m, d in [(28.6, 77, 1.2), (29.8, 74, 0.6), (30.4, 72, 0.6), (31.0, 76, 1.2), (32.2, 79, 1.25)]:
    mx.add(strings([m], d + 0.4, 0.06, 0.5, 5000, 50 + m), t0, 0.7)
for tt in np.arange(28.6, 33.3, beat):
    mx_dry.add(taiko(0.8), tt, 0.55)
mx_dry.add(timpani(36, 2.5, 1.0), 31.0, 0.8)
mx.add(swell(1.3), 32.2, 0.9)

# 33.5: resolution to D major, long tail
mx_dry.add(boom(6.0, 1.0), 33.5, 0.9)
mx_dry.add(timpani(38, 4.0, 1.0), 33.5, 0.9)
mx.add(horns([38, 50, 54, 57, 62], 6.5, 0.2, 3.5, 0.8), 33.5, 0.9)
mx.add(choir([62, 66, 69, 74, 78], 6.5, 0.3, 4.0, 'ah'), 33.5, 1.1)
mx.add(strings([38, 50, 57, 62, 66, 69], 6.5, 0.3, 4.0, 3800, 13), 33.5, 1.0)
mx.add(piano(74, 6.0, 0.7), 34.2, 0.8)
mx.add(piano(78, 6.0, 0.6), 34.9, 0.7)
mx.add(piano(81, 6.0, 0.5), 35.6, 0.6)


# ------------------------------------------------------------------ reverb + master
def hall(L, R, rt=3.8, mix=0.38, pre=0.025):
    n = int(SR * rt)
    tt = np.arange(n) / SR
    out = []
    for c, x in enumerate((L, R)):
        ir = rng.normal(0, 1, n) * np.exp(-tt * 6.9 / rt)
        ir = lp(ir, 6500) * (1 - np.exp(-tt * 60))
        ir = np.concatenate([np.zeros(int(pre * SR)), ir])
        ir /= np.sqrt(np.sum(ir ** 2))
        wet = signal.fftconvolve(x, ir)[:len(x)]
        out.append(x * (1 - mix) + wet * mix * 1.6)
    return out


L1, R1 = hall(mx.L, mx.R, 4.2, 0.42)
L2, R2 = hall(mx_dry.L, mx_dry.R, 2.2, 0.22)
L, R = L1 + L2, R1 + R2
# breath of silence before the drop (keep the hall tail very low)
s0, s1 = int(23.5 * SR), int(24.0 * SR)
duck = np.ones(N)
duck[s0:s1] = np.linspace(0.25, 0.04, s1 - s0)
L, R = L * duck, R * duck
L, R = hp(L, 28), hp(R, 28)
# gentle glue compression + soft clip
peak = max(np.abs(L).max(), np.abs(R).max())
L, R = L / peak, R / peak
L, R = np.tanh(L * 1.6) / np.tanh(1.6), np.tanh(R * 1.6) / np.tanh(1.6)
fade = np.ones(N)
fade[:int(0.05 * SR)] = np.linspace(0, 1, int(0.05 * SR))
fade[-int(3.0 * SR):] = np.linspace(1, 0, int(3.0 * SR)) ** 2
L, R = L * fade * 0.9, R * fade * 0.9
wavfile.write(sys.argv[1] if len(sys.argv) > 1 else 'score.wav', SR, (np.stack([L, R], 1) * 32767).astype(np.int16))
print('ok')
