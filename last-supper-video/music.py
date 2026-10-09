"""Synthesize a 32 s trailer-style score for the Last Supper video.

120 BPM, D minor. Structure (seconds):
  0-8    tension intro: drone, choir pad, ticking clock, heartbeat
  8      BRAAM drop
  8-20   driving section: kick, claps, hats, pulsing bass, toms
  20-24  riser + snare roll, hard cut to silence at 24.0
  24.5   second BRAAM, epic half-time drums + choir until 29
  29-32  tail: choir and reverb ring out under the title
"""
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 44100
DUR = 32.0
N = int(SR * DUR)
BEAT = 0.5
rng = np.random.default_rng(7)


def t_arr(d):
    return np.arange(int(SR * d)) / SR


def place(buf, x, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf):
        return
    n = min(len(x), len(buf) - i)
    buf[i:i + n] += x[:n] * gain


def env_adsr(n, a, d, s, r, sus_len):
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    sl = max(int(sus_len * SR), 0)
    e = np.concatenate([
        np.linspace(0, 1, max(a, 1)),
        np.linspace(1, s, max(d, 1)),
        np.full(sl, s),
        np.linspace(s, 0, max(r, 1)),
    ])
    if len(e) < n:
        e = np.pad(e, (0, n - len(e)))
    return e[:n]


def saw(f, t, phase=0.0):
    return 2 * ((f * t + phase) % 1.0) - 1


def lp(x, fc, order=2):
    b, a = signal.butter(order, min(fc, SR / 2 - 100) / (SR / 2), 'low')
    return signal.lfilter(b, a, x)


def hp(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), 'high')
    return signal.lfilter(b, a, x)


def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], 'band')
    return signal.lfilter(b, a, x)


def note(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------- instruments
def kick(big=False):
    t = t_arr(0.9 if big else 0.45)
    f = 45 + 110 * np.exp(-t * (18 if big else 30))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * (3.5 if big else 7))
    click = hp(rng.standard_normal(len(t)), 2000) * np.exp(-t * 300) * 0.4
    return np.tanh((x + click) * (2.2 if big else 1.6))


def clap():
    t = t_arr(0.4)
    n = rng.standard_normal(len(t))
    e = np.zeros_like(t)
    for off in (0, 0.012, 0.024):
        e += np.where(t >= off, np.exp(-(t - off) * 60), 0)
    e += np.exp(-t * 14) * 0.5
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.5
    return bp(n, 900, 6000) * e * 0.8 + body


def hat(open_=False):
    t = t_arr(0.25 if open_ else 0.06)
    return hp(rng.standard_normal(len(t)), 7000) * np.exp(-t * (14 if open_ else 80)) * 0.35


def tom(f0):
    t = t_arr(0.7)
    f = f0 * (1 + 0.6 * np.exp(-t * 25))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 5)
    x += lp(rng.standard_normal(len(t)), 1200) * np.exp(-t * 20) * 0.3
    return np.tanh(x * 1.8)


def braam(root=26, dur=3.2):
    t = t_arr(dur)
    x = np.zeros_like(t)
    for m, g in ((root, 1.0), (root + 12, 0.8), (root + 19, 0.5), (root + 24, 0.3)):
        for det in (-0.12, 0.0, 0.11):
            x += saw(note(m + det), t, rng.random()) * g
    # filter sweep: open fast, close slowly
    out = np.zeros_like(x)
    seg = 2048
    for i in range(0, len(x), seg):
        tt = i / SR
        fc = 180 + 2600 * np.exp(-tt * 1.6) * min(1, tt * 20 + 0.2)
        out[i:i + seg] = lp(x[max(0, i - 4096):i + seg], fc)[-len(x[i:i + seg]):]
    e = env_adsr(len(t), 0.01, 0.6, 0.55, 1.6, dur - 2.2)
    sub = np.sin(2 * np.pi * note(root - 12) * t) * e * 0.9
    return np.tanh(out * e * 0.5) + sub * 0.6


def choir(chord, dur, bright=1.0):
    """Formant-filtered detuned saws -> 'aah' choir pad."""
    t = t_arr(dur)
    src = np.zeros_like(t)
    vib = 0.004 * np.sin(2 * np.pi * 5.2 * t)
    for m in chord:
        for det in (-0.08, -0.03, 0.02, 0.07):
            f = note(m + det) * (1 + vib)
            ph = np.cumsum(f) / SR + rng.random()
            src += 2 * (ph % 1.0) - 1
    out = (bp(src, 600, 820) * 1.0 + bp(src, 1050, 1350) * 0.55 * bright
           + bp(src, 2400, 2800) * 0.25 * bright)
    out += lp(src, 500) * 0.25
    e = env_adsr(len(t), min(1.2, dur / 3), 0.5, 0.85, min(1.5, dur / 3), dur - 3)
    return out * e / len(chord) * 0.6


def riser(dur):
    t = t_arr(dur)
    n = rng.standard_normal(len(t))
    out = np.zeros_like(t)
    seg = 2048
    for i in range(0, len(t), seg):
        p = i / len(t)
        c = 400 + 9000 * p ** 2
        out[i:i + seg] = bp(n[max(0, i - 4096):i + seg], c * 0.7, min(c * 1.4, 20000))[-len(n[i:i + seg]):]
    f = 110 * 2 ** (t / dur * 3)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    return (out * 0.7 + tone) * (t / dur) ** 2


def whoosh(dur=0.5, rev=False):
    t = t_arr(dur)
    n = rng.standard_normal(len(t))
    e = np.sin(np.pi * t / dur) ** 2
    if rev:
        e = (t / dur) ** 3
    return bp(n, 300, 5000) * e * 0.4


def tick():
    t = t_arr(0.03)
    return bp(rng.standard_normal(len(t)), 2500, 7000) * np.exp(-t * 250) * 0.5


def heartbeat():
    t = t_arr(0.35)
    x = np.sin(2 * np.pi * 50 * t) * np.exp(-t * 18)
    x2 = np.sin(2 * np.pi * 45 * t) * np.exp(-t * 22) * 0.7
    out = np.zeros(int(SR * 0.6))
    place(out, x, 0)
    place(out, x2, 0.2)
    return np.tanh(out * 2)


def bassline(roots, start, end, step=0.125):
    out = np.zeros(N)
    t = t_arr(step * 0.9)
    tm = start
    while tm < end - 1e-6:
        bar = int((tm - start) // 4)
        m = roots[bar % len(roots)]
        x = saw(note(m), t) * 0.6 + np.sin(2 * np.pi * note(m - 12) * t) * 0.8
        x = lp(x, 600) * np.exp(-t * 10)
        # sidechain: duck right after each beat
        duck = 0.35 if abs(tm / BEAT - round(tm / BEAT)) < 1e-6 else 1.0
        place(out, np.tanh(x * 2) * duck, tm, 0.5)
        tm += step
    return out


def reverb(x, rt=2.8, mix=0.3):
    n = int(SR * rt)
    tt = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-tt * 6.9 / rt)
    ir = lp(ir, 5000)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = signal.fftconvolve(x, ir)[:len(x)]
    return x * (1 - mix) + wet * mix * 1.8


# ---------------------------------------------------------------- arrange
drums = np.zeros(N)
music = np.zeros(N)
fx = np.zeros(N)

Dm, Bb, F, C, Gm, A = [50, 53, 57, 62], [46, 50, 53, 58], [45, 48, 53, 57], [48, 52, 55, 60], [43, 50, 55, 58], [45, 49, 52, 57]

# intro 0-8
drone_t = t_arr(8.2)
drone = (np.sin(2 * np.pi * note(26) * drone_t) * 0.5 + lp(saw(note(38), drone_t), 300) * 0.25)
drone *= np.clip(drone_t / 4, 0, 1) * np.clip((8.2 - drone_t) / 0.3, 0, 1)
place(music, drone, 0, 0.7)
place(music, choir(Dm, 4.2, 0.6), 0.0, 0.8)
place(music, choir(Bb, 4.2, 0.8), 4.0, 0.9)
for i in range(32):
    tm = i * 0.25
    place(fx, tick(), tm, 0.25 + 0.5 * tm / 8)
for tm in (0.5, 2.5, 4.5, 6.0, 7.0, 7.5):
    place(drums, heartbeat(), tm, 0.9)
place(fx, riser(2.0), 6.0, 0.6)
place(fx, whoosh(1.0, rev=True), 7.0, 0.8)

# drop at 8
place(music, braam(26, 3.5), 8.0, 1.0)
place(drums, kick(big=True), 8.0, 1.0)
prog = [Dm, Bb, F, C, Dm, Bb, Gm, A]
for i in range(3):
    place(music, choir(prog[i % len(prog)], 4.2, 1.0), 8 + i * 4, 0.6)
music += bassline([38, 34, 41, 36], 8.0, 20.0)

for b in range(int((20 - 8) / BEAT)):
    tm = 8 + b * BEAT
    place(drums, kick(), tm, 0.9)
    if b % 2 == 1:
        place(drums, clap(), tm, 0.55)
    if tm >= 10:
        place(drums, hat(), tm + 0.25, 0.8)
        place(drums, hat(), tm + 0.125, 0.4)
        place(drums, hat(), tm + 0.375, 0.4)
    if b % 8 == 7:
        for k, f0 in enumerate((180, 140, 110, 85)):
            place(drums, tom(f0), tm + k * 0.125, 0.6)

# 20-24 build
music += bassline([38, 34], 20.0, 24.0, step=0.125) * np.r_[np.zeros(int(SR * 20)), np.linspace(0.6, 1.2, N - int(SR * 20))]
place(music, choir(Gm, 2.1, 1.0), 20.0, 0.7)
place(music, choir(A, 2.1, 1.0), 22.0, 0.8)
for b in range(8):
    tm = 20 + b * BEAT
    place(drums, kick(), tm, 0.9)
roll_t = 22.0
while roll_t < 24.0:
    step = 0.125 if roll_t < 23 else 0.0625
    place(drums, clap(), roll_t, 0.25 + 0.35 * (roll_t - 22) / 2)
    roll_t += step
place(fx, riser(4.0), 20.0, 0.9)
# hard silence 24.0 - 24.5
silence_mask = np.ones(N)
s0, s1 = int(24.0 * SR), int(24.5 * SR)
silence_mask[s0:s1] = 0

place(fx, whoosh(0.5, rev=True), 24.0, 1.0)

# 24.5 epic
place(music, braam(26, 4.5), 24.5, 1.2)
place(drums, kick(big=True), 24.5, 1.2)
place(music, choir([50, 57, 62, 65, 69], 7.5, 1.0), 24.5, 0.9)
music += bassline([38, 34], 24.5, 29.0, step=0.25) * 0.8
for b in range(int((29 - 24.5) / BEAT)):
    tm = 24.5 + b * BEAT
    if b % 2 == 0:
        place(drums, kick(big=True), tm, 0.8)
        place(drums, tom(70), tm, 0.5)
    else:
        place(drums, clap(), tm, 0.6)
        place(drums, tom(110), tm, 0.35)
place(music, braam(26, 3.0), 28.5, 0.7)

# tail
place(music, choir([50, 57, 62, 66], 3.6, 0.8), 29.0, 0.7)
place(fx, whoosh(1.2), 28.8, 0.5)

# transition whooshes on main cuts
for tm in (9.6, 11.6, 13.6, 15.6, 17.6, 19.6):
    place(fx, whoosh(0.45), tm, 0.5)

# ---------------------------------------------------------------- mix
mix = reverb(music, 3.2, 0.35) * 0.8 + reverb(drums, 1.2, 0.12) * 0.9 + reverb(fx, 2.0, 0.3) * 0.7
mix *= silence_mask
# keep reverb tail out of the silence but let the 24.5 hit land
mix = hp(mix, 25)
mix = np.tanh(mix * 1.3)
fade = np.ones(N)
fade[-int(SR * 1.5):] = np.linspace(1, 0, int(SR * 1.5)) ** 2
mix *= fade
mix /= np.max(np.abs(mix)) + 1e-9
mix *= 0.89
stereo = np.stack([mix, np.roll(mix, int(SR * 0.0007))], axis=1)
out = sys.argv[1] if len(sys.argv) > 1 else 'score.wav'
wavfile.write(out, SR, (stereo * 32767).astype(np.int16))
print('wrote', out)
