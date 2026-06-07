#!/usr/bin/env python3
"""Pick 'dwell' frames from a handheld whiteboard pan.

Camera moving = motion-blurred + high frame-to-frame diff. Camera paused on a
region = sharp + low diff -> that's where content is readable. We pick, for each
still segment, the sharpest frame; plus gap-fill so coverage never has a hole
longer than GAP seconds.
"""
import sys, glob, json
import numpy as np
from PIL import Image, ImageFilter

paths = sorted(glob.glob("out/frames/f_*.jpg"))
N = len(paths)
sharp = np.zeros(N)
small = np.zeros((N, 90, 160), dtype=np.float32)
for i, p in enumerate(paths):
    im = Image.open(p).convert("L")
    a = np.asarray(im, dtype=np.float32)
    # variance-of-Laplacian sharpness
    lap = (-4 * a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
           + np.roll(a, 1, 1) + np.roll(a, -1, 1))
    sharp[i] = lap[1:-1, 1:-1].var()
    small[i] = np.asarray(im.resize((160, 90)), dtype=np.float32)

diff = np.zeros(N)
for i in range(1, N):
    diff[i] = np.mean(np.abs(small[i] - small[i - 1]))

print("diff percentiles:", {p: round(float(np.percentile(diff[1:], p)), 2)
                            for p in (10, 25, 50, 75, 90)}, file=sys.stderr)
print("sharp percentiles:", {p: round(float(np.percentile(sharp, p)), 1)
                             for p in (10, 50, 90)}, file=sys.stderr)

STILL = float(sys.argv[1]) if len(sys.argv) > 1 else float(np.percentile(diff[1:], 40))
GAP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
print(f"STILL={STILL:.2f} GAP={GAP}", file=sys.stderr)

still = diff < STILL
reps = []
i = 0
while i < N:
    if still[i]:
        j = i
        while j < N and still[j]:
            j += 1
        seg = range(i, j)
        best = max(seg, key=lambda k: sharp[k])
        reps.append(best)
        i = j
    else:
        i += 1

# gap-fill: ensure no hole > GAP seconds (frame index ~= seconds)
filled = []
prev = -GAP
for r in sorted(set(reps)):
    if r - prev > GAP:
        # add sharpest frame in the (prev, r) window
        lo = max(prev + 1, 0)
        if r - 1 > lo:
            g = max(range(lo, r), key=lambda k: sharp[k])
            filled.append(g)
    filled.append(r)
    prev = r
if N - 1 - prev > GAP:
    g = max(range(prev + 1, N), key=lambda k: sharp[k])
    filled.append(g)

reps = sorted(set(filled))

def mmss(s):
    return f"{s//60:02d}:{s%60:02d}"

out = [{"sec": int(r), "ts": mmss(int(r)), "sharp": round(float(sharp[r]), 1)} for r in reps]
print(json.dumps(out))
print(f"\n{len(reps)} dwell frames @ secs: {[int(r) for r in reps]}", file=sys.stderr)
