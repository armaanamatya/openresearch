#!/usr/bin/env python3
"""Collapse 1fps frames into distinct visual states (slides/diagrams).

Anchor-compare strategy: a frame stays in the current state while its perceptual
hash distance from the STATE ANCHOR (first frame of the state) is <= threshold.
Anchor-compare (vs prev-frame compare) catches slow/gradual diagram drift, since
accumulated change eventually exceeds the threshold and opens a new state.

Representative frame for each state = LAST frame of the run (most complete view
of a progressively drawn diagram).
"""
import sys, os, json, glob
from PIL import Image
import numpy as np

FRAMES_DIR = sys.argv[1] if len(sys.argv) > 1 else "out/frames"

def dhash(img, size=16):
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()  # size*size bits

def ahash(img, size=12):
    g = img.convert("L").resize((size, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a > a.mean()).flatten()

paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "f_*.jpg")))
print(f"loaded {len(paths)} frames", file=sys.stderr)
dh, ah = [], []
for p in paths:
    im = Image.open(p)
    dh.append(dhash(im)); ah.append(ahash(im))
dh = np.array(dh); ah = np.array(ah)

def states_for(thresh_d, thresh_a):
    states = []
    anchor = 0
    for i in range(1, len(paths)):
        dd = np.count_nonzero(dh[i] != dh[anchor])
        da = np.count_nonzero(ah[i] != ah[anchor])
        if dd > thresh_d or da > thresh_a:
            states.append((anchor, i - 1))
            anchor = i
    states.append((anchor, len(paths) - 1))
    return states

# Preview state counts across thresholds so we can pick sensitivity.
print("threshold preview (dhash256 / ahash144):", file=sys.stderr)
for td, ta in [(40, 30), (32, 24), (24, 18), (18, 14), (14, 10)]:
    st = states_for(td, ta)
    print(f"  d>{td:>3} a>{ta:>3}  -> {len(st):>3} states", file=sys.stderr)

# Chosen thresholds (override via argv 2,3)
TD = int(sys.argv[2]) if len(sys.argv) > 2 else 24
TA = int(sys.argv[3]) if len(sys.argv) > 3 else 18
st = states_for(TD, TA)

def mmss(s):
    return f"{s//60:02d}:{s%60:02d}"

out = []
for k, (a, b) in enumerate(st):
    rep = b  # last frame of the run = most complete
    rep_sec = rep  # frame index ~= seconds (fps=1, 0-based)
    out.append({
        "state": k + 1,
        "start_sec": a, "end_sec": b, "dur_sec": b - a + 1,
        "start": mmss(a), "end": mmss(b),
        "rep_frame": os.path.basename(paths[rep]),
        "rep_sec": rep_sec, "rep_ts": mmss(rep_sec),
    })
print(json.dumps(out, indent=2))
print(f"\nCHOSEN td>{TD} ta>{TA}: {len(out)} distinct states", file=sys.stderr)
