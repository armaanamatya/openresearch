#!/usr/bin/env python3
"""Generate tight, upscaled crops of dense whiteboard regions for legible reading.
Coords are fractional (l,t,r,b) of each portrait full-res frame."""
import os
from PIL import Image

os.makedirs("out/crops", exist_ok=True)
# (name, src, l, t, r, b, scale)
CROPS = [
    ("01_cost_table",     "out/hi/h_000006.jpg", 0.00, 0.18, 0.44, 0.60, 2.6),
    ("02_rubric",         "out/hi/h_000098.jpg", 0.44, 0.42, 0.70, 0.62, 3.0),
    ("03_issues",         "out/hi/h_000098.jpg", 0.60, 0.02, 1.00, 0.20, 2.6),
    ("04_dataloader",     "out/hi/h_000098.jpg", 0.50, 0.14, 0.78, 0.40, 2.8),
    ("05_error_note",     "out/hi/h_000098.jpg", 0.55, 0.26, 1.00, 0.44, 2.6),
    ("06_bes_def_tdd",    "out/hi/h_000167.jpg", 0.16, 0.28, 0.78, 0.46, 2.4),
    ("07_goaltree_sdar",  "out/hi/h_000167.jpg", 0.44, 0.42, 0.80, 0.60, 3.0),
    ("08_forward_back",   "out/hi/h_000167.jpg", 0.40, 0.52, 0.66, 0.76, 3.0),
    ("09_concerns",       "out/hi/h_000167.jpg", 0.08, 0.50, 0.42, 0.76, 2.6),
    ("10_gtm",            "out/hi/h_000167.jpg", 0.76, 0.20, 1.00, 0.62, 2.6),
    ("11_harness_notes",  "out/hi/h_000241.jpg", 0.12, 0.36, 0.44, 0.66, 2.8),
]
for name, src, l, t, r, b, sc in CROPS:
    im = Image.open(src)
    W, H = im.size
    c = im.crop((int(l*W), int(t*H), int(r*W), int(b*H)))
    c = c.resize((int(c.width*sc), int(c.height*sc)), Image.LANCZOS)
    c.save(f"out/crops/{name}.jpg", quality=92)
    print(name, c.size)
