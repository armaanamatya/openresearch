#!/usr/bin/env python3
"""Contact-sheet montage of dwell frames for visual triage."""
import json, glob
from PIL import Image, ImageDraw

reps = json.load(open("out/reps.json"))
paths = sorted(glob.glob("out/frames/f_*.jpg"))
COLS, TW, TH, PAD, LBL = 6, 380, 214, 4, 20
rows = (len(reps) + COLS - 1) // COLS
W = COLS * (TW + PAD) + PAD
H = rows * (TH + LBL + PAD) + PAD
canvas = Image.new("RGB", (W, H), (20, 20, 20))
d = ImageDraw.Draw(canvas)
for idx, r in enumerate(reps):
    sec = r["sec"]
    im = Image.open(paths[sec]).resize((TW, TH))
    c = idx % COLS
    rw = idx // COLS
    x = PAD + c * (TW + PAD)
    y = PAD + rw * (TH + LBL + PAD)
    d.text((x, y), f"{r['ts']} (s{sec})  sh{r['sharp']}", fill=(0, 255, 0))
    canvas.paste(im, (x, y + LBL))
canvas.save("out/montage.png")
print(f"montage: {W}x{H}, {len(reps)} frames")
