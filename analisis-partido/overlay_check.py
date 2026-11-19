"""Draw the pitch outline from a camera calibration onto one or more 4K frames, to check whether the camera moved.
Usage: python overlay_check.py calib.npz out.jpg frame1.jpg [frame2.jpg ...]   (frames are stacked vertically, downscaled 2x)
"""
import sys, numpy as np
from PIL import Image, ImageDraw
sys.argv_in = list(sys.argv); calib, out, frames = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.argv = ["x", "--calib", calib]; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import pitch_map as pm
L, W = pm.L, pm.W
def poly(pts): return pm.project(pm.P, np.c_[np.array(pts, float), np.zeros(len(pts))])
lines = []
lines.append([(0,0),(L,0),(L,W),(0,W),(0,0)]); lines.append([(L/2,0),(L/2,W)])
t = np.linspace(0, 2*np.pi, 80); lines.append(list(zip(L/2+3*np.cos(t), W/2+3*np.sin(t))))
for gx, s in ((0, 1), (L, -1)):   # penalty area: 6 m quarter circles from each post, joined by a 3 m straight (futsal)
    a = np.linspace(-np.pi/2, np.pi/2, 40) if s == 1 else np.linspace(np.pi/2, 3*np.pi/2, 40)
    arc1 = [(gx+6*np.cos(u), W/2-1.5+6*np.sin(u)) for u in a if (W/2-1.5+6*np.sin(u)) <= W/2-1.5+1e-6]
    arc2 = [(gx+6*np.cos(u), W/2+1.5+6*np.sin(u)) for u in a if (W/2+1.5+6*np.sin(u)) >= W/2+1.5-1e-6]
    lines.append(sorted(arc1+arc2, key=lambda p: p[1]))
    lines.append([(gx, W/2-1.5), (gx, W/2+1.5)])   # goal mouth on the line
ims = [Image.open(f).convert("RGB") for f in frames]
for im, f in zip(ims, frames):
    d = ImageDraw.Draw(im)
    for ln in lines:
        p = poly(ln); d.line([tuple(q) for q in p], fill=(255, 40, 40), width=5)
    d.text((40, 40), f, fill=(255, 255, 0))
w = max(i.width for i in ims)//2; canvas = Image.new("RGB", (w, sum(i.height//2 for i in ims)))
y = 0
for i in ims: j = i.resize((i.width//2, i.height//2)); canvas.paste(j, (0, y)); y += j.height
canvas.save(out, quality=85); print("wrote", out)
