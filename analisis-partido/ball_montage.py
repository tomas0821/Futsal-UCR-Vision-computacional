"""Montage of crops around ball fixes for a visual precision check.
Usage: python ball_montage.py clip.mp4 ball_track.csv out.png [--n 48] [--scale 1] [--seed 0]
Crops 96x96 px around each sampled detection (source == det); the clip must be in the same pixel frame as the track (scale
applies to track coords -> clip coords).
"""
import sys, csv, subprocess, numpy as np, random
from PIL import Image, ImageDraw, ImageFont
clip, trk, out = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[sys.argv.index("--n")+1]) if "--n" in sys.argv else 48
scale = float(sys.argv[sys.argv.index("--scale")+1]) if "--scale" in sys.argv else 1.0
seed = int(sys.argv[sys.argv.index("--seed")+1]) if "--seed" in sys.argv else 0
W = int(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width","-of","csv=p=0",clip]).decode().strip()); H = int(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=height","-of","csv=p=0",clip]).decode().strip())
rows = [r for r in csv.DictReader(open(trk)) if r["source"]=="det"]; random.seed(seed); sample = sorted(random.sample(rows, min(N, len(rows))), key=lambda r:int(r["frame_number"]))
want = {int(r["frame_number"]): r for r in sample}; crops = []
dec = subprocess.Popen(["ffmpeg","-v","error","-i",clip,"-f","rawvideo","-pix_fmt","rgb24","-"], stdout=subprocess.PIPE, bufsize=10**8); n=0; nb=W*H*3; R=48
while len(crops) < len(sample):
    buf = dec.stdout.read(nb)
    if len(buf) < nb: break
    n += 1
    if n in want:
        r = want[n]; x, y = float(r["x_px"])*scale, float(r["y_px"])*scale; fr = np.frombuffer(buf, np.uint8).reshape(H,W,3)
        x0, y0 = int(max(0, x-R)), int(max(0, y-R)); c = Image.fromarray(fr[y0:y0+2*R, x0:x0+2*R].copy()).resize((2*R*2, 2*R*2), Image.NEAREST)
        d = ImageDraw.Draw(c); d.rectangle([2*R*2//2-6, 2*R*2//2-6, 2*R*2//2+6, 2*R*2//2+6], outline=(255,0,0), width=1); crops.append((c, f"f{n} c{float(r['confidence']):.2f}"))
dec.stdout.close(); dec.terminate()
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11); cols = 8; S = 4*R
img = Image.new("RGB", (cols*S, ((len(crops)+cols-1)//cols)*(S+14)), (246,246,242)); d = ImageDraw.Draw(img)
for k,(c,lab) in enumerate(crops): cx, cy = (k%cols)*S, (k//cols)*(S+14); img.paste(c, (cx,cy)); d.text((cx+2, cy+S), lab, fill=(60,60,58), font=font)
img.save(out); print("montage", len(crops), "crops ->", out)
