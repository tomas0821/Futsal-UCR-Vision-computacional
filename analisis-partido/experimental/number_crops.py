"""Extract torso/back crops per stitched identity from the 4K clip for jersey-number reading.
Usage: python number_crops.py clip_4k.mp4 track_details_stitched.txt out_dir [--top 20] [--min-h 70] [--every 5] [--box-scale 2]
Selects the top-N identities by number of boxes, keeps boxes with 1080p height >= min-h (near half), samples every k-th frame per
identity, saves crops (upper 12–50% of the box, central 80% width, upscaled x2) and an index CSV.
"""
import sys, os, csv, subprocess, collections, numpy as np
from PIL import Image, ImageFilter
clip, trk, out = sys.argv[1], sys.argv[2], sys.argv[3]
TOP = int(sys.argv[sys.argv.index("--top")+1]) if "--top" in sys.argv else 20
SEGS = set(int(x) for x in open(sys.argv[sys.argv.index("--segs")+1]).read().split()) if "--segs" in sys.argv else None   # restrict to these seg_ids
UP = int(sys.argv[sys.argv.index("--upscale")+1]) if "--upscale" in sys.argv else 2
SHARPEN = "--sharpen" in sys.argv
MINH = float(sys.argv[sys.argv.index("--min-h")+1]) if "--min-h" in sys.argv else 70
EVERY = int(sys.argv[sys.argv.index("--every")+1]) if "--every" in sys.argv else 5
BS = float(sys.argv[sys.argv.index("--box-scale")+1]) if "--box-scale" in sys.argv else 2.0
os.makedirs(out, exist_ok=True)
rows = [r for r in csv.DictReader(open(trk)) if r["class_name"] in ("juga","port") and r.get("label")]
cnt = collections.Counter(r["label"] for r in rows); chains = [k for k,_ in cnt.most_common(TOP)] if SEGS is None else list(cnt)
print("identities:", ", ".join(f"{k}({cnt[k]})" for k in chains))
byf = collections.defaultdict(list); last = collections.defaultdict(lambda: -10**9)
for r in rows:
    if r["label"] not in chains or float(r["height"]) < MINH: continue
    if SEGS is not None and int(r.get("seg_id") or 0) not in SEGS: continue
    f = int(r["frame_number"])
    if f - last[r["label"]] < EVERY: continue
    last[r["label"]] = f; byf[f].append((r["label"], float(r["x_center"])*BS, float(r["y_center"])*BS, float(r["width"])*BS, float(r["height"])*BS))
W = int(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width","-of","csv=p=0",clip]).decode()); H = int(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=height","-of","csv=p=0",clip]).decode())
dec = subprocess.Popen(["ffmpeg","-v","error","-i",clip,"-f","rawvideo","-pix_fmt","rgb24","-"], stdout=subprocess.PIPE, bufsize=10**8)
idx = open(f"{out}/index.csv","w",newline=""); w = csv.writer(idx); w.writerow(["file","label","frame","x","y","w","h"]); n=0; saved=0; nb=W*H*3
while True:
    buf = dec.stdout.read(nb)
    if len(buf) < nb: break
    n += 1
    if n not in byf: continue
    fr = np.frombuffer(buf, np.uint8).reshape(H,W,3)
    for lab,x,y,bw,bh in byf[n]:
        x1,x2 = int(x-0.40*bw), int(x+0.40*bw); y1,y2 = int(y-0.5*bh+0.12*bh), int(y-0.5*bh+0.50*bh)
        if x1<0 or y1<0 or x2>W or y2>H or x2-x1<8 or y2-y1<8: continue
        c = Image.fromarray(fr[y1:y2, x1:x2].copy()); c = c.resize((c.width*UP, c.height*UP), Image.LANCZOS)
        if SHARPEN: c = c.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
        fn = f"{lab}_{n:06d}.jpg"; c.save(f"{out}/{fn}", quality=92); w.writerow([fn,lab,n,x1,y1,x2-x1,y2-y1]); saved += 1
    if n % 5000 == 0: print(n, "frames,", saved, "crops", flush=True)
idx.close(); dec.stdout.close(); print("done:", saved, "crops ->", out)
