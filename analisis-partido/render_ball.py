"""Draw a ball track (from ball_track.py) onto a video. Green circle = detected, amber dashed = interpolated, short trail.
Usage: python render_ball.py in.mp4 ball_track.csv out.mp4 [--scale 1] [--out-width 1920] [--boxes track_details_stitched.txt --box-scale 0.5]
scale: track px -> input px; box-scale: box px -> input px (people boxes optional).
"""
import sys, csv, subprocess, collections, numpy as np
from PIL import Image, ImageDraw, ImageFont
src, trk, out = sys.argv[1], sys.argv[2], sys.argv[3]
scale = float(sys.argv[sys.argv.index("--scale")+1]) if "--scale" in sys.argv else 1.0
OW = int(sys.argv[sys.argv.index("--out-width")+1]) if "--out-width" in sys.argv else 1920
boxes = sys.argv[sys.argv.index("--boxes")+1] if "--boxes" in sys.argv else None
bscale = float(sys.argv[sys.argv.index("--box-scale")+1]) if "--box-scale" in sys.argv else 1.0
caption = sys.argv[sys.argv.index("--caption")+1] if "--caption" in sys.argv else None  # optional top banner
t0 = float(sys.argv[sys.argv.index("--t0")+1]) if "--t0" in sys.argv else None  # match clock offset (s) shown in the banner
W = int(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width","-of","csv=p=0",src]).decode()); H = int(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=height","-of","csv=p=0",src]).decode())
fps = subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=r_frame_rate","-of","csv=p=0",src]).decode().strip()
OH = int(H*OW/W)//2*2; s = OW/W
ball = {int(r["frame_number"]): (float(r["x_px"])*scale*s, float(r["y_px"])*scale*s, r["source"], r["tracklet"]) for r in csv.DictReader(open(trk))}
byf = collections.defaultdict(list)
if boxes:
    for r in csv.DictReader(open(boxes)):
        if r["class_name"] in ("juga","port","ref"): byf[int(r["frame_number"])].append((r.get("label") or r["class_name"], float(r["x_center"])*bscale*s, float(r["y_center"])*bscale*s, float(r["width"])*bscale*s, float(r["height"])*bscale*s))
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
bfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
FPS = eval(fps)
dec = subprocess.Popen(["ffmpeg","-v","error","-i",src,"-vf",f"scale={OW}:{OH}","-f","rawvideo","-pix_fmt","rgb24","-"], stdout=subprocess.PIPE, bufsize=10**8)
enc = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{OW}x{OH}","-r",fps,"-i","-","-c:v","libx264","-preset","fast","-crf","23","-pix_fmt","yuv420p","-movflags","+faststart",out], stdin=subprocess.PIPE)
n = 0; nb = OW*OH*3
while True:
    buf = dec.stdout.read(nb)
    if len(buf) < nb: break
    n += 1; im = Image.frombuffer("RGB",(OW,OH),buf,"raw","RGB",0,1).copy(); d = ImageDraw.Draw(im)
    for lab,x,y,w,h in byf.get(n,[]):
        c = (31,58,147) if lab.startswith(("A","SR")) else (225,225,215) if lab.startswith(("B","GR")) else (235,104,52) if lab.startswith("GK") else (255,215,0)
        d.rectangle([x-w/2,y-h/2,x+w/2,y+h/2],outline=c,width=1); tw=d.textlength(lab,font=font); d.rectangle([x-w/2,y-h/2-19,x-w/2+tw+6,y-h/2-2],fill=c); d.text((x-w/2+3,y-h/2-19),lab,fill=(255,255,255) if lab.startswith(("A","SR","GK")) else (0,0,0),font=font)
    tr = [ball[k] for k in range(n-25, n) if k in ball and ball[k][3]==ball.get(n,(0,0,0,None))[3]]
    if len(tr) > 1: d.line([(p[0],p[1]) for p in tr], fill=(0,200,0), width=2)
    if n in ball:
        x,y,srcty,_ = ball[n]; r = 14
        if srcty == "det": d.ellipse([x-r,y-r,x+r,y+r], outline=(0,230,0), width=3)
        else:
            for a in range(0,360,45): d.arc([x-r,y-r,x+r,y+r], a, a+25, fill=(255,170,0), width=3)
    if caption:
        txt = caption + (f"   {int((t0+(n-1)/FPS)//60):02d}:{int((t0+(n-1)/FPS)%60):02d}" if t0 is not None else "")
        d.rectangle([0,0,OW,40], fill=(0,0,0)); d.text((12,7), txt, fill=(255,255,255), font=bfont)
    enc.stdin.write(im.tobytes())
    if n % 5000 == 0: print(n, flush=True)
enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait(); print("done", n, "frames ->", out)
