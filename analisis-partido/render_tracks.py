"""Draw compact tracking annotations onto a video from a track_details.txt (1080p coords). Pure ffmpeg pipes + PIL.
Usage: python render_tracks.py in_list.txt|in.mp4 track_details.txt out.mp4 [--fps 50]
in_list.txt = ffmpeg concat list (file '...' lines) for multi-part sources.
"""
import sys, csv, subprocess, collections, numpy as np
from PIL import Image, ImageDraw, ImageFont
src, trk, out = sys.argv[1], sys.argv[2], sys.argv[3]
fps = sys.argv[sys.argv.index("--fps")+1] if "--fps" in sys.argv else "50"
Wd, Hd = 1920, 1080
COL = {"bola":(0,220,0),"juga":(0,200,255),"port":(255,0,255),"ref":(255,215,0)}; SHORT = {"bola":"b","juga":"J","port":"GK","ref":"R"}
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
byf = collections.defaultdict(list)
for r in csv.DictReader(open(trk)):
    byf[int(r["frame_number"])].append((r["class_name"], r.get("label") or str(int(r["track_id"]) % 100000), float(r["x_center"]), float(r["y_center"]), float(r["width"]), float(r["height"])))
inp = ["-f","concat","-safe","0","-i",src] if src.endswith(".txt") else ["-i",src]
dec = subprocess.Popen(["ffmpeg","-v","error"]+inp+["-f","rawvideo","-pix_fmt","rgb24","-"], stdout=subprocess.PIPE, bufsize=10**8)
enc = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{Wd}x{Hd}","-r",fps,"-i","-","-c:v","libx264","-preset","fast","-crf","23","-pix_fmt","yuv420p","-movflags","+faststart",out], stdin=subprocess.PIPE)
n = 0; nbytes = Wd*Hd*3
while True:
    buf = dec.stdout.read(nbytes)
    if len(buf) < nbytes: break
    n += 1; im = Image.frombuffer("RGB",(Wd,Hd),buf,"raw","RGB",0,1).copy(); d = ImageDraw.Draw(im)
    for cls,tid,x,y,w,h in byf.get(n,[]):
        x1,y1,x2,y2 = int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2); c = COL[cls]
        if cls=="bola": r=int(max(w,h)/2)+6; d.ellipse([x-r,y-r,x+r,y+r],outline=c,width=2); continue
        if str(tid).startswith(("A","B","GK")): c = (31,58,147) if str(tid).startswith("A") else (225,225,215) if str(tid).startswith("B") else COL["port"]
        d.rectangle([x1,y1,x2,y2],outline=c,width=1); lab=f"{tid}" if str(tid).startswith(("A","B","GK")) else f"{tid} {SHORT[cls]}"; tw=int(d.textlength(lab,font=font)); th=15
        ty = y1-4 if y1-th-6>0 else y2+th+4
        d.rectangle([x1,ty-th-3,x1+tw+4,ty+2],fill=c); d.text((x1+2,ty-th-2),lab,fill=(0,0,0),font=font)
    if n % 500 == 0: d.text((10,10),"",fill=(0,0,0))
    enc.stdin.write(im.tobytes())
    if n % 5000 == 0: print(f"{n} frames", flush=True)
enc.stdin.close(); enc.wait(); dec.wait(); print("done", n, "frames ->", out)
