"""Top-down animated pitch ('radar') from track_pitch_coords.csv, optionally side by side with the tracked video.
Usage: python radar_video.py track_pitch_coords.csv out.mp4 [--fps 50] [--out-fps 25] [--video tracked.mp4]
"""
import numpy as np, csv, sys, subprocess, collections
from PIL import Image, ImageDraw, ImageFont
L, W = 37.19, 17.2
src, out = sys.argv[1], sys.argv[2]
fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
ofps = float(sys.argv[sys.argv.index("--out-fps")+1]) if "--out-fps" in sys.argv else 25.0
video = sys.argv[sys.argv.index("--video")+1] if "--video" in sys.argv else None
COL = {"juga":(42,120,214),"port":(235,104,52),"ref":(27,175,122),"bola":(237,161,0),"A":(31,58,147),"B":(200,200,192),"GK_left":(235,104,52),"GK_right":(235,104,52)}
PPM = 24; M = 30; CW, CH = int(L*PPM)+2*M, int(W*PPM)+2*M           # 24 px per metre -> 953x473
def P(x,y): return (M+x*PPM, M+y*PPM)
def base():
    im = Image.new("RGB",(CW,CH),(246,246,242)); d = ImageDraw.Draw(im); g=(170,170,164)
    d.rectangle([P(0,0),P(L,W)],outline=g,width=2); d.line([P(L/2,0),P(L/2,W)],fill=g,width=2)
    d.ellipse([P(L/2-3,W/2-3),P(L/2+3,W/2+3)],outline=g,width=2)
    for X,s in ((0,1),(L,-1)):
        y1,y2=W/2-1.5,W/2+1.5
        d.arc([P(X-6,y1-6),P(X+6,y1+6)],270 if s>0 else 180,360 if s>0 else 270,fill=g,width=2)
        d.arc([P(X-6,y2-6),P(X+6,y2+6)],0 if s>0 else 90,90 if s>0 else 180,fill=g,width=2)
        d.line([P(X+6*s,y1),P(X+6*s,y2)],fill=g,width=2); d.line([P(X,y1),P(X,y2)],fill=(90,90,86),width=6)
    return im
rows = [r for r in csv.DictReader(open(src)) if r["pitch_x_m"]!="nan"]
byf = collections.defaultdict(list)
for r in rows: byf[int(r["frame_number"])].append((r.get("team", r["class_name"]) if r.get("team","?") not in ("?","") else r["class_name"], int(r["track_id"]), float(r["pitch_x_m"]), float(r["pitch_y_m"])))
nframes = max(byf)+1; step = int(round(fps/ofps)); font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",16)
trail = collections.defaultdict(lambda: collections.deque(maxlen=20)); template = base()
ff = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{CW}x{CH}","-r",str(ofps),"-i","-","-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p", out if not video else out+".radar.mp4"], stdin=subprocess.PIPE)
for f in range(1, nframes+1, step):
    im = template.copy(); d = ImageDraw.Draw(im)
    for cls,tid,x,y in byf.get(f,[]):
        if not (-1<=x<=L+1 and -1<=y<=W+1): continue
        tr = trail[(cls,tid)]
        if tr and np.hypot(x-tr[-1][0], y-tr[-1][1]) > 2.0: tr.clear()
        tr.append((x,y))
    for (cls,tid),tr in list(trail.items()):
        if not tr: continue
        if len(tr)>1 and cls!="bola": d.line([P(*q) for q in tr], fill=tuple(int(c*0.55+246*0.45) for c in COL.get(cls,(120,120,120))), width=2)
        x,y = tr[-1]; r = 5 if cls=="bola" else 8
        d.ellipse([P(x-r/PPM,y-r/PPM),P(x+r/PPM,y+r/PPM)], fill=COL.get(cls,(120,120,120)), outline=(255,255,255) if cls!="B" else (90,90,86), width=2)
        if cls in ("port","GK_left","GK_right"): d.text(P(x+0.45,y-0.9),"GK",fill=(60,60,58),font=font)
    d.text((M, 6), f"t = {f/fps:6.1f} s", fill=(60,60,58), font=font)
    ff.stdin.write(im.tobytes())
    # drop stale trails (not updated this frame)
    for k in list(trail):
        if trail[k] and (k[0],k[1]) not in [(c,t) for c,t,_,_ in byf.get(f,[])]: trail[k].clear()
ff.stdin.close(); ff.wait()
if video:
    subprocess.run(["ffmpeg","-v","error","-y","-i",video,"-i",out+".radar.mp4","-filter_complex",f"[0:v]fps={ofps},scale={CW}:-2[a];[1:v]pad=iw:ih+16:0:0:color=#f6f6f2[b];[a][b]vstack=2[o]","-map","[o]","-c:v","libx264","-preset","fast","-crf","23","-pix_fmt","yuv420p",out], check=True)
    import os; os.remove(out+".radar.mp4")
print("wrote", out)
