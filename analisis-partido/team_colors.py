"""Team assignment from kit colour, per detection, with track splitting at team flips.
Usage: python team_colors.py clip.mp4 track_details.txt out_prefix [--every 1]
Writes: <pre>_track_details_teams.txt  (track_details + columns team, seg_id)   -- seg_id = track_id*10+k after splitting at flips
        <pre>_teams_summary.csv         (per segment: team, frames, mean colour, confidence)
        <pre>_swatches.png              (crops sorted by team)
Teams: A = darker kit, B = lighter kit. Keepers -> GK_left/GK_right by goal defended. Refs/ball unchanged.
"""
import sys, csv, subprocess, collections, numpy as np
from PIL import Image, ImageDraw, ImageFont
clip, trk, pre = sys.argv[1], sys.argv[2], sys.argv[3]
EVERY = int(sys.argv[sys.argv.index("--every")+1]) if "--every" in sys.argv else 1
W, H = 1920, 1080; WIN = 25; MINSEG = 60   # a team flip must persist >= 1.2 s to split a track
def rgb2lab(rgb):
    r = np.where(rgb>0.04045, ((rgb+0.055)/1.055)**2.4, rgb/12.92)
    M = np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]]); xyz = r @ M.T / np.array([0.95047,1.0,1.08883])
    f = np.where(xyz>0.008856, np.cbrt(xyz), 7.787*xyz+16/116); return np.c_[116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])]
rows = list(csv.DictReader(open(trk))); byf = collections.defaultdict(list)
for i,r in enumerate(rows):
    if r["class_name"] in ("juga","port"): byf[int(r["frame_number"])].append((i, int(r["track_id"]), r["class_name"], float(r["x_center"]), float(r["y_center"]), float(r["width"]), float(r["height"])))
col = {}; crops = {}
dec = subprocess.Popen(["ffmpeg","-v","error","-i",clip,"-f","rawvideo","-pix_fmt","rgb24","-"], stdout=subprocess.PIPE, bufsize=10**8)
n = 0; nb = W*H*3
while True:
    buf = dec.stdout.read(nb)
    if len(buf) < nb: break
    n += 1
    if n % EVERY: continue
    fr = np.frombuffer(buf, np.uint8).reshape(H,W,3)
    for i,tid,cls,x,y,w,h in byf.get(n,[]):
        x1,x2 = int(x-0.2*w), int(x+0.2*w); y1,y2 = int(y-0.5*h+0.2*h), int(y-0.5*h+0.45*h)
        if x2-x1 < 3 or y2-y1 < 3 or x1<0 or y1<0 or x2>W or y2>H: continue
        col[i] = np.median(rgb2lab(fr[y1:y2, x1:x2].reshape(-1,3).astype(np.float32)/255), axis=0)
        if cls=="juga" and h > 60 and (tid not in crops or len(crops[tid]) < 6) and n % 100 == 0:
            crops.setdefault(tid, []).append(Image.fromarray(fr[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)].copy()))
dec.stdout.close(); print(f"frames {n}; coloured detections {len(col)}")
# k=2 on player detections
pidx = [i for i in col if rows[i]["class_name"]=="juga"]; X = np.array([col[i] for i in pidx])
rng = np.random.default_rng(0); best=None
for _ in range(10):
    C = X[rng.choice(len(X),2,replace=False)]
    for _ in range(30):
        lb = np.argmin(((X[:,None,:]-C[None])**2).sum(-1),1); C = np.array([X[lb==j].mean(0) if (lb==j).any() else C[j] for j in range(2)])
    cost = ((X-C[lb])**2).sum()
    if best is None or cost<best[0]: best=(cost,C.copy())
C = best[1]; o = np.argsort(C[:,0]); C = C[o]   # C[0]=dark (A), C[1]=light (B)
d = np.sqrt(((X[:,None,:]-C[None])**2).sum(-1)); margin = (d[:,1]-d[:,0])/np.linalg.norm(C[0]-C[1])   # >0 -> A
det_team = {}; det_margin = {}
for i,m in zip(pidx, margin): det_team[i] = "A" if m>0 else "B"; det_margin[i] = float(m)
print(f"centroids Lab: A={np.round(C[0],1)} B={np.round(C[1],1)}; detections A {sum(v=='A' for v in det_team.values())}, B {sum(v=='B' for v in det_team.values())}; |margin|<0.1: {100*np.mean(np.abs(margin)<0.1):.1f}%")
# per track: running vote along the track, split where the smoothed sign flips for >= MINSEG frames
tracks = collections.defaultdict(list)
for i,r in enumerate(rows):
    if r["class_name"]=="juga": tracks[int(r["track_id"])].append((int(r["frame_number"]), i))
seg_of = {}; seg_team = {}; seg_info = []; nsplit = 0
for tid,lst in tracks.items():
    lst.sort(); idx=[i for _,i in lst]; m = np.array([det_margin.get(i, 0.0) for i in idx])
    k = min(WIN, len(m)); sm = np.convolve(m, np.ones(k)/k, mode="same") if len(m)>=3 else m
    sign = np.where(sm>=0, 1, -1); cuts=[0]
    j=1
    while j < len(sign):
        if sign[j]!=sign[j-1]:
            run = 1
            while j+run < len(sign) and sign[j+run]==sign[j]: run += 1
            if run >= MINSEG and j >= MINSEG: cuts.append(j)
            j += run
        else: j += 1
    cuts.append(len(idx))
    if len(cuts) > 2: nsplit += len(cuts)-2
    for k_,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
        sid = tid*10+k_; votes = m[a:b]; t = "A" if votes.mean()>=0 else "B"; conf = float(np.mean(np.sign(votes)==np.sign(votes.mean()))) if len(votes) else 0
        for i in idx[a:b]: seg_of[i]=sid
        seg_team[sid]=t; seg_info.append((sid,tid,t,b-a,conf,np.mean([col[i] for i in idx[a:b] if i in col],axis=0) if any(i in col for i in idx[a:b]) else np.zeros(3)))
# keepers: side by mean x
gk = collections.defaultdict(list)
for i,r in enumerate(rows):
    if r["class_name"]=="port": gk[int(r["track_id"])].append(float(r["x_center"]))
gk_side = {t:("GK_left" if np.mean(v)<W/2 else "GK_right") for t,v in gk.items()}
with open(f"{pre}_track_details_teams.txt","w",newline="") as f:
    w = csv.writer(f); w.writerow(list(rows[0].keys())+["team","seg_id"])
    for i,r in enumerate(rows):
        c = r["class_name"]; team = seg_team.get(seg_of.get(i), "?") if c=="juga" else gk_side.get(int(r["track_id"]),"GK") if c=="port" else c
        w.writerow(list(r.values())+[team, seg_of.get(i, int(r["track_id"])*10)])
with open(f"{pre}_teams_summary.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["seg_id","track_id","team","frames","vote_agreement","L","a","b"])
    for sid,tid,t,nf,conf,c in sorted(seg_info): w.writerow([sid,tid,t,nf,f"{conf:.2f}",f"{c[0]:.1f}",f"{c[1]:.1f}",f"{c[2]:.1f}"])
print(f"player tracks {len(tracks)} -> segments {len(seg_info)} ({nsplit} cross-team splits); A segments {sum(1 for s in seg_info if s[2]=='A')}, B {sum(1 for s in seg_info if s[2]=='B')}")
# per-frame count check
cnt = collections.Counter()
for fnum,dets in byf.items():
    c = collections.Counter(seg_team.get(seg_of.get(i),"?") for i,tid,cls,*_ in dets if cls=="juga"); cnt[(c["A"],c["B"])]+=1
tot=sum(cnt.values()); print("per-frame (A,B) counts:", [(k,round(100*v/tot,1)) for k,v in cnt.most_common(5)], f"| A==5 {100*sum(v for k,v in cnt.items() if k[0]==5)/tot:.0f}%  B==5 {100*sum(v for k,v in cnt.items() if k[1]==5)/tot:.0f}%")
# swatches: up to 6 crops per track, coloured frame by the team of that moment is not tracked -> use segment majority per track's first segment
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",11); tw,th=48,96; cols=30
items=[]; 
for sid,tid,t,nf,conf,c in sorted(seg_info, key=lambda s:(s[2],-s[3])):
    for im in crops.get(tid,[])[:3]: items.append((t,tid,im))
items=items[:240]; img=Image.new("RGB",(cols*tw,((len(items)+cols-1)//cols)*(th+14)),(246,246,242)); d=ImageDraw.Draw(img)
for k,(t,tid,im) in enumerate(items):
    cx,cy=(k%cols)*tw,(k//cols)*(th+14); img.paste(im.resize((tw,th)),(cx,cy)); d.rectangle([cx,cy,cx+tw-1,cy+th-1],outline=(30,30,30) if t=="A" else (235,235,235),width=3); d.text((cx+2,cy+th),f"{t}{tid}",fill=(60,60,58),font=font)
img.save(f"{pre}_swatches.png"); print("wrote outputs with prefix", pre)
