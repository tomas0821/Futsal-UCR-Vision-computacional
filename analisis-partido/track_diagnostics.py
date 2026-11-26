"""Tracking-quality diagnostics for a track_details.txt (1080p): duplicate boxes, confidences, track lifetimes, stitchable ID switches.
Usage: python track_diagnostics.py track_details.txt [--fps 50]
"""
import csv, collections, sys, numpy as np
src = sys.argv[1]; fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
rows = list(csv.DictReader(open(src))); people = {"juga","port","ref"}
byf = collections.defaultdict(list)
for r in rows: byf[int(r["frame_number"])].append((r["class_name"],int(r["track_id"]),float(r["x_center"]),float(r["y_center"]),float(r["width"]),float(r["height"]),float(r["confidence"])))
def iou(a,b):
    ax1,ay1,ax2,ay2=a[2]-a[4]/2,a[3]-a[5]/2,a[2]+a[4]/2,a[3]+a[5]/2; bx1,by1,bx2,by2=b[2]-b[4]/2,b[3]-b[5]/2,b[2]+b[4]/2,b[3]+b[5]/2
    inter=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1)); return inter/(a[4]*a[5]+b[4]*b[5]-inter+1e-9)
nfr=len(byf); dupf=0; pairs=collections.Counter(); npeople=[]
for f,dets in byf.items():
    P=[d for d in dets if d[0] in people]; npeople.append(len(P)); dup=False
    for i in range(len(P)):
        for j in range(i+1,len(P)):
            if iou(P[i],P[j])>0.45: dup=True; pairs[tuple(sorted((P[i][0],P[j][0])))]+=1
    dupf+=dup
tr=collections.defaultdict(list)
for r in rows:
    if r["class_name"] in people: tr[int(r["track_id"])].append((int(r["frame_number"]),float(r["x_center"]),float(r["y_center"])+float(r["height"])/2))
life=[]; starts=[]; ends=[]
for tid,p in tr.items():
    p.sort(); life.append((p[-1][0]-p[0][0]+1)/fps); starts.append(p[0]); ends.append(p[-1])
life=np.array(life); S=np.array(starts); stitch=0; gaps=[]; last=max(byf)
for f,x,y in ends:
    if f>=last-5 or x<40 or x>1880 or y>1075: continue
    m=(S[:,0]>f)&(S[:,0]<=f+3*fps)
    if m.any():
        d=np.hypot(S[m,1]-x,S[m,2]-y); j=np.argmin(d)
        if d[j]<60: stitch+=1; gaps.append((S[m][j,0]-f)/fps)
conf=np.array([float(r["confidence"]) for r in rows if r["class_name"] in people])
print(f"frames {nfr} | person boxes/frame {np.mean(npeople):.2f} | frames with duplicate box (IoU>0.45) {100*dupf/nfr:.1f}% | same-class dup pairs {sum(v for k,v in pairs.items() if k[0]==k[1])}, cross-class {sum(v for k,v in pairs.items() if k[0]!=k[1])}")
print(f"person tracks {len(tr)} | lifetime median {np.median(life):.1f} s, p90 {np.percentile(life,90):.1f} s, max {life.max():.0f} s | share <2 s {100*(life<2).mean():.0f}% | tracks per minute {len(tr)/(last/fps/60):.0f}")
print(f"ID switches recoverable (new track within 3 s & 60 px of a track end): {stitch} ({100*stitch/len(ends):.0f}% of ends) | gap median {np.median(gaps):.2f} s | conf median {np.median(conf):.2f}, share <0.5 {100*(conf<0.5).mean():.0f}%")
