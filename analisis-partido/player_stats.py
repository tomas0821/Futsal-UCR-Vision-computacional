"""Per-track (per player segment) statistics and small-multiple trajectory plots from track_pitch_coords.csv.
Usage: python player_stats.py track_pitch_coords.csv out_prefix [--fps 50] [--top 12]
Writes <out_prefix>_tracks.csv and <out_prefix>_players.png
"""
import numpy as np, csv, sys, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from scipy.ndimage import median_filter, uniform_filter1d
L, W = 37.19, 17.2
src, pre = sys.argv[1], sys.argv[2]
fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
TOP = int(sys.argv[sys.argv.index("--top")+1]) if "--top" in sys.argv else 12
COL = {"juga":"#2a78d6","port":"#eb6834","ref":"#1baf7a","bola":"#eda100"}; NAME = {"juga":"Player","port":"Goalkeeper","ref":"Referee","bola":"Ball"}
JUMP, VMAX = 0.25, 10.0   # m/frame split threshold (0.25 m/frame = 12.5 m/s at 50 fps); m/s cap for speed stats
KEEP_GAPS = "--keep-gaps" in sys.argv   # stitched identities: do not split at detection gaps

rows = [r for r in csv.DictReader(open(src)) if r["pitch_x_m"]!="nan"]
tracks = collections.defaultdict(list)
labels = {}
for r in rows:
    tracks[(r["class_name"], int(r["track_id"]))].append((int(r["frame_number"]), float(r["pitch_x_m"]), float(r["pitch_y_m"])))
    if "player_id" in r: labels[(r["class_name"], int(r["track_id"]))] = r["player_id"]
def segments(pts):
    pts = np.array(sorted(pts)); cut=[0]
    for i in range(1,len(pts)):
        if np.hypot(*(pts[i,1:]-pts[i-1,1:])) > JUMP*max(1,pts[i,0]-pts[i-1,0]) or (not KEEP_GAPS and pts[i,0]-pts[i-1,0] > 25): cut.append(i)
    cut.append(len(pts)); return [pts[a:b] for a,b in zip(cut[:-1],cut[1:])]

recs = []
for (cls,tid),pts in tracks.items():
    for k,seg in enumerate(segments(pts)):
        if len(seg) < (10 if cls=="bola" else 25): continue
        t = seg[:,0]/fps; xy = seg[:,1:]
        if cls!="bola" and len(xy)>=9: xy = median_filter(xy, size=(9,1), mode="nearest"); xy = uniform_filter1d(xy, 5, axis=0, mode="nearest")
        d = np.hypot(*np.diff(xy,axis=0).T); dt = np.diff(t); v = d/np.maximum(dt,1e-6); ok = (v<=VMAX) & (dt<=2.0)   # ignore steps across gaps > 2 s
        dist = float(d[ok].sum()); dur = float(t[-1]-t[0]); tracked = float(dt[ok].sum())
        recs.append(dict(class_name=cls, track_id=tid, label=labels.get((cls,tid), f"#{tid}"), segment=k, start_s=round(t[0],2), end_s=round(t[-1],2), duration_s=round(dur,2), frames=len(seg),
                         distance_m=round(dist,1), tracked_s=round(tracked,1), mean_speed_ms=round(dist/tracked,2) if tracked>0 else 0, max_speed_ms=round(float(np.percentile(v[ok],98)) if ok.any() else 0,2),
                         mean_x_m=round(float(xy[:,0].mean()),1), mean_y_m=round(float(xy[:,1].mean()),1), share_right_half=round(float((xy[:,0]>L/2).mean()),2), xy=xy, t=t))
recs.sort(key=lambda r:(-r["duration_s"]))
with open(f"{pre}_tracks.csv","w",newline="") as f:
    cols=[c for c in recs[0] if c not in ("xy","t")]; w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in recs: w.writerow({c:r[c] for c in cols})
print(f"{len(recs)} segments -> {pre}_tracks.csv")
print(f"{'class':5s} {'id':>4s} {'seg':>3s} {'start':>6s} {'dur s':>6s} {'dist m':>6s} {'mean m/s':>8s} {'p98 m/s':>7s}")
for r in recs[:TOP+4]: print(f"{r['class_name']:5s} {r['track_id']:4d} {r['segment']:3d} {r['start_s']:6.1f} {r['duration_s']:6.1f} {r['distance_m']:6.1f} {r['mean_speed_ms']:8.2f} {r['max_speed_ms']:7.2f}")

def draw_pitch(ax):
    ax.add_patch(Rectangle((0,0),L,W,fill=False,lw=0.8,color="#b5b5ae")); ax.plot([L/2,L/2],[0,W],lw=0.6,color="#b5b5ae"); ax.add_patch(Circle((L/2,W/2),3,fill=False,lw=0.6,color="#b5b5ae"))
    for X,s in ((0,1),(L,-1)):
        y1,y2=W/2-1.5,W/2+1.5
        ax.add_patch(Arc((X,y1),12,12,theta1=270 if s>0 else 180,theta2=360 if s>0 else 270,lw=0.6,color="#b5b5ae")); ax.add_patch(Arc((X,y2),12,12,theta1=0 if s>0 else 90,theta2=90 if s>0 else 180,lw=0.6,color="#b5b5ae"))
        ax.plot([X+6*s,X+6*s],[y1,y2],lw=0.6,color="#b5b5ae"); ax.plot([X,X],[y1,y2],lw=2,color="#8a8a84")
    ax.set_xlim(-1,L+1); ax.set_ylim(W+1,-1); ax.set_aspect("equal"); ax.axis("off")
people = [r for r in recs if r["class_name"]!="bola"][:TOP]
ncol = 4; nrow = int(np.ceil(len(people)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 2.35*nrow+0.6), facecolor="#ffffff"); axes=np.atleast_2d(axes)
for ax,r in zip(axes.ravel(), people):
    draw_pitch(ax); xy=r["xy"]; c=COL[r["class_name"]]
    gaps = np.flatnonzero(np.diff(r["t"]) > 1.0); a0 = 0
    for g_ in list(gaps)+[len(xy)-1]: ax.plot(xy[a0:g_+1,0],xy[a0:g_+1,1],lw=1.4,color=c,alpha=0.9); a0 = g_+1
    ax.scatter([xy[0,0]],[xy[0,1]],s=22,color="#ffffff",edgecolors=c,zorder=5,linewidths=1.2); ax.scatter([xy[-1,0]],[xy[-1,1]],s=26,color=c,zorder=5,edgecolors="none")
    ax.set_title(f"{NAME[r['class_name']]} {r.get('label', '#'+str(r['track_id']))}  ·  {r['start_s']:.0f}–{r['end_s']:.0f} s", fontsize=9, color="#2b2b28", loc="left", pad=3)
    ax.text(0.0,-0.02,f"{r['distance_m']:.0f} m in {r['tracked_s']:.0f} s tracked (span {r['duration_s']:.0f} s)  ·  mean {r['mean_speed_ms']:.1f} m/s  ·  peak {r['max_speed_ms']:.1f} m/s",transform=ax.transAxes,fontsize=7.5,color="#6b6b66",va="top")
for ax in axes.ravel()[len(people):]: ax.axis("off")
fig.suptitle(f"Longest {len(people)} player segments, first half, full, stitched identities · open circle = start, filled = end · segments split at track-ID switches", fontsize=10, color="#2b2b28", x=0.01, ha="left")
plt.tight_layout(rect=(0,0,1,0.965)); plt.savefig(f"{pre}_players.png", dpi=150, facecolor="#ffffff"); print("saved", f"{pre}_players.png")
