import numpy as np, csv, sys, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import median_filter
L, W = 37.19, 17.2
src = sys.argv[1]; out = sys.argv[2]; fps = 50
COL = {"juga":"#2a78d6","port":"#eb6834","ref":"#1baf7a","bola":"#eda100"}   # reference palette slots 1-4, fixed order
NAME = {"juga":"Player","port":"Goalkeeper","ref":"Referee","bola":"Ball"}
rows = [r for r in csv.DictReader(open(src)) if r["pitch_x_m"]!="nan"]
tracks = collections.defaultdict(list)
for r in rows: tracks[(r["class_name"], r["track_id"])].append((int(r["frame_number"]), float(r["pitch_x_m"]), float(r["pitch_y_m"])))

def draw_pitch(ax):
    ax.add_patch(Rectangle((0,0),L,W,fill=False,lw=1.2,color="#9a9a94"))
    ax.plot([L/2,L/2],[0,W],lw=1,color="#9a9a94"); ax.add_patch(Circle((L/2,W/2),3,fill=False,lw=1,color="#9a9a94"))
    for X,s in ((0,1),(L,-1)):
        y1,y2 = W/2-1.5, W/2+1.5
        ax.add_patch(Arc((X,y1),12,12,theta1=270 if s>0 else 180,theta2=360 if s>0 else 270,lw=1,color="#9a9a94"))
        ax.add_patch(Arc((X,y2),12,12,theta1=0 if s>0 else 90,theta2=90 if s>0 else 180,lw=1,color="#9a9a94"))
        ax.plot([X+6*s,X+6*s],[y1,y2],lw=1,color="#9a9a94"); ax.plot([X,X],[y1,y2],lw=3,color="#6b6b66")
    ax.set_xlim(-1.5,L+1.5); ax.set_ylim(W+1.5,-1.5); ax.set_aspect("equal"); ax.axis("off")

fig, axes = plt.subplots(1,2,figsize=(15,4.9),facecolor="#ffffff")
ax = axes[0]; draw_pitch(ax)
n_used = collections.Counter()
JUMP = 1.0   # metres per frame (50 fps): anything faster is an ID switch, not motion
def segments(pts):
    pts = np.array(sorted(pts)); cut = [0]
    for i in range(1,len(pts)):
        if np.hypot(*(pts[i,1:]-pts[i-1,1:])) > JUMP*max(1,pts[i,0]-pts[i-1,0]) or pts[i,0]-pts[i-1,0] > 25: cut.append(i)
    cut.append(len(pts)); return [pts[a:b] for a,b in zip(cut[:-1],cut[1:])]
for (cls,tid),pts in tracks.items():
    for seg_ in segments(pts):
        if len(seg_) < (15 if cls=="bola" else 50): continue
        xy = seg_[:,1:]
        if cls!="bola" and len(xy)>=7: xy = median_filter(xy, size=(7,1), mode="nearest")
        if cls=="bola": ax.scatter(xy[:,0],xy[:,1],s=9,color=COL[cls],alpha=0.85,zorder=4,edgecolors="none")
        else: ax.plot(xy[:,0],xy[:,1],lw=1.6 if cls!="juga" else 1.1,color=COL[cls],alpha=0.85 if cls!="juga" else 0.6,zorder=3 if cls!="juga" else 2)
        n_used[cls]+=1
for i,c in enumerate(["juga","port","ref","bola"]):
    ax.plot([],[],color=COL[c],lw=2,label=f"{NAME[c]} ({n_used[c]} segments)")
ax.legend(loc="lower center",bbox_to_anchor=(0.5,-0.13),ncol=4,frameon=False,fontsize=9)
ax.set_title("Trajectories, first half, full (0:00-27:25) (feet position, 7-frame median, split at ID switches)",fontsize=10,color="#2b2b28",loc="left")
ax = axes[1]; draw_pitch(ax)
pl = np.array([(float(r["pitch_x_m"]),float(r["pitch_y_m"])) for r in rows if r["class_name"]=="juga"])
Hh,xe,ye = np.histogram2d(pl[:,0],pl[:,1],bins=[np.arange(0,L+1,1.0),np.arange(0,W+1,1.0)])
cmap = LinearSegmentedColormap.from_list("blue",["#ffffff","#cde2fb","#86b6ef","#3987e5","#1c5cab","#0d366b"])
im = ax.imshow((Hh/fps).T,origin="upper",extent=[0,xe[-1],ye[-1],0],cmap=cmap,alpha=0.95,zorder=1,interpolation="nearest")
draw_pitch(ax)
cb = fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02); cb.set_label("player-seconds per m²",fontsize=8); cb.ax.tick_params(labelsize=8); cb.outline.set_visible(False)
ax.set_title("Player occupancy, 27 min (1 m cells)",fontsize=10,color="#2b2b28",loc="left")
fig.text(0.01,0.01,"Attack direction: left goal at x=0, right goal at x=37.19 m. Camera side is the bottom touchline. Calibration: 15 clicked landmarks, fisheye fit (2026-09-17), median 6.6 px ≈ 0.1 m on the ground.",fontsize=7.5,color="#6b6b66")
plt.tight_layout(rect=(0,0.03,1,1)); plt.savefig(out,dpi=150,facecolor="#ffffff"); print("saved",out, dict(n_used))
