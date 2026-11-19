"""Ball position timeline, sprint-intensity timeline and ball heatmap from track_pitch_coords.csv.
Usage: python timeline_plots.py track_pitch_coords.csv out.png [--fps 50] [--t0 600]
"""
import numpy as np, csv, sys, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import median_filter, uniform_filter1d
L, W = 37.19, 17.2
src, out = sys.argv[1], sys.argv[2]
fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
t0 = float(sys.argv[sys.argv.index("--t0")+1]) if "--t0" in sys.argv else 0.0
rows = [r for r in csv.DictReader(open(src)) if r["pitch_x_m"]!="nan"]
# --- ball: per-second median x (only frames with a ball detection inside the pitch)
ball = collections.defaultdict(list)
for r in rows:
    if r["class_name"]=="bola":
        x,y=float(r["pitch_x_m"]),float(r["pitch_y_m"])
        if -1<=x<=L+1 and -1<=y<=W+1: ball[int(int(r["frame_number"])/fps)].append((x,y))
secs = np.arange(0, int(max(int(r["frame_number"]) for r in rows)/fps)+1)
bx = np.array([np.median([p[0] for p in ball[s]]) if ball[s] else np.nan for s in secs]); seen = np.array([len(ball[s])/fps for s in secs])
# --- sprint intensity: per-second number of player segments moving faster than 4 m/s
tracks = collections.defaultdict(list)
for r in rows:
    if r["class_name"]=="juga": tracks[int(r["track_id"])].append((int(r["frame_number"]), float(r["pitch_x_m"]), float(r["pitch_y_m"])))
sprint = np.zeros(len(secs)); moving = np.zeros(len(secs))
for tid,pts in tracks.items():
    pts=np.array(sorted(pts))
    if len(pts)<15: continue
    xy=uniform_filter1d(median_filter(pts[:,1:],size=(9,1),mode="nearest"),5,axis=0,mode="nearest"); t=pts[:,0]/fps
    v=np.hypot(*np.diff(xy,axis=0).T)/np.maximum(np.diff(t),1e-6); v[v>10]=np.nan
    for s in np.unique(t[1:].astype(int)):
        m=(t[1:].astype(int)==s); vv=np.nanmean(v[m]) if np.isfinite(v[m]).any() else 0
        if s<len(secs): sprint[s]+= vv>4.0; moving[s]+= vv>1.5
fig = plt.figure(figsize=(15,7.2), facecolor="#ffffff"); gs = fig.add_gridspec(3,2, width_ratios=[1.9,1], height_ratios=[1,1,0.05], hspace=0.45, wspace=0.12)
ax = fig.add_subplot(gs[0,0]); tt = secs+t0
ax.fill_between(tt, 0, L, where=np.isnan(bx), color="#eeeeea", step="mid", label="ball not detected")
ax.plot(tt, bx, color="#eda100", lw=1.8, label="ball x (per-second median)"); ax.axhline(L/2, color="#b5b5ae", lw=0.8)
ax.set_ylim(0,L); ax.set_ylabel("distance from left goal line (m)"); ax.set_title("Ball position along the pitch", loc="left", fontsize=10)
ax.text(tt[0]+2, L-1.5, "right goal", fontsize=8, color="#6b6b66", va="top"); ax.text(tt[0]+2, 1.5, "left goal", fontsize=8, color="#6b6b66")
ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2); ax.spines[["top","right"]].set_visible(False)
ax2 = fig.add_subplot(gs[1,0], sharex=ax)
ax2.bar(tt, moving, width=1, color="#cde2fb", label="players moving > 1.5 m/s"); ax2.bar(tt, sprint, width=1, color="#2a78d6", label="players sprinting > 4 m/s")
ax2.set_ylim(0,10); ax2.set_ylabel("players"); ax2.set_xlabel("match time, first half (s)"); ax2.set_title("Intensity", loc="left", fontsize=10)
ax2.legend(loc="upper right", fontsize=8, frameon=False, ncol=2); ax2.spines[["top","right"]].set_visible(False)
# --- ball heatmap
ax3 = fig.add_subplot(gs[:2,1])
pb = np.array([p for s in ball for p in ball[s]]) if ball else np.zeros((0,2))
Hh,xe,ye = np.histogram2d(pb[:,0],pb[:,1],bins=[np.arange(0,L+1,2.0),np.arange(0,W+1,2.0)])
cmap = LinearSegmentedColormap.from_list("amber",["#ffffff","#fde9b8","#f7c95c","#eda100","#a86f00"])
im = ax3.imshow((Hh/fps).T, origin="upper", extent=[0,xe[-1],ye[-1],0], cmap=cmap, interpolation="nearest")
g="#9a9a94"; ax3.add_patch(Rectangle((0,0),L,W,fill=False,lw=1.2,color=g)); ax3.plot([L/2,L/2],[0,W],lw=1,color=g); ax3.add_patch(Circle((L/2,W/2),3,fill=False,lw=1,color=g))
for X,s in ((0,1),(L,-1)):
    y1,y2=W/2-1.5,W/2+1.5; ax3.add_patch(Arc((X,y1),12,12,theta1=270 if s>0 else 180,theta2=360 if s>0 else 270,lw=1,color=g)); ax3.add_patch(Arc((X,y2),12,12,theta1=0 if s>0 else 90,theta2=90 if s>0 else 180,lw=1,color=g)); ax3.plot([X+6*s,X+6*s],[y1,y2],lw=1,color=g); ax3.plot([X,X],[y1,y2],lw=3,color="#6b6b66")
ax3.set_xlim(-1,L+1); ax3.set_ylim(W+1,-1); ax3.set_aspect("equal"); ax3.axis("off"); ax3.set_title("Ball heatmap, 2 m cells (seconds with ball seen)", loc="left", fontsize=10)
cb = fig.colorbar(im, ax=ax3, fraction=0.03, pad=0.02); cb.set_label("s", fontsize=8); cb.ax.tick_params(labelsize=8); cb.outline.set_visible(False)
fig.text(0.01,0.02,f"Ball detected in {100*np.mean(~np.isnan(bx)):.0f}% of seconds. Speeds from smoothed feet positions; sprint = segment mean > 4 m/s within the second.", fontsize=8, color="#6b6b66")
plt.savefig(out, dpi=150, facecolor="#ffffff", bbox_inches="tight"); print("saved", out)
