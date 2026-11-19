"""Team-level views from <pre>_track_details_teams.txt: team trajectories/occupancy, on-court counts, possession proxy.
Usage: python team_plots.py track_details_teams.txt calib.npz out.png [--fps 50] [--t0 600] [--scale 2]
"""
import sys, csv, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter1d
sys.argv_saved = list(sys.argv); src, calib, out = sys.argv[1], sys.argv[2], sys.argv[3]
fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
t0 = float(sys.argv[sys.argv.index("--t0")+1]) if "--t0" in sys.argv else 0.0
scale = float(sys.argv[sys.argv.index("--scale")+1]) if "--scale" in sys.argv else 2.0
sys.argv = ["x","--calib",calib]; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from pitch_map import image_to_pitch, L, W
rows = list(csv.DictReader(open(src)))
feet = np.array([[float(r["x_center"]), float(r["y_center"])+float(r["height"])/2] for r in rows])*scale
pm = image_to_pitch(feet); ok = np.isfinite(pm[:,0])
TEAM = {"A":"#1f3a93","B":"#d9d9d0"}; TEAMLINE = {"A":"#1f3a93","B":"#8c8c84"}; NAME = {"A":"Team A (dark kit)","B":"Team B (light kit)"}
if "--names" in sys.argv_saved:   # e.g. --names "A=San Ramón (dark, attacks left);B=Grecia (grey, attacks right)"
    for kv in sys.argv_saved[sys.argv_saved.index("--names")+1].split(";"): k,v = kv.split("=",1); NAME[k.strip()] = v.strip()
def draw_pitch(ax, lw=1):
    g="#9a9a94"; ax.add_patch(Rectangle((0,0),L,W,fill=False,lw=lw,color=g)); ax.plot([L/2,L/2],[0,W],lw=lw,color=g); ax.add_patch(Circle((L/2,W/2),3,fill=False,lw=lw,color=g))
    for X,s in ((0,1),(L,-1)):
        y1,y2=W/2-1.5,W/2+1.5; ax.add_patch(Arc((X,y1),12,12,theta1=270 if s>0 else 180,theta2=360 if s>0 else 270,lw=lw,color=g)); ax.add_patch(Arc((X,y2),12,12,theta1=0 if s>0 else 90,theta2=90 if s>0 else 180,lw=lw,color=g)); ax.plot([X+6*s,X+6*s],[y1,y2],lw=lw,color=g); ax.plot([X,X],[y1,y2],lw=3,color="#6b6b66")
    ax.set_xlim(-1,L+1); ax.set_ylim(W+1,-1); ax.set_aspect("equal"); ax.axis("off")
fig = plt.figure(figsize=(15,8.6), facecolor="#ffffff"); gs = fig.add_gridspec(3,2, height_ratios=[1,1,0.9], hspace=0.35, wspace=0.08)
# occupancy per team
for j,t in enumerate("AB"):
    ax = fig.add_subplot(gs[0,j]); m = ok & np.array([r["team"]==t for r in rows])
    Hh,xe,ye = np.histogram2d(pm[m,0],pm[m,1],bins=[np.arange(0,L+1,1.0),np.arange(0,W+1,1.0)])
    cmap = LinearSegmentedColormap.from_list("c",["#ffffff", TEAM[t] if t=="A" else "#9a9a94"])
    ax.imshow((Hh/fps).T,origin="upper",extent=[0,xe[-1],ye[-1],0],cmap=cmap,interpolation="nearest",vmin=0); draw_pitch(ax)
    ax.set_title(f"{NAME[t]} occupancy, player-seconds per m² (max {Hh.max()/fps:.0f})", loc="left", fontsize=10)
# on-court count per second + possession proxy
secs = np.arange(0, int(max(int(r["frame_number"]) for r in rows)/fps)+1); cntA=np.zeros(len(secs)); cntB=np.zeros(len(secs)); nfr=np.zeros(len(secs))
byf = collections.defaultdict(list)
for i,r in enumerate(rows): byf[int(r["frame_number"])].append(i)
poss = np.full(len(secs), np.nan); ball_x = np.full(len(secs), np.nan); votes = collections.defaultdict(lambda: collections.Counter())
for f,idx in byf.items():
    s = int(f/fps)
    if s >= len(secs): continue
    nfr[s]+=1; tA = [i for i in idx if rows[i]["team"]=="A"]; tB = [i for i in idx if rows[i]["team"]=="B"]; cntA[s]+=len(tA); cntB[s]+=len(tB)
    balls = [i for i in idx if rows[i]["class_name"]=="bola" and ok[i]]
    if balls:
        b = pm[balls[0]]; ball_x[s] = b[0] if np.isnan(ball_x[s]) else ball_x[s]
        players = [i for i in tA+tB if ok[i]]
        if players:
            dd = np.hypot(pm[players,0]-b[0], pm[players,1]-b[1]); k = int(np.argmin(dd))
            if dd[k] < 2.5: votes[s][rows[players[k]]["team"]] += 1
for s in range(len(secs)):
    if votes[s]: poss[s] = "AB".index(votes[s].most_common(1)[0][0])
with np.errstate(invalid="ignore"): cntA/=np.maximum(nfr,1); cntB/=np.maximum(nfr,1)
ax = fig.add_subplot(gs[1,:]); tt=secs+t0
ax.plot(tt, uniform_filter1d(cntA,5), color=TEAMLINE["A"], lw=1.5, label=NAME["A"]); ax.plot(tt, uniform_filter1d(cntB,5), color=TEAMLINE["B"], lw=1.5, label=NAME["B"])
ax.axhline(4, color="#c9c9c2", lw=0.8, ls="--"); ax.set_ylim(0,6); ax.set_ylabel("players detected on court"); ax.set_title("Outfield players detected per team (4 expected; dips = misses/occlusion)", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.spines[["top","right"]].set_visible(False)
ax = fig.add_subplot(gs[2,:], sharex=ax)
for t,j in (("A",0),("B",1)):
    m = poss==j; ax.fill_between(tt, 0, 1, where=m, color=TEAM[t] if t=="A" else "#b5b5ae", step="mid", alpha=0.9)
ax.plot(tt, ball_x/L, color="#eda100", lw=1.2, label="ball position (0 = left goal, 1 = right goal)")
ax.set_ylim(0,1); ax.set_yticks([0,0.5,1]); ax.set_xlabel("match time (s)"); ax.set_title("Possession proxy: team of the player nearest the ball (< 2.5 m), per second", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.spines[["top","right"]].set_visible(False)
valid = ~np.isnan(poss); shareA = np.mean(poss[valid]==0) if valid.any() else float("nan")
fig.text(0.01,0.01,f"Possession proxy over {valid.sum()} s with a ball fix near a player: Team A {100*shareA:.0f}% / Team B {100*(1-shareA):.0f}%. Team = kit colour per detection, tracks split at sustained team flips.", fontsize=8.5, color="#6b6b66")
plt.savefig(out, dpi=150, facecolor="#ffffff", bbox_inches="tight"); print("saved", out, f"| possession A {100*shareA:.0f}% B {100*(1-shareA):.0f}% | mean on-court A {cntA[nfr>0].mean():.2f} B {cntB[nfr>0].mean():.2f}")
