"""Ball trajectory plots from ball_track.py output.
Usage: python ball_plots.py ball_track.csv out.png [--fps 50] [--t0 600] [--stitched stitched.csv]
Panels: (1) trajectory on the pitch coloured by time, (2) x and y vs time with detected/interpolated/missing shading,
(3) speed vs time with kicks (speed > 8 m/s), (4) ball heatmap, (5) kick map: start points and directions of fast segments.
"""
import sys, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter1d
L, W = 37.19, 17.2
src, out = sys.argv[1], sys.argv[2]
fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
t0 = float(sys.argv[sys.argv.index("--t0")+1]) if "--t0" in sys.argv else 0.0
rows = list(csv.DictReader(open(src)))
f = np.array([int(r["frame_number"]) for r in rows]); x = np.array([float(r["pitch_x_m"]) for r in rows]); y = np.array([float(r["pitch_y_m"]) for r in rows])
det = np.array([r["source"]=="det" for r in rows]); tr = np.array([int(r["tracklet"]) for r in rows]); t = f/fps + t0
def draw_pitch(ax, lw=1):
    g="#9a9a94"; ax.add_patch(Rectangle((0,0),L,W,fill=False,lw=lw,color=g)); ax.plot([L/2,L/2],[0,W],lw=lw,color=g); ax.add_patch(Circle((L/2,W/2),3,fill=False,lw=lw,color=g))
    for X,s in ((0,1),(L,-1)):
        y1,y2=W/2-1.5,W/2+1.5; ax.add_patch(Arc((X,y1),12,12,theta1=270 if s>0 else 180,theta2=360 if s>0 else 270,lw=lw,color=g)); ax.add_patch(Arc((X,y2),12,12,theta1=0 if s>0 else 90,theta2=90 if s>0 else 180,lw=lw,color=g)); ax.plot([X+6*s,X+6*s],[y1,y2],lw=lw,color=g); ax.plot([X,X],[y1,y2],lw=3,color="#6b6b66")
    ax.set_xlim(-1.5,L+1.5); ax.set_ylim(W+1.5,-1.5); ax.set_aspect("equal"); ax.axis("off")
# smoothed speed within tracklets
xs, ys = x.copy(), y.copy()
for k in np.unique(tr):
    m = tr==k
    if m.sum() >= 9: xs[m] = uniform_filter1d(x[m], 9, mode="nearest"); ys[m] = uniform_filter1d(y[m], 9, mode="nearest")
v = np.full(len(f), np.nan); same = (tr[1:]==tr[:-1]) & (np.diff(f) <= 3)
v[1:][same] = (np.hypot(np.diff(xs), np.diff(ys))/np.maximum(np.diff(f),1)*fps)[same]
KICK = 8.0
fig = plt.figure(figsize=(16, 12), facecolor="#ffffff"); gs = fig.add_gridspec(4, 2, height_ratios=[1.35, 0.8, 0.6, 1.1], hspace=0.42, wspace=0.1)
# 1 trajectory coloured by time
ax = fig.add_subplot(gs[0, :]); draw_pitch(ax)
cmap = LinearSegmentedColormap.from_list("t", ["#cde2fb", "#2a78d6", "#0d366b"])
for k in np.unique(tr):
    m = np.flatnonzero(tr==k)
    if len(m) < 2: continue
    pts = np.c_[xs[m], ys[m]].reshape(-1,1,2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(t.min(), t.max()), linewidths=1.4, alpha=0.9); lc.set_array(t[m][:-1]); ax.add_collection(lc)
    ax.scatter(xs[m][0], ys[m][0], s=14, color="#ffffff", edgecolors="#2a78d6", zorder=5, linewidths=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t.min(), t.max())); cax = ax.inset_axes([0.30, -0.06, 0.40, 0.03]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_label("match time (s)", fontsize=8); cb.ax.tick_params(labelsize=8); cb.outline.set_visible(False)
ax.set_title(f"Ball trajectory, {t.min():.0f}–{t.max():.0f} s · {len(np.unique(tr))} continuous tracklets (open circle = tracklet start) · left goal x = 0, right goal x = {L} m", loc="left", fontsize=10)
# 2 x/y vs time
ax = fig.add_subplot(gs[1, :]); allf = np.arange(f.min(), f.max()+1); have = np.zeros(len(allf), bool); have[f - f.min()] = True; isdet = np.zeros(len(allf), bool); isdet[f[det]-f.min()] = True
ta = allf/fps + t0
ax.fill_between(ta, 0, L, where=~have, color="#eeeeea", step="mid", lw=0, label="ball not located"); ax.fill_between(ta, 0, L, where=have & ~isdet, color="#fde9b8", step="mid", lw=0, label="interpolated (gap ≤ 0.5 s)")
ax.plot(t, xs, color="#2a78d6", lw=1.2, label="x, along the pitch"); ax.plot(t, ys, color="#eb6834", lw=1.0, label="y, across the pitch"); ax.axhline(L/2, color="#c9c9c2", lw=0.8, ls="--")
ax.set_ylim(0, L); ax.set_ylabel("m"); ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=4); ax.spines[["top","right"]].set_visible(False); ax.set_title("Ball position vs time", loc="left", fontsize=10)
# 3 speed
ax2 = fig.add_subplot(gs[2, :], sharex=ax); ax2.plot(t, v, color="#6b6b66", lw=0.8); ax2.axhline(KICK, color="#e4572e", lw=0.8, ls="--")
cand = np.flatnonzero((v[1:] >= KICK) & (v[:-1] < KICK)) + 1
kicks = []; last = -10**9
for k in cand:
    seg = v[k:k+5]
    if len(seg) == 5 and np.all(seg >= KICK) and f[k] - last >= fps: kicks.append(k); last = f[k]
kicks = np.array(kicks, int); ax2.scatter(t[kicks], v[kicks], s=18, color="#e4572e", zorder=5, label=f"{len(kicks)} kicks (speed ≥ {KICK:.0f} m/s for ≥ 0.1 s, one per second)")
ax2.set_ylim(0, min(30, np.nanmax(v)*1.05 if np.isfinite(np.nanmax(v)) else 30)); ax2.set_ylabel("m/s"); ax2.set_xlabel("match time (s)"); ax2.legend(loc="upper right", fontsize=8, frameon=False); ax2.spines[["top","right"]].set_visible(False); ax2.set_title("Ball speed (9-frame smoothing, within tracklets)", loc="left", fontsize=10)
# 4 heatmap
ax = fig.add_subplot(gs[3, 0]); H, xe, ye = np.histogram2d(x[det], y[det], bins=[np.arange(0, L+1, 1.0), np.arange(0, W+1, 1.0)])
ax.imshow((H/fps).T, origin="upper", extent=[0, xe[-1], ye[-1], 0], cmap=LinearSegmentedColormap.from_list("a", ["#ffffff", "#fde9b8", "#f7c95c", "#eda100", "#a86f00"]), interpolation="nearest"); draw_pitch(ax)
ax.set_title(f"Ball heatmap, 1 m cells (seconds detected; max {H.max()/fps:.1f} s)", loc="left", fontsize=10)
# 5 kick map
ax = fig.add_subplot(gs[3, 1]); draw_pitch(ax)
for k in kicks:
    m = (tr == tr[k]) & (f >= f[k]) & (f <= f[k] + int(0.6*fps))
    if m.sum() < 3: continue
    px_, py_ = xs[m], ys[m]; vmax = np.nanmax(v[m]) if np.isfinite(v[m]).any() else 0
    ax.annotate("", xy=(px_[-1], py_[-1]), xytext=(px_[0], py_[0]), arrowprops=dict(arrowstyle="->", color="#e4572e" if vmax >= 15 else "#2a78d6", lw=1.2, alpha=0.85))
ax.set_title("Kick map: ball path over 0.6 s after each speed-up (red = peak ≥ 15 m/s, shot-like)", loc="left", fontsize=10)
fig.text(0.01, 0.005, f"Source: {src.split('/')[-1]} · ball located in {100*have.mean():.0f}% of frames ({100*isdet.mean():.0f}% detected). Airborne balls are mapped as if on the floor (position error grows with height).", fontsize=8, color="#6b6b66")
plt.savefig(out, dpi=150, facecolor="#ffffff", bbox_inches="tight"); print("saved", out, "| kicks", len(kicks), "| tracklets", len(np.unique(tr)))
