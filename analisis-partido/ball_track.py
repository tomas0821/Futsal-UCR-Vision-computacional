"""Ball trajectory tracker with pitch-plane gating over per-frame candidates.
Usage: python ball_track.py candidates.csv out.csv --calib <match>_camera.npz [--fps 50] [--scale 1] [--nframes N]
candidates.csv: frame_number,x_center,y_center,width,height,confidence[,tile]  (pixel coords; --scale 2 if 1080p)
Output: frame_number,x_px,y_px,pitch_x_m,pitch_y_m,confidence,tracklet,source  (source = det | interp)
"""
import sys, csv, collections, numpy as np
src, out = sys.argv[1], sys.argv[2]
fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
scale = float(sys.argv[sys.argv.index("--scale")+1]) if "--scale" in sys.argv else 1.0
calib = sys.argv[sys.argv.index("--calib")+1]
NF = int(sys.argv[sys.argv.index("--nframes")+1]) if "--nframes" in sys.argv else None
CMIN = float(sys.argv[sys.argv.index("--cmin")+1]) if "--cmin" in sys.argv else 0.0
VMAX = float(sys.argv[sys.argv.index("--vmax")+1]) if "--vmax" in sys.argv else 0.4   # m per frame (0.4 = 20 m/s at 50 fps)
QUIET = "--quiet" in sys.argv
sys.argv = ["x","--calib",calib]; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from pitch_map import image_to_pitch, L, W
rows = [r for r in csv.DictReader(open(src)) if float(r["confidence"]) >= CMIN]
px = np.array([[float(r["x_center"]), float(r["y_center"])] for r in rows])*scale; conf = np.array([float(r["confidence"]) for r in rows]); fr = np.array([int(r["frame_number"]) for r in rows])
pm = image_to_pitch(px); MARGIN = 2.5
inpitch = np.isfinite(pm[:,0]) & (pm[:,0] > -MARGIN) & (pm[:,0] < L+MARGIN) & (pm[:,1] > -MARGIN) & (pm[:,1] < W+MARGIN)
# static-spot suppression: a 0.25 m cell holding a candidate in > 25% of frames is a painted mark / spare ball, not the ball in play
cell = np.floor(np.nan_to_num(pm, nan=-999)/0.25).astype(int); occ = collections.defaultdict(set)
for i in np.flatnonzero(inpitch): occ[(cell[i,0], cell[i,1])].add(fr[i])
nfr_all = len(set(fr.tolist())); static_cells = {c for c, fs_ in occ.items() if len(fs_) > 0.25*nfr_all}
is_static = np.array([inpitch[i] and (cell[i,0], cell[i,1]) in static_cells for i in range(len(rows))])
keep = inpitch & ~is_static
byf = collections.defaultdict(list)
for i in np.flatnonzero(keep): byf[fr[i]].append(i)
nframes = NF or int(fr.max())
RESET = 25; CONF_START = max(0.30, CMIN); CONF_START2 = max(0.12, CMIN)
pos = None; vel = np.zeros(2); last_f = -999; tid = 0; pending = None; track = {}
for f in range(1, nframes+1):
    cands = byf.get(f, [])
    chosen = None
    if pos is not None and f - last_f <= RESET and cands:
        dt = f - last_f; pred = pos + vel*min(dt, 10)*np.exp(-0.05*dt); rad = 0.3 + VMAX*dt
        best = None
        for i in cands:
            d = np.linalg.norm(pm[i]-pred)
            if d <= rad and np.linalg.norm(pm[i]-pos) <= rad:
                s = d/rad - 0.8*conf[i]
                if best is None or s < best[0]: best = (s, i)
        if best: chosen = best[1]
    if chosen is None and cands and (pos is None or f - last_f > RESET):
        hi = max(cands, key=lambda i: conf[i])
        if conf[hi] >= CONF_START or (pending is not None and pending[0] == f-1 and np.linalg.norm(pm[hi]-pending[1]) < 1.0 and conf[hi] >= CONF_START2):
            if pos is None or f - last_f > RESET: tid += 1
            chosen = hi; pos = pm[hi]; vel = np.zeros(2); last_f = f
        pending = (f, pm[hi]) if conf[hi] >= CONF_START2 else None
    if chosen is not None:
        dt = max(f - last_f, 1)
        if pos is not None and f != last_f:
            vel = 0.6*vel + 0.4*(pm[chosen]-pos)/dt; sp = np.linalg.norm(vel)
            if sp > VMAX: vel *= VMAX/sp
        pos = pm[chosen]; last_f = f; track[f] = (chosen, tid, "det")
# interpolate gaps <= RESET within the same tracklet
keys = sorted(track); filled = dict(track)
for a, b in zip(keys, keys[1:]):
    if track[a][1] == track[b][1] and 1 < b-a <= RESET:
        ia, ib = track[a][0], track[b][0]
        for f in range(a+1, b):
            t = (f-a)/(b-a); filled[f] = (None, track[a][1], "interp", pm[ia]*(1-t)+pm[ib]*t, px[ia]*(1-t)+px[ib]*t)
with open(out, "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["frame_number","x_px","y_px","pitch_x_m","pitch_y_m","confidence","tracklet","source"])
    for f in sorted(filled):
        v = filled[f]
        if v[2] == "det": i = v[0]; w.writerow([f, f"{px[i,0]:.1f}", f"{px[i,1]:.1f}", f"{pm[i,0]:.2f}", f"{pm[i,1]:.2f}", f"{conf[i]:.3f}", v[1], "det"])
        else: w.writerow([f, f"{v[4][0]:.1f}", f"{v[4][1]:.1f}", f"{v[3][0]:.2f}", f"{v[3][1]:.2f}", "", v[1], "interp"])
det = len(track); tot = len(filled); ntr = tid
fs = np.array(sorted(filled)); pts = np.array([filled[f][3] if filled[f][2]=="interp" else pm[filled[f][0]] for f in fs]); tids = np.array([filled[f][1] for f in fs])
same = tids[1:]==tids[:-1]; v = (np.hypot(*np.diff(pts,axis=0).T)/np.maximum(np.diff(fs),1)*fps)[same]
static = np.mean(v < 0.3) if len(v) else 0
print(f"candidates {len(rows)} | in-pitch {inpitch.sum()} ({100*inpitch.mean():.0f}%), static-suppressed {is_static.sum()} in {len(static_cells)} cells | frames with ball: detected {det}/{nframes} ({100*det/nframes:.0f}%), interp {tot}/{nframes} ({100*tot/nframes:.0f}%) | tracklets {ntr} | speed p50/p95 {np.percentile(v,50):.1f}/{np.percentile(v,95):.1f} m/s | static share {100*static:.0f}%")
