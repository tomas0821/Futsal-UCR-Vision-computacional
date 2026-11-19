"""Stitch team-labelled track segments into player identities (offline, per team) by global assignment.
Usage: python stitch_tracks.py track_pitch_coords_teams.csv out_prefix [--fps 50]
Writes <pre>_stitched.csv (track_pitch_coords format with track_id = chain id, plus player_id) and prints a report.
"""
import sys, csv, collections, numpy as np
from scipy.optimize import linear_sum_assignment
src, pre = sys.argv[1], sys.argv[2]; fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
MIN_SEG = 10          # frames; shorter segments are dropped
GAP_MAX = 25.0        # s, longest bridgeable gap
NOLINK = 14.0         # cost of leaving a segment end unlinked
rows = [r for r in csv.DictReader(open(src)) if r["pitch_x_m"] != "nan"]
segs = collections.defaultdict(list)
for i,r in enumerate(rows):
    if r["class_name"] == "juga" and r["team"] in ("A","B"): segs[(r["team"], int(r["seg_id"]))].append((int(r["frame_number"]), float(r["pitch_x_m"]), float(r["pitch_y_m"]), i))
    elif r["class_name"] == "port": segs[(r["team"], int(r["seg_id"]))].append((int(r["frame_number"]), float(r["pitch_x_m"]), float(r["pitch_y_m"]), i))
S = {}
for k,pts in segs.items():
    pts.sort(); 
    if len(pts) < MIN_SEG: continue
    a = np.array(pts, float); n = min(10, len(a)-1)
    v_end = (a[-1,1:3]-a[-1-n,1:3]) / max((a[-1,0]-a[-1-n,0])/fps, 1e-6) if n>0 else np.zeros(2)
    S[k] = dict(t0=a[0,0]/fps, t1=a[-1,0]/fps, p0=a[0,1:3], p1=a[-1,1:3], v1=np.clip(v_end,-6,6), idx=[p[3] for p in pts], n=len(pts))
def link_cost(a, b):
    g = b["t0"] - a["t1"]
    if g <= 0 or g > GAP_MAX: return None
    pred = a["p1"] + a["v1"]*min(g, 0.6)                      # short extrapolation, then treat as unknown
    d = np.linalg.norm(b["p0"] - pred)
    allow = 1.5 + 6.0*min(g, 3.0) + 0.5*max(g-3.0, 0)         # reachable radius
    if d > allow: return None
    return d/(1.0+0.5*g) + 0.35*g
chains = {}; report = []
for group in sorted(set(k[0] for k in S)):
    keys = [k for k in S if k[0]==group]; keys.sort(key=lambda k: S[k]["t0"]); n=len(keys)
    if group.startswith("GK"):          # one keeper per goal: a single identity, no assignment needed
        chains[(group, 1)] = keys; report.append((group, n, 1, S[keys[-1]]["t1"]-S[keys[0]]["t0"], S[keys[-1]]["t1"]-S[keys[0]]["t0"], sum(S[k]["t1"]-S[k]["t0"] for k in keys), 0, 0)); continue
    C = np.full((n, n), 1e6)
    for i,ki in enumerate(keys):
        for j,kj in enumerate(keys):
            if i==j: continue
            c = link_cost(S[ki], S[kj])
            if c is not None: C[i,j] = c
    # augment: each row may go to a dummy (no successor) at cost NOLINK; each column may come from a dummy (no predecessor)
    big = np.full((2*n, 2*n), 1e6); big[:n,:n] = C; big[:n, n:] = np.where(np.eye(n)>0, NOLINK, 1e6); big[n:, :n] = np.where(np.eye(n)>0, NOLINK, 1e6); big[n:, n:] = 0
    ri, ci = linear_sum_assignment(big)
    succ = {keys[i]: keys[j] for i,j in zip(ri,ci) if i<n and j<n and big[i,j] < 1e5}
    pred = {v:k for k,v in succ.items()}
    heads = [k for k in keys if k not in pred]; cid = 0
    for h in heads:
        cid += 1; k = h; members=[]
        while k is not None: members.append(k); k = succ.get(k)
        chains[(group, cid)] = members
    # per-frame concurrency check for this group
    occ = collections.Counter()
    for (g_,c),members in chains.items():
        if g_!=group: continue
        for k in members:
            for f in range(int(S[k]["t0"]*fps), int(S[k]["t1"]*fps)+1): occ[f]+=1
    over = sum(1 for f,c in occ.items() if c>4); dur = [sum(S[k]["t1"]-S[k]["t0"] for k in m) for (g_,c),m in chains.items() if g_==group]
    span = [S[m[-1]]["t1"]-S[m[0]]["t0"] for (g_,c),m in chains.items() if g_==group]
    report.append((group, n, len(heads), np.median(span), max(span), np.mean(dur), over, len(occ)))
print(f"{'group':9s} {'segs':>5s} {'chains':>6s} {'span med/max s':>15s} {'on-court/chain s':>16s} {'frames>4 concurrent':>20s}")
for g,n,h,ms,mx,md,over,nf in report: print(f"{g:9s} {n:5d} {h:6d} {ms:7.0f}/{mx:5.0f} {md:16.0f} {over:10d} of {nf}")
# write stitched csv
pid_of = {}
for (g,c),members in chains.items():
    for k in members: pid_of[k] = (g,c)
base = {"A":1000,"B":2000,"GK_left":3000,"GK_right":3100}
with open(f"{pre}_stitched.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["frame_number","track_id","class_name","team","player_id","confidence","pitch_x_m","pitch_y_m"])
    for k,(g,c) in pid_of.items():
        for i in S[k]["idx"]:
            r = rows[i]; w.writerow([r["frame_number"], base.get(g,9000)+c, r["class_name"], r["team"], f"{g}{c}", r["confidence"], r["pitch_x_m"], r["pitch_y_m"]])
    for i,r in enumerate(rows):
        if r["class_name"] in ("ref","bola"): w.writerow([r["frame_number"], r["track_id"], r["class_name"], r["team"], r["class_name"], r["confidence"], r["pitch_x_m"], r["pitch_y_m"]])
print("wrote", f"{pre}_stitched.csv")
