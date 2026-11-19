"""Match-level analytics from stitched player tracks + ball track, one or more halves.
Usage: python match_analytics.py out_prefix --names "A=San Ramon;B=Grecia" [--fps 50]
         --half 1T stitched.csv ball_track.csv A=left,B=right [--half 2T stitched.csv ball_track.csv A=right,B=left] ...
  stitched.csv : frame_number,track_id,class_name,team,player_id,confidence,pitch_x_m,pitch_y_m   (stitch_tracks.py)
  ball_track   : frame_number,x_px,y_px,pitch_x_m,pitch_y_m,confidence,tracklet,source           (ball_track.py)
  A=left means team A attacks the left goal (x = 0) in that half.
Outputs: <out>_events.csv (shots, goals, kick-offs), <out>_players.csv, <out>_teams.csv, <out>_report.png, <out>_shotmap.png, <out>_heatmaps.png
Events: kick = ball speed >= 8 m/s sustained; shot = kick aimed at the goal mouth (+-3 m) from <= 20 m that gets within 4 m of the line
(or is lost on the way); on target = aimed inside the 3 m goal and reaching <= 1 m; kick-off = ball still at the centre spot with each team
in its own half (>= 0.6 s); goal = kick-off (other than the first of a half) preceded (<= 120 s) by a shot; the team kicking off conceded.
"""
import sys, csv, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from scipy.ndimage import gaussian_filter, uniform_filter1d
L, W = 37.19, 17.2; GOAL = 1.5   # half goal width (3 m goal)
out = sys.argv[1]; fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
names = dict(kv.split("=") for kv in (sys.argv[sys.argv.index("--names")+1] if "--names" in sys.argv else "A=A;B=B").split(";"))
halves = []
for i, a in enumerate(sys.argv):
    if a == "--half": halves.append(dict(name=sys.argv[i+1], stitched=sys.argv[i+2], ball=sys.argv[i+3], attack=dict(kv.split("=") for kv in sys.argv[i+4].split(","))))
def mmss(t): return f"{int(t)//60:02d}:{int(t)%60:02d}"
def load_half(h):
    rows = list(csv.DictReader(open(h["stitched"]))); nfr = max(int(r["frame_number"]) for r in rows)
    pl = collections.defaultdict(list)   # frame -> [(team, pid, x, y)] outfield players
    traj = collections.defaultdict(list) # pid -> [(frame, x, y)]
    for r in rows:
        if r["pitch_x_m"] in ("nan", ""): continue
        f = int(r["frame_number"]); x, y = float(r["pitch_x_m"]), float(r["pitch_y_m"])
        if r["class_name"] == "juga" and r["team"] in ("A", "B"): pl[f].append((r["team"], r["player_id"], x, y))
        if r["class_name"] in ("juga", "port"): traj[(r["team"] if r["class_name"] == "juga" else "GK", r["player_id"])].append((f, x, y))
    bx = np.full(nfr + 2, np.nan); by = np.full(nfr + 2, np.nan); bsrc = np.zeros(nfr + 2, dtype=np.int8)
    for r in csv.DictReader(open(h["ball"])):
        f = int(r["frame_number"])
        if f <= nfr: bx[f], by[f] = float(r["pitch_x_m"]), float(r["pitch_y_m"]); bsrc[f] = 1 if r["source"] == "det" else 2
    return dict(h, nfr=nfr, pl=pl, traj=traj, bx=bx, by=by, bsrc=bsrc)
def speed(bx, by):
    k = 5; vx = np.full_like(bx, np.nan); vx[k:] = (bx[k:] - bx[:-k]) * fps / k; vy = np.full_like(by, np.nan); vy[k:] = (by[k:] - by[:-k]) * fps / k
    return vx, vy, np.hypot(vx, vy)
events = []; shots_all = []; team_poss = collections.defaultdict(lambda: collections.Counter()); heat = {t: np.zeros((int(W*2)+1, int(L*2)+1)) for t in ("A", "B")}
pheat = collections.defaultdict(lambda: np.zeros((int(W*2)+1, int(L*2)+1)))
pstats = {}
for h in halves:
    H = load_half(h); n = H["nfr"]; bx, by, pl = H["bx"], H["by"], H["pl"]; att = H["attack"]; hn = H["name"]
    if "auto" in att.values():
        # first 6 minutes: frames where someone is at the centre spot and the two teams sit on opposite sides -> defending sides
        votes = collections.Counter()
        for f in range(1, min(n, int(360*fps)) + 1, 5):
            P = pl[f]
            if len(P) < 6 or not any(np.hypot(x-L/2, y-W/2) <= 2.5 for t, pid, x, y in P): continue
            mx = {t: np.mean([x for tt, pid, x, y in P if tt == t and np.hypot(x-L/2, y-W/2) > 3.0] or [np.nan]) for t in ("A", "B")}
            if np.isfinite(mx["A"]) and np.isfinite(mx["B"]) and (mx["A"] - L/2) * (mx["B"] - L/2) < 0 and abs(mx["A"] - mx["B"]) > 6: votes["A_left" if mx["A"] < L/2 else "A_right"] += 1
        if not votes: sys.exit(f"[{hn}] could not infer the attacking directions (no kick-off formation in the first 6 min); pass them explicitly")
        a_def = "left" if votes["A_left"] >= votes["A_right"] else "right"; att = {"A": "right" if a_def == "left" else "left", "B": a_def}; h["attack"] = att
        print(f"[{hn}] attacking directions inferred from kick-off formations ({dict(votes)}): A attacks {att['A']}, B attacks {att['B']}")
    goal_x = {t: (0.0 if att[t] == "left" else L) for t in ("A", "B")}      # goal each team attacks
    own_half = {t: (lambda x: x > L/2) if att[t] == "left" else (lambda x: x < L/2) for t in ("A", "B")}
    vx, vy, sp = speed(bx, by)
    # --- possession proxy per 5-min window + heatmaps + player stats
    for f in range(1, n+1):
        for t, pid, x, y in pl[f]:
            if 0 <= x <= L and 0 <= y <= W: heat[t][int(y*2), int(x*2)] += 1; pheat[(t, pid)][int(y*2), int(x*2)] += 1
        if np.isfinite(bx[f]) and pl[f]:
            d = [(np.hypot(x-bx[f], y-by[f]), t) for t, pid, x, y in pl[f]]; dm, tm = min(d)
            if dm <= 2.5: team_poss[(hn, int((f/fps)//300))][tm] += 1
    for (t, pid), v in H["traj"].items():
        v.sort(); fr = np.array([a[0] for a in v]); xy = np.array([[a[1], a[2]] for a in v])
        step = np.hypot(np.diff(xy[:,0]), np.diff(xy[:,1])); dt = np.diff(fr) / fps
        ok = (dt <= 2.0) & (step / np.maximum(dt, 1e-6) <= 12.5)   # ignore gaps > 2 s and impossible jumps
        dist = step[ok].sum(); ttime = dt[ok].sum()
        sm = uniform_filter1d(xy, 25, axis=0) if len(xy) > 25 else xy; vs = np.hypot(np.diff(sm[:,0]), np.diff(sm[:,1])) / np.maximum(dt, 1e-6); vs = vs[ok] if len(vs) == len(ok) else vs
        k = (t, pid); s = pstats.setdefault(k, dict(team=t, player=pid, halves=set(), time_s=0, dist_m=0, top_speed=0, sprints=0))
        s["halves"].add(hn); s["time_s"] += ttime; s["dist_m"] += dist; s["top_speed"] = max(s["top_speed"], float(np.percentile(vs, 99)) if len(vs) else 0)
        s["sprints"] += int(((vs > 5.5).astype(int) if len(vs) else np.array([0])).sum() / (fps * 0.5))   # sprint-seconds / 0.5 s
    # --- kicks: fast runs
    fast = np.isfinite(sp) & (sp >= 8.0); f = 1; kicks = []
    while f <= n:
        if fast[f]:
            g = f
            while g + 1 <= n and (fast[g+1] or (np.isfinite(sp[g+1]) and sp[g+1] >= 5.0 and g + 1 - f < int(fps))): g += 1
            if g - f + 1 >= 3: kicks.append((f, g))
            f = g + 1
        else: f += 1
    # --- shots
    shots = []
    for (f0, f1) in kicks:
        d0 = max(1, f0 - 3)
        if not np.isfinite(bx[d0]): d0 = f0
        x0, y0 = bx[d0], by[d0]; dx, dy = np.nanmean(vx[f0:min(f0+10, f1+1)]), np.nanmean(vy[f0:min(f0+10, f1+1)])
        if not (np.isfinite(dx) and abs(dx) > 1e-3): continue
        for gx in (0.0, L):
            if (gx - x0) * dx <= 0: continue
            tt = (gx - x0) / dx; y_int = y0 + dy * tt
            if abs(y_int - W/2) > 3.0 or abs(gx - x0) > 20.0: continue
            seg = slice(f0, min(n, f1 + int(1.5*fps)) + 1); xs = bx[seg]; fin = np.isfinite(xs)
            if not fin.any(): continue
            closest = np.nanmin(np.abs(xs - gx)); last_seen = np.abs(xs[fin][-1] - gx)
            seg_e = slice(f0, min(n, f1 + int(0.5*fps)) + 1); xe = bx[seg_e]; ie = int(np.nanargmin(np.abs(xe - gx))) if np.isfinite(xe).any() else 0
            x_end, y_end = float(bx[seg_e][ie]), float(by[seg_e][ie])
            lost = (~fin[-int(0.5*fps):]).all() if len(fin) >= int(0.5*fps) else False
            if not (closest <= 4.0 or (lost and last_seen <= 10.0)): continue
            # shooter: nearest outfield player to the kick point in the 10 frames before
            cand = [(np.hypot(x-x0, y-y0), t, pid) for ff in range(max(1, f0-10), f0+1) for t, pid, x, y in pl[ff]]
            if not cand: continue
            dm, team, pid = min(cand)
            if dm > 3.0 or goal_x[team] != gx: continue    # not attributable, or a clearance toward the own goal
            on_t = abs(y_int - W/2) <= GOAL and closest <= 1.0
            crossed = False; y_cross = np.nan; beyond = (xs - gx) * (1 if gx == 0 else -1) <= -0.3   # ball beyond the goal line
            if np.any(beyond & fin):
                j = int(np.argmax(beyond & fin)); ys = by[seg]
                if j > 0 and np.isfinite(xs[j-1]) and np.isfinite(ys[j-1]) and xs[j] != xs[j-1]:
                    w_ = (gx - xs[j-1]) / (xs[j] - xs[j-1]); y_cross = float(ys[j-1] + w_ * (ys[j] - ys[j-1]))
                else: y_cross = float(ys[j])
                crossed = abs(y_cross - W/2) <= 2.0        # inside the goal mouth (3 m goal + calibration slack)
            shots.append(dict(half=hn, frame=f0, t=f0/fps, team=team, player=pid, x=x0, y=y0, x_end=x_end, y_end=y_end, speed=float(np.nanmax(sp[f0:f1+1])), goal_x=gx, y_at_line=y_int, on_target=on_t, crossed_line=crossed, y_cross=y_cross, dist=float(np.hypot(gx-x0, W/2-y0))))
            break
    # --- kick-offs: formation = each team >= 3 outfield players in its own half and <= 1 in the other, someone within 2.5 m of the
    # centre spot, players nearly still; the ball at the centre spot is supporting evidence (not required: it is often not detected there)
    def formation(f, att_dir):
        own = {t: (lambda x: x > L/2) if att_dir[t] == "left" else (lambda x: x < L/2) for t in ("A", "B")}
        P = pl[f]
        if len(P) < 6: return False
        atc = [(t, pid, x, y) for t, pid, x, y in P if np.hypot(x - L/2, y - W/2) <= 3.0]          # kick-off takers at the centre spot
        rest = [(t, pid, x, y) for t, pid, x, y in P if np.hypot(x - L/2, y - W/2) > 3.0]
        cnt = collections.Counter(t for t, pid, x, y in rest if own[t](x)); tot = collections.Counter(t for t, pid, x, y in rest)
        return 1 <= len(atc) <= 3 and all(cnt[t] >= 2 and tot[t] - cnt[t] == 0 for t in ("A", "B")) and sum(cnt.values()) >= 5
    ball_centre = np.isfinite(bx) & (np.hypot(bx - L/2, by - W/2) <= 1.5) & ~(np.isfinite(sp) & (sp > 1.5))
    form = np.zeros(n + 2, bool)
    for f in range(1, n+1, 2): form[f] = formation(f, att)
    form[2:n+1:2] = form[1:n:2]
    # players nearly still: mean displacement of outfield players over 1 s
    def still_players(f):
        a = {(t, pid): (x, y) for t, pid, x, y in pl[f]}; b = {(t, pid): (x, y) for t, pid, x, y in pl[max(1, f - int(fps))]}
        d = [np.hypot(a[k][0]-b[k][0], a[k][1]-b[k][1]) for k in a if k in b]
        return bool(d) and np.mean(d) < 1.5
    kos = []; f = 1
    while f <= n:
        if form[f]:
            g = f
            while g + 1 <= n and form[min(n, g+1):min(n, g+int(fps))+1].any(): g += 1
            dur = (g - f + 1) / fps; bc = ball_centre[f:g+1].mean() > 0 and ball_centre[f:g+1].sum() >= 0.3*fps
            st = any(still_players(ff) for ff in range(f + int(0.5*fps), g + 1, int(0.5*fps))) or still_players(min(n, f + int(0.5*fps)))
            if __import__("os").environ.get("MA_DEBUG"): print(f"      formation run {mmss(f/fps)} dur {dur:.1f}s still {st} ball_at_centre {bool(bc)} -> {'KICK-OFF' if (dur >= 1.0 and st and (bc or dur >= 2.0)) else 'rejected'}")
            if (dur >= 1.0 and st and (bc or dur >= 2.0)):
                # kicking team = team of the player nearest the centre spot (the takers stand on the ball; opponents must stay 3 m away)
                near = collections.Counter()
                for ff in range(f, g+1, 5):
                    c = [(np.hypot(x-L/2, y-W/2), t) for t, pid, x, y in pl[ff]]
                    if c and min(c)[0] <= 1.5: near[min(c)[1]] += 1
                kicking = near.most_common(1)[0][0] if near else "?"
                kos.append(dict(half=hn, frame=f, t=f/fps, end=g/fps, kicking=kicking, ball_seen=bool(bc), dur=dur, votes=dict(near)))
            f = g + 1
        else: f += 1
    merged = []
    for k in kos:
        if merged and k["t"] - merged[-1]["end"] < 30:
            merged[-1]["end"] = k["end"]; merged[-1]["ball_seen"] |= k["ball_seen"]; v = collections.Counter(merged[-1]["votes"]); v.update(k["votes"]); merged[-1]["votes"] = dict(v); merged[-1]["kicking"] = v.most_common(1)[0][0] if v else merged[-1]["kicking"]
        else: merged.append(k)
    kos = merged
    # --- goals: every kick-off except the first of the half means a goal was just scored by the team NOT kicking off.
    # evidence: 'shot' (a shot by that team <= 120 s before), 'crossed' (that shot's ball crossed the line inside the mouth), 'formation only'
    goals = []
    for i, k in enumerate(kos):
        if i == 0 and k["t"] < 180: k["note"] = "start of half"; continue
        scorer = {"A": "B", "B": "A"}.get(k["kicking"], "?")
        prev = [s for s in shots if 0 < k["t"] - s["t"] <= 120 and s["team"] == scorer]
        anyshot = [s for s in shots if 0 < k["t"] - s["t"] <= 120]
        if prev:
            s0 = max(prev, key=lambda s: s["t"]); s0["goal"] = True; ev = "shot crossed the line" if s0["crossed_line"] else "shot seen"
            goals.append(dict(half=hn, t=s0["t"], team=scorer, player=s0["player"], x=s0["x"], y=s0["y"], kickoff_t=k["t"], kickoff_team=k["kicking"], evidence=ev))
        elif anyshot:
            s0 = max(anyshot, key=lambda s: s["t"]); s0["goal"] = True   # shooter team label disagreed with the kick-off: trust the kick-off
            goals.append(dict(half=hn, t=s0["t"], team=scorer, player="?", x=s0["x"], y=s0["y"], kickoff_t=k["t"], kickoff_team=k["kicking"], evidence="shot seen but attributed to the other team"))
        else:
            goals.append(dict(half=hn, t=max(0.0, k["t"] - 25), team=scorer, player="?", x=np.nan, y=np.nan, kickoff_t=k["t"], kickoff_team=k["kicking"], evidence="kick-off formation only (shot not tracked)"))
        k["note"] = f"after a goal by {names.get(scorer, scorer)} ({goals[-1]['evidence']})"
    # a shot whose ball crossed the line but with no kick-off afterwards is NOT counted (checked on video: usually a save on the line);
    # it is listed as 'possible goal (review)' because instant restarts after a real goal leave no kick-off formation
    possible = [s0 for s0 in shots if s0["crossed_line"] and not s0.get("goal")]
    for s0 in possible: s0["possible"] = True
    goals.sort(key=lambda g: g["t"])
    for g in goals:
        if g["player"] == "?" and not np.isfinite(g["x"]): events.append(dict(half=hn, time=mmss(g["t"]), t=round(g["t"],1), type="goal (inferred)", team=names.get(g["team"], g["team"]), player="?", x="", y="", detail=g["evidence"] + f"; kick-off at {mmss(g['kickoff_t'])}"))
    for s in shots: s.setdefault("goal", False); shots_all.append(s)
    for s in shots: events.append(dict(half=hn, time=mmss(s["t"]), t=round(s["t"],1), type=("goal" if s["goal"] else ("shot on target, possible goal (review)" if s.get("possible") else ("shot on target" if s["on_target"] else "shot"))), team=names.get(s["team"], s["team"]), player=s["player"], x=round(s["x"],1), y=round(s["y"],1), detail=f"speed {s['speed']:.1f} m/s, from {s['dist']:.1f} m, aimed {s['y_at_line']-W/2:+.1f} m from the goal centre" + (f", crossed the line at {s['y_cross']-W/2:+.1f} m" if np.isfinite(s['y_cross']) else "")))
    for k in kos: events.append(dict(half=hn, time=mmss(k["t"]), t=round(k["t"],1), type="kick-off", team=names.get(k["kicking"], k["kicking"]), player="", x=round(L/2,1), y=round(W/2,1), detail=k.get("note", "")))
    H["shots"] = shots; H["goals"] = goals; H["kos"] = kos; H["sp"] = sp; h["goals"] = goals
    print(f"[{hn}] frames {n} ({mmss(n/fps)}), kicks {len(kicks)}, shots {len(shots)} (on target {sum(s['on_target'] for s in shots)}), kick-offs {len(kos)}, goals {len(goals)}, possible goals to review {len(possible)}: {[mmss(s0['t']) + ' ' + names.get(s0['team'], s0['team']) for s0 in possible]}")
    for g in goals: print(f"   GOAL {names.get(g['team'], g['team'])} {g['player']} at ~{mmss(g['t'])} ({('kick-off ' + mmss(g['kickoff_t']) + ' by ' + names.get(g['kickoff_team'], g['kickoff_team']) + '; ') if np.isfinite(g['kickoff_t']) else ''}{g['evidence']})")
    for k in kos: print(f"   kick-off {mmss(k['t'])} by {names.get(k['kicking'], k['kicking'])} (centre votes {k['votes']}, ball seen {k['ball_seen']}, {k['dur']:.1f} s): {k.get('note','')}")
events.sort(key=lambda e: (e["half"], e["t"]))
with open(f"{out}_events.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(events[0].keys()) if events else ["half"]); w.writeheader(); w.writerows(events)
# --- team table
teams = {}
for t in ("A", "B"):
    sh = [s for s in shots_all if s["team"] == t]; pf = sum(c[t] for c in team_poss.values()); pa = sum(sum(c.values()) for c in team_poss.values())
    teams[t] = dict(team=names.get(t, t), goals=sum(1 for h in halves for g in h.get("goals", []) if g["team"] == t), shots=len(sh), on_target=sum(s["on_target"] for s in sh), possession=f"{100*pf/max(pa,1):.0f}%", mean_shot_dist=f"{np.mean([s['dist'] for s in sh]):.1f}" if sh else "", distance_km=f"{sum(v['dist_m'] for v in pstats.values() if v['team']==t)/1000:.1f}")
with open(f"{out}_teams.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(teams["A"].keys())); w.writeheader(); w.writerows(teams.values())
print("teams:", teams)
# --- player table
prow = []
for (t, pid), s in sorted(pstats.items(), key=lambda kv: -kv[1]["dist_m"]):
    if s["time_s"] < 30: continue
    sh = [x for x in shots_all if x["team"] == t and x["player"] == pid]
    prow.append(dict(team=names.get(t, t), player=pid, halves="+".join(sorted(s["halves"])), time_min=round(s["time_s"]/60, 1), distance_m=round(s["dist_m"]), mean_speed=round(s["dist_m"]/max(s["time_s"],1), 2), top_speed=round(s["top_speed"], 1), sprint_s=round(s["sprints"]*0.5, 1), shots=len(sh), on_target=sum(x["on_target"] for x in sh), goals=sum(x["goal"] for x in sh)))
with open(f"{out}_players.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(prow[0].keys())); w.writeheader(); w.writerows(prow)
# --- drawing helpers
def pitch(ax):
    ax.add_patch(Rectangle((0, 0), L, W, fill=False, lw=1.2, color="k")); ax.plot([L/2, L/2], [0, W], "k", lw=1); ax.add_patch(Circle((L/2, W/2), 3, fill=False, lw=1, color="k"))
    for gx, s in ((0, 1), (L, -1)):
        ax.add_patch(Arc((gx, W/2-1.5), 12, 12, theta1=270 if s==1 else 180, theta2=360 if s==1 else 270, lw=1, color="k")); ax.add_patch(Arc((gx, W/2+1.5), 12, 12, theta1=0 if s==1 else 90, theta2=90 if s==1 else 180, lw=1, color="k"))
        ax.plot([gx+6*s, gx+6*s], [W/2-1.5, W/2+1.5], "k", lw=1); ax.add_patch(Rectangle((gx-0.8 if s==1 else gx, W/2-1.5), 0.8, 3, fill=True, color="0.6"))
    ax.set_xlim(-1.5, L+1.5); ax.set_ylim(W+1, -1); ax.set_aspect("equal"); ax.axis("off")
col = {"A": "#1f4e9c", "B": "#8c8c8c"}
# heatmaps
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for a, t in zip(ax, ("A", "B")):
    pitch(a); hm = gaussian_filter(heat[t], 1.5); a.imshow(hm, extent=(0, L, W, 0), cmap="Reds" if t=="A" else "Greys", alpha=0.85, zorder=0, vmax=np.percentile(hm, 99.5))
    a.set_title(f"{names.get(t,t)} occupancy, all halves (attacking " + "/".join(f"{h['name']}:{h['attack'][t]}" for h in halves) + ")")
fig.tight_layout(); fig.savefig(f"{out}_heatmaps.png", dpi=130)
# shot map
fig, ax = plt.subplots(figsize=(12, 6)); pitch(ax)
for s in shots_all:
    c = col[s["team"]]; m = "*" if s["goal"] else ("o" if s["on_target"] else "x"); sz = 320 if s["goal"] else 70
    ax.scatter(s["x"], s["y"], s=sz, marker=m, color=c, edgecolor="k" if s["goal"] else None, zorder=3); ax.plot([s["x"], s["x_end"]], [s["y"], s["y_end"]], color=c, lw=0.8, alpha=0.6)
    ax.annotate(f"{s['half']} {mmss(s['t'])}", (s["x"], s["y"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_title(f"Shot map: {names.get('A','A')} (blue) vs {names.get('B','B')} (grey). star = goal, o = on target, x = off target. Teams change ends between halves; each shot is drawn where it happened.")
fig.tight_layout(); fig.savefig(f"{out}_shotmap.png", dpi=130)
# report: possession per 5 min, timeline of events, top players
fig = plt.figure(figsize=(14, 9)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.4])
ax = fig.add_subplot(gs[0, :]); keys = sorted(team_poss); labels = [f"{h} {5*w}-{5*w+5}'" for h, w in keys]
pa = [100*team_poss[k]["A"]/max(sum(team_poss[k].values()), 1) for k in keys]
ax.bar(range(len(keys)), pa, color=col["A"], label=names.get("A","A")); ax.bar(range(len(keys)), [100-p for p in pa], bottom=pa, color=col["B"], label=names.get("B","B"))
ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, rotation=30, fontsize=8); ax.set_ylabel("possession proxy %"); ax.legend(loc="upper right", fontsize=8); ax.set_title("Possession proxy (team of the nearest player within 2.5 m of the ball) per 5-min window")
ax = fig.add_subplot(gs[1, :]); t_off = 0
for h in halves:
    H = [x for x in halves if x is h][0]; n = max(int(r["frame_number"]) for r in csv.DictReader(open(h["stitched"]))); dur = n / fps
    ax.axvspan(t_off, t_off + dur, color="0.95"); ax.text(t_off + dur/2, 1.9, h["name"], ha="center", fontsize=10)
    for s in shots_all:
        if s["half"] != h["name"]: continue
        y = 1 if s["team"] == "A" else 0; ax.scatter(t_off + s["t"], y, marker="*" if s["goal"] else ("o" if s["on_target"] else "x"), s=250 if s["goal"] else 50, color=col[s["team"]], zorder=3)
    for e in events:
        if e["half"] == h["name"] and e["type"] == "kick-off": ax.axvline(t_off + e["t"], color="green", lw=0.8, ls="--")
        if e["half"] == h["name"] and e["type"] == "goal (inferred)": ax.scatter(t_off + e["t"], 1 if e["team"] == names.get("A","A") else 0, marker="*", s=250, facecolor="none", edgecolor=col["A" if e["team"] == names.get("A","A") else "B"], zorder=3)
    t_off += dur + 60
ax.set_yticks([0, 1]); ax.set_yticklabels([names.get("B","B"), names.get("A","A")]); ax.set_ylim(-0.6, 2.2); ax.set_xlabel("recording time (s), halves separated by 60 s"); ax.set_title("Shots (x off, o on target, star goal, hollow star = goal inferred from a kick-off) and kick-offs (green dashed)")
ax = fig.add_subplot(gs[2, :]); ax.axis("off")
cell = [[r["team"], r["player"], r["time_min"], r["distance_m"], r["mean_speed"], r["top_speed"], r["sprint_s"], r["shots"], r["on_target"], r["goals"]] for r in prow[:14]]
tb = ax.table(cellText=cell, colLabels=["team", "player", "min on court", "distance m", "mean m/s", "top m/s (p99)", "sprint s (>5.5 m/s)", "shots", "on target", "goals"], loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.25)
ax.set_title("Top identities by distance (identities are stitched tracks, not yet jersey numbers)")
fig.suptitle(f"{names.get('A','A')} {teams['A']['goals']} - {teams['B']['goals']} {names.get('B','B')} (goals = kick-offs seen; instant restarts are missed)   |   shots {teams['A']['shots']} ({teams['A']['on_target']} on target) vs {teams['B']['shots']} ({teams['B']['on_target']})   |   possession {teams['A']['possession']} vs {teams['B']['possession']}", fontsize=13)
fig.tight_layout(); fig.savefig(f"{out}_report.png", dpi=130); print("wrote", f"{out}_{{events,teams,players}}.csv and _report/_shotmap/_heatmaps.png")
