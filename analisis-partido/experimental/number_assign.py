"""Assign jersey numbers per tracker segment from OCR reads, with a per-frame uniqueness rule.
Usage: python number_assign.py track_details_stitched.txt crops/index.csv numbers_reads.csv out_track_details.txt [--roster-min 10]
Steps: reads -> (frame, chain) -> the segment (seg_id) that owned that box; roster per team = numbers with >= roster-min reads;
segment number = majority of its roster reads (>= 2 reads, >= 70%); segments without reads inherit a number only when both
their chain-neighbours (previous and next numbered segment in the same stitched chain, within 60 s) agree; per frame and team,
a number may appear once: on conflict the segment with more own reads keeps it, the other becomes '?'.
Labels: 'SR n' / 'GR n' / 'GK L' / 'GK R' / 'SR ?' / 'GR ?'.
"""
import sys, csv, collections, numpy as np
src, idx_f, reads_f, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
RMIN = int(sys.argv[sys.argv.index("--roster-min")+1]) if "--roster-min" in sys.argv else 10
rows = list(csv.DictReader(open(src))); cols = list(rows[0].keys())
def prefix(lab): return "SR" if lab.startswith("A") else "GR" if lab.startswith("B") else "GK"
# (frame, chain) -> seg_id, and segment spans / chain order
seg_of = {}; seg_frames = collections.defaultdict(list); seg_chain = {}
for r in rows:
    if r["class_name"] == "juga":          # keepers are labelled GK L/R directly; their seg_ids can collide with player seg_ids
        k = (int(r["frame_number"]), r["label"]); sid = int(r["seg_id"]); seg_of[k] = sid; seg_frames[sid].append(int(r["frame_number"])); seg_chain[sid] = r["label"]
span = {s: (min(f), max(f)) for s, f in seg_frames.items()}
# reads -> segments; kit colour of each read crop (mean lightness of the torso crop) decides the team when it disagrees with the chain
CROP_DIRS = [d for d in sys.argv[sys.argv.index("--crop-dirs")+1].split(",")] if "--crop-dirs" in sys.argv else []
def crop_L(fn):
    for d in CROP_DIRS:
        try:
            from PIL import Image; im = Image.open(f"{d}/{fn}").convert("L"); w, h = im.size
            return float(np.asarray(im)[int(0.15*h):int(0.85*h), int(0.2*w):int(0.8*w)].mean())
        except FileNotFoundError: continue
    return None
reads = collections.defaultdict(list); seg_L = collections.defaultdict(list)
for r in csv.DictReader(open(reads_f)):
    sid = seg_of.get((int(r["frame"]), r["label"]))
    if sid is not None and r["number"] != "0":
        reads[sid].append(r["number"])
        if CROP_DIRS:
            L = crop_L(r["file"])
            if L is not None: seg_L[sid].append(L)
kit_team = {}
if CROP_DIRS:
    L_DARK, L_LIGHT = 90.0, 140.0    # 8-bit lightness of the torso crop: dark kit < 90, grey kit > 140 (gap in between = undecided)
    for sid, Ls in seg_L.items():
        if len(Ls) >= 2:
            m = float(np.median(Ls)); kit_team[sid] = "SR" if m < L_DARK else "GR" if m > L_LIGHT else None
    flips = {sid: t for sid, t in kit_team.items() if t and prefix(seg_chain[sid]) in ("SR", "GR") and t != prefix(seg_chain[sid])}
    print(f"kit-colour check: {sum(1 for t in kit_team.values() if t)} segments decided, {len(flips)} team labels overridden by kit colour: {sorted(flips.items())[:12]}")
    _pf = prefix
    def prefix(lab, _sid=None):
        return _pf(lab)
    seg_team = {sid: (flips.get(sid) or _pf(ch)) for sid, ch in seg_chain.items()}
else:
    seg_team = {sid: prefix(ch) for sid, ch in seg_chain.items()}
team_reads = collections.defaultdict(collections.Counter)
for sid, ns in reads.items():
    for n in ns: team_reads[seg_team[sid]][n] += 1
roster = {t: {n for n, c in cnt.items() if c >= RMIN} for t, cnt in team_reads.items()}
print("roster:", {t: sorted(r, key=int) for t, r in roster.items() if t != "GK"})
seg_num = {}; seg_votes = {}
for sid, ns in reads.items():
    c = collections.Counter(n for n in ns if n in roster.get(seg_team[sid], set()))
    if not c: continue
    mc = c.most_common(2); n, v = mc[0]; v2 = mc[1][1] if len(mc) > 1 else 0
    if v >= 2 and v >= 2 * v2 and v / sum(c.values()) >= 0.5: seg_num[sid] = n; seg_votes[sid] = v   # winner at least twice the runner-up
n_read = len(seg_num)
# inherit within chains when both neighbours agree
by_chain = collections.defaultdict(list)
for sid, ch in seg_chain.items(): by_chain[ch].append(sid)
n_inh = 0
for ch, sids in by_chain.items():
    sids.sort(key=lambda s: span[s][0])
    for i, sid in enumerate(sids):
        if sid in seg_num: continue
        prev = next((s for s in reversed(sids[:i]) if s in seg_num), None); nxt = next((s for s in sids[i+1:] if s in seg_num), None)
        gp = span[sid][0] - span[prev][1] if prev else 10**9; gn = span[nxt][0] - span[sid][1] if nxt else 10**9
        if prev and nxt and seg_num[prev] == seg_num[nxt] and gp <= 3000 and gn <= 3000: seg_num[sid] = seg_num[prev]; seg_votes[sid] = 0; n_inh += 1
        elif prev and gp <= 500 and (not nxt or gn > 500 or seg_num[nxt] == seg_num[prev]): seg_num[sid] = seg_num[prev]; seg_votes[sid] = 0; n_inh += 1
        elif nxt and gn <= 500 and (not prev or gp > 500): seg_num[sid] = seg_num[nxt]; seg_votes[sid] = 0; n_inh += 1
# position-continuity propagation (needs --pitch track_pitch_coords_teams.csv): an unnamed segment inherits from a numbered
# segment of the same team whose end is within 1 s before its start and within reach (1.5 m + 6 m/s x gap), if that candidate is unique
if "--pitch" in sys.argv:
    pc = collections.defaultdict(list)
    for r in csv.DictReader(open(sys.argv[sys.argv.index("--pitch")+1])):
        if r["class_name"] == "juga" and r["pitch_x_m"] != "nan": pc[int(r["seg_id"])].append((int(r["frame_number"]), float(r["pitch_x_m"]), float(r["pitch_y_m"])))
    ends = {}; starts = {}
    for s, v in pc.items():
        v.sort(); ends[s] = v[-1]; starts[s] = v[0]
    n_pos = 0
    for it in range(3):
        changed = 0
        for s in list(seg_chain):
            if s in seg_num or s not in starts or seg_team[s] == "GK": continue
            f0, x0, y0 = starts[s]; cands = []
            for t in seg_num:
                if seg_team[t] != seg_team[s] or t not in ends: continue
                f1, x1, y1 = ends[t]; gap = (f0 - f1) / 50.0
                if 0 < gap <= 1.0 and np.hypot(x0 - x1, y0 - y1) <= 1.5 + 6.0 * gap: cands.append((t, gap))
            nums = {seg_num[t] for t, _ in cands}
            if len(nums) == 1: seg_num[s] = nums.pop(); seg_votes[s] = 0; changed += 1
        # backwards: an unnamed segment whose end leads into a numbered segment's start
        for s in list(seg_chain):
            if s in seg_num or s not in ends or seg_team[s] == "GK": continue
            f1, x1, y1 = ends[s]; cands = []
            for t in seg_num:
                if seg_team[t] != seg_team[s] or t not in starts: continue
                f0, x0, y0 = starts[t]; gap = (f0 - f1) / 50.0
                if 0 < gap <= 1.0 and np.hypot(x0 - x1, y0 - y1) <= 1.5 + 6.0 * gap: cands.append((t, gap))
            nums = {seg_num[t] for t, _ in cands}
            if len(nums) == 1: seg_num[s] = nums.pop(); seg_votes[s] = 0; changed += 1
        n_pos += changed
        if not changed: break
    print(f"position-continuity propagation: {n_pos} segments numbered")
# per-frame uniqueness within a team
byf = collections.defaultdict(list)
for r in rows:
    if r["class_name"] == "juga":
        sid = int(r["seg_id"])
        if sid in seg_num: byf[int(r["frame_number"])].append(sid)
banned = set(); conflicts = 0
for f, sids in byf.items():
    seen = collections.defaultdict(list)
    for s in sids: seen[(seg_team[s], seg_num[s])].append(s)
    for key, ss in seen.items():
        if len(ss) > 1:
            conflicts += 1; ss.sort(key=lambda s: (-seg_votes[s], -(span[s][1]-span[s][0])))
            for s in ss[1:]: banned.add((s, f))
out_rows = []; used = collections.Counter()
for r in rows:
    lab = r["label"]
    if r["class_name"] in ("juga", "port") and not lab.startswith(("R", "b")):
        sid = int(r["seg_id"]); f = int(r["frame_number"]); p = "GK" if r["class_name"] == "port" else seg_team.get(sid, prefix(lab))
        if p == "GK": new = "GK L" if "left" in lab else "GK R"
        else:
            n = seg_num.get(sid); new = f"{p} {n}" if n and (sid, f) not in banned else f"{p} ?"
        used[new] += 1; r = dict(r); r["label"] = new
    out_rows.append(r)
with open(out, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(out_rows)
tot_seg = len([s for s in seg_chain if seg_team[s] != "GK"])
print(f"segments {tot_seg}: numbered from own reads {n_read}, inherited {n_inh}; per-frame same-number conflicts {conflicts} (loser set to '?')")
print("labels used:", sorted(used.items(), key=lambda kv: (kv[0][:2], kv[0])))
