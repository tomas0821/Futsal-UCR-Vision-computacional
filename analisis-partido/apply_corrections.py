"""Apply identity corrections from review_identities.html to the stitched data.
Usage: python apply_corrections.py track_details_stitched.txt identity_corrections.json out_prefix [--calib calibration/<match>_camera.npz] [--scale 2]
Writes <out>_track_details.txt (1080p boxes with corrected label) and <out>_stitched.csv (pitch coords with player_id),
so render_tracks.py / player_stats.py / chain_gantt.py can be re-run on the corrected identities.
Operations: relabel (whole segment), split (segment from a frame on), swap (two identities from a frame on, all segments),
delete (segment), mark (manual feet position of an identity at a frame; interpolated linearly between marks, only where the
identity has no detection within 1 s).
"""
import sys, csv, json, collections, numpy as np
src, cj, pre = sys.argv[1], sys.argv[2], sys.argv[3]
scale = float(sys.argv[sys.argv.index("--scale")+1]) if "--scale" in sys.argv else 2.0
calib = sys.argv[sys.argv.index("--calib")+1] if "--calib" in sys.argv else None
J = json.load(open(cj)); corr = J["corrections"]; fps = J.get("fps", 50)
rows = list(csv.DictReader(open(src)))
def pid_of(r):
    sid = int(r.get("seg_id") or r["track_id"]); f = int(r["frame_number"]); pid = r.get("label") or ""
    if r["class_name"] not in ("juga", "port"): return pid, False
    for c in corr:
        if c["op"] == "swap" and f >= c["frame"]:
            if pid == c["a"]: pid = c["b"]
            elif pid == c["b"]: pid = c["a"]
        if c.get("seg") != sid: continue
        if c["op"] == "relabel": pid = c["to"]
        elif c["op"] == "split" and f >= c["frame"]: pid = c["to"]
        elif c["op"] == "delete": return pid, True
    return pid, False
out_rows = []; n_del = 0; changed = 0
for r in rows:
    pid, deleted = pid_of(r)
    if deleted: n_del += 1; continue
    if pid != (r.get("label") or ""): changed += 1
    r = dict(r); r["label"] = pid; out_rows.append(r)
# manual marks -> synthetic boxes (feet position), interpolated per identity where no detection within 1 s
marks = collections.defaultdict(list)
for c in corr:
    if c["op"] == "mark": marks[c["to"]].append((c["frame"], c["x"], c["y"]))
have = collections.defaultdict(set)
for r in out_rows: have[r["label"]].add(int(r["frame_number"]))
n_syn = 0; step = 2
for pid, ms in marks.items():
    ms.sort()
    pairs = list(zip(ms, ms[1:])) if len(ms) > 1 else []
    for (f0, x0, y0), (f1, x1, y1) in pairs:
        for f in range(f0, f1 + 1, step):
            if any(abs(f - g) <= fps for g in have[pid]): continue
            t = (f - f0) / max(f1 - f0, 1); x = x0 + (x1 - x0) * t; y = y0 + (y1 - y0) * t
            out_rows.append({"frame_number": str(f), "track_id": "0", "class_id": "1", "class_name": "juga", "x_center": f"{x:.1f}", "y_center": f"{y - 45:.1f}", "width": "40", "height": "90", "confidence": "1.0", "team": pid[0], "seg_id": "0", "label": pid}); n_syn += 1
    if not pairs:   # a single mark: one synthetic box
        f0, x0, y0 = ms[0]
        if not any(abs(f0 - g) <= fps for g in have[pid]):
            out_rows.append({"frame_number": str(f0), "track_id": "0", "class_id": "1", "class_name": "juga", "x_center": f"{x0:.1f}", "y_center": f"{y0 - 45:.1f}", "width": "40", "height": "90", "confidence": "1.0", "team": pid[0], "seg_id": "0", "label": pid}); n_syn += 1
out_rows.sort(key=lambda r: (int(r["frame_number"]), r["label"]))
with open(f"{pre}_track_details.txt", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
# pitch coords
sys.argv = ["x"] + (["--calib", calib] if calib else []); sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from pitch_map import image_to_pitch
feet = np.array([[float(r["x_center"]), float(r["y_center"]) + float(r["height"]) / 2] for r in out_rows]) * scale; pm = image_to_pitch(feet)
base = {"A": 1000, "B": 2000, "G": 3000}
with open(f"{pre}_stitched.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame_number", "track_id", "class_name", "team", "player_id", "confidence", "pitch_x_m", "pitch_y_m"])
    for r, (x, y) in zip(out_rows, pm):
        lab = r["label"]; num = base.get(lab[0], 9000) + (int("".join(ch for ch in lab if ch.isdigit()) or 0)) if r["class_name"] in ("juga", "port") else int(r["track_id"])
        w.writerow([r["frame_number"], num, r["class_name"], r.get("team", ""), lab if r["class_name"] in ("juga", "port") else r["class_name"], r["confidence"], f"{x:.2f}", f"{y:.2f}"])
print(f"corrections {len(corr)}: labels changed on {changed} boxes, {n_del} boxes deleted, {n_syn} synthetic boxes from marks -> {pre}_track_details.txt, {pre}_stitched.csv")
