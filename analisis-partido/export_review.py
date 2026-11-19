"""Export stitched identities + boxes to a compact JSON for review_identities.html.
Usage: python export_review.py track_details_stitched.txt out_review.json [--fps 50] [--step 2]
  track_details_stitched.txt: 1080p boxes with columns team, seg_id, label (from the stitching step)
  --step 2: keep every 2nd frame (25 fps review is enough; halves the file)
"""
import sys, csv, json, collections
src, out = sys.argv[1], sys.argv[2]
fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
step = int(sys.argv[sys.argv.index("--step")+1]) if "--step" in sys.argv else 2
frames = collections.defaultdict(list); segs = {}; ids = collections.Counter()
for r in csv.DictReader(open(src)):
    f = int(r["frame_number"])
    if f % step: continue
    cls = r["class_name"]; lab = r.get("label") or ""
    if cls == "bola":
        frames[f].append([0, "ball", "bola", round(float(r["x_center"])), round(float(r["y_center"])), round(float(r["width"])), round(float(r["height"]))]); continue
    if cls == "ref":
        frames[f].append([0, "R", "ref", round(float(r["x_center"])), round(float(r["y_center"])), round(float(r["width"])), round(float(r["height"]))]); continue
    sid = int(r.get("seg_id") or r["track_id"]); pid = lab or f"?{sid}"
    frames[f].append([sid, pid, cls, round(float(r["x_center"])), round(float(r["y_center"])), round(float(r["width"])), round(float(r["height"]))])
    s = segs.setdefault(sid, {"pid": pid, "team": r.get("team",""), "f0": f, "f1": f, "n": 0}); s["f1"] = f; s["n"] += 1; ids[pid] += 1
data = {"fps": fps, "step": step, "width": 1920, "height": 1080, "source": src,
        "identities": sorted(ids, key=lambda k: (k[0], len(k), k)),
        "segments": segs, "frames": {str(f): v for f, v in sorted(frames.items())}}
json.dump(data, open(out, "w"), separators=(",", ":"))
print(f"frames {len(frames)} (every {step}), boxes {sum(len(v) for v in frames.values())}, segments {len(segs)}, identities {len(ids)} -> {out}")
