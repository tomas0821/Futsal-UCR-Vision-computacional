"""Merge track_details.txt files from consecutive clips of one recording into a single file.
Usage: python merge_track_parts.py out.txt part1.txt:0 part2.txt:40816 part3.txt:68822
Each argument is <file>:<frame offset of that part's first frame within the full recording>. Track IDs get +100000*part_index so
they stay unique (BotSort restarts numbering in each part).
"""
import sys, csv
out = sys.argv[1]; parts = [a.rsplit(":",1) for a in sys.argv[2:]]
n = 0
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame_number","track_id","class_id","class_name","x_center","y_center","width","height","confidence"])
    for k,(path,off) in enumerate(parts):
        off = int(off); m = 0
        for r in csv.DictReader(open(path)):
            w.writerow([int(r["frame_number"])+off, int(r["track_id"])+100000*k, r["class_id"], r["class_name"], r["x_center"], r["y_center"], r["width"], r["height"], r["confidence"]]); m += 1
        print(f"part {k+1}: {path}  offset {off}  rows {m}"); n += m
print("total rows", n, "->", out)
