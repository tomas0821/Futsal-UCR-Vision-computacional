"""Read jersey numbers on torso crops with EasyOCR and vote per identity.
Usage: python number_ocr.py crops_dir out_prefix [--min-conf 0.4] [--min-votes 3] [--gpu]
Reads crops_dir/index.csv, writes <out>_reads.csv (every read) and <out>_numbers.csv (per identity: number, votes, agreement),
and <out>_montage.png (per identity: up to 6 crops that produced the winning read).
"""
import sys, os, csv, collections, re, numpy as np
from PIL import Image, ImageDraw, ImageFont
crops, pre = sys.argv[1], sys.argv[2]
MINC = float(sys.argv[sys.argv.index("--min-conf")+1]) if "--min-conf" in sys.argv else 0.4
MINV = int(sys.argv[sys.argv.index("--min-votes")+1]) if "--min-votes" in sys.argv else 3
import easyocr
reader = easyocr.Reader(["en"], gpu="--gpu" in sys.argv, verbose=False)
idx = list(csv.DictReader(open(f"{crops}/index.csv")))
reads = []; per = collections.defaultdict(collections.Counter); ex = collections.defaultdict(list)
for i, r in enumerate(idx):
    im = np.array(Image.open(f"{crops}/{r['file']}").convert("RGB"))
    res = reader.readtext(im, allowlist="0123456789", detail=1, paragraph=False, text_threshold=0.5, low_text=0.3, mag_ratio=1.5)
    for box, txt, conf in res:
        txt = re.sub(r"\D", "", txt)
        if not txt or len(txt) > 2 or conf < MINC: continue
        h = max(p[1] for p in box) - min(p[1] for p in box)
        if h < 0.25 * im.shape[0]: continue          # a number spans a good part of the torso crop
        reads.append((r["file"], r["label"], r["frame"], txt, round(float(conf), 3)))
        per[r["label"]][txt] += 1; ex[(r["label"], txt)].append((r["file"], float(conf)))
    if (i + 1) % 500 == 0: print(i + 1, "crops,", len(reads), "reads", flush=True)
with open(f"{pre}_reads.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "label", "frame", "number", "conf"]); w.writerows(reads)
labels = sorted(set(r["label"] for r in idx), key=lambda k: (k[0], len(k), k))
print(f"\n{'identity':10s} {'crops':>5s} {'reads':>5s} {'number':>6s} {'votes':>5s} {'agree':>6s}  other reads")
rows = []
ncr = collections.Counter(r["label"] for r in idx)
for lab in labels:
    c = per[lab]; tot = sum(c.values())
    if tot == 0: rows.append((lab, ncr[lab], 0, "", 0, 0.0, "")); print(f"{lab:10s} {ncr[lab]:5d} {0:5d} {'-':>6s}"); continue
    num, v = c.most_common(1)[0]; agree = v / tot; others = ", ".join(f"{k}×{n}" for k, n in c.most_common()[1:4])
    verdict = num if (v >= MINV and agree >= 0.5) else ""
    rows.append((lab, ncr[lab], tot, verdict, v, round(agree, 2), others))
    print(f"{lab:10s} {ncr[lab]:5d} {tot:5d} {num if verdict else '('+num+')':>6s} {v:5d} {agree:6.0%}  {others}")
with open(f"{pre}_numbers.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["identity", "crops", "reads", "number", "votes", "agreement", "other_reads"]); w.writerows(rows)
# montage
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12); TW, TH = 96, 96; cols = 7
named = [r for r in rows if r[2] > 0]
img = Image.new("RGB", (cols * TW, max(1, len(named)) * (TH + 16)), (246, 246, 242)); d = ImageDraw.Draw(img)
for row, (lab, ncrops, tot, verdict, v, agree, others) in enumerate(named):
    num = per[lab].most_common(1)[0][0]; y = row * (TH + 16)
    d.text((2, y + TH + 2), f"{lab}: {num if verdict else '('+num+')'} ×{v}/{tot}", fill=(30, 30, 28), font=font)
    for k, (fn, cf) in enumerate(sorted(ex[(lab, num)], key=lambda t: -t[1])[:cols - 1]):
        c = Image.open(f"{crops}/{fn}").convert("RGB"); c.thumbnail((TW, TH)); img.paste(c, ((k + 1) * TW, y))
img.save(f"{pre}_montage.png"); print("wrote", f"{pre}_numbers.csv", f"{pre}_montage.png")
