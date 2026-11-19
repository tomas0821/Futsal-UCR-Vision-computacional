"""Apply the trained jersey-number classifier to a crops folder and write reads in number_ocr's CSV format.
Usage: <ocr venv python> apply_digits.py model.pt crops_dir out_reads.csv [--min-prob 0.8] [--batch 256]
A crop yields a read when the top class is a number (not 'none') with probability >= min-prob; conf = that probability.
"""
import sys, csv, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms, models
mp, crops, out = sys.argv[1], sys.argv[2], sys.argv[3]
MINP = float(sys.argv[sys.argv.index("--min-prob")+1]) if "--min-prob" in sys.argv else 0.8
BS = int(sys.argv[sys.argv.index("--batch")+1]) if "--batch" in sys.argv else 256
ck = torch.load(mp, map_location="cpu", weights_only=False); classes = ck["classes"]; W, H = ck["size"]
m = models.resnet18(); m.fc = nn.Linear(m.fc.in_features, len(classes)); m.load_state_dict(ck["state"]); m.eval()
tf = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
idx = list(csv.DictReader(open(f"{crops}/index.csv"))); reads = []; none_i = classes.index("none") if "none" in classes else -1
with torch.no_grad():
    for i in range(0, len(idx), BS):
        batch = idx[i:i+BS]; x = torch.stack([tf(Image.open(f"{crops}/{r['file']}").convert("RGB")) for r in batch])
        p = torch.softmax(m(x), 1); prob, cls = p.max(1)
        for r, pr, c in zip(batch, prob.tolist(), cls.tolist()):
            if c != none_i and pr >= MINP: reads.append((r["file"], r["label"], r["frame"], classes[c], round(pr, 3)))
        if (i // BS) % 10 == 0: print(i + len(batch), "crops,", len(reads), "reads", flush=True)
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "label", "frame", "number", "conf"]); w.writerows(reads)
print(f"done: {len(reads)} reads from {len(idx)} crops ({100*len(reads)/max(len(idx),1):.0f}%) -> {out}")
