"""Train a small jersey-number classifier on torso crops (classes = roster numbers + 'none').
Usage: <ocr venv python> train_digits.py dataset_dir out_model.pt [--epochs 25] [--size 64x96]
Also writes out_model.json with the class list and the validation confusion summary.
"""
import sys, os, json, time, collections, torch, torch.nn as nn
from torchvision import datasets, transforms, models
root, out = sys.argv[1], sys.argv[2]
EP = int(sys.argv[sys.argv.index("--epochs")+1]) if "--epochs" in sys.argv else 25
W, H = (64, 96) if "--size" not in sys.argv else tuple(int(x) for x in sys.argv[sys.argv.index("--size")+1].split("x"))
torch.manual_seed(0); torch.set_num_threads(max(1, os.cpu_count() - 2))
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tr_tf = transforms.Compose([transforms.Resize((H, W)), transforms.RandomAffine(6, translate=(0.06, 0.06), scale=(0.9, 1.1)), transforms.ColorJitter(0.3, 0.3, 0.2, 0.02), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3), transforms.ToTensor(), norm])
va_tf = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor(), norm])
tr = datasets.ImageFolder(f"{root}/train", tr_tf); va = datasets.ImageFolder(f"{root}/val", va_tf); classes = tr.classes
counts = collections.Counter(tr.targets); weights = torch.tensor([1.0 / counts[i] ** 0.5 for i in range(len(classes))]); weights /= weights.mean()
tl = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True, num_workers=2); vl = torch.utils.data.DataLoader(va, batch_size=128, num_workers=2)
m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); m.fc = nn.Linear(m.fc.in_features, len(classes))
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * len(tl)); lossf = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
best = 0; t0 = time.time()
for ep in range(EP):
    m.train(); tot = 0; n = 0
    for x, y in tl:
        opt.zero_grad(); l = lossf(m(x), y); l.backward(); opt.step(); sched.step(); tot += l.item() * len(y); n += len(y)
    m.eval(); correct = 0; conf = collections.Counter(); per = collections.defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for x, y in vl:
            p = m(x).argmax(1)
            for a, b in zip(y.tolist(), p.tolist()): per[a][1] += 1; per[a][0] += a == b; conf[(a, b)] += a != b
            correct += (p == y).sum().item()
    acc = correct / len(va)
    if acc >= best: best = acc; torch.save({"state": m.state_dict(), "classes": classes, "size": (W, H)}, out)
    print(f"epoch {ep+1:2d}  loss {tot/n:.3f}  val acc {acc:.3f}  ({time.time()-t0:.0f} s)", flush=True)
per_class = {classes[a]: f"{c}/{t}" for a, (c, t) in sorted(per.items())}; worst = [(classes[a], classes[b], k) for (a, b), k in conf.most_common(6)]
json.dump({"classes": classes, "best_val_acc": best, "per_class_val": per_class, "top_confusions": worst, "size": [W, H]}, open(out.replace(".pt", ".json"), "w"), indent=1)
print("best val acc", round(best, 3), "| per class", per_class, "| top confusions", worst)
