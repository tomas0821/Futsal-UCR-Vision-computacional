"""Grecia-7: merge the stray classes (gk->port, player->juga) into the 4-class scheme of v5 and build two data yamls:
data_4cls.yaml (Roboflow split) and data_heldout.yaml (train = grecia1T + caribe from all splits, val = all loschiles)."""
import glob, os, yaml
D = "Grecia-7"; names6 = yaml.safe_load(open(f"{D}/data.yaml"))["names"]; target = ["bola", "juga", "port", "ref"]
alias = {"gk": "port", "player": "juga"}; remap = {i: target.index(alias.get(n, n)) for i, n in enumerate(names6)}
print("remap", {names6[i]: target[j] for i, j in remap.items()})
changed = 0
for f in glob.glob(f"{D}/*/labels/*.txt"):
    out = []
    for line in open(f):
        p = line.split()
        if not p: continue
        new = remap[int(p[0])]; changed += new != int(p[0]); out.append(" ".join([str(new)] + p[1:]) + "\n")
    open(f, "w").write("".join(out))
print("labels rewritten; boxes remapped:", changed)
root = os.path.abspath(D)
yaml.safe_dump({"path": root, "train": "train/images", "val": "valid/images", "test": "test/images", "nc": 4, "names": target}, open(f"{D}/data_4cls.yaml", "w"))
imgs = sorted(glob.glob(f"{root}/*/images/*"))
held = [p for p in imgs if os.path.basename(p).startswith("loschiles")]; rest = [p for p in imgs if p not in set(held)]
open(f"{D}/train_heldout.txt", "w").write("\n".join(rest) + "\n"); open(f"{D}/val_heldout.txt", "w").write("\n".join(held) + "\n")
yaml.safe_dump({"path": root, "train": f"{root}/train_heldout.txt", "val": f"{root}/val_heldout.txt", "nc": 4, "names": target}, open(f"{D}/data_heldout.yaml", "w"))
print("held-out split: train", len(rest), "(grecia1T + caribe), val", len(held), "(loschiles)")
