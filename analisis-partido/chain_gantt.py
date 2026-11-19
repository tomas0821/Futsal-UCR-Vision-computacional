"""Timeline of stitched player identities per team (Gantt) + concurrency check.  Usage: python chain_gantt.py stitched.csv out.png [--fps 50] [--t0 600]"""
import sys, csv, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]; fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0; t0 = float(sys.argv[sys.argv.index("--t0")+1]) if "--t0" in sys.argv else 0.0
rows = list(csv.DictReader(open(src))); frames = collections.defaultdict(set)
for r in rows:
    if r["class_name"] in ("juga","port"): frames[(r["team"], r["player_id"])].add(int(r["frame_number"]))
groups = [("A","San Ramón (dark)","#1f3a93"),("B","Grecia (grey)","#8c8c84"),("GK_left","GK left","#eb6834"),("GK_right","GK right","#eb6834")]
fig, axes = plt.subplots(len(groups), 1, figsize=(15, 9), sharex=True, facecolor="#ffffff", gridspec_kw={"height_ratios":[9,9,2,2]})
last = max(max(v) for v in frames.values())
for ax,(g,name,col) in zip(axes, groups):
    ids = sorted([k for k in frames if k[0]==g], key=lambda k: -len(frames[k]))
    occ = np.zeros(last+2)
    for row,k in enumerate(ids):
        fr = np.array(sorted(frames[k])); occ[fr] += 1
        # draw runs (gaps > 1 s broken)
        cuts = np.flatnonzero(np.diff(fr) > fps) ; starts = np.r_[fr[0], fr[cuts+1]]; ends = np.r_[fr[cuts], fr[-1]]
        for s,e in zip(starts, ends): ax.plot([s/fps+t0, e/fps+t0], [row, row], lw=5, color=col, solid_capstyle="butt", alpha=0.9)
        ax.text(t0-2, row, k[1], ha="right", va="center", fontsize=7, color="#6b6b66")
    over = occ > (4 if g in "AB" else 1)
    if over.any(): ax.fill_between(np.arange(len(occ))/fps+t0, -0.6, len(ids)-0.4, where=over, color="#e4572e", alpha=0.18, step="mid", lw=0)
    ax.set_ylim(len(ids)-0.4, -0.6); ax.set_yticks([]); ax.spines[["top","right","left"]].set_visible(False)
    ax.set_title(f"{name}: {len(ids)} identities · shaded = more than {4 if g in "AB" else 1} active at once ({100*over[:last].mean():.0f}% of frames)", loc="left", fontsize=9)
axes[-1].set_xlabel("match time (s)"); plt.tight_layout(); plt.savefig(out, dpi=150, facecolor="#ffffff"); print("saved", out)
