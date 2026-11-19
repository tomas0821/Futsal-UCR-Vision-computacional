"""Compare two tracking runs of the same clip (track_details.txt, 1080p) + two ball tracks (ball_track.py output).
Usage: python compare_runs.py A_name A_track_details.txt A_ball_track.csv B_name B_track_details.txt B_ball_track.csv out.png [--fps 50] [--t0 600]
Prints a table (detections per class, per-frame person counts vs the expected 8 players + 2 keepers, track lifetimes,
ball coverage) and draws per-second timelines: people on court per team run, ball detected/interpolated/missing.
"""
import sys, csv, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a = sys.argv[1:4]; b = sys.argv[4:7]; out = sys.argv[7]
fps = float(sys.argv[sys.argv.index("--fps")+1]) if "--fps" in sys.argv else 50.0
t0 = float(sys.argv[sys.argv.index("--t0")+1]) if "--t0" in sys.argv else 0.0
def load(name, td, bt):
    rows = list(csv.DictReader(open(td))); nfr = max(int(r["frame_number"]) for r in rows)
    cls = collections.Counter(r["class_name"] for r in rows)
    perf = collections.defaultdict(lambda: collections.Counter())
    for r in rows: perf[int(r["frame_number"])][r["class_name"]] += 1
    juga = np.array([perf[f]["juga"] for f in range(1, nfr + 1)]); port = np.array([perf[f]["port"] for f in range(1, nfr + 1)])
    tracks = collections.defaultdict(list)
    for r in rows:
        if r["class_name"] in ("juga", "port", "ref"): tracks[int(r["track_id"])].append(int(r["frame_number"]))
    life = np.array([(max(v) - min(v) + 1) / fps for v in tracks.values()])
    ball = list(csv.DictReader(open(bt))); det = {int(r["frame_number"]) for r in ball if r["source"] == "det"}; interp = {int(r["frame_number"]) for r in ball if r["source"] == "interp"}
    ntl = len({r["tracklet"] for r in ball})
    return dict(name=name, nfr=nfr, cls=cls, juga=juga, port=port, life=life, det=det, interp=interp, ntl=ntl, ntracks=len(tracks))
A, B = load(*a), load(*b)
def row(label, f, fmt="{}"):
    print(f"{label:48s} {fmt.format(f(A)):>14s} {fmt.format(f(B)):>14s}")
print(f"{'':48s} {A['name']:>14s} {B['name']:>14s}")
for c in ("juga", "port", "ref", "bola"): row(f"detections {c}", lambda R, c=c: R["cls"][c])
row("frames", lambda R: R["nfr"])
row("players/frame mean (expect 8)", lambda R: R["juga"].mean(), "{:.2f}")
row("frames with exactly 8 players", lambda R: 100 * (R["juga"] == 8).mean(), "{:.0f}%")
row("frames with < 7 players", lambda R: 100 * (R["juga"] < 7).mean(), "{:.0f}%")
row("frames with > 8 players (subs/bench or dup)", lambda R: 100 * (R["juga"] > 8).mean(), "{:.0f}%")
row("frames with exactly 2 keepers", lambda R: 100 * (R["port"] == 2).mean(), "{:.0f}%")
row("person tracks", lambda R: R["ntracks"])
row("track life median s", lambda R: np.median(R["life"]), "{:.1f}")
row("track life p90 s", lambda R: np.percentile(R["life"], 90), "{:.1f}")
row("tracks > 60 s", lambda R: int((R["life"] > 60).sum()))
row("ball detected (frames)", lambda R: 100 * len(R["det"]) / R["nfr"], "{:.0f}%")
row("ball detected + interpolated", lambda R: 100 * (len(R["det"]) + len(R["interp"])) / R["nfr"], "{:.0f}%")
row("ball tracklets", lambda R: R["ntl"])
# figure: per-second timelines
sec = int(np.ceil(A["nfr"] / fps)); t = t0 + np.arange(sec)
fig, ax = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)
for R, col in ((A, "#1f77b4"), (B, "#d62728")):
    ps = [R["juga"][int(s*fps):int((s+1)*fps)].mean() for s in range(sec)]
    ax[0].plot(t, ps, color=col, lw=1, label=R["name"])
    bd = [100 * np.mean([f in R["det"] for f in range(int(s*fps)+1, int((s+1)*fps)+1)]) for s in range(sec)]
    ax[1].plot(t, bd, color=col, lw=1, label=R["name"])
    bi = [100 * np.mean([(f in R["det"]) or (f in R["interp"]) for f in range(int(s*fps)+1, int((s+1)*fps)+1)]) for s in range(sec)]
    ax[2].plot(t, bi, color=col, lw=1, label=R["name"])
ax[0].axhline(8, color="k", ls=":", lw=0.8); ax[0].set_ylabel("players detected\nper frame (8 expected)"); ax[0].legend(loc="lower right", ncol=2, fontsize=9)
ax[1].set_ylabel("ball detected\n% of frames / s"); ax[2].set_ylabel("ball det.+interp.\n% of frames / s"); ax[2].set_xlabel("match time (s)")
for x in ax: x.grid(alpha=0.3)
fig.suptitle(f"5-min clip, {A['name']} vs {B['name']}: players per frame and ball coverage"); fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out)
