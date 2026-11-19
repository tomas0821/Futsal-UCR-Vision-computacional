"""Fit the fixed-camera fisheye model from landmarks clicked with click_landmarks.html.

Usage:
    python fit_from_clicks.py <match>_landmarks.json <frame.jpg|png> [--out calibration/<match>_camera.npz]

Prints the reprojection error per landmark (px and approx. metres on the ground), writes
<match>_camera.npz (+ .json summary) and <match>_overlay.jpg with the pitch outline drawn on the frame.
Needs >= 8 clicked points including both goals for a reliable fit.
"""
import json, sys, os, numpy as np
from scipy import optimize
from scipy.spatial.transform import Rotation as Rot
from PIL import Image, ImageDraw

def project(p, P3, W):
    rv, C, f, k1, k2, cx, cy = p[:3], p[3:6], p[6], p[7], p[8], p[9], p[10]
    R = Rot.from_rotvec(rv).as_matrix()
    Pw = np.c_[P3[:,0], W - P3[:,1], P3[:,2]] - C          # model frame: Y away from camera
    Pc = Pw @ R.T; z = np.maximum(Pc[:,2], 1e-6)
    rho = np.hypot(Pc[:,0], Pc[:,1]); th = np.arctan2(rho, z); thd = th*(1 + k1*th**2 + k2*th**4)
    s = np.where(rho>1e-9, f*thd/np.maximum(rho,1e-9), 0)
    return np.c_[cx + Pc[:,0]*s, cy + Pc[:,1]*s]

def main():
    jpath, ipath = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else os.path.join(os.path.dirname(os.path.abspath(jpath)), f"{json.load(open(jpath))['match']}_camera.npz")
    J = json.load(open(jpath)); L, W = J["pitch"]["length_m"], J["pitch"]["width_m"]
    pts = [p for p in J["points"] if p["u"] is not None]
    if len(pts) < 8: sys.exit(f"only {len(pts)} clicked points; need at least 8 (include both goals)")
    P3 = np.array([[p["X"], p["Y"], p["Z"]] for p in pts], float); uv = np.array([[p["u"], p["v"]] for p in pts], float)
    ids = [p["id"] for p in pts]
    im = Image.open(ipath).convert("RGB"); IW, IH = im.size
    if J.get("image") and (J["image"]["width"], J["image"]["height"]) != (IW, IH):
        print(f"warning: JSON was clicked on {J['image']['width']}x{J['image']['height']}, frame is {IW}x{IH}; scaling clicks")
        uv *= np.array([IW/J["image"]["width"], IH/J["image"]["height"]])

    # initial pose: camera behind the near touchline, above, looking at the far third
    C0 = np.array([L/2, -5.0, 4.5]); fwd = np.array([L/2, W-2.0, 0]) - C0; fwd /= np.linalg.norm(fwd)
    down = np.array([0,0,-1.0]); down -= fwd*np.dot(down,fwd); down /= np.linalg.norm(down); right = np.cross(down, fwd)
    p0 = np.r_[Rot.from_matrix(np.vstack([right,down,fwd])).as_rotvec(), C0, 0.3*IW, 0.0, 0.0, IW/2, IH/2]
    lo = np.r_[-np.pi*np.ones(3), -10, -30, 1.0, 0.12*IW, -1.0, -1.0, 0.42*IW, 0.40*IH]
    hi = np.r_[ np.pi*np.ones(3), L+10, -0.3, 15.0, 0.9*IW, 1.0, 1.0, 0.58*IW, 0.60*IH]
    state = {"P3": P3, "uv": uv}
    res = lambda p: (project(p, state["P3"], W) - state["uv"]).ravel()
    best = None
    for f0 in (0.22, 0.3, 0.4):                       # a few focal starts to avoid a bad basin
        p0[6] = f0*IW
        r = optimize.least_squares(res, np.clip(p0, lo+1e-9, hi-1e-9), bounds=(lo,hi), loss="soft_l1", f_scale=8, max_nfev=5000)
        if best is None or r.cost < best.cost: best = r
    p = best.x
    err = np.linalg.norm(project(p, P3, W) - uv, axis=1)
    dropped = [ids[i] for i in np.flatnonzero(err > 40)]
    if dropped and len(ids) - len(dropped) >= 8:
        print(f"dropping {len(dropped)} point(s) > 40 px and refitting: {', '.join(dropped)}  (optional points on the wrong marking are the usual cause)")
        keep = err <= 40; P3, uv, ids = P3[keep], uv[keep], [i for i,k in zip(ids,keep) if k]; state["P3"], state["uv"] = P3, uv
        r = optimize.least_squares(res, p, bounds=(lo,hi), loss="soft_l1", f_scale=8, max_nfev=5000); p = r.x
        err = np.linalg.norm(project(p, P3, W) - uv, axis=1)

    # local ground scale (px per m) at each ground point -> error in metres
    def m_per_px(X, Y):
        a = project(p, np.array([[X,Y,0],[X+0.2,Y,0],[X,Y+0.2,0]]), W); return 0.2/max(np.linalg.norm(a[1]-a[0]), np.linalg.norm(a[2]-a[0]), 1e-6)
    print(f"{'id':6s} {'px err':>7s} {'~m err':>7s}")
    for i,pid in enumerate(ids):
        m = err[i]*m_per_px(P3[i,0], P3[i,1]) if P3[i,2]==0 else float('nan')
        print(f"{pid:6s} {err[i]:7.1f} {m:7.2f}")
    print(f"median {np.median(err):.1f} px, max {err.max():.1f} px | camera at X={p[3]:.1f} m, {-p[4]:.1f} m behind near touchline, {p[5]:.1f} m high | f={p[6]:.0f} k=({p[7]:.3f},{p[8]:.3f}) c=({p[9]:.0f},{p[10]:.0f})")
    if err.max() > 40: print("!! one or more points are still > 40 px off: re-check them in the clicker (wrong post, wrong corner, mirrored goal)")

    np.savez(out, params=p, L=L, W=W, image_size=(IW,IH), ids=np.array(ids), uv=uv, err=err)
    json.dump({"match": J["match"], "source_json": os.path.basename(jpath), "frame": os.path.basename(ipath), "pitch_m": [L,W],
               "camera": {"X": p[3], "behind_near_touchline_m": -p[4], "height_m": p[5]}, "f": p[6], "k1": p[7], "k2": p[8], "cx": p[9], "cy": p[10],
               "reprojection_px": {"median": float(np.median(err)), "max": float(err.max())}, "per_point_px": dict(zip(ids, err.round(1).tolist())), "dropped": dropped},
              open(out.replace(".npz",".json"),"w"), indent=1)

    # overlay
    d = ImageDraw.Draw(im); s = IW/1920
    def poly(P):  # draw a polyline of ground points
        q = project(p, np.c_[P, np.zeros(len(P))], W)
        for (x,y) in q:
            if 0<=x<IW and 0<=y<IH: d.ellipse([x-1.5*s,y-1.5*s,x+1.5*s,y+1.5*s], fill=(255,40,40))
    t = np.linspace(0,1,600)[:,None]
    for a,b in [((0,0),(L,0)),((0,W),(L,W)),((0,0),(0,W)),((L,0),(L,W)),((L/2,0),(L/2,W))]: poly(np.array(a)+t*(np.array(b)-np.array(a)))
    ang = np.linspace(0,2*np.pi,400); poly(np.c_[L/2+3*np.cos(ang), W/2+3*np.sin(ang)])
    for X,sg in ((0,1),(L,-1)):
        y1,y2 = W/2-J["goal"]["width_m"]/2, W/2+J["goal"]["width_m"]/2
        a1 = np.linspace(-np.pi/2, 0, 100) if sg>0 else np.linspace(np.pi, 3*np.pi/2, 100); poly(np.c_[X+6*np.cos(a1), y1+6*np.sin(a1)])
        a2 = np.linspace(0, np.pi/2, 100) if sg>0 else np.linspace(np.pi/2, np.pi, 100); poly(np.c_[X+6*np.cos(a2), y2+6*np.sin(a2)])
        poly(np.c_[np.full(50, X+6*sg), np.linspace(y1,y2,50)])
        for Y in (y1,y2):
            q = project(p, np.array([[X,Y,0],[X,Y,J["goal"]["height_m"]]]), W); d.line([tuple(q[0]),tuple(q[1])], fill=(255,255,0), width=int(3*s))
    for (x,y),pid in zip(uv, ids): d.ellipse([x-5*s,y-5*s,x+5*s,y+5*s], outline=(0,255,120), width=int(2*s)); d.text((x+7*s,y-7*s), pid, fill=(0,255,120))
    ov = out.replace(".npz","_overlay.jpg"); im.save(ov, quality=90)
    print("wrote", out, "and", ov)

if __name__ == "__main__":
    main()
