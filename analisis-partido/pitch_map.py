"""Image -> pitch-plane mapping for the fixed Osmo camera (Grecia 2026-05-09 position).
Pitch frame: X along the far touchline from the left goal line (m), Y from the far touchline toward the camera (m).
Usage: python pitch_map.py track_details.txt out.csv [--scale 2] [--calib calibration/<match>_camera.npz]
  --scale 2  : tracks are in 1080p coordinates (frame is 4K)
  --calib    : per-match calibration from fit_from_clicks.py (default: grecia_2026-05-09_camera.npz)
"""
import numpy as np, sys, csv
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import LinearNDInterpolator
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
_calib = sys.argv[sys.argv.index("--calib")+1] if "--calib" in sys.argv else None
if _calib:
    _z = np.load(_calib); P = _z["params"]; L, W = float(_z["L"]), float(_z["W"])
else:
    _z = np.load(f"{HERE}/calibraciones/grecia_2026-05-09_camera.npz"); P = _z["params"]; L, W = float(_z["L"]), float(_z["W"])   # default: clicked Grecia calibration

def project(p, P3):
    rv, C, f, k1, k2, cx, cy = p[:3], p[3:6], p[6], p[7], p[8], p[9], p[10]; k3 = p[11] if len(p) > 11 else 0.0
    R = Rot.from_rotvec(rv).as_matrix()
    Pw = np.c_[P3[:,0], W - P3[:,1], P3[:,2]] - C
    Pc = Pw @ R.T; z = np.maximum(Pc[:,2], 1e-6)
    rho = np.hypot(Pc[:,0], Pc[:,1]); th = np.arctan2(rho, z); thd = th*(1 + k1*th**2 + k2*th**4 + k3*th**6)
    s = np.where(rho>1e-9, f*thd/np.maximum(rho,1e-9), 0)
    return np.c_[cx + Pc[:,0]*s, cy + Pc[:,1]*s]

# dense inverse on the ground plane (with margin), via interpolation
_gx, _gy = np.meshgrid(np.arange(-4, L+4, 0.1), np.arange(-3, W+6, 0.1))
_G = np.c_[_gx.ravel(), _gy.ravel(), np.zeros(_gx.size)]
_I = project(P, _G)
_ok = (_I[:,0]>-500)&(_I[:,0]<4340)&(_I[:,1]>-500)&(_I[:,1]<2660)
_inv = LinearNDInterpolator(_I[_ok], _G[_ok,:2])

def image_to_pitch(xy_4k):
    """xy_4k: (N,2) pixel coords in the 3840x2160 frame -> (N,2) pitch metres (nan outside)."""
    return _inv(np.asarray(xy_4k, float))

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[sys.argv.index("--scale")+1]) if "--scale" in sys.argv else 1.0
    rows = list(csv.DictReader(open(src)))
    feet = np.array([[float(r["x_center"]), float(r["y_center"]) + float(r["height"])/2] for r in rows])*scale
    pm = image_to_pitch(feet)
    with open(dst, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_number","track_id","class_id","class_name","confidence","img_x","img_y","pitch_x_m","pitch_y_m","inside"])
        for r,(ix,iy),(px,py) in zip(rows, feet, pm):
            inside = int(np.isfinite(px) and -0.5<=px<=L+0.5 and -0.5<=py<=W+0.5)
            w.writerow([r["frame_number"], r["track_id"], r["class_id"], r["class_name"], r["confidence"], f"{ix:.1f}", f"{iy:.1f}", f"{px:.2f}", f"{py:.2f}", inside])
    ok = np.isfinite(pm[:,0]); ins = ok & (pm[:,0]>=-0.5)&(pm[:,0]<=L+0.5)&(pm[:,1]>=-0.5)&(pm[:,1]<=W+0.5)
    print(f"{len(rows)} detections -> {ins.sum()} inside pitch (+0.5 m margin), {(~ok).sum()} unmapped")
    for c in sorted(set(r["class_name"] for r in rows)):
        m = np.array([r["class_name"]==c for r in rows]); print(f"  {c:5s}: inside {ins[m].mean()*100:5.1f}%   X range {np.nanmin(pm[m,0]):6.1f}..{np.nanmax(pm[m,0]):5.1f}  Y range {np.nanmin(pm[m,1]):5.1f}..{np.nanmax(pm[m,1]):5.1f}")
