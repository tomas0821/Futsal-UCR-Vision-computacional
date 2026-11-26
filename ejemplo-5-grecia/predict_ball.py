"""Ball candidate detection on a 4K clip: full-frame or tiled inference, low confidence floor, all candidates kept.
Env: MODEL (weights), MODE=full|tiled, IMGSZ (input size), CONF (floor, default 0.05), TAG
Usage: python predict_ball.py videos/clip_4k.mp4 -> futsal_prediction/ball_<TAG>_<clip>/ball_candidates.txt
Output CSV: frame_number,x_center,y_center,width,height,confidence,tile   (4K pixel coords)
"""
import os, sys, cv2, numpy as np
from ultralytics import YOLO
MODEL = os.environ.get("MODEL", "runs/detect/futsal_grecia/yolov8m_grecia_v5/weights/best.pt")
MODE = os.environ.get("MODE", "tiled"); IMGSZ = int(os.environ.get("IMGSZ", "1920")); CONF = float(os.environ.get("CONF", "0.05")); TAG = os.environ.get("TAG", "v5tiled")
VIDEO = sys.argv[1]; OUT = f"futsal_prediction/ball_{TAG}_" + os.path.splitext(os.path.basename(VIDEO))[0]; os.makedirs(OUT, exist_ok=True)
model = YOLO(MODEL); names = model.names
ball_ids = [k for k, v in names.items() if str(v).lower() in ("bola", "ball", "balon", "sports ball")] or [0]
print("model", MODEL, "classes", names, "ball ids", ball_ids, "mode", MODE, "imgsz", IMGSZ, flush=True)
cap = cv2.VideoCapture(VIDEO); W = int(cap.get(3)); H = int(cap.get(4)); n = 0
tw, th = int(W*0.575), int(H*0.575)
tiles = [(0, 0, W, H)] if MODE == "full" else [(x0, y0, x0 + tw, y0 + th) for x0 in (0, W - tw) for y0 in (0, H - th)]
with open(f"{OUT}/ball_candidates.txt", "w") as f:
    f.write("frame_number,x_center,y_center,width,height,confidence,tile\n")
    while True:
        ok, frame = cap.read()
        if not ok: break
        n += 1
        crops = [frame[y0:y1, x0:x1] for (x0, y0, x1, y1) in tiles]
        res = model.predict(crops, imgsz=IMGSZ, conf=CONF, classes=ball_ids, device=0, verbose=False, half=True)
        for (x0, y0, x1, y1), r, ti in zip(tiles, res, range(len(tiles))):
            if r.boxes is None: continue
            for (x, y, w, h), c in zip(r.boxes.xywh.tolist(), r.boxes.conf.tolist()):
                f.write(f"{n},{x + x0:.1f},{y + y0:.1f},{w:.1f},{h:.1f},{c:.4f},{ti}\n")
        if n % 1000 == 0: print(n, flush=True)
print("frames", n, "->", OUT)
