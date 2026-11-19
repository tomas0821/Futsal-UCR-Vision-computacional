"""Inferencia con tracking (BotSort) usando el modelo del Ejemplo 5.

Genera:
  - video anotado (mp4)
  - track_details.txt (CSV: frame_number,track_id,class_id,class_name,x_center,y_center,width,height,confidence)
"""
from ultralytics import YOLO
import cv2, os, sys

MODEL_PATH = "runs/detect/futsal_grecia/yolov8m_grecia_v5/weights/best.pt"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "videos/grecia_1T_600s_30s_upright_1080p.mp4"
OUTPUT_DIR = "futsal_prediction/v5_" + os.path.splitext(os.path.basename(VIDEO_PATH))[0]
CONF       = 0.4   # was 0.25: low-conf duplicate boxes created spurious tracks

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_video = os.path.join(OUTPUT_DIR, os.path.basename(VIDEO_PATH))
output_txt   = os.path.join(OUTPUT_DIR, "track_details.txt")


# Colores por clase (BGR) y dibujo discreto: caja fina + etiqueta pequena "id clase".
COLORS = {"bola": (0, 220, 0), "juga": (255, 200, 0), "port": (255, 0, 255), "ref": (0, 215, 255)}
SHORT  = {"bola": "b", "juga": "J", "port": "GK", "ref": "R"}
FONT, FSCALE, FTHICK, LW = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1, 1

def draw(frame, boxes, names):
    if boxes is None or len(boxes) == 0:
        return frame
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
    for (x1, y1, x2, y2), tid, cls in zip(boxes.xyxy.int().tolist(), ids, boxes.cls.int().tolist()):
        name = names[cls]; color = COLORS.get(name, (255, 255, 255))
        if name == "bola":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = max((x2 - x1), (y2 - y1)) // 2 + 6
            cv2.circle(frame, (cx, cy), r, color, 2)
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LW)
        label = f"{tid} {SHORT[name]}" if tid is not None else SHORT[name]
        (tw, th), base = cv2.getTextSize(label, FONT, FSCALE, FTHICK)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
        cv2.rectangle(frame, (x1, ty - th - 3), (x1 + tw + 4, ty + base - 1), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty - 1), FONT, FSCALE, (0, 0, 0), FTHICK, cv2.LINE_AA)
    return frame

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)
fps   = cap.get(cv2.CAP_PROP_FPS)
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
out   = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print(f"Procesando {VIDEO_PATH} ({w}x{h} @ {fps:.1f} fps) -> {output_video}")
n = 0
with open(output_txt, "w") as f:
    f.write("frame_number,track_id,class_id,class_name,x_center,y_center,width,height,confidence\n")
    for result in model.track(source=VIDEO_PATH, conf=CONF, iou=0.5, agnostic_nms=True, imgsz=1280, tracker="botsort_tuned.yaml",
                              persist=True, stream=True, device=0, verbose=False):
        n += 1
        out.write(draw(result.orig_img.copy(), result.boxes, model.names))
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        for (x, y, bw, bh), tid, cls, cf in zip(boxes.xywh.tolist(), boxes.id.int().tolist(),
                                                 boxes.cls.int().tolist(), boxes.conf.tolist()):
            f.write(f"{n},{tid},{cls},{model.names[cls]},{x:.1f},{y:.1f},{bw:.1f},{bh:.1f},{cf:.4f}\n")

out.release()
print(f"Frames procesados: {n}")
print(f"Video guardado en: {output_video}")
print(f"Tracks guardados en: {output_txt}")
