from ultralytics import YOLO
import cv2, os

MODEL_PATH = 'runs/pose/futsal_pitch_detection/yolov8m_pitch_field3/weights/best.pt'
VIDEO_PATH = '../ejemplo-2-jugadores-y-arbitros/videos/video2.mp4'
OUTPUT_DIR = 'futsal_pitch_prediction/resultado_keypoints'
CONF       = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, os.path.basename(VIDEO_PATH))

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)
fps   = cap.get(cv2.CAP_PROP_FPS)
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out   = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print(f'Procesando {VIDEO_PATH} → {output_path}')
for result in model.predict(source=VIDEO_PATH, conf=CONF, stream=True, verbose=False):
    out.write(result.plot())

cap.release()
out.release()
print(f'Video guardado en: {output_path}')
