from ultralytics import YOLO
import cv2, os

MODEL_PATH = 'runs/detect/futsal_jugadores_arbitros/yolov8m_sin_balon/weights/best.pt'
VIDEO_PATH = 'videos/video2.mp4'
OUTPUT_DIR = 'futsal_prediction/resultado_jugadores_arbitros'
CONF       = 0.5

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
