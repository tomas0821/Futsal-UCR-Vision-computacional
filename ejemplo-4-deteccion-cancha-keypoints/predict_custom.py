import cv2
from ultralytics import YOLO
import os
import sys
import numpy as np

def run_custom_inference():
    print("Iniciando script de inferencia optimizada (Fidelidad Total)...", flush=True)
    model_path = 'runs/pose/futsal_pitch_detection/yolov8m_pitch_field3/weights/best.pt'
    model = YOLO(model_path)
    
    input_path = 'videos/CAM_20260316011334_0014_D.MP4'
    cap = cv2.VideoCapture(input_path)
    
    # Trabajaremos todo a 1280x720 para máxima velocidad y calce perfecto
    target_w = 1280
    target_h = 720
    output_path = 'videos/resultado_keypoints_mejorado.avi'
    
    print(f"Dimensiones de procesamiento: {target_w}x{target_h}", flush=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (target_w, target_h))
    
    if not out.isOpened():
        print("Error crítico: No se pudo crear el archivo AVI.", flush=True)
        return

    frame_count = 0
    detections_found = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        # 1. Redimensionar primero (más eficiente que dárselo a YOLO en 4K)
        frame_res = cv2.resize(frame, (target_w, target_h))

        # 2. Inferencia sobre el frame ya redimensionado
        results = model(frame_res, conf=0.1, verbose=False, device=0)[0]

        if frame_count % 100 == 0:
            print(f"Procesando frame {frame_count}...", flush=True)

        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            # Los puntos ya están en el espacio 1280x720
            kpts = results.keypoints.xy[0].cpu().numpy()
            confs = results.keypoints.conf[0].cpu().numpy()
            
            if kpts.shape[0] > 0:
                detections_found += 1
                if detections_found == 1:
                    print(f"¡Primera detección precisa en frame {frame_count}!", flush=True)

                for i, (kpt, conf) in enumerate(zip(kpts, confs)):
                    if conf > 0.1:
                        # Dibujo directo (calce perfecto)
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(frame_res, (x, y), 10, (0, 255, 255), -1)
                        cv2.putText(frame_res, str(i), (x + 15, y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        out.write(frame_res)

    cap.release()
    out.release()
    print(f"¡TERMINADO! Frames: {frame_count}, Detecciones: {detections_found}", flush=True)
    print(f"Archivo guardado en: {output_path}", flush=True)

if __name__ == "__main__":
    run_custom_inference()
