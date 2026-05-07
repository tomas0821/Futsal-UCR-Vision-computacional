import cv2
from ultralytics import YOLO
import os

def run_combined():
    # 1. Cargar ambos modelos
    model_ball = YOLO('../ejemplo-1-deteccion-basica/runs/detect/futsal_training/yolov8m_ball_detection/weights/best.pt')
    model_people = YOLO('../ejemplo-2-jugadores-y-arbitros/runs/detect/futsal_jugadores_arbitros/yolov8m_sin_balon/weights/best.pt')

    # 2. Configurar el video de entrada y salida
    input_video = '../ejemplo-2-jugadores-y-arbitros/videos/video2.mp4'
    cap = cv2.VideoCapture(input_video)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = 'videos/video_combinado.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Procesando video: {input_video}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Inferencia del modelo de balón
        results_ball = model_ball(frame, verbose=False)[0]
        
        # 4. Inferencia del modelo de personas (gk, player, ref)
        results_people = model_people(frame, verbose=False)[0]

        # 5. Dibujar resultados en el frame
        # Dibujamos primero las personas
        annotated_frame = results_people.plot()
        
        # Dibujamos el balón encima (usamos el frame ya anotado)
        # Para evitar conflictos de IDs de clase, extraemos las cajas del balón manualmente
        for box in results_ball.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > 0.5:
                # Dibujar círculo o caja para el balón (Color Verde)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ball {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Frames procesados: {frame_count}")

    cap.release()
    out.release()
    print(f"Proceso finalizado. Video guardado en: {output_path}")

if __name__ == "__main__":
    run_combined()
