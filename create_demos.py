import cv2
import os

def extract_demo(input_path, output_path, start_sec, duration_sec=10):
    if not os.path.exists(input_path):
        print(f"Saltando: {input_path} (No existe)")
        return

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Redimensionar a 720p para que pese poco en GitHub
    target_w, target_h = 1280, 720
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Extrayendo demo de {input_path}...")
    count = 0
    while count < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret: break
        
        frame_res = cv2.resize(frame, (target_w, target_h))
        out.write(frame_res)
        count += 1
        
    cap.release()
    out.release()
    print(f"¡Demo guardada en: {output_path}!")

if __name__ == "__main__":
    # Demo 1: Balón
    extract_demo('./ejemplo-1-deteccion-basica/runs/detect/futsal_prediction/match_video_result_yolov8m/20250719_193728.avi', 
                 './ejemplo-1-deteccion-basica/demo_balon.mp4', 20)
    
    # Demo 2: Personas
    extract_demo('./ejemplo-2-jugadores-y-arbitros/runs/detect/futsal_prediction/resultado_jugadores_arbitros/video2.avi', 
                 './ejemplo-2-jugadores-y-arbitros/demo_personas.mp4', 10)
    
    # Demo 3: Combinado
    extract_demo('./ejemplo-3-modelos-combinados/videos/video_combinado.mp4', 
                 './ejemplo-3-modelos-combinados/demo_combinado.mp4', 5)
    
    # Demo 4: Cancha
    extract_demo('./ejemplo-4-deteccion-cancha-keypoints/videos/resultado_keypoints_mejorado.avi', 
                 './ejemplo-4-deteccion-cancha-keypoints/demo_cancha.mp4', 0)
