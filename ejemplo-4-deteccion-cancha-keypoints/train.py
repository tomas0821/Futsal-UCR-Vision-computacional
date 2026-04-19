import os
from ultralytics import YOLO
import torch

def train():
    # Verificar GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Cargar el modelo YOLOv8-pose (Medium)
    model = YOLO('yolov8m-pose.pt') 

    # Buscar el data.yaml
    possible_data_yaml = None
    for root, dirs, files in os.walk('.'):
        if 'data.yaml' in files:
            possible_data_yaml = os.path.join(root, 'data.yaml')
            break
    
    if not possible_data_yaml:
        print("Error: No se encontró data.yaml.")
        return

    print(f"Entrenando detección de cancha con: {possible_data_yaml}")
    
    # Iniciar entrenamiento de pose/keypoints
    results = model.train(
        data=possible_data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project='futsal_pitch_detection',
        name='yolov8m_pitch_keypoints'
    )
    
    print("Entrenamiento de keypoints completado.")

if __name__ == "__main__":
    train()
