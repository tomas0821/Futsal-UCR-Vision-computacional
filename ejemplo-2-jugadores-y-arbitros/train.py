import os
from ultralytics import YOLO
import torch

def train():
    # Verificar GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Cargar el modelo Medium
    model = YOLO('yolov8m.pt') 

    # Ruta al archivo filtrado
    data_yaml = os.path.join(os.getcwd(), 'futsal-ucr-8/data_sin_balon.yaml')
    
    if not os.path.exists(data_yaml):
        print(f"Error: No se encontró {data_yaml}")
        return

    print(f"Entrenando con: {data_yaml}")
    
    # Iniciar entrenamiento (gk, player, ref)
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project='futsal_jugadores_arbitros',
        name='yolov8m_sin_balon'
    )
    
    print("Entrenamiento completado.")

if __name__ == "__main__":
    train()
