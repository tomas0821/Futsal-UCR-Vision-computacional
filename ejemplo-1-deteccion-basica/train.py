import os
from ultralytics import YOLO
import torch

def train():
    # Verificar si hay una GPU disponible
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Cargar el modelo
    # Se recomienda cargar un modelo pre-entrenado para iniciar el entrenamiento
    model = YOLO('yolov8n.pt') 

    # Entrenar el modelo
    # Buscamos el archivo data.yaml en la carpeta del dataset descargado
    # Roboflow suele crear una carpeta con el nombre del proyecto y versión
    
    possible_data_yaml = None
    for root, dirs, files in os.walk('.'):
        if 'data.yaml' in files:
            possible_data_yaml = os.path.join(root, 'data.yaml')
            break
    
    if not possible_data_yaml:
        print("Error: No se encontró data.yaml. Asegúrate de que el dataset esté descargado.")
        return

    print(f"Usando configuración de datos desde: {possible_data_yaml}")
    
    # Iniciar el proceso de entrenamiento
    results = model.train(
        data=possible_data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project='futsal_training',
        name='yolov8_ball_detection'
    )
    
    print("Entrenamiento completado.")

if __name__ == "__main__":
    train()
