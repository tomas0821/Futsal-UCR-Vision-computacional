import os
from ultralytics import YOLO
import torch

def train():
    # Verificar GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Modelo base Medium (mismo que ejemplos 1 y 2)
    model = YOLO("yolov8m.pt")

    # Dataset Grecia v5: 4 clases (bola, juga, port, ref), 840 train imgs 1280x1280 (2x aug)
    data_yaml = os.path.join(os.getcwd(), "Grecia-5/data.yaml")
    if not os.path.exists(data_yaml):
        print(f"Error: No se encontro {data_yaml}")
        return
    print(f"Entrenando con: {data_yaml}")

    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=1280,
        batch=8,
        degrees=5,
        close_mosaic=15,
        device=device,
        project="futsal_grecia",
        name="yolov8m_grecia_v5",
    )
    print("Entrenamiento completado.")

if __name__ == "__main__":
    train()
