import os, sys
from ultralytics import YOLO
import torch
variant = os.environ.get("VARIANT", "4cls")
data = os.path.join(os.getcwd(), "Grecia-7", f"data_{variant}.yaml")
model = YOLO("yolov8m.pt")
model.train(data=data, epochs=100, imgsz=1280, batch=8, degrees=5, close_mosaic=15, device=0 if torch.cuda.is_available() else "cpu", project="futsal_grecia", name=f"yolov8m_grecia_v7_{variant}")
print("done", variant)
