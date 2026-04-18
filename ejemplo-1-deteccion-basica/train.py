import os
from ultralytics import YOLO
import torch

def train():
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # The data path is usually the dataset folder name/data.yaml
    # We'll assume the dataset is downloaded to the current directory's 'datasets' or similar.
    # Roboflow usually creates a folder like project-version
    
    # We'll look for data.yaml in the downloaded dataset folder
    possible_data_yaml = None
    for root, dirs, files in os.walk('.'):
        if 'data.yaml' in files:
            possible_data_yaml = os.path.join(root, 'data.yaml')
            break
    
    if not possible_data_yaml:
        print("Error: data.yaml not found. Make sure dataset is downloaded.")
        return

    print(f"Using data configuration from: {possible_data_yaml}")
    
    results = model.train(
        data=possible_data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        project='futsal_training',
        name='yolov8_ball_detection'
    )
    
    print("Training complete.")

if __name__ == "__main__":
    train()
