import os
from roboflow import Roboflow
from dotenv import load_dotenv

def download():
    # Cargar variables desde el .env del proyecto principal
    load_dotenv("../ejemplo-1-deteccion-basica/.env")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    # NUEVO Dataset: ucr-futsal-field3 versión 3
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("football-ucr").project("ucr-futsal-field3")
    dataset = project.version(3).download("yolov8")
    
    print(f"Dataset descargado en: {dataset.location}")

if __name__ == "__main__":
    download()
