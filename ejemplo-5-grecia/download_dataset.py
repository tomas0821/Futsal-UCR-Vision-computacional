import os
from roboflow import Roboflow
from dotenv import load_dotenv

def download():
    # Cargar variables desde el .env del proyecto principal
    load_dotenv("../ejemplo-1-deteccion-basica/.env")
    api_key = os.getenv("ROBOFLOW_API_KEY")

    # Dataset específico: grecia versión 7 (frames del partido San Ramón - Grecia, 9 de mayo 2026)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("football-ucr").project("grecia")
    print("Tipo de proyecto:", project.type)
    dataset = project.version(7).download("yolov8")
    print(f"Dataset descargado en: {dataset.location}")

if __name__ == "__main__":
    download()
