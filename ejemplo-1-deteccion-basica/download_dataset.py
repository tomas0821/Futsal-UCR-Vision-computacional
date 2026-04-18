import os
from roboflow import Roboflow
from dotenv import load_dotenv

def download():
    # Cargar variables de entorno desde el archivo .env
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    project_full_name = os.getenv("ROBOFLOW_PROJECT") # formato: espacio_trabajo/proyecto/version
    
    parts = project_full_name.split("/")
    if len(parts) != 3:
        print("Error: ROBOFLOW_PROJECT debe estar en el formato 'espacio_trabajo/proyecto/version'")
        return
    
    workspace_name, project_id, version_number = parts
    
    # Inicializar la conexión con Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_id)
    
    # Descargar la versión específica del dataset para YOLOv8
    dataset = project.version(int(version_number)).download("yolov8")
    
    print(f"Dataset descargado en: {dataset.location}")

if __name__ == "__main__":
    download()
