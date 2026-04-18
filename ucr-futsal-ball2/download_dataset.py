import os
from roboflow import Roboflow
from dotenv import load_dotenv

def download():
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    project_full_name = os.getenv("ROBOFLOW_PROJECT") # workspace/project/version
    
    parts = project_full_name.split("/")
    if len(parts) != 3:
        print("Error: ROBOFLOW_PROJECT must be in format 'workspace/project/version'")
        return
    
    workspace_name, project_id, version_number = parts
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_id)
    dataset = project.version(int(version_number)).download("yolov8")
    
    print(f"Dataset downloaded to: {dataset.location}")

if __name__ == "__main__":
    download()
