#!/bin/bash
#SBATCH --job-name=pitch_yolo_cli
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pitch_cli_%j.log
#SBATCH --error=pitch_cli_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al Ejemplo 4
cd /home/tomas.rojas_s/futsal/ejemplo-4-deteccion-cancha-keypoints

# Inferencia usando la herramienta oficial de YOLO
yolo task=pose mode=predict \
  model=runs/pose/futsal_pitch_detection/yolov8m_pitch_field3/weights/best.pt \
  source=videos/CAM_20260316011334_0014_D.MP4 \
  imgsz=1280 \
  conf=0.1 \
  save=True \
  save_txt=True \
  project=futsal_pitch_prediction \
  name=resultado_oficial_yolo \
  device=0
