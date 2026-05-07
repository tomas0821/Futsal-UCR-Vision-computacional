#!/bin/bash
#SBATCH --job-name=futsal_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=futsal_predict_%j.log
#SBATCH --error=futsal_predict_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al directorio del proyecto (Ejemplo 1)
cd /home/tomas.rojas_s/futsal/ejemplo-1-deteccion-basica

# Ejecutar inferencia
# model: Ruta al mejor peso obtenido en el entrenamiento (YOLOv8m)
# source: Video de entrada
# conf: Umbral de confianza
yolo task=detect mode=predict \
  model=runs/detect/futsal_training/yolov8m_ball_detection/weights/best.pt \
  source=videos/20250719_193728.mp4 \
  save=True \
  conf=0.6 \
  project=futsal_prediction \
  name=match_video_result_yolov8m
