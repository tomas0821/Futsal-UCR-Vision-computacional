#!/bin/bash
#SBATCH --job-name=pitch_strict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=pitch_strict_%j.log
#SBATCH --error=pitch_strict_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al Ejemplo 4
cd /home/tomas.rojas_s/futsal/ejemplo-4-deteccion-cancha-keypoints

# Inferencia Estricta
yolo task=pose mode=predict \
  model=runs/pose/futsal_pitch_detection/yolov8m_pitch_field3/weights/best.pt \
  source=videos/CAM_20260316011334_0014_D.MP4 \
  imgsz=640 \
  conf=0.5 \
  save=True \
  project=futsal_pitch_prediction \
  name=resultado_estricto \
  device=0
