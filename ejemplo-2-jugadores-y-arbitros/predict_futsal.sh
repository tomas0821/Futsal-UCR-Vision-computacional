#!/bin/bash
#SBATCH --job-name=players_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=players_predict_%j.log
#SBATCH --error=players_predict_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al directorio del Ejemplo 2
cd /home/tomas.rojas_s/futsal/ejemplo-2-jugadores-y-arbitros

# Ejecutar inferencia
# model: Pesos del modelo Medium entrenado sin balón
yolo task=detect mode=predict \
  model=runs/detect/futsal_jugadores_arbitros/yolov8m_sin_balon/weights/best.pt \
  source=videos/video2.mp4 \
  save=True \
  conf=0.5 \
  project=futsal_prediction \
  name=resultado_jugadores_arbitros
