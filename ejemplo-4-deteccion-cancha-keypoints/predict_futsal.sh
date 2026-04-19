#!/bin/bash
#SBATCH --job-name=futsal_pitch_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=futsal_pitch_predict_%j.log
#SBATCH --error=futsal_pitch_predict_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al directorio del Ejemplo 4
cd /home/tomas.rojas_s/futsal/ejemplo-4-deteccion-cancha-keypoints

# Inferencia de Pose (Puntos clave de la cancha)
yolo task=pose mode=predict   model=runs/pose/futsal_pitch_detection/yolov8m_pitch_keypoints/weights/best.pt   source=../ejemplo-2-jugadores-y-arbitros/videos/video2.mp4   save=True   conf=0.3   project=futsal_pitch_prediction   name=resultado_keypoints_video2
