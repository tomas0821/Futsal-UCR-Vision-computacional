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

# Load conda environment
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navigate to project directory
cd /home/tomas.rojas_s/futsal/ucr-futsal-ball2

# Run inference
yolo task=detect mode=predict \
  model=runs/detect/futsal_training/yolov8_ball_detection/weights/best.pt \
  source=videos/20250719_193728.mp4 \
  save=True \
  conf=0.6 \
  project=futsal_prediction \
  name=match_video_result_conf06
