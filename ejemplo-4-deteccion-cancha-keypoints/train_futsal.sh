#!/bin/bash
#SBATCH --job-name=futsal_pitch_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=futsal_pitch_%j.log
#SBATCH --error=futsal_pitch_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al directorio del Ejemplo 4
cd /home/tomas.rojas_s/futsal/ejemplo-4-deteccion-cancha-keypoints

# Iniciar el entrenamiento de keypoints
python train.py
