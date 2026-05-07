#!/bin/bash
#SBATCH --job-name=pitch_custom
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pitch_custom_%j.log
#SBATCH --error=pitch_custom_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al Ejemplo 4
cd /home/tomas.rojas_s/futsal/ejemplo-4-deteccion-cancha-keypoints

# Ejecutar inferencia personalizada
python -u predict_custom.py
