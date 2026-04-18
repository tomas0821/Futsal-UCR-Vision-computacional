#!/bin/bash
#SBATCH --job-name=futsal_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=futsal_train_%j.log
#SBATCH --error=futsal_train_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al directorio del proyecto (Ejemplo 1)
cd /home/tomas.rojas_s/futsal/ejemplo-1-deteccion-basica

# Paso 1: Descargar el dataset (si no se ha hecho antes)
python download_dataset.py

# Paso 2: Iniciar el entrenamiento
python train.py
