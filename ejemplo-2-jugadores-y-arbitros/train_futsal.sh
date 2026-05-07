#!/bin/bash
#SBATCH --job-name=futsal_players_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=futsal_players_%j.log
#SBATCH --error=futsal_players_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al directorio del Ejemplo 2
cd /home/tomas.rojas_s/futsal/ejemplo-2-jugadores-y-arbitros

# Paso 1: Descargar y Filtrar (ya lo hicimos manualmente, pero lo dejamos por si acaso)
# python download_dataset.py
# python filter_ball.py

# Paso 2: Iniciar el entrenamiento sin balón
python train.py
