#!/bin/bash
#SBATCH --job-name=futsal_grecia_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=futsal_grecia_%j.log
#SBATCH --error=futsal_grecia_%j.err

# Cargar el entorno de conda
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navegar al directorio del Ejemplo 5
cd "$SLURM_SUBMIT_DIR"

# Paso 1: Descargar el dataset (ya descargado; descomentar si hace falta)
# python download_dataset.py

# Paso 2: Entrenar (bola, juga, port, ref)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --exact -n 1 -c $SLURM_CPUS_PER_TASK python train.py
