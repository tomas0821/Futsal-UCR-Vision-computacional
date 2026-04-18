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

# Load conda environment
CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

# Navigate to project directory
cd /home/tomas.rojas_s/futsal/ejemplo-1-deteccion-basica

# Step 1: Download dataset (if not already downloaded)
python download_dataset.py

# Step 2: Start training
python train.py
