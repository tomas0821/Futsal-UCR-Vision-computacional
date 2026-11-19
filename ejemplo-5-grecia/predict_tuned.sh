#!/bin/bash
#SBATCH --job-name=grecia_tuned
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=grecia_tuned_%j.log
#SBATCH --error=grecia_tuned_%j.err

CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

cd "$SLURM_SUBMIT_DIR"

# Uso: sbatch predict_futsal.sh [videos/mi_video.mp4]
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --exact -n 1 -c $SLURM_CPUS_PER_TASK python predict_tuned.py "$@"
