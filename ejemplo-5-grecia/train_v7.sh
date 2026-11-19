#!/bin/bash
#SBATCH --job-name=futsal_v7
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=futsal_v7_%j.log
#SBATCH --error=futsal_v7_%j.err
CONDA_PATH=$(conda info --base); source $CONDA_PATH/etc/profile.d/conda.sh; conda activate futsal_env
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ "$VARIANT" = "4cls" ]; then srun --exact -n 1 -c $SLURM_CPUS_PER_TASK python prepare_v7.py; fi
srun --exact -n 1 -c $SLURM_CPUS_PER_TASK python train_v7.py
