#!/bin/bash
#SBATCH --job-name=combined_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=combined_predict_%j.log
#SBATCH --error=combined_predict_%j.err

CONDA_PATH=$(conda info --base)
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate futsal_env

cd "$(dirname "$(realpath "$0")")" 

python combined_inference.py
