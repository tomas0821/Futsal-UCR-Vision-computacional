#!/bin/bash
#SBATCH --job-name=v7_clip
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=v7_clip_%j.log
#SBATCH --error=v7_clip_%j.err
CONDA_PATH=$(conda info --base); source $CONDA_PATH/etc/profile.d/conda.sh; conda activate futsal_env
cd "$SLURM_SUBMIT_DIR"; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MODEL=runs/detect/futsal_grecia/yolov8m_grecia_v7_4cls/weights/best.pt
CONF=0.4 IOU=0.5 TRACKER=botsort_tuned.yaml TAG=v7tuned srun --exact -n 1 -c $SLURM_CPUS_PER_TASK python predict_tuned.py videos/grecia_1T_600s_300s_upright_1080p.mp4
MODE=full IMGSZ=2560 CONF=0.05 TAG=v7full2560 srun --exact -n 1 -c $SLURM_CPUS_PER_TASK python predict_ball.py videos/grecia_1T_600s_300s_upright_4k.mp4
echo ALLDONE
