#!/bin/bash
#
# ejemplo_gpu.sh - Ejemplo de trabajo SLURM con GPU
#
# Este script demuestra cómo enviar un trabajo que utiliza GPU
# para aceleración de cálculos.
#

#SBATCH --partition=gpu             # Partición con GPUs
#SBATCH --nodes=1                   # Un nodo
#SBATCH --ntasks=1                  # Una tarea
#SBATCH --cpus-per-task=4           # 4 CPUs para la tarea
#SBATCH --gres=gpu:1                # 1 GPU (adjust si necesitas más)
#SBATCH --time=00:10:00             # Tiempo: 10 minutos
#SBATCH --job-name=futsal_gpu       # Nombre del trabajo
#SBATCH --output=salida_gpu_%j.log  # Archivo de salida
#SBATCH --error=error_gpu_%j.log    # Archivo de errores

# Información del trabajo
echo "================================"
echo "Trabajo SLURM con GPU - Futsal UCR"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $SLURM_NODELIST"
echo "CPUs asignados: $SLURM_CPUS_ON_NODE"
echo "GPUs asignadas: $SLURM_GPUS"
echo "Tiempo: $(date)"
echo "================================"
echo ""

# Verificar GPU disponible
echo "Estado de GPUs:"
nvidia-smi
echo ""

# Cargar ambiente conda si es necesario
# eval "$(conda shell.bash hook)"
# conda activate video-rotation

# Ejecutar script que usa GPU
python3 script_con_gpu.py

echo ""
echo "Trabajo completado: $(date)"
