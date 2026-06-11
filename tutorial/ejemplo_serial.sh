#!/bin/bash
#
# ejemplo_serial.sh - Ejemplo básico de trabajo SLURM en partición serial
#
# Este script demuestra cómo enviar un trabajo simple al cluster
# sin usar paralelismo ni GPU.
#

#SBATCH --partition=serial          # Partición para trabajos secuenciales
#SBATCH --nodes=1                   # Número de nodos (1 para serial)
#SBATCH --ntasks=1                  # Una tarea
#SBATCH --cpus-per-task=1           # Un CPU
#SBATCH --time=00:05:00             # Tiempo máximo: 5 minutos
#SBATCH --job-name=futsal_serial    # Nombre del trabajo
#SBATCH --output=salida_%j.log      # Archivo de salida (%j = job ID)
#SBATCH --error=error_%j.log        # Archivo de errores

# Información del trabajo
echo "================================"
echo "Trabajo SLURM Serial - Futsal UCR"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Tiempo: $(date)"
echo "================================"
echo ""

# Cargar ambiente (si es necesario)
# module load python/3.11

# Ejecutar script Python
python3 simple_script.py

echo ""
echo "Trabajo completado: $(date)"
