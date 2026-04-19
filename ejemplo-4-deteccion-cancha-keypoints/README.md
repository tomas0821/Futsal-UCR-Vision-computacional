# Ejemplo 4: Detección de Keypoints de la Cancha

Este ejemplo utiliza **YOLOv8-pose** para detectar la geometría de la cancha de futsal mediante 22 puntos clave estratégicos.

## 📊 Resultados Obtenidos (YOLOv8m-pose)
*   **mAP50 Pose:** **0.995** (99.5% de precisión en detección de puntos).
*   **mAP50-95 Pose:** **0.982** (Localización milimétrica de las esquinas).
*   **Velocidad:** ~1.8ms por frame.

## 🎯 Objetivo Técnico
La detección de estos 22 puntos clave permite realizar procesos de **Homografía**, lo cual es el paso previo necesario para convertir la imagen de la cámara en un mapa táctico 2D (vista cenital) y calcular distancias reales recorridas por los jugadores.

## 📁 Contenido de esta Carpeta
*   `download_dataset.py`: Descarga el dataset de geometría de cancha.
*   `train.py`: Entrenamiento de estimación de pose.
*   `train_futsal.sh`: Script para SLURM (Entrenamiento).
*   `predict_futsal.sh`: Script para SLURM (Inferencia en video).

## 🚀 Cómo ejecutar
1. Navega a esta carpeta: `cd ejemplo-4-deteccion-cancha-keypoints`
2. Descarga los datos: `python download_dataset.py`
3. Entrena: `sbatch train_futsal.sh`
4. Infiere: `sbatch predict_futsal.sh`
