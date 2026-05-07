# Ejemplo 4: Detección de Keypoints de la Cancha (Optimizado)

Este ejemplo utiliza **YOLOv8-pose** para detectar la geometría de la cancha de futsal. Se ha actualizado para usar el dataset optimizado con 10 puntos clave estratégicos.

## 📊 Resultados Finales (YOLOv8m-pose + field3)
*   **Dataset:** ucr-futsal-field3 (Versión 3)
*   **Keypoints:** 10 puntos (esquinas y puntos de interés clave).
*   **mAP50 Pose:** **0.995** (99.5% de precisión).
*   **mAP50-95 Pose:** **0.982** (Localización de extrema exactitud).
*   **Velocidad de Inferencia:** ~1.7ms por frame (A100).

## 🎯 Objetivo Técnico
La reducción de 22 a 10 puntos clave permite un modelo más robusto y menos propenso a errores en ángulos de cámara difíciles, manteniendo la precisión necesaria para realizar la **Homografía** y el mapeo táctico 2D.

## 📁 Contenido de esta Carpeta
*   `download_dataset.py`: Descarga el dataset `ucr-futsal-field3`.
*   `train.py`: Entrenamiento configurado para 10 keypoints.
*   `train_futsal.sh`: Script para SLURM (Entrenamiento).
*   `predict_futsal.sh`: Inferencia en videos anteriores.
*   `predict_new_video.sh`: Inferencia en el nuevo video del sistema CAM.

## 🚀 Cómo ejecutar
1. Navega a esta carpeta: `cd ejemplo-4-deteccion-cancha-keypoints`
2. Descarga los datos: `python download_dataset.py`
3. Entrena: `sbatch train_futsal.sh`
4. Infiere (nuevo video): `sbatch predict_new_video.sh`

## 🎬 Demostración de Resultados (Beta)
En este clip se puede observar la estructura de los 10 puntos clave detectados en la cancha:

https://github.com/tomas0821/Futsal-UCR-Vision-computacional/raw/master/ejemplo-4-deteccion-cancha-keypoints/demo_cancha.mp4

> **Nota importante:** Debido al tamaño limitado del dataset (20 imágenes), el modelo puede presentar falsos positivos en estructuras similares del techo. Se recomienda ampliar el dataset para mayor robustez.
