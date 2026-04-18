# Ejemplo 1: Detección Básica de Balón

Este ejemplo documenta el proceso inicial de entrenamiento e inferencia para detectar un balón de futsal utilizando la arquitectura YOLOv8.

## 📊 Resultados Obtenidos
*   **mAP50:** 0.891
*   **Precisión:** 94.1%
*   **Velocidad de Inferencia:** ~0.4ms por frame (A100).

## 📁 Contenido de esta Carpeta
*   `download_dataset.py`: Descarga el dataset de Roboflow (UCR-Futsal-Ball2).
*   `train.py`: Lógica de entrenamiento.
*   `train_futsal.sh`: Script para enviar el entrenamiento al GPU del cluster.
*   `predict_futsal.sh`: Script para procesar videos y generar detecciones.
*   `videos/`: Carpeta para los videos originales de prueba.

## 🚀 Cómo ejecutar
1. Navega a esta carpeta: `cd ejemplo-1-deteccion-basica`
2. Descarga los datos: `python download_dataset.py`
3. Entrena: `sbatch train_futsal.sh`
4. Infiere: `sbatch predict_futsal.sh`
