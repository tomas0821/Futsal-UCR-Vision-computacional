# Futsal Ball Detection - UCR

Este repositorio contiene el flujo de trabajo para el entrenamiento y la inferencia de un modelo de detección de balones de futsal utilizando YOLOv8. Está diseñado como referencia para estudiantes asistentes del proyecto.

## 🚀 Resumen del Modelo

*   **Arquitectura:** YOLOv8n (Nano) - Elegido por su alta velocidad para procesamiento en tiempo real.
*   **Dataset:** [football-ucr/ucr-futsal-ball2](https://roboflow.com) (Versión 6).
*   **Hardware de entrenamiento:** NVIDIA A100 (Cluster SLURM).

## 📊 Resultados del Entrenamiento (100 Epochs)

El modelo alcanzó una precisión excelente, ideal para seguimiento en video:

| Métrica | Valor | Descripción |
| :--- | :--- | :--- |
| **mAP50** | **0.891** | Precisión media en detección estándar. |
| **mAP50-95** | **0.518** | Precisión media con alta exigencia de solapamiento. |
| **Precision** | **94.1%** | Muy bajo índice de falsas alarmas (pocos "fantasmas"). |
| **Recall** | **81.6%** | Capacidad de encontrar el balón en la mayoría de los frames. |

## 🛠️ Estructura del Proyecto

*   `download_dataset.py`: Script para descargar la última versión del dataset desde Roboflow.
*   `train.py`: Script de entrenamiento en Python configurado para YOLOv8.
*   `train_futsal.sh`: Script de envío para SLURM para ejecutar el entrenamiento en GPU.
*   `predict_futsal.sh`: Script para ejecutar inferencia sobre videos de partidos.
*   `runs/detect/futsal_training/`: Contiene los pesos (`best.pt`) y las gráficas de rendimiento.

## 📋 Instrucciones para Estudiantes

### 1. Preparación
Instalar las dependencias necesarias:
```bash
pip install ultralytics roboflow python-dotenv
```

### 2. Entrenamiento
Para volver a entrenar el modelo en el cluster:
```bash
sbatch train_futsal.sh
```

### 3. Inferencia
Para procesar un nuevo video de futsal:
1. Sube el video a la carpeta `videos/`.
2. Ejecuta:
```bash
sbatch predict_futsal.sh
```

Los resultados se guardarán en la carpeta `futsal_prediction/`.
