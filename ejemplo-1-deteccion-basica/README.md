# Ejemplo 1: Detección Básica de Balón

Este ejemplo documenta el proceso inicial de entrenamiento e inferencia para detectar un balón de futsal utilizando la arquitectura YOLOv8. Hemos probado dos versiones del modelo para comparar rendimiento.

## 📊 Comparativa de Modelos (Resultados Finales)

| Métrica | YOLOv8n (Nano) | YOLOv8m (Medium) | Impacto |
| :--- | :--- | :--- | :--- |
| **mAP50** | 0.891 | **0.930** | +4.4% |
| **mAP50-95** | 0.518 | **0.609** | **+17.5%** |
| **Precision** | 94.1% | **99.2%** | +5.4% |
| **Recall** | 81.6% | **86.8%** | +6.3% |
| **Inferencia (A100)** | **0.4ms** | 1.6ms | Mayor carga computacional |

> **Nota:** El modelo **Medium** es significativamente más preciso en la localización de la caja (mAP50-95), lo cual es crucial para análisis de trayectoria precisos.

## 📁 Contenido de esta Carpeta
*   `download_dataset.py`: Descarga el dataset de Roboflow (UCR-Futsal-Ball2).
*   `train.py`: Lógica de entrenamiento (configurado por defecto para YOLOv8m).
*   `train_futsal.sh`: Script para enviar el entrenamiento al GPU del cluster.
*   `predict_futsal.sh`: Script para procesar videos y generar detecciones.
*   `videos/`: Carpeta para los videos originales de prueba.

## 🚀 Cómo ejecutar

### 1. Configuración del Entorno (Solo la primera vez)
```bash
conda create -n futsal_env python=3.11 -y
conda activate futsal_env
pip install ultralytics roboflow python-dotenv
```

### 2. Preparación de Credenciales
Crea un archivo `.env` con tu `ROBOFLOW_API_KEY`.

### 3. Ejecución
1. Navega a esta carpeta: `cd ejemplo-1-deteccion-basica`
2. Descarga los datos: `python download_dataset.py`
3. Entrena: `sbatch train_futsal.sh`
4. Infiere: `sbatch predict_futsal.sh`
