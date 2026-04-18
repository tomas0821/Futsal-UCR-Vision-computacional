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

### 1. Configuración del Entorno (Solo la primera vez)
Cada estudiante debe crear su propio entorno virtual para asegurar que todas las librerías sean compatibles. Ejecuta los siguientes comandos en la terminal:

```bash
# Crear el entorno con Python 3.11
conda create -n futsal_env python=3.11 -y

# Activar el entorno
conda activate futsal_env

# Instalar dependencias necesarias
pip install ultralytics roboflow python-dotenv
```

### 2. Preparación de Credenciales
Para descargar el dataset, necesitas una API Key de Roboflow.
1. Crea un archivo llamado `.env` en esta carpeta.
2. Agrega tu llave y el ID del proyecto (puedes solicitar estos datos al encargado):
   ```text
   ROBOFLOW_API_KEY=tu_llave_aqui
   ROBOFLOW_PROJECT=football-ucr/ucr-futsal-ball2/6
   ```

### 3. Ejecución
Una vez configurado el entorno:
1. Navega a esta carpeta: `cd ejemplo-1-deteccion-basica`
2. Descarga los datos: `python download_dataset.py`
3. Entrena en el cluster: `sbatch train_futsal.sh`
4. Infiere sobre video: `sbatch predict_futsal.sh`
