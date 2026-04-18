# Ejemplo 2: Detección de Jugadores, Porteros y Árbitros (Sin Balón)

En este ejemplo se utiliza un dataset multiclase pero se aplica un filtrado selectivo para ignorar la clase 'ball' y reentrenar el modelo solo con el personal en cancha.

## 📊 Resultados Obtenidos (YOLOv8m)
*   **mAP50:** **0.963** (96.3% de éxito general).
*   **mAP50-95:** **0.629** (Localización de cajas muy precisa).
*   **Precision:** 96.5%
*   **Recall:** 93.2%

### Desglose por Clase:
*   **Porteros (gk):** 98.4% mAP50.
*   **Jugadores (player):** 95.5% mAP50.
*   **Árbitros (ref):** 95.8% mAP50.

## 🛠️ Técnica de Filtrado
Se utilizó un script (`filter_ball.py`) para eliminar las anotaciones del balón y reindexar las clases restantes (remapeo de IDs) antes del entrenamiento.

## 📁 Contenido de esta Carpeta
*   `download_dataset.py`: Descarga el dataset de Roboflow (Futsal-UCR v8).
*   `filter_ball.py`: Script de procesamiento para limpiar el dataset.
*   `train.py`: Lógica de entrenamiento configurada para 3 clases (gk, player, ref).
*   `train_futsal.sh`: Script de envío para SLURM.
*   `predict_futsal.sh`: Script de inferencia para procesar videos con el nuevo modelo.

## 🚀 Cómo ejecutar
1. Navega a esta carpeta: `cd ejemplo-2-jugadores-y-arbitros`
2. Prepara el dataset:
   ```bash
   python download_dataset.py
   python filter_ball.py
   ```
3. Entrena: `sbatch train_futsal.sh`
4. Infiere: `sbatch predict_futsal.sh`
