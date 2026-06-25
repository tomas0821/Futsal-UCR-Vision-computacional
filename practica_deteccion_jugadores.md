# Práctica: Detección de Jugadores de Futsal con YOLOv8

**Curso:** Análisis de Video con Inteligencia Artificial  
**Duración estimada:** 2–3 horas  
**Modalidad:** Individual o parejas  
**Plataforma:** Cluster UCR (GPU A100)

---

## Objetivos

Al finalizar esta práctica, podrás:
- Preparar un dataset multiclase para detección de objetos
- Entrenar un modelo YOLOv8m en el cluster con GPU
- Interpretar las métricas de entrenamiento (mAP, precisión, recall)
- Correr inferencia sobre un video y analizar los resultados
- Evaluar el efecto de modificar el umbral de confianza

---

## Contexto

En esta práctica trabajarás con el **Ejemplo 2** del repositorio del proyecto, que entrena un modelo capaz de detectar tres tipos de personas en una cancha de futsal:

| Clase | ID | Descripción |
|-------|----|-------------|
| `gk` | 0 | Portero |
| `player` | 1 | Jugador de campo |
| `ref` | 2 | Árbitro |

El dataset proviene de videos reales de partidos de la UCR y fue anotado manualmente en Roboflow. El balón está excluido de este modelo — existe un modelo separado para eso.

---

## Parte 1 — Preparación del entorno

### 1.1 Conectarse al cluster

```bash
ssh tu_usuario@172.16.24.2
```

> Si estás fuera de la red UCR, primero conéctate a la VPN en `acceso.ucr.ac.cr`.

### 1.2 Activar el entorno conda

```bash
conda activate futsal_env
```

Si el entorno no existe aún:

```bash
conda create -n futsal_env python=3.11 -y
conda activate futsal_env
pip install ultralytics roboflow python-dotenv
```

### 1.3 Navegar al ejemplo

```bash
cd ~/futsal/ejemplo-2-jugadores-y-arbitros
ls
```

Deberías ver: `download_dataset.py`, `filter_ball.py`, `train.py`, `train_futsal.sh`, `predict_futsal.sh`, `videos/`.

---

## Parte 2 — Preparar el dataset

### 2.1 Descargar el dataset

```bash
python download_dataset.py
```

Esto descarga el dataset **Futsal-UCR v8** desde Roboflow al directorio `futsal-ucr-8/`. Contiene imágenes de partidos con anotaciones de 4 clases: `ball`, `gk`, `player`, `ref`.

### 2.2 Filtrar el balón

El modelo que entrenaremos detecta **únicamente personas**. Necesitamos eliminar las anotaciones del balón y reindexar las clases restantes:

```bash
python filter_ball.py
```

**¿Qué hace este script internamente?**  
Lee cada archivo `.txt` de anotaciones, descarta las líneas donde `class_id == 0` (balón) y resta 1 al ID de las demás clases para que queden como `gk=0, player=1, ref=2`.

**Pregunta 1:** ¿Por qué crees que es mejor tener un modelo separado para el balón en lugar de detectar todo con un solo modelo?

---

## Parte 3 — Entrenar el modelo

### 3.1 Revisar el script de entrenamiento

Abre `train.py` y observa los parámetros principales:

```python
model.train(
    data='futsal-ucr-8/data_sin_balon.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=device,
    project='futsal_jugadores_arbitros',
    name='yolov8m_sin_balon'
)
```

**Pregunta 2:** ¿Qué significan los parámetros `epochs`, `imgsz` y `batch`? ¿Qué efecto tiene aumentar o disminuir cada uno?

### 3.2 Enviar el job al cluster

```bash
sbatch train_futsal.sh
```

Anota el **Job ID** que te devuelve el sistema (por ejemplo: `Submitted batch job 12345`).

### 3.3 Monitorear el progreso

```bash
squeue -u $USER
```

Para ver el log en tiempo real:

```bash
tail -f futsal_train_<JOB_ID>.log
```

El entrenamiento toma aproximadamente 15–20 minutos en la A100. Cuando termine verás `Entrenamiento completado.` en el log.

---

## Parte 4 — Analizar las métricas

Una vez finalizado el entrenamiento, los resultados están en:

```
runs/detect/futsal_jugadores_arbitros/yolov8m_sin_balon/
```

### 4.1 Revisar el archivo de métricas

```bash
cat runs/detect/futsal_jugadores_arbitros/yolov8m_sin_balon/results.csv | head -5
```

Las columnas más importantes son: `metrics/mAP50(B)` y `metrics/mAP50-95(B)`.

### 4.2 Interpretar las imágenes de resultados

Descarga los siguientes archivos a tu computadora con `scp` o mediante el Google Drive del proyecto:

| Archivo | Qué muestra |
|---------|-------------|
| `results.png` | Curvas de pérdida y mAP durante el entrenamiento |
| `confusion_matrix_normalized.png` | Qué clases se confunden entre sí |
| `BoxPR_curve.png` | Curva Precisión-Recall por clase |

**Pregunta 3:** Observa la matriz de confusión. ¿Qué clase es más difícil de detectar? ¿Por qué crees que ocurre eso?

**Pregunta 4:** El modelo entrenado obtuvo los siguientes resultados globales. Compara con los resultados de tu entrenamiento:

| Métrica | Resultado de referencia | Tu resultado |
|---------|------------------------|--------------|
| mAP50 | 96.3% | ___ |
| mAP50-95 | 62.9% | ___ |
| Precisión | 96.5% | ___ |
| Recall | 93.2% | ___ |

---

## Parte 5 — Correr inferencia en un video

### 5.1 Inferencia con parámetros base

Abre `predict_futsal.sh` y revisa los parámetros del comando `yolo`:

```bash
yolo task=detect mode=predict \
  model=runs/detect/futsal_jugadores_arbitros/yolov8m_sin_balon/weights/best.pt \
  source=videos/video2.mp4 \
  save=True \
  conf=0.5 \
  project=futsal_prediction \
  name=resultado_v1
```

Envía el job:

```bash
sbatch predict_futsal.sh
```

El video anotado se guardará en `futsal_prediction/resultado_v1/`.

### 5.2 Experimento — Efecto del umbral de confianza

Crea dos variantes del script modificando únicamente el parámetro `conf`:

**Variante A — umbral bajo (conf=0.25):**
```bash
yolo task=detect mode=predict \
  model=runs/detect/futsal_jugadores_arbitros/yolov8m_sin_balon/weights/best.pt \
  source=videos/video2.mp4 \
  save=True conf=0.25 project=futsal_prediction name=resultado_conf025
```

**Variante B — umbral alto (conf=0.80):**
```bash
yolo task=detect mode=predict \
  model=runs/detect/futsal_jugadores_arbitros/yolov8m_sin_balon/weights/best.pt \
  source=videos/video2.mp4 \
  save=True conf=0.80 project=futsal_prediction name=resultado_conf080
```

Puedes correr ambos directamente en el nodo de GPU con `srun`:

```bash
srun --partition=gpu --gres=gpu:a100:1 --pty bash
conda activate futsal_env
cd ~/futsal/ejemplo-2-jugadores-y-arbitros
# pega el comando yolo aquí
```

**Pregunta 5:** Compara los tres videos de salida. ¿Qué pasa con las detecciones cuando bajas el umbral a 0.25? ¿Y cuando lo subes a 0.80? ¿Cuál configuración usarías en producción y por qué?

---

## Parte 6 — Preguntas de reflexión

Responde en un párrafo cada una:

1. El script `filter_ball.py` elimina el balón del dataset antes de entrenar. ¿Qué pasaría si en vez de eso simplemente le pidieras al modelo que ignore la clase balón durante la inferencia?

2. El modelo detecta porteros con 98.4% mAP50, mucho mejor que jugadores (95.5%). ¿A qué se podría deber esta diferencia? ¿Cómo lo resolverías?

3. El entrenamiento usó `imgsz=640`. Si los videos de la cámara son de resolución 1920×1080, ¿qué implica esto para la detección de objetos pequeños como el balón?

---

## Entrega

Sube a la plataforma del curso:

- [ ] Captura de pantalla del `results.png` de tu entrenamiento
- [ ] Captura de pantalla de `confusion_matrix_normalized.png`
- [ ] La tabla de métricas de la Pregunta 4 completada
- [ ] Respuestas a las preguntas 1–5 y las 3 preguntas de reflexión (documento PDF)
- [ ] El Job ID de tu entrenamiento

---

## Referencia rápida de comandos SLURM

```bash
squeue -u $USER          # Ver mis jobs activos
scancel <JOB_ID>         # Cancelar un job
sinfo -p gpu             # Ver estado de la partición GPU
tail -f mi_log.log       # Ver log en tiempo real
```
