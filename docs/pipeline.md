# Pipeline: del video 4K a las estadísticas del partido

Este documento describe, etapa por etapa, cómo se procesa un tiempo de un partido. Las etapas de GPU corren en el clúster (carpeta `ejemplo-5-grecia/`); el resto corre en cualquier máquina con Python (carpeta `analisis-partido/`). Los tiempos indicados corresponden al primer tiempo San Ramón vs Grecia (27 min, 82 252 cuadros a 50 fps).

## 0. Preparar el video

La cámara Osmo graba invertida (180°) y en 4K. Se generan dos versiones enderezadas a 50 fps: una a 1080p para las personas y otra 4K para el balón.

```bash
ffmpeg -i partido_1T.MP4 -vf "hflip,vflip,fps=50,scale=1920:1080" -c:v libx264 -crf 20 -pix_fmt yuv420p -an 1T_upright_1080p.mp4
ffmpeg -i partido_1T.MP4 -vf "hflip,vflip,fps=50"                 -c:v libx264 -crf 21 -pix_fmt yuv420p -an 1T_upright_4k.mp4
```

La recodificación 4K necesita unos 3.6 GB de memoria; conviene hacerla en el clúster. Las grabaciones largas vienen en varios archivos; los `track_details.txt` de cada parte se unen después con `merge_track_parts.py`.

## 1. Pase de personas (clúster, GPU)

```bash
cd ejemplo-5-grecia
MODEL=runs/detect/futsal_grecia/yolov8m_grecia_v7_4cls/weights/best.pt \
CONF=0.4 IOU=0.5 TRACKER=botsort_tuned.yaml TAG=v7 \
sbatch --export=ALL predict_tuned.sh videos/1T_upright_1080p.mp4
```

`predict_tuned.py` corre YOLOv8m a 1280 px con BotSort y escribe `futsal_prediction/v7_<clip>/track_details.txt` más un video anotado. El tracker ajustado (`botsort_tuned.yaml`) recuerda una pista perdida durante 3 s (150 cuadros), exige confianza 0.7 para abrir pistas nuevas y desactiva la compensación de movimiento porque la cámara es fija. Con `conf=0.4`, `iou=0.5` y NMS agnóstica de clase las cajas dobles bajan del 20 % al 0.3 % de los cuadros.

Rendimiento: 1 500 cuadros en 1 min 37 s en una A100; el tiempo completo, en tres partes, unos 30 min.

## 2. Pase de balón (clúster, GPU)

```bash
MODEL=runs/detect/futsal_grecia/yolov8m_grecia_v7_4cls/weights/best.pt \
MODE=full IMGSZ=2560 CONF=0.05 TAG=v7full2560 \
sbatch --export=ALL predict_ball.sh videos/1T_upright_4k.mp4
```

`predict_ball.py` solo predice la clase `bola`, sobre el cuadro 4K completo a 2560 px, con un piso de confianza de 0.05, y guarda **todos** los candidatos en `ball_candidates.txt` (coordenadas en píxeles 4K). El filtrado se hace después, en la cancha. `MODE=tiled` divide el cuadro en 2 × 2 mosaicos con 15 % de solape; da cobertura parecida y se descartó.

Rendimiento: 53 min en una A100 para 27 min de video 4K.

## 3. Calibración de la cámara (una vez por posición de cámara)

1. Abrir `analisis-partido/click_landmarks.html` en un navegador, cargar un cuadro 4K limpio y marcar los puntos en el orden indicado (bases y extremos de ambas porterías, extremos de la línea central, esquinas visibles). Descargar el JSON.
2. Ajustar el modelo de lente:
   ```bash
   python analisis-partido/fit_from_clicks.py <partido>_landmarks.json cuadro.jpg --out analisis-partido/calibraciones/<partido>_camera.npz
   ```
   Imprime el error por punto; más de 40 px indica un clic equivocado. Genera además una imagen con el contorno de la cancha dibujado sobre el cuadro, que hay que mirar.
3. Si hay duda de que la cámara se movió entre partes, `overlay_check.py` dibuja el contorno sobre varios cuadros.

Modelo: ojo de pez equidistante con polinomio de dos términos en θ; se ajustan pose, focal y punto principal. Marco de la cancha: X a lo largo de la banda lejana desde la línea de meta izquierda, Y desde la banda lejana hacia la cámara, en metros; cancha de 37.19 × 17.2 m. Error típico: ±0.3 m en el centro, hasta ±1 m en las líneas de meta.

La calibración de la posición de Grecia (9 de mayo de 2026) está en `analisis-partido/calibraciones/`.

## 4. Posprocesamiento (local, CPU)

```bash
bash analisis-partido/posproceso_tiempo.sh RUN_DIR 1T_upright_1080p.mp4 PREFIJO RUN_DIR/ball_candidates.csv analisis-partido/calibraciones/grecia_2026-05-09_camera.npz
```

El script encadena cinco pasos, que también se pueden correr por separado:

| Paso | Script | Qué hace | Salida |
| :--- | :--- | :--- | :--- |
| Proyección | `pitch_map.py` | Pies de cada caja (centro inferior) → cancha en metros. `--scale 2` cuando las pistas son de 1080p y la calibración de 4K. | `track_pitch_coords.csv` |
| Equipos | `team_colors.py` | Color mediano del torso (Lab) por detección; A = camiseta oscura, B = clara; porteros por portería defendida. Parte las pistas donde cambia el equipo. | `<pre>_track_details_teams.txt`, muestrario de recortes |
| Unión | (en el script) | Junta equipo y coordenadas de cancha; imprime qué mitad ocupa cada equipo en los primeros dos minutos. | `track_pitch_coords_teams.csv` |
| Identidades | `stitch_tracks.py` | Por equipo, enlaza segmentos fin → inicio por asignación global (húngaro con opción "no enlazar") según distancia y tiempo. | `<pre>_stitched.csv` |
| Balón | `ball_track.py` | Filtra candidatos en el plano de la cancha (velocidad máxima, continuidad), arma tracklets e interpola huecos de hasta 0.5 s. | `ball_track_c0.2.csv` |

## 5. Analítica del partido

```bash
python analisis-partido/match_analytics.py salida/partido --names "A=San Ramon;B=Grecia" \
  --half 1T RUN_1T/PREFIJO_stitched.csv RUN_1T/ball_track_c0.2.csv A=left,B=right \
  --half 2T RUN_2T/PREFIJO_stitched.csv RUN_2T/ball_track_c0.2.csv A=right,B=left
```

`A=left` significa que el equipo A ataca la portería izquierda (x = 0) en ese tiempo; si se omite, se infiere de la formación del saque inicial. Produce `_events.csv` (tiros, goles, saques de centro), `_players.csv`, `_teams.csv` y tres figuras: reporte, mapa de tiros y mapas de calor.

Definiciones:

- **Posesión**: equipo del jugador más cercano al balón (≤ 2.5 m), acumulado por ventanas de 5 min.
- **Golpe**: velocidad del balón ≥ 8 m/s sostenida. **Tiro**: golpe desde ≤ 20 m dirigido a la boca de la portería (±3 m) que llega a 4 m de la línea. **A puerta**: ±1.5 m del centro y a 1 m de la línea.
- **Gol**: formación de saque de centro (1 a 3 jugadores a menos de 3 m del centro, el resto en su mitad, todos quietos ≥ 1 s). El equipo que saca es el del jugador más cercano al punto central; el gol se atribuye al contrario. Un balón que cruza la línea sin saque posterior queda como "posible gol, revisar".
- **Distancia**: suma de pasos entre cuadros consecutivos (huecos ≤ 2 s, saltos ≤ 12.5 m/s). Ver la advertencia en [resultados.md](resultados.md#distancias-y-velocidades).

## 6. Visualización y control de calidad

| Script | Uso |
| :--- | :--- |
| `render_tracks.py` | Video 1080p con cajas finas e identidades por equipo (ffmpeg + PIL, sin OpenCV). |
| `render_ball.py` | Círculo verde (detectado) o ámbar (interpolado) con estela; opcionalmente las cajas de personas, un rótulo y el reloj del partido. |
| `ball_plots.py` | Trayectoria coloreada por tiempo, x/y contra tiempo, velocidad con golpes, mapa de calor del balón. |
| `chain_gantt.py` | Línea de tiempo de identidades por equipo; detecta más de 4 + 1 simultáneos. |
| `player_stats.py`, `team_plots.py`, `timeline_plots.py` | Estadísticas por segmento, ocupación por equipo, líneas de tiempo. |
| `radar_video.py` | Vista cenital animada, opcionalmente junto al video. |
| `example_frames.py` | Cuadros de ejemplo: detección completa de personas y casos difíciles del balón. |
| `track_diagnostics.py`, `compare_runs.py` | Cajas dobles, vida de las pistas, cobertura del balón; comparación de dos corridas. |
| `export_review.py`, `review_identities.html`, `apply_corrections.py` | Revisión manual de identidades en el navegador y aplicación de las correcciones. |

## Formatos de datos

**`track_details.txt`** (pase de personas, píxeles 1080p, con encabezado):

```
frame_number,track_id,class_id,class_name,x_center,y_center,width,height,confidence
```

**`ball_candidates.txt`** (pase de balón, píxeles 4K): `frame_number,x_center,y_center,width,height,confidence,tile`.

**`track_pitch_coords.csv`**: `frame_number,track_id,class_id,class_name,confidence,img_x,img_y,pitch_x_m,pitch_y_m,inside`.

**`<pre>_track_details_teams.txt`**: `track_details` más `team` y `seg_id` (`seg_id = 10 × track_id + k` tras partir la pista).

**`<pre>_stitched.csv`**: `frame_number,track_id,class_name,team,player_id,confidence,pitch_x_m,pitch_y_m`, con `track_id` = cadena (1000 + n para A, 2000 + n para B, 3000 / 3100 porteros) y `player_id` = `A4`, `B2`, `GK_left1`, etc. Para volver a las cajas originales se une con `track_pitch_coords.csv` por `(frame_number, confidence, pitch_x_m, pitch_y_m)`.

**`ball_track_c0.2.csv`**: `frame_number,x_px,y_px,pitch_x_m,pitch_y_m,confidence,tracklet,source` con `source` = `det` o `interp`; la confianza va vacía en las filas interpoladas y los huecos de más de 0.5 s no tienen filas.
