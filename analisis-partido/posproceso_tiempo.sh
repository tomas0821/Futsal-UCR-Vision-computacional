#!/bin/bash
# Posprocesamiento de personas y balón para un tiempo de partido.
# Uso: bash analisis-partido/posproceso_tiempo.sh RUN_DIR VIDEO_1080P PREFIJO BALL_CANDIDATES_CSV [CALIB_NPZ]
# Necesita RUN_DIR/track_details.txt (1080p, ya unido si la grabación vino en partes).
# Escribe RUN_DIR/track_pitch_coords.csv, RUN_DIR/PREFIJO_track_details_teams.txt, RUN_DIR/track_pitch_coords_teams.csv,
# RUN_DIR/PREFIJO_stitched.csv y RUN_DIR/ball_track_c0.2.csv, e imprime qué mitad defiende cada equipo en los primeros 2 min.
set -e
cd "$(dirname "$0")/.."
O=$1; VID=$2; P=$3; BC=$4; C=${5:-analisis-partido/calibraciones/grecia_2026-05-09_camera.npz}
A=analisis-partido
[ -s "$O/track_details.txt" ] || { echo "falta $O/track_details.txt"; exit 1; }

echo "$(date) proyección a la cancha"; python3 $A/pitch_map.py $O/track_details.txt $O/track_pitch_coords.csv --scale 2 --calib $C | head -1
echo "$(date) equipos por color"; python3 $A/team_colors.py $VID $O/track_details.txt $O/$P --every 1 | tail -3
echo "$(date) unión equipos + coordenadas"; python3 - "$O" "$P" "$C" "$A" <<'PY'
import csv, sys, numpy as np, collections
O, P, C, A = sys.argv[1:5]; sys.argv = ["x", "--calib", C]; sys.path.insert(0, A); from pitch_map import image_to_pitch, L
rows = list(csv.DictReader(open(f"{O}/{P}_track_details_teams.txt")))
feet = np.array([[float(r["x_center"]), float(r["y_center"]) + float(r["height"]) / 2] for r in rows]) * 2; pm = image_to_pitch(feet)
with open(f"{O}/track_pitch_coords_teams.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame_number", "track_id", "class_name", "team", "seg_id", "confidence", "pitch_x_m", "pitch_y_m"])
    for r, (x, y) in zip(rows, pm): w.writerow([r["frame_number"], r["seg_id"], r["class_name"], r["team"], r["seg_id"], r["confidence"], f"{x:.2f}", f"{y:.2f}"])
print("unidas", len(rows), "detecciones")
mx = collections.defaultdict(list)
for r, (x, y) in zip(rows, pm):
    if r["class_name"] == "juga" and int(r["frame_number"]) <= 6000 and np.isfinite(x): mx[r["team"]].append(x)
for t in sorted(mx):
    lado = "izquierda" if np.mean(mx[t]) < L / 2 else "derecha"; ataca = "derecha" if lado == "izquierda" else "izquierda"
    print(f"primeros 2 min: equipo {t} x medio {np.mean(mx[t]):.1f} m -> defiende la {lado}, ataca a la {ataca}")
PY
echo "$(date) identidades"; python3 $A/stitch_tracks.py $O/track_pitch_coords_teams.csv $O/$P | head -5
echo "$(date) trayectoria del balón"; python3 $A/ball_track.py $BC $O/ball_track_c0.2.csv --calib $C --scale 1 --cmin 0.2
echo "$(date) POSPROCESO LISTO"
