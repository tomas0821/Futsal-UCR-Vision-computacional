# Ejemplo 3: Inferencia Combinada (Múltiples Modelos)

Este ejemplo enseña cómo utilizar la API de Python de YOLOv8 para cargar dos modelos especializados y fusionar sus resultados en un único video de salida.

## 🧠 Arquitectura de la Solución
1.  **Modelo A (Balón):** Entrenado en el Ejemplo 1.
2.  **Modelo B (Personas):** Entrenado en el Ejemplo 2.
3.  **Fusión:** Un script de Python lee el video frame por frame, aplica ambos modelos y dibuja todas las detecciones juntas.

## 🚀 Cómo ejecutar
1. Asegúrate de que los Ejemplos 1 y 2 tengan sus modelos entrenados en las carpetas `runs/`.
2. Navega a esta carpeta: `cd ejemplo-3-modelos-combinados`
3. Ejecuta la inferencia combinada:
   ```bash
   sbatch predict_combined.sh
   ```

El video resultante se guardará en `videos/video_combinado.mp4`.
