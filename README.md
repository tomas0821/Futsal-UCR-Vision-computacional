# Futsal Analysis Project - UCR

Este repositorio es una guía técnica y un catálogo de ejemplos para el análisis de video en futsal utilizando Inteligencia Artificial. Está diseñado para ser utilizado por estudiantes asistentes e investigadores.

## 📂 Catálogo de Ejemplos

| Ejemplo | Descripción | Modelo | Estado |
| :--- | :--- | :--- | :--- |
| **[Ejemplo 1: Detección Básica](./ejemplo-1-deteccion-basica/)** | Entrenamiento de un modelo YOLOv8m para detectar el balón. | YOLOv8m | ✅ Completado |
| **[Ejemplo 2: Jugadores y Árbitros](./ejemplo-2-jugadores-y-arbitros/)** | Detección multiclase de personas en cancha (excluyendo el balón). | YOLOv8m | ✅ Completado |
| **[Ejemplo 3: Modelos Combinados](./ejemplo-3-modelos-combinados/)** | Tutorial avanzado: Inferencia simultánea usando múltiples modelos. | Multimodelo | ✅ Completado |
| **[Ejemplo 4: Keypoints de Cancha](./ejemplo-4-deteccion-cancha-keypoints/)** | Estimación de pose para detectar la geometría y 22 puntos de la cancha. | YOLOv8m-pose | ✅ Completado |

## 🛠️ Requisitos Generales

1.  **Entorno:** Se recomienda usar un entorno Conda (ej. `futsal_env`).
2.  **Librerías:** `ultralytics`, `roboflow`, `python-dotenv`, `opencv-python`.
3.  **Hardware:** Acceso al cluster SLURM para entrenamiento en GPU.

## 🤝 Contribuciones
Los estudiantes asistentes deben seguir la estructura de carpetas `ejemplo-X-...` para agregar nuevos módulos al proyecto.
