# Futsal Analysis Project - UCR

Este repositorio es una guía técnica y un catálogo de ejemplos para el análisis de video en futsal utilizando Inteligencia Artificial. Está diseñado para ser utilizado por estudiantes asistentes e investigadores.

## 📂 Catálogo de Ejemplos

| Ejemplo | Descripción | Modelo | Estado |
| :--- | :--- | :--- | :--- |
| **[Ejemplo 1: Detección Básica](./ejemplo-1-deteccion-basica/)** | Entrenamiento de un modelo YOLOv8n para detectar el balón en video. | YOLOv8n | ✅ Completado |
| **Ejemplo 2: Seguimiento (Tracking)** | *Próximamente* - Implementación de BoTSORT para seguir la trayectoria del balón. | - | ⏳ Pendiente |

## 🛠️ Requisitos Generales

1.  **Entorno:** Se recomienda usar un entorno Conda (ej. `futsal_env`).
2.  **Librerías:** `ultralytics`, `roboflow`, `python-dotenv`.
3.  **Hardware:** Acceso al cluster SLURM para entrenamiento en GPU.

## 🤝 Contribuciones
Los estudiantes asistentes deben seguir la estructura de carpetas `ejemplo-X-...` para agregar nuevos módulos al proyecto.
