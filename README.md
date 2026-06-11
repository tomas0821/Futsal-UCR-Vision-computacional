# Futsal Analysis Project - UCR

Este repositorio es una guía técnica y un catálogo de ejemplos para el análisis de video en futsal utilizando Inteligencia Artificial. Está diseñado para ser utilizado por estudiantes asistentes e investigadores.

---

## 🎓 Tutorial: Cálculos en el Cluster UCR

**¡NUEVO!** Aprende a ejecutar análisis de futsal en el cluster UCR con GPU.

### 📚 [Ver Tutorial Completo →](./README_TUTORIAL.md)

**Módulos disponibles:**
1. **GitHub** - Acceso a archivos y clonación
2. **SLURM Serial** - Ejecutar trabajos en la partición serial
3. **GPU** - Acelerar cálculos con GPU
4. **Git Avanzado** - Branches, merge y colaboración

**Archivos de ejemplo en `tutorial/`:**
- `ejemplo_serial.sh` - Script básico para SLURM
- `ejemplo_gpu.sh` - Script con GPU
- `simple_script.py` - Python para trabajo serial
- `script_con_gpu.py` - Python con GPU (PyTorch/TensorFlow)

**Duración total:** ~45 minutos

---

## 📂 Catálogo de Ejemplos

| Ejemplo | Descripción | Modelo | Estado |
| :--- | :--- | :--- | :--- |
| **[Ejemplo 1: Detección Básica](./ejemplo-1-deteccion-basica/)** | Entrenamiento de un modelo YOLOv8m para detectar el balón. | YOLOv8m | ✅ Completado |
| **[Ejemplo 2: Jugadores y Árbitros](./ejemplo-2-jugadores-y-arbitros/)** | Detección multiclase de personas en cancha (excluyendo el balón). | YOLOv8m | ✅ Completado |
| **[Ejemplo 3: Modelos Combinados](./ejemplo-3-modelos-combinados/)** | Tutorial avanzado: Inferencia simultánea usando múltiples modelos. | Multimodelo | ✅ Completado |
| **[Ejemplo 4: Keypoints de Cancha](./ejemplo-4-deteccion-cancha-keypoints/)** | Estimación de pose para detectar la geometría y 22 puntos de la cancha. | YOLOv8m-pose | ✅ Completado |

## 📖 Documentación de Investigación

*   **[Análisis Bibliográfico](./analisis-bibliografico/):** Investigación técnica realizada por las estudiantes sobre el estado del arte en visión computacional deportiva.

## 📺 Video Tutoriales de Referencia

Para profundizar en las técnicas utilizadas en este proyecto, se recomiendan los siguientes tutoriales:

1.  **[YOLOv8 Object Detection Tutorial](https://www.youtube.com/watch?v=aBVGKoNZQUw):** Introducción completa a la arquitectura y entrenamiento.
2.  **[Advanced Computer Vision for Sports](https://www.youtube.com/watch?v=neBZ6huolkg&t=2140s):** Técnicas avanzadas aplicadas al análisis deportivo y seguimiento.

## 🛠️ Requisitos del Sistema

### Hardware Recomendado

| Componente | Mínimo | Recomendado |
|-----------|--------|------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | CPU Intel/AMD | NVIDIA RTX 3050+ |
| **Disco** | 50 GB | 200+ GB |

### Software Requerido

- **Python:** 3.9+ (recomendado 3.11)
- **Conda:** Miniconda o Anaconda
- **Git:** 2.25+
- **SSH:** Para acceso al cluster (opcional)

### Librerías Principales

| Librería | Versión | Propósito |
|----------|---------|----------|
| `ultralytics` | ≥8.0 | YOLOv8/v11 training & inference |
| `opencv-python` | ≥4.8 | Video processing |
| `numpy` | ≥1.24 | Numerical operations |
| `torch` | ≥2.0 | GPU acceleration (opcional) |
| `roboflow` | ≥1.1 | Dataset management (opcional) |

### Acceso al Cluster (Opcional pero Recomendado)

```
Host: 172.16.24.2
Puerto SSH: 22
Usuario: tu_usuario_ucr
VPN: acceso.ucr.ac.cr (si es remoto)
```

## 🔧 Instalación y Configuración

### Paso 1: Instalar Conda (si no lo tienes)

#### Opción A: Miniconda (recomendado - ligero)

```bash
# Descargar Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Instalar
bash Miniconda3-latest-Linux-x86_64.sh

# Reiniciar terminal o ejecutar
source ~/.bashrc
```

#### Opción B: Anaconda (completo)

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-Linux-x86_64.sh
bash Anaconda3-2024.02-Linux-x86_64.sh
source ~/.bashrc
```

**Verificar instalación:**
```bash
conda --version
```

---

### Paso 2: Clonar el Repositorio

```bash
git clone https://github.com/tomas0821/Futsal-UCR-Vision-computacional
cd Futsal-UCR-Vision-computacional
```

---

### Paso 3: Crear Entorno Conda

#### Opción A: Crear entorno desde cero

```bash
# Crear entorno con Python 3.11
conda create -n futsal python=3.11 -y

# Activar entorno
conda activate futsal
```

#### Opción B: Crear desde archivo (si existe environment.yml)

```bash
conda env create -f environment.yml
conda activate futsal
```

**Verificar activación:**
```bash
python --version  # Debe mostrar Python 3.11.x
which python      # Debe mostrar ruta del entorno
```

---

### Paso 4: Instalar Dependencias

#### Dependencias Base (obligatorias)

```bash
pip install --upgrade pip
pip install ultralytics opencv-python numpy pandas matplotlib
```

#### Para Trabajo con GPU

**PyTorch con CUDA (recomendado):**
```bash
# Para GPU NVIDIA (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Para GPU NVIDIA (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Solo CPU (si no tienes GPU)
pip install torch torchvision torchaudio
```

**O TensorFlow:**
```bash
pip install tensorflow[and-cuda]  # Con GPU
# o
pip install tensorflow             # Solo CPU
```

#### Dependencias Adicionales (opcionales)

```bash
# Manejo de datos
pip install roboflow python-dotenv

# Visualización avanzada
pip install seaborn scikit-learn

# Desarrollo
pip install jupyter ipython

# Verificación de código
pip install black flake8
```

---

### Paso 5: Verificar Instalación

```bash
# Verificar Python
python -c "import sys; print(f'Python: {sys.version}')"

# Verificar YOLOv8
python -c "from ultralytics import YOLO; print('✓ YOLO OK')"

# Verificar OpenCV
python -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"

# Verificar PyTorch (si lo instalaste)
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA disponible: {torch.cuda.is_available()}')"

# Verificar TensorFlow (si lo instalaste)
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__}')"
```

---

### Paso 6: Crear Archivo de Configuración (Opcional)

Si usas API keys (Roboflow, etc.):

```bash
# Crear archivo .env
cat > .env << 'EOF'
ROBOFLOW_API_KEY=tu_api_key_aqui
CLUSTER_USER=tu_usuario
CLUSTER_HOST=172.16.24.2
EOF

# NO COMMITEAR este archivo
echo ".env" >> .gitignore
```

---

## ⚡ Quick Start - Primeros Pasos

### Para principiantes (sin experiencia con cluster):

```bash
# 1. Clonar repositorio
git clone https://github.com/tomas0821/Futsal-UCR-Vision-computacional
cd Futsal-UCR-Vision-computacional

# 2. Crear y activar entorno
conda create -n futsal python=3.11 -y
conda activate futsal

# 3. Instalar dependencias
pip install ultralytics opencv-python torch torchvision

# 4. Leer el tutorial
cat README_TUTORIAL.md

# 5. Conectar al cluster
ssh usuario@172.16.24.2

# 6. Enviar tu primer trabajo
sbatch tutorial/ejemplo_serial.sh

# 7. Monitorear
squeue -u $USER
```

### Para usuarios avanzados:

```bash
# Ir directo a GPU
sbatch tutorial/ejemplo_gpu.sh

# Ver disponibilidad GPU
sinfo -p gpu

# Crear tu rama para experimentos
git checkout -b mi-experimento
```

---

## 🤝 Contribuciones

Las estudiantes asistentes deben seguir la estructura de carpetas `ejemplo-X-...` para agregar nuevos módulos al proyecto.

**Para trabajar en el cluster:**
1. Crear una rama: `git checkout -b mi-feature`
2. Hacer cambios y commits
3. Mergear a master: `git merge mi-feature`
4. Subir: `git push origin master`

---

## 🐛 Troubleshooting - Instalación

### "conda: command not found"

```bash
# Conda no está en el PATH
source ~/miniconda3/bin/activate
# O reinicia la terminal después de instalar
bash
```

### "No module named 'ultralytics'"

```bash
# Verificar que el entorno esté activado
conda activate futsal
# Reinstalar
pip install --upgrade ultralytics
```

### Error de GPU / CUDA

```bash
# Verificar CUDA
nvidia-smi

# Si no funciona GPU, instalar CPU-only
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Error con OpenCV

```bash
# En sistemas Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libglib2.0-0
pip install --upgrade opencv-python
```

### Conflicto de versiones

```bash
# Crear nuevo entorno limpio
conda env remove -n futsal
conda create -n futsal python=3.11 -y
conda activate futsal
pip install -r requirements.txt  # Si existe
```

### GitHub - "Authentication failed"

```bash
# Usar HTTPS con token (recomendado)
git config --global credential.helper store
# O generar SSH key
ssh-keygen -t ed25519 -C "tu_email@ejemplo.com"
```

---

## 📞 Soporte

- **Preguntas sobre setup:** Ver sección [🔧 Instalación](#-instalación-y-configuración)
- **Preguntas sobre el tutorial:** Ver [README_TUTORIAL.md](./README_TUTORIAL.md#troubleshooting)
- **Problemas con cluster:** Contactar administrador del cluster
- **Issues técnicos:** Abrir un issue en [GitHub Issues](https://github.com/tomas0821/Futsal-UCR-Vision-computacional/issues)

---

## 📚 Recursos Adicionales

- [Conda Cheat Sheet](https://conda.io/projects/conda/en/latest/_downloads/misc/conda-cheatsheet.pdf)
- [Git Workflow Guide](https://git-scm.com/book/en/v2)
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyTorch Getting Started](https://pytorch.org/get-started/locally/)
- [UCR Cluster Documentation](./cluster/)
