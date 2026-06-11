# 🚀 Tutorial: Cálculos en el Cluster UCR con GitHub

Tutorial práctico para aprender a ejecutar cálculos en el cluster UCR usando SLURM, GitHub y GPU.

---

## 📋 Tabla de Contenidos

1. [Módulo 1: GitHub](#módulo-1-github--acceso-a-los-archivos)
2. [Módulo 2: SLURM Serial](#módulo-2-fundamentos-slurm---trabajo-serial)
3. [Módulo 3: GPU](#módulo-3-trabajo-en-gpu)
4. [Módulo 4: Git Avanzado](#módulo-4-git-avanzado---branches-y-merge)

---

## Módulo 1: GitHub – Acceso a los Archivos

### 📚 ¿Qué es GitHub?

GitHub es una plataforma de control de versiones que permite:
- Compartir código y archivos
- Colaborar en proyectos
- Mantener un historial de cambios
- Trabajar en equipo de forma organizada

### 🔽 Clonar el Repositorio

Abre tu terminal y ejecuta:

```bash
git clone https://github.com/tomas0821/Futsal-UCR-Vision-computacional
cd Futsal-UCR-Vision-computacional
```

### 📁 Explorar la Estructura

```
Futsal-UCR-Vision-computacional/
├── tutorial/                    ← Tu estás aquí
│   ├── ejemplo_serial.sh       # Script SLURM básico
│   ├── ejemplo_gpu.sh          # Script SLURM con GPU
│   ├── simple_script.py        # Python para serial
│   └── script_con_gpu.py       # Python para GPU
├── cluster/                     # Configuración del cluster
├── futsal-ucr/                 # Análisis de futsal
└── README.md                    # Documentación principal
```

### 🔄 Descargar Cambios (Pull)

Para obtener los últimos cambios:

```bash
git pull origin master
```

---

## Módulo 2: Fundamentos SLURM – Trabajo Serial

### 🏗️ ¿Qué es SLURM?

SLURM es el gestor de recursos del cluster. Administra:
- **Colas (Particiones)**: serial, gpu, normal, etc.
- **Recursos**: CPU, GPU, memoria, tiempo
- **Trabajos**: Ordena y ejecuta tareas

### 📍 Particiones Disponibles

| Partición | Uso | CPU | GPU | Tiempo Máx |
|-----------|-----|-----|-----|-----------|
| `serial` | Trabajos sin paralelismo | 1-4 | No | 1 día |
| `gpu` | Trabajos con GPU | 4-8 | 1-2 | 1 día |
| `normal` | General | Varios | Opcional | 2 días |

### 📝 Script SLURM Básico

Archivo: `tutorial/ejemplo_serial.sh`

```bash
#!/bin/bash
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

echo "¡Hola desde el cluster!"
python3 simple_script.py
```

### ⏱️ Parámetros Importantes

- `--partition`: Qué cola usar
- `--nodes`: Número de nodos
- `--ntasks`: Número de tareas paralelas
- `--cpus-per-task`: CPUs por tarea
- `--time`: Tiempo máximo (HH:MM:SS)
- `--job-name`: Nombre descriptivo
- `--output`: Archivo de salida

### 🚀 Enviar un Trabajo

```bash
sbatch tutorial/ejemplo_serial.sh
```

**Respuesta esperada:**
```
Submitted batch job 12345
```

### 📊 Monitorear Trabajos

Ver todos tus trabajos:

```bash
squeue -u $USER
```

**Output:**
```
JOBID PARTITION NAME              ST  TIME NODES CPUS
12345 serial    futsal_serial     R   0:15 1     1
```

Significado de estados:
- `PD` = Pendiente (esperando recursos)
- `R` = Ejecutándose
- `CA` = Cancelado
- `CD` = Completado
- `F` = Falló

### 📖 Ver Resultados

Los archivos de salida van a tu directorio actual:

```bash
# Ver salida
cat salida_12345.log

# Ver errores
cat error_12345.log

# Ver en tiempo real
tail -f salida_12345.log
```

### 🎯 Práctica: Tu Primer Trabajo

```bash
# 1. Conectate al cluster
ssh usuario@172.16.24.2

# 2. Clona el repo (si no lo has hecho)
git clone https://github.com/tomas0821/Futsal-UCR-Vision-computacional
cd Futsal-UCR-Vision-computacional

# 3. Envía el trabajo serial
sbatch tutorial/ejemplo_serial.sh

# 4. Monitorea
squeue -u $USER

# 5. Ve los resultados (cuando esté CD - Completado)
cat salida_*.log
```

---

## Módulo 3: Trabajo en GPU

### 🖥️ ¿Por Qué GPU?

Las GPUs aceleran:
- **Redes neuronales**: 10-100x más rápido
- **Procesamiento de video**: 50-200x más rápido
- **Cálculos matriciales**: 20-100x más rápido

### 📝 Script SLURM con GPU

Archivo: `tutorial/ejemplo_gpu.sh`

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1        ← GPU disponible
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# Verificar GPU
nvidia-smi

# Ejecutar con GPU
python3 script_con_gpu.py
```

**Parámetro clave:**
- `--gres=gpu:1` = Solicitar 1 GPU (máximo 2)

### 🚀 Enviar Trabajo con GPU

```bash
sbatch tutorial/ejemplo_gpu.sh
```

### 📊 Verificar GPU Disponible

Antes de enviar:

```bash
# Ver GPUs del cluster
sinfo -p gpu

# Ver GPU en tiempo real mientras ejecuta
watch -n 1 nvidia-smi
```

### 💻 Usar GPU en Python

**Ejemplo con PyTorch:**

```python
import torch

# Verificar GPU
print(torch.cuda.is_available())    # True si hay GPU
print(torch.cuda.get_device_name(0)) # Nombre GPU

# Crear tensor en GPU
device = torch.device("cuda")
x = torch.randn(1000, 1000, device=device)
y = x @ x  # Multiplicación matricial
```

**Ejemplo con TensorFlow:**

```python
import tensorflow as tf

# Verificar GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs disponibles: {len(gpus)}")

# Operación automáticamente en GPU
A = tf.random.normal([1000, 1000])
B = tf.random.normal([1000, 1000])
C = tf.matmul(A, B)
```

### 🎯 Práctica: Tu Primer Trabajo con GPU

```bash
# 1. Envía el trabajo GPU
sbatch tutorial/ejemplo_gpu.sh

# 2. Monitorea
squeue -u $USER

# 3. Ver GPU en uso (en otra ventana)
watch -n 1 nvidia-smi

# 4. Ver resultados
cat salida_gpu_*.log
```

---

## Módulo 4: Git Avanzado – Branches y Merge

### 🌿 ¿Qué es una Rama (Branch)?

Una rama permite:
- Hacer cambios sin afectar `master`
- Trabajar en paralelo con compañeros
- Organizar experimentos

### 📋 Workflow Básico

```
master (rama principal)
  ↓
  └─→ mi-experimento (tu rama)
       ├─ Cambio 1
       ├─ Cambio 2
       └─ Cambio 3
  ↓
master (después de merge)
```

### 🔄 Crear y Cambiar de Rama

```bash
# Ver todas las ramas
git branch

# Crear nueva rama
git checkout -b mi-experimento

# Cambiar a otra rama
git checkout master
```

### ✏️ Hacer Cambios en tu Rama

```bash
# Crear o editar archivo
echo "nuevo contenido" > mi_archivo.py

# Ver cambios
git status

# Agregar cambios
git add mi_archivo.py

# Commit (guardar cambios localmente)
git commit -m "Agregué nuevo experimento"

# Ver historial
git log --oneline
```

### 🔗 Mergear Cambios a Master

```bash
# Cambiar a master
git checkout master

# Asegurar que master esté actualizado
git pull origin master

# Mergear tu rama
git merge mi-experimento

# Subir cambios al repositorio
git push origin master
```

### 🧹 Limpiar Rama Vieja

```bash
# Eliminar rama local
git branch -d mi-experimento

# Eliminar rama en el repositorio (si existe)
git push origin --delete mi-experimento
```

### 🎯 Ejemplo Completo: Experimento con Futsal

```bash
# 1. Crear rama para tu experimento
git checkout -b analisis-futsal-mejoras

# 2. Hacer cambios
# Editar futsal-ucr/inference.py
# Hacer experimentos
# Crear resultados

# 3. Commit cambios
git add futsal-ucr/inference.py
git commit -m "Mejoré detección de jugadores con nuevo modelo"

# 4. Mergear a master
git checkout master
git pull origin master
git merge analisis-futsal-mejoras

# 5. Subir al repositorio
git push origin master

# 6. Limpiar
git branch -d analisis-futsal-mejoras
```

### 📊 Ver Historial de Cambios

```bash
# Log simple
git log --oneline

# Log con ramas visualizadas
git log --oneline --graph --all

# Ver diferencias entre ramas
git diff master mi-experimento
```

---

## 🔗 Comandos Útiles Rápidos

### SLURM

```bash
sbatch script.sh           # Enviar trabajo
squeue -u $USER            # Ver trabajos activos
scancel JOBID              # Cancelar trabajo
sacct -S 2024-01-01        # Ver histórico
sinfo                      # Ver estado del cluster
```

### Git

```bash
git clone URL              # Clonar repositorio
git pull origin master     # Actualizar desde web
git add archivo            # Preparar cambios
git commit -m "msg"        # Guardar cambios locales
git push origin master     # Subir a GitHub
git status                 # Ver estado
git log --oneline          # Ver historial
```

### SSH al Cluster

```bash
ssh usuario@172.16.24.2    # Conectar
ssh -i ~/.ssh/id_rsa usuario@172.16.24.2  # Con llave privada
scp archivo usuario@172.16.24.2:~/        # Copiar archivo
```

---

## 📚 Recursos Adicionales

- [SLURM Documentación](https://slurm.schedmd.com/)
- [Git Documentación](https://git-scm.com/doc)
- [GitHub Guía](https://guides.github.com/)
- [PyTorch en GPU](https://pytorch.org/docs/stable/cuda.html)
- [TensorFlow en GPU](https://www.tensorflow.org/guide/gpu)

---

## ❓ Troubleshooting

### "Submitted batch job" pero no aparece en squeue

```bash
# Está esperando recursos (estado PD)
# Verifica disponibilidad
sinfo -p serial
sinfo -p gpu
```

### Error: "sbatch: command not found"

```bash
# SLURM no está cargado
module load slurm
```

### GPU no disponible

```bash
# Verifica GPU instaladas
nvidia-smi

# En script, instala drivers si es necesario
conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch -c nvidia
```

### "fatal: not a git repository"

```bash
# No estás en un directorio git
cd Futsal-UCR-Vision-computacional
git status
```

---

## 📝 Notas Finales

- **Guarda tu trabajo**: Siempre haz commit con `git add` + `git commit`
- **Nombres descriptivos**: Los commits y ramas ayudan a recordar qué hiciste
- **Monitorea recursos**: Usa `squeue` y `sinfo` regularmente
- **Lee los logs**: Siempre revisa `salida_*.log` para entender qué pasó

---

**¡Bienvenido al cluster UCR! 🎉**

¿Preguntas? Revisa los scripts en `tutorial/` o consulta la documentación oficial.
