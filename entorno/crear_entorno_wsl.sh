#!/usr/bin/env bash
# Crea el entorno de entrenamiento con GPU en Linux / WSL2, con las mismas
# versiones exactas para todo el equipo (entorno/requirements-gpu.lock).
#
# Requisitos:
#   - Driver de NVIDIA instalado en Windows (WSL2 lo comparte; `nvidia-smi`
#     debe funcionar dentro de Ubuntu). No hace falta instalar CUDA: viene en
#     los paquetes nvidia-* de pip.
#   - uv (https://docs.astral.sh/uv/). Si no lo tienes:
#       curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Uso, desde la raíz del repo dentro de WSL:
#   bash entorno/crear_entorno_wsl.sh                 # en ~/venvs/clasificacion_basura
#   bash entorno/crear_entorno_wsl.sh /otra/ruta      # en otra carpeta
#
# El entorno va en el disco de Linux y no dentro del repo en /mnt/c, porque
# leer miles de archivos desde /mnt/c es mucho más lento.
#
# Para regenerar el lock después de cambiar requirements.txt (desde Windows o
# Linux, con uv):
#   uv pip compile entorno/requirements-gpu.in --python-version 3.12 \
#     --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header \
#     -o entorno/requirements-gpu.lock

set -euo pipefail

VENV="${1:-$HOME/venvs/clasificacion_basura}"
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="3.12"

command -v uv >/dev/null || { echo "Falta uv; ver instrucciones arriba." >&2; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python "$PYTHON_VERSION" "$VENV"
fi
uv pip install --python "$VENV/bin/python" --require-hashes \
    -r "$RAIZ/entorno/requirements-gpu.lock"

# TensorFlow 2.21 busca las librerías de CUDA instaladas por pip en las
# carpetas de su RUNPATH, pero ahí falta nvidia/cusolver/lib: sin esto falla
# con "Could not load dynamic library 'libcusolver.so.11'" y no usa la GPU.
# Se agregan todas las carpetas nvidia/*/lib a LD_LIBRARY_PATH al activar.
MARCA="# crear_entorno_wsl.sh: librerias de NVIDIA"
if ! grep -qF "$MARCA" "$VENV/bin/activate"; then
    cat >> "$VENV/bin/activate" <<EOF

$MARCA
_NV="\$VIRTUAL_ENV/lib/python$PYTHON_VERSION/site-packages/nvidia"
export LD_LIBRARY_PATH="\$(ls -d "\$_NV"/*/lib | paste -sd:)\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
unset _NV
EOF
fi

echo
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python - <<'PY'
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import platform
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"Python {platform.python_version()} | TensorFlow {tf.__version__}")
print(f"GPUs: {[g.name for g in gpus] or 'NINGUNA'}")
if gpus:
    print("GPU:", tf.config.experimental.get_device_details(gpus[0]).get("device_name"))
PY
echo
echo "Listo. Actívalo con: source $VENV/bin/activate"
