#!/usr/bin/env bash
# setup_cuda_torch.sh
# Run after `uv sync` to ensure the CUDA-enabled torch is installed.
# Downloads and extracts the cu121 torch wheel into the venv.

set -e

WHEEL_URL="https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl"
WHEEL_NAME="torch-2.5.1+cu121-cp311-cp311-win_amd64.whl"
CACHE_DIR="${LOCALAPPDATA:-/tmp}/torch-cu121"
WHEEL_PATH="${CACHE_DIR}/${WHEEL_NAME}"
SITE_PKGS=".venv/Lib/site-packages"
PYTHON=".venv/Scripts/python.exe"

mkdir -p "$CACHE_DIR"

if [ ! -f "$WHEEL_PATH" ] || [ $(stat -c%s "$WHEEL_PATH") -lt 1000000000 ]; then
    echo "Downloading CUDA torch wheel (~2.3 GB)..."
    curl -L -o "$WHEEL_PATH" "$WHEEL_URL"
else
    echo "Using cached wheel: $WHEEL_PATH ($(du -h "$WHEEL_PATH" | cut -f1))"
fi

echo "Installing CUDA torch into venv..."

# Extract wheel directly into site-packages (avoids uv pip overwrite issue)
python -c "
import zipfile, os, shutil

wheel = r'${WHEEL_PATH}'
site = r'${SITE_PKGS}'

# Remove existing torch and its dist-info
shutil.rmtree(os.path.join(site, 'torch'), ignore_errors=True)
for d in os.listdir(site):
    if d.startswith('torch-'):
        shutil.rmtree(os.path.join(site, d), ignore_errors=True)

# Extract new wheel
with zipfile.ZipFile(wheel) as z:
    z.extractall(site)

print('Installed successfully')
"

echo "Verifying..."
"$PYTHON" -c "import torch; print(f'torch {torch.__version__} | CUDA {torch.cuda.is_available()}')"