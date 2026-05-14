#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export UV_LINK_MODE=copy

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is not installed or not on PATH."
  echo "Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "[INFO] Creating virtual environment..."
  uv venv --python 3.12
fi

echo "[INFO] Syncing project dependencies with uv..."
uv sync --extra dev

echo "[INFO] Downloading local model files..."
./.venv/bin/python -m app.scripts.download_models

echo "[INFO] Verifying GPU availability..."
./.venv/bin/python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"

echo "[INFO] Installation completed."
