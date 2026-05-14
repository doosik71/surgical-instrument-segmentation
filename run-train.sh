#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] Virtual environment not found."
  echo "Run install.sh first."
  exit 1
fi

exec ./.venv/bin/python -m scripts.train_interpolation_model "$@"
