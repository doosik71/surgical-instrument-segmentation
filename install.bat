@echo off
setlocal

cd /d "%~dp0"
set UV_LINK_MODE=copy

where uv >nul 2>nul
if errorlevel 1 (
  echo [ERROR] uv is not installed or not on PATH.
  echo Install uv first: https://docs.astral.sh/uv/
  exit /b 1
)

if not exist ".venv" (
  echo [INFO] Creating virtual environment...
  uv venv --python 3.12
  if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"

echo [INFO] Syncing project dependencies with uv...
uv sync --extra dev
if errorlevel 1 exit /b 1

echo [INFO] Downloading local model files...
python -m app.scripts.download_models
if errorlevel 1 exit /b 1

echo [INFO] Verifying GPU availability...
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
if errorlevel 1 exit /b 1

echo [INFO] Installation completed.
exit /b 0
