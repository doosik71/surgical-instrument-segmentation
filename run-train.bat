@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Run install.bat first.
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m scripts.train_interpolation_model
exit /b %errorlevel%
