@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found.
  echo Run install.bat first.
  exit /b 1
)

".venv\Scripts\python.exe" -m scripts.compare_speed %*
exit /b %errorlevel%
