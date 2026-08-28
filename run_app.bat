@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "APP_PYTHON=.venv\Scripts\python.exe"
) else (
  set "APP_PYTHON=python"
)

"%APP_PYTHON%" -m src.gui.app
if errorlevel 1 (
  echo.
  echo GuitarBapu failed to start. Install requirements.txt first.
  pause
)

endlocal
