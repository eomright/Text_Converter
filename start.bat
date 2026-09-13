@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "VENV=%USERPROFILE%\korean-stt-runtime\.venv"

if not exist "%VENV%\Scripts\python.exe" (
  echo [ERROR] venv not found: %VENV%
  echo Run setup first:
  echo   py -3.12 -m venv "%USERPROFILE%\korean-stt-runtime\.venv"
  echo   "%USERPROFILE%\korean-stt-runtime\.venv\Scripts\python.exe" -m pip install -r server\requirements.txt
  pause
  exit /b 1
)

echo Starting Korean STT server...
echo Browser will open at http://127.0.0.1:8000
echo Press Ctrl+C to stop.
echo.

start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000"
"%VENV%\Scripts\python.exe" -m uvicorn server.main:app --host 127.0.0.1 --port 8000

pause
