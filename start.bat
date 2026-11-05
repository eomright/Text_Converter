@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if not defined KSTT_RUNTIME set "KSTT_RUNTIME=%USERPROFILE%\korean-stt-runtime"
set "VENV=%KSTT_RUNTIME%\.venv"

if not exist "%VENV%\Scripts\python.exe" (
  echo [ERROR] Not set up yet: %VENV%
  echo Run setup.bat first.
  pause
  exit /b 1
)

rem ffmpeg installed by winget may not be on PATH in this window yet - add it.
where ffmpeg >nul 2>nul
if errorlevel 1 (
  for /d %%p in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do for /d %%b in ("%%p\*") do if exist "%%b\bin\ffmpeg.exe" set "PATH=%%b\bin;%PATH%"
  if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [ERROR] ffmpeg not found. Run setup.bat first.
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
