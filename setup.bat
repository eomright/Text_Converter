@echo off
rem Korean STT - first-time setup.
rem Installs Python 3.12 and ffmpeg with winget if missing, creates the venv,
rem installs packages, and downloads the default Whisper models.
rem Safe to run again: steps that are already done are skipped.
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not defined KSTT_RUNTIME set "KSTT_RUNTIME=%USERPROFILE%\korean-stt-runtime"
set "VENV=%KSTT_RUNTIME%\.venv"

echo ================================================
echo   Korean STT - first-time setup
echo   Install folder: %KSTT_RUNTIME%
echo ================================================
echo.

rem ---------------------------------------------------------------- 1. Python
call :find_python
if defined PYEXE goto have_python
echo [1/5] Python 3.12 not found. Installing with winget...
call :need_winget
if errorlevel 1 goto fail
winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
call :find_python
if defined PYEXE goto have_python
echo.
echo [ERROR] Python was installed but could not be found yet.
echo         Close this window and run setup.bat once more.
goto fail
:have_python
echo [1/5] Python OK: %PYEXE%

rem ---------------------------------------------------------------- 2. ffmpeg
call :find_ffmpeg
if defined FFOK goto have_ffmpeg
echo [2/5] ffmpeg not found. Installing with winget...
call :need_winget
if errorlevel 1 goto fail
winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
call :find_ffmpeg
if defined FFOK goto have_ffmpeg
echo.
echo [ERROR] ffmpeg was installed but could not be found yet.
echo         Close this window and run setup.bat once more.
goto fail
:have_ffmpeg
echo [2/5] ffmpeg OK

rem ---------------------------------------------------------------- 3. venv
if exist "%VENV%\Scripts\python.exe" goto have_venv
echo [3/5] Creating virtual environment...
"%PYEXE%" -m venv "%VENV%"
if errorlevel 1 (
  echo [ERROR] Failed to create the virtual environment.
  goto fail
)
:have_venv
echo [3/5] venv OK: %VENV%

rem ---------------------------------------------------------------- 4. packages
echo [4/5] Installing packages - this takes a few minutes the first time...
"%VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r server\requirements.txt
if errorlevel 1 (
  echo [ERROR] Package install failed. Check your internet connection and run setup.bat again.
  goto fail
)
"%VENV%\Scripts\python.exe" -c "import faster_whisper, fastapi, uvicorn, multipart"
if errorlevel 1 (
  echo [ERROR] Packages installed but could not be imported.
  goto fail
)
echo [4/5] Packages OK

rem ---------------------------------------------------------------- 5. models
echo [5/5] Downloading speech models - about 1GB, only the first time...
"%VENV%\Scripts\python.exe" -c "from server import config; from faster_whisper.utils import download_model; [print('  -', m, '->', download_model(config.AVAILABLE_MODELS[m]['repo'], cache_dir=str(config.MODEL_DIR))) for m in dict.fromkeys([config.DEFAULT_MODEL, config.LIVE_MODEL])]"
if errorlevel 1 (
  echo [ERROR] Model download failed. Check your internet connection and run setup.bat again.
  goto fail
)
echo [5/5] Models OK
echo.
echo ================================================
echo   Setup complete. Double-click start.bat to run.
echo   Lecture mode downloads one more model - about 3GB - the first time you use it.
echo ================================================
echo.
pause
exit /b 0

:fail
echo.
echo Setup did not finish. Fix the message above and run setup.bat again.
echo.
pause
exit /b 1

rem ================================================================ helpers
:find_python
set "PYEXE="
for /f "delims=" %%i in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
if defined PYEXE exit /b 0
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PYEXE exit /b 0
if exist "%ProgramFiles%\Python312\python.exe" set "PYEXE=%ProgramFiles%\Python312\python.exe"
exit /b 0

:find_ffmpeg
rem winget may put ffmpeg on PATH only for new windows, so also look where it installs.
set "FFOK="
where ffmpeg >nul 2>nul && set "FFOK=1"
if defined FFOK exit /b 0
for /d %%p in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do for /d %%b in ("%%p\*") do if exist "%%b\bin\ffmpeg.exe" set "FFOK=1"
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" set "FFOK=1"
exit /b 0

:need_winget
where winget >nul 2>nul
if not errorlevel 1 exit /b 0
echo.
echo [ERROR] winget is not available on this PC, so automatic install is not possible.
echo         Install these by hand, then run setup.bat again:
echo           Python 3.12 : https://www.python.org/downloads/release/python-3120/
echo           ffmpeg      : https://www.gyan.dev/ffmpeg/builds/   - add its bin folder to PATH
exit /b 1
