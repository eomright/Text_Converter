#!/bin/bash
# Korean STT - first-time setup for macOS.
# Installs Homebrew, Python 3.12 and ffmpeg if missing, creates the venv,
# installs packages, and downloads the default Whisper models.
# Safe to run again: steps that are already done are skipped.

cd "$(dirname "$0")" || exit 1
export PYTHONUTF8=1
RUNTIME="${KSTT_RUNTIME:-$HOME/korean-stt-runtime}"
VENV="$RUNTIME/.venv"

fail() {
  echo
  echo "[ERROR] $1"
  echo "Setup did not finish. Fix the message above and run setup.command again."
  echo
  read -r -p "Press Enter to close..." _
  exit 1
}

load_brew() {
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    if [ -x "$b" ]; then eval "$("$b" shellenv)"; return 0; fi
  done
  command -v brew >/dev/null 2>&1
}

find_python() {
  PYEXE=""
  for p in \
    "$(command -v python3.12 2>/dev/null)" \
    "${HOMEBREW_PREFIX:-/opt/homebrew}/opt/python@3.12/bin/python3.12" \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12; do
    if [ -n "$p" ] && [ -x "$p" ]; then PYEXE="$p"; return 0; fi
  done
  return 1
}

echo "================================================"
echo "  Korean STT - first-time setup (macOS)"
echo "  Install folder: $RUNTIME"
echo "================================================"
echo

# Let start.command open with a double-click later (files from the internet are blocked).
xattr -dr com.apple.quarantine . 2>/dev/null
chmod +x setup.command start.command 2>/dev/null

# ---------------------------------------------------------------- 0. Homebrew
load_brew
find_python; HAVE_PY=$?
command -v ffmpeg >/dev/null 2>&1; HAVE_FF=$?
if [ $HAVE_PY -ne 0 ] || [ $HAVE_FF -ne 0 ]; then
  if ! load_brew; then
    echo "[0/5] Homebrew not found. Installing it - your Mac password will be asked..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
      || fail "Homebrew install failed. See https://brew.sh"
    load_brew || fail "Homebrew was installed but could not be found. Run setup.command again."
  fi
  echo "[0/5] Homebrew OK"
fi

# ---------------------------------------------------------------- 1. Python
if ! find_python; then
  echo "[1/5] Python 3.12 not found. Installing with Homebrew..."
  brew install python@3.12 || fail "Python install failed."
  find_python || fail "Python was installed but could not be found. Run setup.command again."
fi
echo "[1/5] Python OK: $PYEXE"

# ---------------------------------------------------------------- 2. ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[2/5] ffmpeg not found. Installing with Homebrew..."
  brew install ffmpeg || fail "ffmpeg install failed."
  command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg was installed but could not be found. Run setup.command again."
fi
echo "[2/5] ffmpeg OK"

# ---------------------------------------------------------------- 3. venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "[3/5] Creating virtual environment..."
  mkdir -p "$RUNTIME"
  "$PYEXE" -m venv "$VENV" || fail "Failed to create the virtual environment."
fi
echo "[3/5] venv OK: $VENV"

# ---------------------------------------------------------------- 4. packages
echo "[4/5] Installing packages - this takes a few minutes the first time..."
"$VENV/bin/python" -m pip install --disable-pip-version-check -q -r server/requirements.txt \
  || fail "Package install failed. Check your internet connection and run setup.command again."
"$VENV/bin/python" -c "import faster_whisper, fastapi, uvicorn, multipart" \
  || fail "Packages installed but could not be imported."
echo "[4/5] Packages OK"

# ---------------------------------------------------------------- 5. models
echo "[5/5] Downloading speech models - about 1GB, only the first time..."
KSTT_RUNTIME="$RUNTIME" "$VENV/bin/python" -c "from server import config; from faster_whisper.utils import download_model; [print('  -', m, '->', download_model(config.AVAILABLE_MODELS[m]['repo'], cache_dir=str(config.MODEL_DIR))) for m in dict.fromkeys([config.DEFAULT_MODEL, config.LIVE_MODEL])]" \
  || fail "Model download failed. Check your internet connection and run setup.command again."
echo "[5/5] Models OK"

echo
echo "================================================"
echo "  Setup complete. Double-click start.command to run."
echo "  Lecture mode downloads one more model - about 3GB - the first time you use it."
echo "================================================"
echo
read -r -p "Press Enter to close..." _
