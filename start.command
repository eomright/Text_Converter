#!/bin/bash
# Korean STT - start the server on macOS and open the browser.

cd "$(dirname "$0")" || exit 1
RUNTIME="${KSTT_RUNTIME:-$HOME/korean-stt-runtime}"
VENV="$RUNTIME/.venv"

# A double-clicked script may not have Homebrew on PATH - add it so ffmpeg is found.
for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
  if [ -x "$b" ]; then eval "$("$b" shellenv)"; break; fi
done

if [ ! -x "$VENV/bin/python" ]; then
  echo "[ERROR] Not set up yet: $VENV"
  echo "Run setup.command first."
  read -r -p "Press Enter to close..." _
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg not found. Run setup.command first."
  read -r -p "Press Enter to close..." _
  exit 1
fi

echo "Starting Korean STT server..."
echo "Browser will open at http://127.0.0.1:8000"
echo "Press Ctrl+C to stop."
echo

(sleep 4; open "http://127.0.0.1:8000") &
"$VENV/bin/python" -m uvicorn server.main:app --host 127.0.0.1 --port 8000

read -r -p "Server stopped. Press Enter to close..." _
