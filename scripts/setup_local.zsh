#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
# Explicit override supports Python installations outside the usual PATH.
if [[ ! -x .venv/bin/python ]]; then
  PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-}"
  if [[ -z "$PYTHON_BOOTSTRAP" ]]; then
    for candidate in python3.12 python3.11; do
      if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BOOTSTRAP="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$PYTHON_BOOTSTRAP" ]]; then
    print -u2 'Install native Apple Silicon Python 3.11 or 3.12 first.'
    exit 1
  fi
  "$PYTHON_BOOTSTRAP" -m venv .venv
fi
source scripts/common.zsh
require_python
"$PYTHON_BIN" -c 'import platform,sys; assert platform.system()=="Darwin" and platform.machine()=="arm64", "Native Apple Silicon required"; assert (3,11)<=sys.version_info[:2]<(3,13), "Use Python 3.11 or 3.12"'
"$PYTHON_BIN" -m pip install -e '.[mac]'
"$PYTHON_BIN" -m inference_lab doctor
if [[ ! -f "$MODEL_DIR/manifest.json" ]]; then
  "$PYTHON_BIN" -m inference_lab prepare --models "$MODEL_DIR"
else
  print "Reusing prepared model directory: $MODEL_DIR"
fi
# A local lock may contain an editable-install path; it stays under ignored models/.
"$PYTHON_BIN" -m pip freeze > "$MODEL_DIR/environment-lock.txt"
print 'Setup complete. Start the server: zsh scripts/serve_local.zsh'
