# Shared helpers; sourced after each entry point changes to the repository root.
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-models/qwen}"
require_python() {
  if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    print -u2 "Python environment missing. Run: zsh scripts/setup_local.zsh"
    return 1
  fi
}
require_models() {
  if [[ ! -f "$MODEL_DIR/manifest.json" ]]; then
    print -u2 "Prepared models missing at $MODEL_DIR. Run: zsh scripts/setup_local.zsh"
    return 1
  fi
}
