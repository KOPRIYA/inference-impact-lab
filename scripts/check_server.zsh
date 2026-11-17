#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
source scripts/common.zsh
require_python
"$PYTHON_BIN" scripts/check_server.py --base-url "${BASE_URL:-http://127.0.0.1:8000/v1}" --model "${MODEL_ID:-default_model}"
