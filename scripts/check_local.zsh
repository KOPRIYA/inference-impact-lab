#!/bin/zsh
# Offline checks and a clearly simulated pipeline run; no model download or GPU needed.
set -euo pipefail
cd "${0:A:h:h}"
source scripts/common.zsh
require_python
"$PYTHON_BIN" -m unittest discover -s tests -v
mkdir -p results
RUN_DIR=$(mktemp -d results/offline-check-XXXXXXXX)
"$PYTHON_BIN" -m inference_lab demo --out "$RUN_DIR"
"$PYTHON_BIN" -m inference_lab verify-bundle evidence/apple-m5-v1
print "Offline checks complete. Simulated report: $RUN_DIR/report.html"
