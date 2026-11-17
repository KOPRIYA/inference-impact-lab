#!/bin/zsh
# Run against an already running OpenAI-compatible streaming inference server.
set -euo pipefail
cd "${0:A:h:h}"
source scripts/common.zsh
require_python
MODEL_ID="${MODEL_ID:-default_model}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
"$PYTHON_BIN" scripts/check_server.py --base-url "$BASE_URL" --model "$MODEL_ID"
mkdir -p results
RUN_DIR=$(mktemp -d results/http-load-XXXXXXXX)
print "HTTP measurements: $RUN_DIR"
"$PYTHON_BIN" -m inference_lab eval-server --model "$MODEL_ID" --base-url "$BASE_URL" --out "$RUN_DIR/development-quality.json"
# Quality results are retained even if poor; load figures do not imply a quality pass.
# Capture server hardware, revision, startup flags and cache metrics separately.
# These are burst submissions with bounded client concurrency, not a Poisson arrival process.
for round in 0 1 2; do
  for concurrency in 1 2 4 8; do
    for reuse in 0 0.5 1; do
      "$PYTHON_BIN" -m inference_lab load-test --model "$MODEL_ID" --base-url "$BASE_URL" \
        --requests 24 --concurrency "$concurrency" --reuse "$reuse" \
        --out "$RUN_DIR/c${concurrency}-reuse${reuse}-round${round}.json"
    done
  done
done
print "Completed HTTP measurements: $RUN_DIR"
