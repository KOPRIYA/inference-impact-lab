#!/bin/zsh
# Run against an already running OpenAI-compatible streaming inference server.
set -euo pipefail
cd "${0:A:h:h}"
: "${MODEL_ID:?Set MODEL_ID to the served model identifier}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
RUN_DIR="results/http-load-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
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
