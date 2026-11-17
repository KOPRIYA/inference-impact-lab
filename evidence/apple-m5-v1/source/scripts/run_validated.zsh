#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-models/qwen}"
RUN_DIR="results/validated-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m inference_lab native-check --models "$MODEL_DIR" --out "$RUN_DIR/native.json"
# This passing development assessment freezes the source/workload/model identity.
"$PYTHON_BIN" -m inference_lab assess --models "$MODEL_DIR" --split development --out "$RUN_DIR/freeze.json"
# Stop on a failed heldout baseline, never tune the prompt on this result.
"$PYTHON_BIN" -m inference_lab assess --models "$MODEL_DIR" --split heldout --freeze "$RUN_DIR/freeze.json" --out "$RUN_DIR/heldout.json"
for mode in fixed-prose natural-json; do
  "$PYTHON_BIN" -m inference_lab matrix --models "$MODEL_DIR" --out "$RUN_DIR/$mode" --rounds 3 --tokens 128 --quality-split heldout --freeze "$RUN_DIR/freeze.json" --performance-mode "$mode"
done
"$PYTHON_BIN" -m inference_lab cache-study --models "$MODEL_DIR" --out "$RUN_DIR/cache-study.json"
for mode in fixed-prose natural-json; do
  "$PYTHON_BIN" -m inference_lab bundle "$RUN_DIR/$mode" --out "$RUN_DIR/$mode-evidence" --validation-dir "$RUN_DIR"
done
open "$RUN_DIR/natural-json/report.html"
print "Completed: $RUN_DIR"
