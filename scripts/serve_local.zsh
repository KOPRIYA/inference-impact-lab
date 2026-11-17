#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
source scripts/common.zsh
require_python
require_models
PRECISION="${PRECISION:-int4}"
if [[ "$PRECISION" != fp16 && "$PRECISION" != int4 ]]; then
  print -u2 'PRECISION must be fp16 or int4.'
  exit 1
fi
# Validate prepared artifacts and native GPU access before serving.
if ! "$PYTHON_BIN" -c 'import sys; from inference_lab.runtime import MLXRunner; MLXRunner(sys.argv[1],sys.argv[2])' "$MODEL_DIR" "$PRECISION"; then
  print -u2 'Model/GPU preflight failed. For No Metal device errors, run this script in a normal macOS Terminal on Apple Silicon.'
  exit 1
fi
print 'Model listing: http://127.0.0.1:8000/v1/models'
print 'Chat endpoint: http://127.0.0.1:8000/v1/chat/completions'
print 'Client model alias: default_model. Stop the foreground server with Ctrl+C.'
# Bind only to loopback. No remote-code trust or KV quantization is enabled.
exec "$PYTHON_BIN" -m mlx_lm server \
  --model "$MODEL_DIR/$PRECISION" --host 127.0.0.1 --port 8000 \
  --decode-concurrency 8 --prompt-concurrency 4 --prefill-step-size 256 \
  --prompt-cache-size 8 --prompt-cache-bytes 512MB --max-tokens 128 \
  --allowed-origins http://127.0.0.1:8000
