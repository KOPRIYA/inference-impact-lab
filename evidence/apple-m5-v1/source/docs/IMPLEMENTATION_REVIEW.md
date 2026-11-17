# Implementation and measurement design

## Project 1 — Quantized retail-support inference

The `prepare` command resolves Qwen2.5-1.5B-Instruct to a Hugging Face commit SHA,
creates FP16 weights, then derives 4-bit weights from those exact FP16 weights.
It writes hashes for every model artifact. The runner verifies them before measurement.

This reduces precision for supported weight layers using MLX affine quantization, group
size 64. It is weight quantization, not INT4 KV cache or activation quantization. Some
parameters remain higher precision; do not claim the entire model is exactly four times smaller.
Model storage and peak allocator memory are measured separately.

The main comparison is `fp16` versus `int4` with identical prompts, generation limits,
temperature and runtime parameters. The output preserves negative results. The technique
may hurt decision quality or fail to improve speed on a particular device.

## Project 2 — Prefix-aware policy assistant

All requests share a policy and reference catalogue. The code finds the exact longest
common token prefix, prefills it in 256-token chunks, and stores a snapshot of the resulting
per-layer attention KV state. Each request receives a deep copy and runs only its remaining
suffix plus normal generation. Its answer is never cached.

The snapshot binds model-artifact identity, policy/workload hash and namespace. A mismatched
identity or token prefix is rejected. Generated tokens cannot mutate another request's
snapshot. This is an isolated single-process benchmark cache, not a production multi-tenant
cache service, eviction policy or authorization system.

The main comparison is `fp16` versus `fp16-prefix`. Setup time is reported separately and
included in an amortized throughput/cost denominator. `int4-prefix` measures the combination.
Warm-only TTFT should not be presented as the latency of a cold request.

## Code tour

| File | Responsibility |
|---|---|
| `inference_lab/runtime.py` | Hardware provenance, model conversion, tokenization, MLX generation and prefix prefill |
| `inference_lab/prefix.py` | Identity checks, token matching and request isolation |
| `inference_lab/experiment.py` | Four-variant experiment matrix, warmup, randomized order, process isolation |
| `inference_lab/metrics.py` | TTFT/ITL/TPOT, percentiles, throughput, SLO rate, cost assumptions, paired bootstrap |
| `inference_lab/workload.py` | Original fictional support policies, prompts, expected-decision oracle |
| `inference_lab/quality.py` | Strict JSON scoring, per-category quality, regression gate |
| `inference_lab/report.py` | Comparable-run validation, evidence tables, offline HTML and evidence summary |
| `inference_lab/server_eval.py` | Same quality checks against a supplied OpenAI-compatible endpoint |
| `tests/` | Metric arithmetic, malformed outputs, cache isolation, provenance, endpoint contracts and stream accounting |

## Validation boundaries

A native Apple M5 run completed; see [VALIDATION.md](VALIDATION.md) and
[RESULTS.md](RESULTS.md). All local quality gates failed. Native execution does not
establish task suitability, broad model quality, or production readiness.

The cache is a single-process experiment without eviction or authorization. Performance
uses fixed-length prose while quality uses short JSON; results are not an end-to-end
benchmark of a production JSON service. NVIDIA serving remains unvalidated.
