# Inference Impact Lab

Two controlled LLM inference optimization projects for an e-commerce support workload:
**4-bit weight quantization** and **exact-prefix KV cache reuse**.

The implementation targets **Qwen2.5-1.5B-Instruct on Apple Silicon / MLX** first.
It includes a separate NVIDIA/vLLM runbook for serving experiments at concurrency.
The fictional retailer and policies are original synthetic data, not Flipkart policies.

## Current evidence status

- Core metrics, quality evaluation, cache isolation, CLI and fixture reports have been tested locally.
- The development machine reports an Apple M5, 10 GPU cores, 16 GB unified memory.
- **Actual MLX inference has not run in this session.** Package-registry access was unavailable,
  MLX/model weights were not installed, and native integration remains unverified.
- Demo numbers are explicitly simulated, identical across variants, and unusable as performance claims.
- No production deployment, cloud rental, measured speedup, or dollar savings is claimed.
- Nothing has been pushed to GitHub. Owner confirmation is required before any push.

## Why these two projects

| Project | Intervention | Bottleneck hypothesis | Comparison |
|---|---|---|---|
| Quantized support inference | Convert the same pinned model revision from FP16 to MLX affine 4-bit weights, group size 64 | Reduce weight storage and memory traffic during decode | `fp16` → `int4` |
| Prefix-aware policy assistant | Prefill the shared policy once; clone its KV state for each request and process only the suffix | Avoid repeated prefill of the shared context | `fp16` → `fp16-prefix` |
| Combined experiment | Apply both changes | Measure whether benefits combine without unacceptable quality loss | `fp16` → `int4-prefix` |

The project orchestrates existing MLX kernels; it does not claim a novel quantization algorithm.
Caching reuses attention state, not completed answers. Every request still runs the model.
Reduced precision is not guaranteed to be faster; kernel overhead and workload shape matter.
Prefix caching is expected to target TTFT, not eliminate autoregressive decode cost.

## Run the offline validation now

From this directory, Python 3.9+ is sufficient; no downloads needed:

```bash
python3 -m unittest discover -s tests -v
python3 -m inference_lab doctor
python3 -m inference_lab demo --out results/demo
open results/demo/report.html
```

Use a new output directory for subsequent demo runs. Existing evidence is never silently replaced.

## Run actual inference on your Mac

Use Python **3.11 or 3.12** with native Apple Silicon architecture. The system Python 3.9
can run the offline tests but is not the recommended MLX environment. If needed, install
Python from [python.org](https://www.python.org/downloads/macos/) first.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mac]'
python -m inference_lab doctor
python -m inference_lab prepare --models models/qwen
python -m pip freeze > models/qwen/environment-lock.txt
python -m inference_lab matrix --models models/qwen --out results/local --rounds 3
open results/local/report.html
```

The download plus original/converted models needs several GB of disk. Conversion and
benchmarking use the GPU and unified memory; close other heavy applications. Start with
the 1.5B model on a 16 GB machine. Do not assume a larger model will fit without swapping.
Do not upload model weights or the Hugging Face cache to GitHub.

The MLX package version is pinned to 0.32.2. Native APIs were checked against upstream
documentation/source, but need execution validation on the installed runtime. The manifest
records the resolved model SHA and file hashes; the environment lock records all dependencies.
If an MLX API incompatibility appears, retain the error and dependency lock rather than
silently switching versions and combining results.

A smaller real smoke run before the full matrix:

```bash
python -m inference_lab matrix --out results/local-smoke --rounds 1 --tokens 16 --warmups 1
```

Use at least three rounds at 128 output tokens for the main comparison. The smoke run
still checks quality; it is not sufficient evidence for a performance claim.

## What gets measured

- 12 performance prompts × 3 rounds × 4 variants = 144 requests by default.
- Separate process per variant/round; deterministic randomized variant and case order.
- Two warmup requests per process; model load and tokenization outside timing.
- Fixed 128 generated non-EOS tokens for performance, so shorter answers cannot masquerade as speedups.
- TTFT starts before cache checkout/copy; ends at first model token available to the Python caller.
- ITL is the interval between consecutive token events; EOS is excluded.
- Throughput is total output tokens / total measured wall time, not mean per-request tokens/s.
- p50/p95 TTFT, pooled p50/p95 ITL, per-request TPOT, end-to-end time and SLO request rate.
- MLX peak allocated memory and model disk footprint, clearly distinguished.
- Warm-prefix performance plus an alternate denominator including one prefix fill per round.
- Original token IDs, output text and timestamps retained for audit.
- Paired case-bootstrap confidence intervals for TTFT changes; repeated rounds are averaged within case.

This is an **in-process, concurrency-1 microbenchmark**, not HTTP latency or maximum serving
capacity. TTFT excludes tokenization and network latency. Warm-cache numbers assume a 100%
shared-prefix hit workload. A production system must also measure misses, eviction, multiple
tenants, long tails, burst load, queueing and memory pressure. See [NVIDIA_RUNBOOK.md](docs/NVIDIA_RUNBOOK.md).

## Quality verification

52 deterministic synthetic cases cover eligible/ineligible decisions, inclusive boundary days,
invalid field types, missing fields, unknown categories and prompt injection. Each has an
independent rule-based expected decision. The expected answer is never placed in the model input.
This set is not used for model training, calibration or fine-tuning. It is a small regression
suite, not a representative external test set or a statistical non-inferiority study.

Quality decoding uses temperature 0, natural EOS and a 96-token cap. Extra prose, fenced JSON,
duplicate keys, extra keys or incorrect decisions fail the strict validator.

Candidate gates require baseline and candidate accuracy >=95%, at most 2 percentage points
absolute accuracy loss, 100% valid schema and no failures on missing/invalid/injection cases.
At 52 cases, one error is about 1.92 percentage points: inspect every regression.
Cache token/decision agreement is also recorded. Deterministic settings do not guarantee
bitwise equivalence across different floating-point execution paths.

If the 1.5B baseline fails, report the failure. Improve the common prompt or choose a stronger
model, regenerate the workload hash, and rerun **all** variants; do not loosen the gate after
seeing results. Add a new held-out test set after any prompt tuning.

## Cost accounting

No monetary price is assumed for an owned Mac. Optionally provide a documented USD/hour
allocation via `--hourly-cost`. The estimate is:

`cost / 1M output tokens = hourly_machine_cost × measured_wall_seconds / 3600 × 1,000,000 / output_tokens`

Reports also include setup-inclusive costs. This is an allocated **machine-time estimate**,
not an electricity bill or realized savings. Model loading, idle uptime, storage and network
are excluded. Include those in a deployment cost analysis. Lower per-query cost only implies
cash savings if resources are reduced, autoscaled or used for additional valuable traffic.

## Outputs and review

- `*-round-*.json`: raw measured requests, quality outputs and provenance.
- `comparison.json`: aggregated metrics, quality gates, cache equivalence and paired TTFT intervals.
- `report.md` / `report.html`: offline reviewable evidence.
- `interview_draft.txt`: an evidence-status-aware starting point; no fabricated accomplishments.

See [IMPLEMENTATION_REVIEW.md](docs/IMPLEMENTATION_REVIEW.md) for the code tour,
[INTERVIEW_GUIDE.md](docs/INTERVIEW_GUIDE.md) for the application answer,
and [SOURCES.md](docs/SOURCES.md) for the book-to-code mapping.

No script commits, pushes, publishes, rents a GPU or creates cloud resources.
