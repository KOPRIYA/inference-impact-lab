# Apple M5 local benchmark

## Evidence and configuration

Reviewed on 17 November 2026 from the local `results/local` run: 12 raw run files,
three rounds, four variants, 12 performance requests per variant per round (144 total).
The reviewed raw run is exported in `evidence/apple-m5-v1/`, with checksums and
a source review snapshot. The old run did not capture runner-code provenance; the
export marks that gap. The committed example fixture remains simulated. These are
v1 workload results and must not be relabeled as results of the revised v2 protocol.

- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Apple M5, 10 GPU cores, 16 GB unified memory.
- Python 3.11.18; MLX 0.32.4; MLX-LM 0.32.2;
  Transformers 5.23.0; Hugging Face Hub 1.33.0.
- Concurrency 1; temperature 0; two warmups per process; 128 fixed output tokens.
- Quality: 52 cases per variant in round zero, natural EOS, 96-token cap.

## Measurements

| Variant | TTFT p50 / p95 ms | ITL p50 / p95 ms | Output tokens/s | Peak MLX GB | Correct decisions |
|---|---:|---:|---:|---:|---:|
| FP16 | 768.023 / 894.270 | 28.773 / 41.076 | 27.482 | 3.359 | 14/52 |
| FP16 + prefix | 92.713 / 113.531 | 28.185 / 34.903 | 33.675 | 3.186 | 14/52 |
| INT4 | 630.194 / 762.501 | 10.774 / 13.811 | 62.205 | 1.313 | 15/52 |
| INT4 + prefix | 71.546 / 98.834 | 10.868 / 13.451 | 84.984 | 1.070 | 15/52 |

Compared with FP16, the combined configuration has approximately 90.7% lower median
TTFT, 62.2% lower median ITL, 3.09x output throughput, and 68.2% lower peak MLX allocation.
Quantization alone gives 2.26x throughput. Prefix reuse alone reduces median TTFT by
87.9% while median ITL changes only slightly, consistent with avoiding repeated prefill.
These observations do not independently prove the hardware-level bottleneck.

Including one prefix fill per round changes cached throughput to 33.30 tokens/s for
FP16 and 82.30 tokens/s for INT4. Model files occupy about 3.099 GB versus 0.880 GB;
file size is distinct from allocator memory and total system RAM.

## Quality is the blocking result

**Every variant fails the predeclared quality gate, including the baseline.**
This is a useful negative result, not a quality-preserving deployment optimization.

| Case group | FP16 correct | INT4 correct |
|---|---:|---:|
| Boundary decisions | 14/24 | 15/24 |
| Missing fields | 0/8 | 0/8 |
| Invalid inputs | 0/12 | 0/12 |
| Injection cases | 0/8 | 0/8 |

Caching gives the same quality results as the corresponding uncached model.
All 52 quality token sequences match within each precision pair. Matching answers
can still be wrong. Valid schema rates are 51/52 for FP16 and 52/52 for INT4.

Examples from raw outputs:

- `fashion-01`: an item with `unused=false` is incorrectly approved.
- `fashion-06`: negative delivery days produce `deny_return` / `MANUAL` instead of
  the required `escalate` / `MANUAL`.
- Missing fields are incorrectly treated as sufficient grounds to approve.
- One FP16 answer contains a fenced JSON block, violating the strict output schema.

Most errors are decision failures, not JSON formatting failures. All responses ended
normally; these failures are not explained by hitting the 96-token output cap.
An empty regression list means no baseline-correct cases became incorrect. It does
not mean the baseline was accurate. The INT4 gain is only one additional correct case.

## Interpretation limits and next experiment

Timing excludes model loading, tokenization, and network latency. Cached rows are warm
shared-prefix hits, not a realistic mixed-hit service. ITL pools token gaps; throughput
includes prefill and measured request overhead, so it is not simply the inverse of ITL.
Paired confidence intervals concern mean case-level changes, not differences of medians.

Performance asks for prose, while quality asks for JSON. The current performance prompt
also retains the earlier JSON instruction before appending a prose override. A future
workload should remove that conflict and measure natural JSON response latency as a
separate task. This would change the workload: preserve these results and rerun all
variants in new directories. Select/tune the baseline on development cases, then evaluate
on a new held-out set. Do not weaken validation or retrospectively rescore these results.

No hourly cost was supplied, so dollar cost is unmeasured, not zero. No production use,
external quality evaluation, or NVIDIA serving result is established.
