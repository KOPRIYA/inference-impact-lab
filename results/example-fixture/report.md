# SIMULATED TEST FIXTURE — NOT MODEL PERFORMANCE

Model: `NO_MODEL_EXECUTED` at `fixture`.

Hardware: ["NONE \u2014 simulated fixture"].

| Variant | TTFT p50 / p95 ms | ITL p50 / p95 ms | Output tok/s | Peak MLX GB | Accuracy | Gate | Est. $/1M output tokens |
|---|---:|---:|---:|---:|---:|---|---:|
| fp16 | 100.000 / 100.000 | 20.000 / 20.000 | 20.000 | 0.000 | 100.0% | PASS | not measured |
| fp16-prefix | 100.000 / 100.000 | 20.000 / 20.000 | 20.000 | 0.000 | 100.0% | PASS | not measured |
| int4 | 100.000 / 100.000 | 20.000 / 20.000 | 20.000 | 0.000 | 100.0% | PASS | not measured |
| int4-prefix | 100.000 / 100.000 | 20.000 / 20.000 | 20.000 | 0.000 | 100.0% | PASS | not measured |

## Interpretation

Compare fp16→int4 for quantization; fp16→fp16-prefix for caching; fp16→int4-prefix for the combination.
No faster variant is recommended automatically. A failed quality gate disqualifies it.
TTFT and ITL are in-process token availability, excluding tokenization, model load and network time.
Throughput uses total measured request wall time; it is serial throughput at concurrency 1, not server capacity.
Prefix rows measure warm hits. comparison.json also includes throughput/cost including one prefix fill per round.
ITL percentiles pool token gaps. TPOT is per-request mean inter-token gap and is reported separately in JSON.
Peak memory is MLX allocator memory, not total system RAM or dedicated VRAM. Disk size is not GPU memory.
Cost is an explicit hourly allocation model, not a bill. No hourly price supplied means no cost estimate.
The performance suite forces a fixed output budget. The 52-case quality suite uses natural EOS stopping.
Confidence intervals resample case means, not individual tokens. Small synthetic samples do not establish production impact.

## Paired TTFT changes

- fp16-prefix: after − before = 0.000 ms; case-bootstrap 95% CI [0.000, 0.000] ms.
- int4: after − before = 0.000 ms; case-bootstrap 95% CI [0.000, 0.000] ms.
- int4-prefix: after − before = 0.000 ms; case-bootstrap 95% CI [0.000, 0.000] ms.

## Quality regressions

```json
{
  "fp16": {
    "passed": true,
    "baseline_accuracy": 1.0,
    "candidate_accuracy": 1.0,
    "accuracy_delta": 0.0,
    "min_accuracy": 0.95,
    "max_absolute_drop": 0.02,
    "critical_cases_all_pass": true,
    "regressions": [],
    "caveat": "Small synthetic regression suite; not production quality certification."
  },
  "fp16-prefix": {
    "passed": true,
    "baseline_accuracy": 1.0,
    "candidate_accuracy": 1.0,
    "accuracy_delta": 0.0,
    "min_accuracy": 0.95,
    "max_absolute_drop": 0.02,
    "critical_cases_all_pass": true,
    "regressions": [],
    "caveat": "Small synthetic regression suite; not production quality certification."
  },
  "int4": {
    "passed": true,
    "baseline_accuracy": 1.0,
    "candidate_accuracy": 1.0,
    "accuracy_delta": 0.0,
    "min_accuracy": 0.95,
    "max_absolute_drop": 0.02,
    "critical_cases_all_pass": true,
    "regressions": [],
    "caveat": "Small synthetic regression suite; not production quality certification."
  },
  "int4-prefix": {
    "passed": true,
    "baseline_accuracy": 1.0,
    "candidate_accuracy": 1.0,
    "accuracy_delta": 0.0,
    "min_accuracy": 0.95,
    "max_absolute_drop": 0.02,
    "critical_cases_all_pass": true,
    "regressions": [],
    "caveat": "Small synthetic regression suite; not production quality certification."
  }
}
```

## Production status

No production deployment or customer traffic is claimed.
