# Local inference measurements — not production

Model: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

Hardware: [{"sppci_model": "Apple M5", "sppci_cores": "10"}].

| Variant | TTFT p50 / p95 ms | ITL p50 / p95 ms | Output tok/s | Peak MLX GB | Accuracy | Gate | Est. $/1M output tokens |
|---|---:|---:|---:|---:|---:|---|---:|
| fp16 | 768.023 / 894.270 | 28.773 / 41.076 | 27.482 | 3.359 | 26.9% | FAIL | not measured |
| fp16-prefix | 92.713 / 113.531 | 28.185 / 34.903 | 33.675 | 3.186 | 26.9% | FAIL | not measured |
| int4 | 630.194 / 762.501 | 10.774 / 13.811 | 62.205 | 1.313 | 28.8% | FAIL | not measured |
| int4-prefix | 71.546 / 98.834 | 10.868 / 13.451 | 84.984 | 1.070 | 28.8% | FAIL | not measured |

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

- fp16-prefix: after − before = -627.875 ms; case-bootstrap 95% CI [-673.371, -582.282] ms.
- int4: after − before = -87.324 ms; case-bootstrap 95% CI [-152.844, -25.781] ms.
- int4-prefix: after − before = -644.611 ms; case-bootstrap 95% CI [-690.119, -599.684] ms.

## Quality diagnostics

Accuracy requires the exact expected decision and a valid schema. Valid JSON alone is insufficient.
No regressions means no previously correct cases became wrong; it does not mean the baseline is useful.

| Variant | Correct / total | Valid schema | Boundary | Missing | Invalid | Injection |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 14 / 52 | 98.1% | 58.3% | 0.0% | 0.0% | 0.0% |
| fp16-prefix | 14 / 52 | 98.1% | 58.3% | 0.0% | 0.0% | 0.0% |
| int4 | 15 / 52 | 100.0% | 62.5% | 0.0% | 0.0% | 0.0% |
| int4-prefix | 15 / 52 | 100.0% | 62.5% | 0.0% | 0.0% | 0.0% |

## Cache agreement

Matching outputs can still be incorrect. Token agreement measures cache consistency, not task accuracy.
```json
{
  "fp16": {
    "n": 52,
    "token_sequence_matches": 52,
    "valid_parsed_answer_matches": 51
  },
  "int4": {
    "n": 52,
    "token_sequence_matches": 52,
    "valid_parsed_answer_matches": 52
  }
}
```

## Quality regressions

```json
{
  "fp16": {
    "passed": false,
    "baseline_accuracy": 0.2692307692307692,
    "candidate_accuracy": 0.2692307692307692,
    "accuracy_delta": 0.0,
    "min_accuracy": 0.95,
    "max_absolute_drop": 0.02,
    "critical_cases_all_pass": false,
    "regressions": [],
    "caveat": "Small synthetic regression suite; not production quality certification."
  },
  "fp16-prefix": {
    "passed": false,
    "baseline_accuracy": 0.2692307692307692,
    "candidate_accuracy": 0.2692307692307692,
    "accuracy_delta": 0.0,
    "min_accuracy": 0.95,
    "max_absolute_drop": 0.02,
    "critical_cases_all_pass": false,
    "regressions": [],
    "caveat": "Small synthetic regression suite; not production quality certification."
  },
  "int4": {
    "passed": false,
    "baseline_accuracy": 0.2692307692307692,
    "candidate_accuracy": 0.28846153846153844,
    "accuracy_delta": 0.019230769230769218,
    "min_accuracy": 0.95,
    "max_absolute_drop": 0.02,
    "critical_cases_all_pass": false,
    "regressions": [],
    "caveat": "Small synthetic regression suite; not production quality certification."
  },
  "int4-prefix": {
    "passed": false,
    "baseline_accuracy": 0.2692307692307692,
    "candidate_accuracy": 0.28846153846153844,
    "accuracy_delta": 0.019230769230769218,
    "min_accuracy": 0.95,
    "max_absolute_drop": 0.02,
    "critical_cases_all_pass": false,
    "regressions": [],
    "caveat": "Small synthetic regression suite; not production quality certification."
  }
}
```

## Production status

No production deployment or customer traffic is claimed.
