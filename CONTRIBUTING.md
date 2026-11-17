# Contributing

Inference Impact Lab measures inference tradeoffs on a synthetic retail workload.
Bug reports should include the command, Python/runtime versions, hardware, and traceback.
Remove credentials, private paths, and customer data before posting logs.

## Local checks

The offline tests need Python 3.9+ and no model downloads:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q inference_lab scripts tests
python3 -m inference_lab demo --out results/contributor-demo
```

Use a new output directory each time. Native MLX runs need Apple Silicon and the
`mac` optional dependencies; follow the README. CI tests the offline pipeline only.

## Changes and evidence

- Keep behavioral fixes covered by focused regression tests.
- Preserve original raw measurements. Never replace old results with a new workload
  under the same run directory or present simulated fixtures as measurements.
- State model revision, environment, input/output lengths, concurrency, cache state,
  quality outcomes, and cost assumptions when reporting performance.
- Prompt tuning requires rerunning all variants and adding an independent held-out
  evaluation. Do not lower the quality gate after observing a failed result.
- Keep model weights, environments, credentials, and personal notes out of commits.
- Raw local runs are ignored by default. Publish a reviewed, sanitized evidence bundle
  separately when independent reproduction of a measured claim is required.

Source code is licensed under the [MIT License](LICENSE). Dependencies and model
weights have separate terms.
