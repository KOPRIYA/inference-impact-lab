# Validation record

## Native execution evidence reviewed on 17 November 2026

Local `results/local` contains a complete measured Apple M5 matrix: all four variants,
three rounds, 144 performance requests, and 52 quality cases per variant. This establishes
successful native model preparation/generation and report creation for the recorded
environment. Cached/uncached quality token sequences agree in all 52 cases per precision.
All task-quality gates fail. See [RESULTS.md](RESULTS.md) for numbers and limitations.

The public `results/example-fixture` is simulated pipeline data only. Local model artifacts,
raw benchmark runs, virtual environments, and personal notes are excluded by `.gitignore`.

## Automated validation

Publication cleanup: 26 offline tests passed on Python 3.9.6; compilation and
report regeneration succeeded. Compilation used a writable temporary bytecode cache
because the sandbox disallows the system Python cache location. Native inference was
not rerun during the cleanup; the existing raw measured runs were preserved.

Run `python3 -m unittest discover -s tests -v` for offline checks and
`python3 -m compileall -q inference_lab scripts tests` for syntax validation.
CI runs the offline tests on Python 3.9, 3.11, and 3.12; it does not exercise a GPU.

Coverage includes timestamp arithmetic, missing-price behavior, strict JSON, cache
invalidation/isolation, quality gates, provenance validation, report generation,
invalid-output cache agreement, EOS token accounting, and HTTP endpoint contracts.

## Not established

NVIDIA/vLLM execution, realistic serving concurrency, external held-out quality,
production traffic, actual monetary savings, and automated native GPU CI remain open.

## Version 2 implementation validation

The revised development/holdout protocol, fixed-prose/natural-JSON modes, native checks,
controlled cache-hit study, HTTP concurrency driver and evidence exporter are implemented.
The new native check was attempted with Python 3.11.18 and recorded as blocked because
MLX cannot access a Metal device inside this execution sandbox. The installed packages
are discoverable, but that does not establish GPU availability. A passing v2 baseline,
fresh timings, mixed-cache measurements and live HTTP results are not yet established.
Run `zsh scripts/run_validated.zsh` in a normal Mac Terminal. It fails closed on baseline
quality failures. The original v1 measurements have not been overwritten or relabeled.

## Local server workflow cleanup

Offline checks were rerun using `.venv/bin/python` (Python 3.11): tests passed, the
simulated report pipeline completed, and the archived v1 evidence checksums verified.
The real `serve_local.zsh` startup was attempted; MLX reported no Metal device in this
sandbox. The HTTP server and live concurrent GPU responses are therefore not validated
here. Server flags and `default_model` alias were checked against installed MLX-LM
0.32.2 source. Model weights and prior raw measurements were preserved.
