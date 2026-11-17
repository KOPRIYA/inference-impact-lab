# Validation record — 17 November 2026

## Executed successfully

- `python3 -m unittest discover -s tests -v`: **25 tests passed** on Python 3.9.6.
- `python3 -m inference_lab doctor`: identified Apple M5, 10 GPU cores, 16 GB unified memory;
  confirmed that MLX, MLX LM, Transformers and Hugging Face Hub were not installed.
- `python3 -m inference_lab demo --out results/example-fixture`: generated all four fixture
  variants across three rounds, the comparison JSON, Markdown/HTML report and guarded interview draft.
- `python3 -m compileall -q inference_lab scripts tests`: completed successfully.
- NVIDIA command generator: emitted the proposed server/load-test commands without executing them.

The session needed a workspace `PYTHONPYCACHEPREFIX` because the default Python bytecode cache
location was not writable. This is an execution-environment constraint, not a project requirement.

## Test coverage

Metric arithmetic and timestamp validity; empty and one-token output; failed-request denominators;
cost validation and missing-price behavior; percentile and paired-bootstrap arithmetic; exact-prefix
matching; cache invalidation by model/policy/namespace; request-state isolation; boundary decisions;
strict JSON and duplicate-key rejection; absence of expected labels in prompts; quality-regression
gates; evidence provenance checks; overwrite protection; terminal EOS versus final length token;
HTTP request contract and endpoint timeout scoring.

## Not executed / not established

Native MLX installation, weight download/conversion, actual generation, native KV snapshot copying,
performance or model quality on the M5, NVIDIA/vLLM installation/serving, cloud cost, external quality
benchmarks, production traffic, or production savings.

An attempted MLX installation failed to reach the package registry (name-resolution/connection
failure in the restricted session). No measurements were substituted. The fixture numbers are
equal across variants and explicitly marked simulated. They verify plumbing only.

## GitHub status

No remote repository has been created, no commit pushed, and no pull request published.
The owner requested an explanation and confirmation before a push. The implementation review
is in `IMPLEMENTATION_REVIEW.md`; the first actual model run remains the next validation step.
