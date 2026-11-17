# Version 2 evaluation protocol

## Baseline selection and holdout

The original 52 synthetic cases are development data. The revised prompt validates inputs
before applying deadlines, includes a few illustrative examples distinct from the test
orders, and removes the irrelevant catalogue from quality prompts. No rule-based answer
is passed into model input and no fallback repairs model answers. This is an unvalidated
candidate prompt: a quality pass must come from actual inference, not the Python oracle.

The new 64-case holdout contains different interior/late values, distracting fields,
float/null input errors, missing data and new injection wording. Full input objects do
not overlap development cases. It is a synthetic holdout of related policy rules, not
an independent real-world dataset. Its generator and criteria are fixed before execution.
The oracle is shared, so external/manual validation remains valuable.

`assess --split development` writes immutable evidence with the source/workload/model
fingerprints and the unchanged quality gate. Only a passing file can serve as `--freeze`
for heldout assessment or a heldout matrix. Source/workload/baseline-model changes invalidate
that freeze. The raw heldout answers are never used to silently revise the prompt.
If the heldout fails, report it; tuning against it consumes the split and requires a new
independently authored holdout. The software cannot police human inspection of public data.

Gate: baseline and candidate accuracy >=95%, <=2 percentage-point absolute regression,
100% candidate schema validity, and no invalid/missing/injection failures. Natural EOS with
96-token cap is used for quality. Cached and uncached candidates are compared to FP16.

## Performance comparisons

`--performance-mode fixed-prose` uses only prose-format instructions, the shared reference
catalogue and exactly 128 generated tokens. `--performance-mode natural-json` uses the
same prompt format as quality and natural stopping (maximum 128 tokens for performance).
The performance sample is drawn from development cases, never selected for heldout results.
Compare natural request latency alongside output lengths: shorter responses can affect throughput.
Do not compare v1 and v2 timings as if only an inference optimization changed.

Each matrix runs four variants in separate processes, three randomized rounds, and two
warmups per process. Reports reject mixed runner fingerprints and incomplete quality splits.
Old files without a runner fingerprint remain readable and are labeled legacy in exports.

## Native and serving experiments

`native-check` exercises FP16 and INT4 loading, fixed-length token accounting, natural EOS,
cache repeatability after an intervening request, cached/uncached token equality, monotonic
timestamps and memory counters. Exact token equality is a deliberately strict diagnostic;
a difference requires investigation, not automatic proof of a cache bug. Lack of GPU access
produces a blocked record and stops the workflow.

`cache-study` executes three shuffled rounds per precision for 0%, 50%, and 100% actual
snapshot use. Misses deliberately bypass the snapshot. It records per-request decisions,
setup-inclusive throughput and peak allocation. This is a controlled serial sensitivity
study, not an eviction algorithm, a production hit-rate measurement, or server concurrency.

`load-test` submits bursts through a bounded HTTP thread pool. It records client queue,
first content delay, chunk gaps, errors, wall throughput and server-reported completion
tokens when supplied. Chunk gaps are not token ITL. `--reuse` controls offered shared-prefix
traffic; first-use misses, eviction, server settings and concurrent prefill races determine
actual hits. Every run has a unique namespace to avoid warm cache carryover. Record server
hardware/model revision/runtime/startup flags and cache metrics separately. No server is
installed or provisioned automatically. The sweep uses 24 requests, three repetitions,
concurrency 1/2/4/8 and offered reuse 0/50/100%; it is a small burst test, not a capacity SLA.

## Evidence and licensing

`bundle RUN_DIRECTORY --out NEW_DIRECTORY` exports allowlisted raw runs and reports,
source files, direct runtime version pins, metadata and SHA-256 checksums. The optional
`--validation-dir` includes the native, development-freeze, heldout and cache-study records
and requires their runner fingerprints to match. The end-to-end script supplies it. It rejects
obvious private paths and credential patterns in evidence. No weights, tokens or arbitrary
local files are included. `verify-bundle` detects altered, missing or added files; checksums
provide integrity relative to the manifest, not authorship or a cryptographic signature.
Direct version pins are not a complete transitive dependency lock.

The original v1 bundle has a known gap: no exact runner fingerprint was captured when it
ran. The included source is a current review snapshot, explicitly not asserted to be the
original runner. New-run bundles assert source correspondence only when fingerprints match.

Project code uses MIT. Downloaded model weights and third-party packages keep their own licenses.
