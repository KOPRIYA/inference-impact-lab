# Optional NVIDIA serving experiment — not executed

This extension is a **runbook plus command generator**, not a validated NVIDIA deployment.
It requires an existing GPU host and an installed compatible vLLM environment. It does not
rent a GPU or create cloud resources. Package installation must follow the
[official installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
for that host's driver, CUDA, Python and GPU combination.

Suggested experiment: **Qwen2.5-7B-Instruct on one NVIDIA A10G 24 GB**, starting at
4096 context and 16 maximum sequences. This is a target configuration, not an observed
machine or guaranteed memory fit. Verify actual free memory and engine allocation first.
Start with Qwen2.5-1.5B-Instruct if memory is insufficient. Never mix model sizes in one comparison.

## Capture provenance

```bash
mkdir -p results/nvidia
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > results/nvidia/hardware.csv
python -m pip freeze > results/nvidia/environment-lock.txt
vllm serve --help > results/nvidia/serve-help.txt
vllm bench serve --help > results/nvidia/bench-help.txt
```

Resolve and record the Hugging Face model commit SHA. Use the same model/tokenizer revision
on server and benchmark client. Record the exact startup command, attention backend, dtype,
engine version, device count, context cap and the hourly **total instance** price and source/date.
The generator below expects an immutable SHA, not `main`.

## Compare prefix cache off/on

```bash
python scripts/nvidia_commands.py --revision YOUR_RESOLVED_SHA --mode baseline --rate 2 --round 0
python scripts/nvidia_commands.py --revision YOUR_RESOLVED_SHA --mode prefix --rate 2 --round 0
```

These commands **print** proposed server and benchmark commands for review. Execute on the
GPU host after checking the captured CLI help. vLLM versions vary: some expose
`--no-enable-prefix-caching`, others `--disable-prefix-caching`. Select the matching flag:

```bash
python scripts/nvidia_commands.py --revision YOUR_RESOLVED_SHA --mode baseline \
  --disable-flag=--no-enable-prefix-caching
```

The workload uses eight repeated 2048-token prefixes, 128-token suffixes, fixed 128-token
outputs, 256 requests and maximum concurrency 16. Sweep offered request rates 1, 2, 4, 8;
repeat each rate at least three times in randomized baseline/prefix order. Use a new filename
for each run. A rate above saturation is still useful evidence: retain errors and queue delays.

For cold-start cache comparisons, restart the server between runs and use the same warmup
protocol for each variant. The built-in benchmark may send a preflight request; record this,
since it can warm one prefix. For explicitly warm-cache comparisons, prime all eight prefixes
and report priming separately. Do not label a mixed-hit run as 100% warm.

Add a low-prefix-reuse control using the CLI's random dataset with the same total input/output
lengths. Prefix caching should not be sold as a universal gain. Read `/metrics` for cache-hit,
queueing, running-request and memory statistics; metric names vary by version.

## Verify quality independently of synthetic load

The vLLM performance dataset contains synthetic token sequences; it is not a quality test.
Run the retail decision suite against each server configuration:

```bash
python -m inference_lab eval-server --model Qwen/Qwen2.5-7B-Instruct \
  --out results/nvidia/baseline-quality.json
```

After restarting with caching enabled, repeat with `prefix-quality.json`. Endpoint failures
count as quality failures. API keys, if needed, come from `INFERENCE_API_KEY` and are never saved.
The client sends only the bundled fictional dataset. This quality command is not a timing benchmark.

## Report interpretation

Use vLLM's exported TTFT, TPOT, ITL, throughput, success/error count and full duration.
HTTP chunks can contain more than one token, especially with speculative decoding: do not
call inter-chunk time exact model token ITL. The local MLX report and vLLM results have different
measurement boundaries and must not be combined into a single before/after result.

For hourly total instance price H, duration T seconds and successful output token count N:
`estimated $ / million output tokens = H × T / 3600 × 1,000,000 / N`.
For serving decisions, include idle provisioned time and actual traffic. Compare cost per
successful request that meets the stated SLO, not only peak token throughput.

Before a production claim: extend to a held-out real-world dataset, varied context lengths,
misses/evictions, mixed tenants, sustained/burst load, authenticated access, rollout monitoring,
fallback and a genuine production observation period. None of those are asserted as completed.
