# Answering “Describe your most impactful LLM inference optimization”

## What is truthful today

“I have implemented a benchmark study for 4-bit weight quantization and shared-prefix KV
caching using Qwen2.5-1.5B-Instruct, targeting an Apple M5 GPU with 16 GB unified memory.
The benchmark and evaluation pipeline has been tested, but native model runs are pending,
so I do not yet claim measured latency, throughput or cost improvements. This is a portfolio
project and has not run in production.”

That is a project-status answer, not yet a strong measured-impact answer. Do not replace
missing metrics with targets, published third-party benchmark results or demo fixtures.

## Template after successful real runs

“My strongest measured inference optimization was for a [local prototype / actual deployment]
of an e-commerce support assistant using **[exact model + revision]** on **[GPU SKU/count/memory]**.
I identified **[prefill/decode/queueing]** as the bottleneck and applied **[winning technique]**.
At **[concurrency, input/output token lengths and cache-hit rate]**, across **[requests and rounds]**,
TTFT p50/p95 changed from **[before]** to **[after] ms**, ITL p50/p95 from **[before]** to **[after] ms**,
and output throughput from **[before]** to **[after] tokens/s**. Estimated cost changed from
**[before]** to **[after] per million output tokens**, using **[explicit price and accounting basis]**.
I verified quality using **[test-set size/composition]**: exact decision accuracy was **[before]**
versus **[after]**, with **[regressions and critical-case outcomes]**. **[Production status, exactly.]**”

If no meaningful hourly price exists for the Mac, say cost was **not measured**. You can
report machine-time reduction separately; it is not automatically a cash saving.

## How to select the result

1. Check that the baseline is useful: it must meet the quality threshold too.
2. Compare one change at a time, then the combination. Quantization may improve decode;
   prefix reuse may improve TTFT. Explain the measured bottleneck, not just the technique.
3. Choose a candidate with repeatable improvements that passes the quality gate. A lower
   memory footprint alone is still a useful result, but label it as such.
4. Read the raw regression cases and uncertainty interval. Do not claim a small difference
   as a robust gain if run variance is large.
5. State local prototype status. Having production-shaped code is not production use.

## Using your earlier CPQ work

Your reported CPQ assistant cost of roughly $0.005/query and p95 response latency around
7 seconds may support a separate application/system efficiency story if backed by logs.
They do not establish before/after model TTFT or ITL, a GPU SKU, or quantization/speculation
that you did not actually perform. Keep the new portfolio experiment separate from Oracle
production work and identify exactly what you personally built, measured and deployed.

## Making this competitive beyond a local portfolio experiment

Run the serving extension on an actual NVIDIA GPU; use a larger representative model,
load sweeps, an external held-out evaluation and a realistic prefix-hit distribution.
Report goodput at explicit latency SLOs as well as maximum throughput. For a production
claim, collect a genuine deployment period, traffic volumes, error/rollback data and actual
allocated cost. A project cannot retrospectively create production experience.
