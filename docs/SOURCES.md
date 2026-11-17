# Reading-to-implementation map

These are references and original implementation ideas, not copied book chapters.
Technical APIs were checked on 17 November 2026. Books explain principles; runtime documentation
is authoritative for version-specific APIs. Only public material was consulted; no claim is
made to having read inaccessible/paywalled chapters in full.

| Source | Relevant material | Application here |
|---|---|---|
| [Philip Kiely, Inference Engineering, chapter 5](https://www.baseten.co/inference-engineering/book/05-techniques/) | Quantization, speculation, caching, parallelism and disaggregation | Controlled experiments for weight quantization and KV reuse; avoid applying every technique at once |
| [Chip Huyen, AI Engineering, chapter 9](https://www.oreilly.com/library/view/ai-engineering/9781098166298/ch09.html) | Inference metrics; model and service optimization | Separate latency, throughput, quality and cost; keep application-level evaluation beside kernel/runtime changes |
| [Chris Fregly, AI Systems Performance Engineering](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/) | Profiling, reproducibility and efficient inference/serving | Record software/hardware, isolate runs, capture raw evidence and SLO-constrained performance |
| [MLX LM](https://github.com/ml-explore/mlx-lm) | Conversion, generation and prompt caching APIs | Actual Apple Silicon adapter |
| [MLX generation source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) | Streaming response and EOS behavior | Count text-token events, including final length token, excluding terminal EOS |
| [MLX conversion source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/convert.py) | FP16 and affine weight conversion | Convert one resolved source revision, retain hashes |
| [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Model identity and chat format | Small local baseline; use tokenizer's native chat template |
| [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) | Repeated-prefix prefill reuse and limitations | NVIDIA extension and explicit warm-hit interpretation |
| [vLLM benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/) | TTFT, ITL, TPOT and streaming measurement | Keep token intervals distinct from HTTP chunk arrival intervals |

Speculative decoding is deliberately a follow-on experiment, not an implemented claim.
Its benefit depends on draft acceptance, extra draft cost, batching and target GPU utilization.
Distillation requires an additional training/evaluation study. Neither should be added merely
to list more optimization techniques in an interview answer.
