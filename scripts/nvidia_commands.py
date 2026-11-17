"""Print a vLLM experiment plan; never provision hardware or execute commands."""
import argparse
import shlex


def build_commands(model, revision, mode, disable_flag, rate, round_number):
    if mode not in ("baseline", "prefix"):
        raise ValueError("unknown variant")
    if not revision or revision == "main":
        raise ValueError("supply the resolved Hugging Face commit SHA")
    server = ["vllm", "serve", model, "--revision", revision, "--tokenizer-revision", revision,
              "--host", "127.0.0.1", "--port", "8000", "--dtype", "half",
              "--max-model-len", "4096", "--max-num-seqs", "16",
              "--gpu-memory-utilization", "0.9",
              "--enable-prefix-caching" if mode == "prefix" else disable_flag]
    bench = ["vllm", "bench", "serve", "--backend", "vllm", "--model", model,
             "--tokenizer", model, "--revision", revision,
             "--base-url", "http://127.0.0.1:8000", "--endpoint", "/v1/completions",
             "--dataset-name", "prefix_repetition", "--prefix-repetition-prefix-len", "2048",
             "--prefix-repetition-suffix-len", "128", "--prefix-repetition-output-len", "128",
             "--prefix-repetition-num-prefixes", "8", "--num-prompts", "256",
             "--request-rate", str(rate), "--max-concurrency", "16", "--ignore-eos",
             "--seed", "17", "--temperature", "0", "--save-result", "--save-detailed",
             "--result-dir", "results/nvidia",
             "--result-filename", "%s-rate-%s-round-%s.json" % (mode, rate, round_number)]
    return server, bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mode", choices=["baseline", "prefix"], required=True)
    parser.add_argument("--disable-flag", choices=["--disable-prefix-caching", "--no-enable-prefix-caching"],
                        default="--disable-prefix-caching")
    parser.add_argument("--rate", type=float, default=2)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    if args.rate <= 0 or args.round < 0:
        parser.error("rate must be positive and round nonnegative")
    server, bench = build_commands(args.model, args.revision, args.mode, args.disable_flag, args.rate, args.round)
    print("# Proposed commands only. Native NVIDIA execution is unvalidated.")
    print("# Terminal 1: start a fresh server for each independently cold benchmark.")
    print(shlex.join(server))
    print("# Terminal 2: after readiness, run the benchmark. Check installed CLI flags first.")
    print(shlex.join(bench))


if __name__ == "__main__":
    main()
