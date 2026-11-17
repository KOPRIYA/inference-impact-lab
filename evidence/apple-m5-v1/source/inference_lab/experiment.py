import json
import random
import subprocess
import sys
import time
from pathlib import Path

from .metrics import summarize
from .quality import quality_summary, score
from .runtime import MLXRunner, environment
from .workload import cases, workload_hash
from .provenance import provenance

VARIANTS = {"fp16": ("fp16", False), "fp16-prefix": ("fp16", True),
            "int4": ("int4", False), "int4-prefix": ("int4", True)}


def run_variant(args):
    precision, use_cache = VARIANTS[args.variant]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / (args.variant + "-round-%02d.json" % args.round)
    if dest.exists():
        raise ValueError("Refusing to overwrite evidence: " + str(dest))
    frozen = None
    if args.quality_split == "heldout":
        if not args.freeze:
            raise ValueError("Heldout matrix requires a passing development --freeze")
        from .validation import check_freeze
        frozen = check_freeze(args.freeze)
    engine = MLXRunner(args.models, precision)
    if frozen and precision == "fp16" and frozen["model_identity"] != engine.model_identity:
        raise ValueError("Baseline model differs from freeze")
    all_cases = cases(args.quality_split)
    fixed = args.performance_mode == "fixed-prose"
    # Stratified subset: 3 cases from every category, fixed across all variants.
    perf_cases = [c for c in cases() if c["id"].endswith(("-00", "-04", "-10")) and not c["id"].startswith("unknown")]
    random.Random(args.seed + args.round).shuffle(perf_cases)
    prompts = [engine.tokenize(c, performance=fixed) for c in perf_cases]
    for _ in range(args.warmups):
        engine.generate(prompts[0], args.tokens, fixed_length=fixed)
    snapshot, setup_seconds = None, 0
    if use_cache:
        snapshot, setup_seconds = engine.build_prefix(prompts, "performance")
    engine.mx.reset_peak_memory()
    requests = []
    start = time.perf_counter()
    for case, prompt in zip(perf_cases, prompts):
        row = engine.generate(prompt, args.tokens, snapshot, fixed_length=fixed)
        row["id"] = case["id"]
        requests.append(row)
    wall_seconds = time.perf_counter() - start
    peak_bytes = engine.mx.get_peak_memory()
    payload = {
        "schema_version": 1, "evidence": "measured_local_inference", "production": False,
        "variant": args.variant, "round": args.round, "seed": args.seed,
        "environment": environment(), "provenance": provenance(), "source_model": engine.manifest["source_model"],
        "source_revision": engine.manifest["source_revision"],
        "model_identity": engine.model_identity, "workload_hash": workload_hash(),
        "config": {"output_tokens": args.tokens, "warmups": args.warmups, "concurrency": 1,
                   "temperature": 0, "prefill_chunk_tokens": 256, "fixed_length_performance": fixed,
                   "performance_mode": args.performance_mode, "quality_split": args.quality_split,
                   "ttft_slo_ms": args.ttft_slo_ms, "tpot_slo_ms": args.tpot_slo_ms},
        "load_seconds_excluded": engine.load_seconds,
        "model_bytes_on_disk": engine.model_bytes_on_disk,
        "peak_mlx_allocated_bytes": peak_bytes,
        "cached_prefix_tokens": len(snapshot.tokens) if snapshot else 0,
        "prefix_setup_seconds": setup_seconds,
        "summary": summarize(requests, wall_seconds, args.hourly_cost,
                             args.ttft_slo_ms, args.tpot_slo_ms, setup_seconds),
        "requests": requests, "quality": [],
    }
    # Release performance cache before quality; this is not a cache reuse across suites.
    snapshot = None
    if args.round == 0:
        qprompts = [engine.tokenize(c) for c in all_cases]
        qsnap = engine.build_prefix(qprompts, "quality")[0] if use_cache else None
        for case, prompt in zip(all_cases, qprompts):
            generated = engine.generate(prompt, 96, qsnap, namespace="quality")
            payload["quality"].append({"id": case["id"], "tag": case["tag"],
                "text": generated["text"], "token_ids": generated["token_ids"],
                "finish_reason": generated["finish_reason"], "expected": case["expected"],
                "score": score(generated["text"], case["expected"])})
    payload["quality_summary"] = quality_summary(payload["quality"])
    # Exclusive creation protects earlier runs; reports only read completed files.
    with dest.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"saved": str(dest), "summary": payload["summary"],
                      "quality": payload["quality_summary"]}), flush=True)


def matrix(args):
    out = Path(args.out)
    if out.exists() and any(out.glob("*-round-*.json")):
        raise ValueError("Choose a new --out directory for a complete, unmixed experiment")
    for round_number in range(args.rounds):
        variants = list(VARIANTS)
        random.Random(args.seed + round_number).shuffle(variants)
        for variant in variants:
            command = [sys.executable, "-m", "inference_lab", "run", "--models", args.models,
                       "--out", args.out, "--variant", variant, "--round", str(round_number),
                       "--seed", str(args.seed), "--quality-split", args.quality_split,
                       "--performance-mode", args.performance_mode, "--tokens", str(args.tokens),
                       "--warmups", str(args.warmups), "--ttft-slo-ms", str(args.ttft_slo_ms),
                       "--tpot-slo-ms", str(args.tpot_slo_ms)]
            if args.freeze:
                command += ["--freeze", args.freeze]
            if args.hourly_cost is not None:
                command += ["--hourly-cost", str(args.hourly_cost)]
            print("Running %s, round %s (separate process)" % (variant, round_number), flush=True)
            subprocess.run(command, check=True)
    from .report import write_report
    write_report(out)
