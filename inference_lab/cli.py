import argparse
import json
import sys


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inference Impact Lab: quantization and prefix-cache experiments")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="Print dependency and hardware readiness, without serial numbers")
    demo = subs.add_parser("demo", help="Offline fixture pipeline; NOT real inference")
    demo.add_argument("--out", default="results/demo")
    prepare = subs.add_parser("prepare", help="Download one revision and derive FP16 + INT4 models")
    prepare.add_argument("--models", default="models/qwen")
    prepare.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    prepare.add_argument("--revision", default="main", help="Resolved to a commit SHA in manifest.json")
    from .experiment import VARIANTS
    for name in ("run", "matrix"):
        sub = subs.add_parser(name)
        sub.add_argument("--models", default="models/qwen")
        sub.add_argument("--out", default="results/local")
        sub.add_argument("--tokens", type=positive, default=128)
        sub.add_argument("--warmups", type=positive, default=2)
        sub.add_argument("--seed", type=int, default=17)
        sub.add_argument("--quality-split", choices=["development", "heldout"], default="development")
        sub.add_argument("--freeze")
        sub.add_argument("--performance-mode", choices=["fixed-prose", "natural-json"], default="fixed-prose")
        sub.add_argument("--hourly-cost", type=float, default=None, help="Explicit allocated machine cost in USD/hour; no default")
        sub.add_argument("--ttft-slo-ms", type=float, default=2000)
        sub.add_argument("--tpot-slo-ms", type=float, default=100)
        if name == "run":
            sub.add_argument("--variant", choices=VARIANTS, required=True)
            sub.add_argument("--round", type=int, default=0)
        else:
            sub.add_argument("--rounds", type=positive, default=3)
    report = subs.add_parser("report")
    report.add_argument("directory")
    server = subs.add_parser("eval-server", help="Quality-only checks against your OpenAI-compatible server")
    server.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    server.add_argument("--model", required=True)
    server.add_argument("--out", required=True)
    for name in ("assess", "native-check", "cache-study"):
        sub = subs.add_parser(name)
        sub.add_argument("--models", default="models/qwen")
        sub.add_argument("--out", required=True)
        if name == "assess":
            sub.add_argument("--split", choices=["development", "heldout"], default="development")
            sub.add_argument("--freeze")
    bundle = subs.add_parser("bundle")
    bundle.add_argument("directory")
    bundle.add_argument("--out", required=True)
    bundle.add_argument("--validation-dir", help="Include native/freeze/heldout/cache-study evidence from this directory")
    verify = subs.add_parser("verify-bundle")
    verify.add_argument("directory")
    load = subs.add_parser("load-test")
    load.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    load.add_argument("--model", required=True)
    load.add_argument("--out", required=True)
    load.add_argument("--requests", type=positive, default=24)
    load.add_argument("--concurrency", type=positive, default=4)
    load.add_argument("--reuse", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        if args.command in ("assess", "native-check"):
            from .validation import assess, native_check
            (assess if args.command == "assess" else native_check)(args)
        elif args.command == "cache-study":
            from .cache_study import run_study
            run_study(args)
        elif args.command in ("bundle", "verify-bundle"):
            from .evidence import bundle, verify
            (bundle(args.directory, args.out, args.validation_dir) if args.command == "bundle" else verify(args.directory))
        elif args.command == "load-test":
            from .load_test import run_load
            run_load(args)
        elif args.command == "doctor":
            from .runtime import environment
            data = environment()
            data["mlx_ready"] = all(data["packages"].get(p) for p in ("mlx", "mlx-lm"))
            print(json.dumps(data, indent=2))
        elif args.command == "prepare":
            from .runtime import prepare_models
            prepare_models(args.models, args.model, args.revision)
        elif args.command == "demo":
            from .demo import demo
            demo(args.out)
        elif args.command == "report":
            from .report import write_report
            write_report(args.directory)
        elif args.command == "eval-server":
            from .server_eval import evaluate_server
            evaluate_server(args)
        else:
            import math
            if args.hourly_cost is not None and (not math.isfinite(args.hourly_cost) or args.hourly_cost < 0):
                parser.error("--hourly-cost must be finite and nonnegative")
            if any(not math.isfinite(n) or n <= 0 for n in (args.ttft_slo_ms, args.tpot_slo_ms)):
                parser.error("SLOs must be finite and positive")
            if args.command == "run":
                if args.round < 0:
                    parser.error("--round cannot be negative")
                from .experiment import run_variant
                run_variant(args)
            else:
                from .experiment import matrix
                matrix(args)
    except (ImportError, FileNotFoundError) as exc:
        print("Cannot run: %s. See README.md for installation and model preparation." % exc, file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print("Cannot run: " + str(exc), file=sys.stderr)
        return 2
    return 0
