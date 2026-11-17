"""Deterministic test fixtures, never an inference benchmark."""
import json
from pathlib import Path

from .experiment import VARIANTS
from .metrics import request_metrics, summarize
from .quality import quality_summary, score
from .workload import cases, workload_hash


def demo(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Choose a new empty demo directory")
    directory.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        for round_number in range(3):
            requests = []
            for case in cases()[:12]:
                # Identical fixture values across all variants: deliberately no fabricated speedup.
                requests.append(dict(request_metrics(0, [.1, .12, .14], .15), id=case["id"], text="fixture"))
            quality = []
            if round_number == 0:
                for case in cases():
                    text = json.dumps(case["expected"])
                    quality.append(dict(id=case["id"], tag=case["tag"], text=text, token_ids=[],
                                        score=score(text, case["expected"])))
            payload = {"schema_version": 1, "evidence": "SIMULATED_TEST_FIXTURE", "production": False,
                "variant": variant, "round": round_number, "seed": 17,
                "environment": {"gpus": ["NONE — simulated fixture"]},
                "source_model": "NO_MODEL_EXECUTED", "source_revision": "fixture", "model_identity": "fixture",
                "workload_hash": workload_hash(), "config": {"output_tokens": 3, "ttft_slo_ms": 2000, "tpot_slo_ms": 100},
                "model_bytes_on_disk": 0, "peak_mlx_allocated_bytes": 0, "prefix_setup_seconds": 0,
                "summary": summarize(requests, 1.8), "requests": requests, "quality": quality,
                "quality_summary": quality_summary(quality)}
            (directory / (variant + "-round-%02d.json" % round_number)).write_text(json.dumps(payload, indent=2) + "\n")
    from .report import write_report
    return write_report(directory)
