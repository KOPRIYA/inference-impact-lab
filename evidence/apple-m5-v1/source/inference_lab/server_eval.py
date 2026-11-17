"""Quality-only evaluation for a user-supplied OpenAI-compatible endpoint."""
import json
import os
import urllib.request
from pathlib import Path

from .quality import quality_summary, score
from .workload import cases, messages, workload_hash


def evaluate_server(args):
    dest = Path(args.out)
    if dest.exists():
        raise ValueError("Refusing to overwrite existing quality evidence")
    rows = []
    for case in cases():
        headers = {"Content-Type": "application/json"}
        if os.environ.get("INFERENCE_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["INFERENCE_API_KEY"]
        request = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps({"model": args.model, "messages": messages(case),
                             "temperature": 0, "max_tokens": 96, "stream": False}).encode(),
            headers=headers, method="POST")
        error = None
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            text = result["choices"][0]["message"]["content"]
        except Exception as exc:
            text, error = "", type(exc).__name__
        rows.append({"id": case["id"], "tag": case["tag"], "text": text,
                     "error": error, "score": score(text, case["expected"])})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as stream:
        json.dump({"model": args.model, "workload_hash": workload_hash(),
                   "evidence": "endpoint_quality_only", "quality": rows,
                   "summary": quality_summary(rows)}, stream, indent=2)
    print(json.dumps(quality_summary(rows), indent=2))
