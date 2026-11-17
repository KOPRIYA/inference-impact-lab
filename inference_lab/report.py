"""Reject incomparable evidence; expose negative results and quality failures."""
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .experiment import VARIANTS
from .metrics import paired_bootstrap_delta, summarize
from .quality import gate, quality_summary


def read_runs(directory):
    paths = sorted(Path(directory).glob("*-round-*.json"))
    runs = [json.loads(p.read_text()) for p in paths]
    if not runs:
        raise ValueError("No completed runs found")
    signatures = set()
    for run in runs:
        signatures.add(json.dumps({k: run[k] for k in
            ("evidence", "source_model", "source_revision", "workload_hash", "config", "environment", "seed")}, sort_keys=True))
    if len(signatures) != 1:
        raise ValueError("Runs differ in provenance, hardware, software, workload or configuration")
    seen = set()
    grouped = defaultdict(list)
    for run in runs:
        key = (run["variant"], run["round"])
        if key in seen:
            raise ValueError("Duplicate variant / round")
        seen.add(key)
        grouped[run["variant"]].append(run)
    if set(grouped) != set(VARIANTS):
        raise ValueError("All four variants are required")
    round_sets = {tuple(sorted(r["round"] for r in group)) for group in grouped.values()}
    if len(round_sets) != 1 or 0 not in next(iter(round_sets)):
        raise ValueError("Variants must have matching rounds including round zero")
    request_sets = {tuple(sorted(r["id"] for r in run["requests"])) for run in runs}
    if len(request_sets) != 1 or any(len({r["id"] for r in run["requests"]}) != len(run["requests"]) for run in runs):
        raise ValueError("Performance request IDs must match and be unique per run")
    hourly_costs = {r["summary"]["hourly_cost_assumption"] for r in runs}
    if len(hourly_costs) != 1:
        raise ValueError("Cost assumptions differ")
    for precision in ("fp16", "int4"):
        identities = {r["model_identity"] for variant in (precision, precision + "-prefix")
                      for r in grouped[variant]}
        if len(identities) != 1:
            raise ValueError("Weights differ between cached and uncached runs")
    return grouped


def write_report(directory):
    directory = Path(directory)
    grouped = read_runs(directory)
    summaries, quality = {}, {}
    for variant, runs in grouped.items():
        rows = [row for run in runs for row in run["requests"]]
        example = runs[0]
        summaries[variant] = summarize(rows, sum(r["summary"]["wall_seconds"] for r in runs),
            example["summary"]["hourly_cost_assumption"],
            example["config"]["ttft_slo_ms"], example["config"]["tpot_slo_ms"],
            sum(r["prefix_setup_seconds"] for r in runs))
        summaries[variant]["max_peak_mlx_gb"] = max(r["peak_mlx_allocated_bytes"] for r in runs) / 1e9
        summaries[variant]["model_disk_gb"] = example["model_bytes_on_disk"] / 1e9
        quality[variant] = next(r for r in runs if r["round"] == 0)["quality"]
    gates = {v: gate(quality["fp16"], quality[v]) for v in VARIANTS}
    cache_equivalence = {}
    for precision in ("fp16", "int4"):
        before = {r["id"]: r for r in quality[precision]}
        after = {r["id"]: r for r in quality[precision + "-prefix"]}
        cache_equivalence[precision] = {
            "n": len(before),
            "token_sequence_matches": sum(before[k].get("token_ids") == after[k].get("token_ids") for k in before),
            "valid_parsed_answer_matches": sum(
                before[k]["score"]["valid_json_schema"] and after[k]["score"]["valid_json_schema"]
                and before[k]["score"]["parsed"] == after[k]["score"]["parsed"] for k in before),
        }
    # Average repeated observations within case before bootstrapping: avoid pseudo-replication.
    case_means = {}
    for variant, runs in grouped.items():
        buckets = defaultdict(list)
        for run in runs:
            for row in run["requests"]:
                buckets[row["id"]].append(row["ttft_ms"])
        if any(value is None for values in buckets.values() for value in values):
            raise ValueError("Missing TTFT; inspect failed requests before comparison")
        case_means[variant] = {key: statistics.mean(vals) for key, vals in buckets.items()}
    keys = sorted(case_means["fp16"])
    deltas = {v: paired_bootstrap_delta([case_means["fp16"][k] for k in keys],
                                      [case_means[v][k] for k in keys]) for v in VARIANTS if v != "fp16"}
    example = grouped["fp16"][0]
    simulated = example["evidence"] != "measured_local_inference"
    comparison = {"evidence": example["evidence"], "production": False,
                  "summaries": summaries, "quality_gates": gates, "ttft_paired_ci": deltas,
                  "cache_equivalence": cache_equivalence,
                  "quality_summaries": {v: quality_summary(rows) for v, rows in quality.items()}}
    (directory / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    def fmt(value):
        return "not measured" if value is None else "%.3f" % value
    title = "SIMULATED TEST FIXTURE — NOT MODEL PERFORMANCE" if simulated else "Local inference measurements — not production"
    lines = ["# " + title, "", "Model: `%s` at `%s`." % (example["source_model"], example["source_revision"]),
             "", "Hardware: " + json.dumps(example["environment"].get("gpus", [])) + ".",
             "", "| Variant | TTFT p50 / p95 ms | ITL p50 / p95 ms | Output tok/s | Peak MLX GB | Accuracy | Gate | Est. $/1M output tokens |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for v in VARIANTS:
        s, q = summaries[v], quality_summary(quality[v])
        lines.append("| %s | %s / %s | %s / %s | %s | %s | %.1f%% | %s | %s |" % (
            v, fmt(s["ttft_ms"]["p50"]), fmt(s["ttft_ms"]["p95"]),
            fmt(s["itl_ms"]["p50"]), fmt(s["itl_ms"]["p95"]),
            fmt(s["output_tokens_per_second"]), fmt(s["max_peak_mlx_gb"]), q["accuracy"] * 100,
            "PASS" if gates[v]["passed"] else "FAIL", fmt(s["estimated_cost_per_1m_output_tokens"])))
    lines += ["", "## Interpretation", "",
        "Compare fp16→int4 for quantization; fp16→fp16-prefix for caching; fp16→int4-prefix for the combination.",
        "No faster variant is recommended automatically. A failed quality gate disqualifies it.",
        "TTFT and ITL are in-process token availability, excluding tokenization, model load and network time.",
        "Throughput uses total measured request wall time; it is serial throughput at concurrency 1, not server capacity.",
        "Prefix rows measure warm hits. comparison.json also includes throughput/cost including one prefix fill per round.",
        "ITL percentiles pool token gaps. TPOT is per-request mean inter-token gap and is reported separately in JSON.",
        "Peak memory is MLX allocator memory, not total system RAM or dedicated VRAM. Disk size is not GPU memory.",
        "Cost is an explicit hourly allocation model, not a bill. No hourly price supplied means no cost estimate.",
        "The performance suite forces a fixed output budget. The 52-case quality suite uses natural EOS stopping.",
        "Confidence intervals resample case means, not individual tokens. Small synthetic samples do not establish production impact.",
        "", "## Paired TTFT changes", ""]
    for v, delta in deltas.items():
        lines.append("- %s: after − before = %.3f ms; case-bootstrap 95%% CI [%.3f, %.3f] ms." %
                     (v, delta["mean_after_minus_before"], *delta["ci95"]))
    lines += ["", "## Quality diagnostics", "",
              "Accuracy requires the exact expected decision and a valid schema. Valid JSON alone is insufficient.",
              "No regressions means no previously correct cases became wrong; it does not mean the baseline is useful.",
              "", "| Variant | Correct / total | Valid schema | Boundary | Missing | Invalid | Injection |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        q = quality_summary(quality[v])
        by_tag = q["by_tag"]
        tag_values = ["%.1f%%" % (by_tag[tag]["accuracy"] * 100) if tag in by_tag else "n/a"
                      for tag in ("boundary", "missing", "invalid", "injection")]
        lines.append("| %s | %d / %d | %.1f%% | %s |" % (
            v, sum(r["score"]["correct"] for r in quality[v]), q["n"],
            q["schema_valid_rate"] * 100, " | ".join(tag_values)))
    lines += ["", "## Cache agreement", "",
              "Matching outputs can still be incorrect. Token agreement measures cache consistency, not task accuracy.",
              "```json", json.dumps(cache_equivalence, indent=2), "```"]
    lines += ["", "## Quality regressions", "", "```json", json.dumps(gates, indent=2), "```", "",
              "## Production status", "", "No production deployment or customer traffic is claimed."]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    # Self-contained, offline HTML evidence viewer, no external JavaScript or fonts.
    rows = "".join("<tr><td>%s</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%s</td></tr>" %
        (html.escape(v), s["ttft_ms"]["p50"], s["itl_ms"]["p50"],
         s["output_tokens_per_second"], "PASS" if gates[v]["passed"] else "FAIL")
        for v, s in summaries.items())
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Inference Impact Lab</title><style>body{font:16px system-ui;max-width:1100px;margin:50px auto;padding:24px;background:#101823;color:#e5edf5}h1{color:#71e0c0}table{border-collapse:collapse;width:100%%}td,th{padding:14px;text-align:left;border-bottom:1px solid #405060}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1c2938;padding:24px}summary{cursor:pointer;margin:24px 0}p{line-height:1.7}</style>
<h1>Inference Impact Lab</h1><p><strong>%s</strong></p><p>Two controlled experiments: 4-bit weight quantization and exact-prefix KV reuse.</p>
<table><tr><th>Variant</th><th>TTFT p50 (ms)</th><th>ITL p50 (ms)</th><th>Output tokens/s</th><th>Quality gate</th></tr>%s</table>
<details><summary>Full methodology and evidence</summary><pre>%s</pre></details></html>""" % (html.escape(title), rows, html.escape("\n".join(lines)))
    (directory / "report.html").write_text(page)
    if simulated:
        summary = "SIMULATED TEST FIXTURE: no model performance was measured.\n"
    else:
        passed = [v for v in VARIANTS if gates[v]["passed"]]
        summary = ("Measured local inference for %s. Quality-passing variants: %s. "
                   "Inspect the report for timing boundaries, quality failures, and uncertainty. "
                   "No production deployment or monetary savings are established.\n") % (
                       example["source_model"], ", ".join(passed) if passed else "none")
    (directory / "evidence_summary.txt").write_text(summary)
    print("Report written to", directory / "report.html")
    return comparison
