"""Exact structured evaluation. No LLM judge, substring match, or answer leakage."""
import json
from collections import defaultdict


def score(text, expected):
    try:
        def unique_keys(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("duplicate JSON key")
                obj[key] = value
            return obj
        obj = json.loads(text.strip(), object_pairs_hook=unique_keys)
        valid = (isinstance(obj, dict) and set(obj) == {"action", "policy_id"}
                 and obj["action"] in ("approve_return", "deny_return", "escalate")
                 and obj["policy_id"] in ("FASHION_30", "ELECTRONICS_7", "BEAUTY_14", "GROCERY_2", "MANUAL"))
    except (ValueError, TypeError):
        obj, valid = None, False
    return {"valid_json_schema": valid, "correct": bool(valid and obj == expected), "parsed": obj}


def quality_summary(rows):
    by_tag = defaultdict(list)
    for row in rows:
        by_tag[row["tag"]].append(bool(row["score"]["correct"]))
    n = len(rows)
    return {"n": n,
            "accuracy": sum(r["score"]["correct"] for r in rows) / n if n else None,
            "schema_valid_rate": sum(r["score"]["valid_json_schema"] for r in rows) / n if n else None,
            "by_tag": {tag: {"n": len(vals), "accuracy": sum(vals) / len(vals)}
                       for tag, vals in sorted(by_tag.items())}}


def gate(baseline, candidate, min_accuracy=.95, max_drop=.02):
    before = {r["id"]: r for r in baseline}
    after = {r["id"]: r for r in candidate}
    if not before or set(before) != set(after) or len(before) != len(baseline) or len(after) != len(candidate):
        raise ValueError("quality cases must have matching, unique IDs")
    b, a = quality_summary(baseline), quality_summary(candidate)
    critical = [r for r in candidate if r["tag"] in ("missing", "invalid", "injection")]
    critical_ok = all(r["score"]["correct"] for r in critical)
    return {"passed": a["accuracy"] >= min_accuracy and b["accuracy"] >= min_accuracy
            and a["accuracy"] >= b["accuracy"] - max_drop
            and a["schema_valid_rate"] == 1 and critical_ok,
            "baseline_accuracy": b["accuracy"], "candidate_accuracy": a["accuracy"],
            "accuracy_delta": a["accuracy"] - b["accuracy"],
            "min_accuracy": min_accuracy, "max_absolute_drop": max_drop,
            "critical_cases_all_pass": critical_ok,
            "regressions": [key for key in before if before[key]["score"]["correct"]
                            and not after[key]["score"]["correct"]],
            "caveat": "Small synthetic regression suite; not production quality certification."}
