"""Metrics use monotonic timestamps, real token events, and wall-clock denominators."""
import math
import random
import statistics


def percentile(values, q):
    if not 0 <= q <= 1:
        raise ValueError("q must be between zero and one")
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def distribution(values):
    return {"n": len(values), "p50": percentile(values, .5),
            "p95": percentile(values, .95),
            "mean": statistics.mean(values) if values else None}


def request_metrics(start, token_times, end):
    times = [start] + token_times + [end]
    if any(not math.isfinite(t) for t in times):
        raise ValueError("timestamps must be finite")
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must be monotonic")
    gaps = [(b - a) * 1000 for a, b in zip(token_times, token_times[1:])]
    return {
        "output_tokens": len(token_times),
        "ttft_ms": (token_times[0] - start) * 1000 if token_times else None,
        "itl_ms": gaps,
        "tpot_ms": statistics.mean(gaps) if gaps else None,
        "e2e_ms": (end - start) * 1000,
    }


def summarize(requests, wall_seconds, hourly_cost=None, ttft_slo_ms=2000,
              tpot_slo_ms=100, setup_seconds=0):
    if wall_seconds <= 0 or setup_seconds < 0:
        raise ValueError("wall time must be positive and setup time nonnegative")
    if hourly_cost is not None and (not math.isfinite(hourly_cost) or hourly_cost < 0):
        raise ValueError("hourly cost must be finite and nonnegative")
    successful = [r for r in requests if not r.get("error") and r["output_tokens"] > 0]
    tokens = sum(r["output_tokens"] for r in successful)
    slo_pass = [r for r in successful if r["ttft_ms"] <= ttft_slo_ms
                and r["tpot_ms"] is not None and r["tpot_ms"] <= tpot_slo_ms]
    # Failures consume wall time and cost but do not earn throughput credit.
    cost = None if hourly_cost is None else hourly_cost * wall_seconds / 3600
    total_cost = None if hourly_cost is None else hourly_cost * (wall_seconds + setup_seconds) / 3600
    return {
        "attempted": len(requests), "successful": len(successful),
        "failed": len(requests) - len(successful), "output_tokens": tokens,
        "wall_seconds": wall_seconds, "setup_seconds": setup_seconds,
        "ttft_ms": distribution([r["ttft_ms"] for r in successful]),
        "itl_ms": distribution([g for r in successful for g in r["itl_ms"]]),
        "tpot_ms": distribution([r["tpot_ms"] for r in successful if r["tpot_ms"] is not None]),
        "e2e_ms": distribution([r["e2e_ms"] for r in successful]),
        "output_tokens_per_second": tokens / wall_seconds,
        "requests_per_second": len(successful) / wall_seconds,
        "slo_requests_per_second": len(slo_pass) / wall_seconds,
        "setup_inclusive_tokens_per_second": tokens / (wall_seconds + setup_seconds),
        "estimated_cost_per_1m_output_tokens": cost * 1e6 / tokens if cost is not None and tokens else None,
        "setup_inclusive_cost_per_1m_output_tokens": total_cost * 1e6 / tokens if total_cost is not None and tokens else None,
        "estimated_cost_per_1k_requests": cost * 1000 / len(successful) if cost is not None and successful else None,
        "hourly_cost_assumption": hourly_cost,
    }


def paired_bootstrap_delta(before, after, repetitions=2000, seed=17):
    """CI on mean paired difference. Caller supplies independent case-level pairs."""
    if len(before) != len(after) or not before:
        raise ValueError("nonempty paired samples required")
    deltas = [a - b for b, a in zip(before, after)]
    rng = random.Random(seed)
    samples = [statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(repetitions)]
    return {"mean_after_minus_before": statistics.mean(deltas),
            "ci95": [percentile(samples, .025), percentile(samples, .975)],
            "independent_cases": len(deltas), "bootstrap_seed": seed}
