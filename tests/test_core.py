import json
import math
import tempfile
import unittest
from pathlib import Path

from inference_lab.demo import demo
from inference_lab.metrics import paired_bootstrap_delta, percentile, request_metrics, summarize
from inference_lab.prefix import PrefixSnapshot, common_prefix
from inference_lab.quality import gate, score
from inference_lab.report import read_runs
from inference_lab.workload import cases, messages, oracle


class MetricTests(unittest.TestCase):
    def test_token_timing(self):
        m = request_metrics(10, [10.1, 10.12, 10.15], 10.2)
        self.assertAlmostEqual(m["ttft_ms"], 100)
        self.assertAlmostEqual(m["tpot_ms"], 25)
        self.assertAlmostEqual(m["e2e_ms"], 200)

    def test_single_token_no_itl(self):
        self.assertIsNone(request_metrics(0, [.1], .2)["tpot_ms"])

    def test_empty_completion_not_success(self):
        s = summarize([request_metrics(0, [], 1)], 1)
        self.assertEqual(s["failed"], 1)
        self.assertIsNone(s["ttft_ms"]["p50"])

    def test_invalid_timestamps(self):
        for args in ((0, [.2, .1], 1), (0, [.1], .05), (0, [math.nan], 1)):
            with self.assertRaises(ValueError):
                request_metrics(*args)

    def test_cost_wall_time_and_failures(self):
        good = request_metrics(0, [.1, .2, .3], .4)
        failed = dict(request_metrics(0, [.1], .5), error="timeout")
        s = summarize([good, failed], 2, hourly_cost=3.6, setup_seconds=1)
        self.assertEqual(s["output_tokens_per_second"], 1.5)
        self.assertAlmostEqual(s["estimated_cost_per_1m_output_tokens"], 2000 / 3)
        self.assertEqual(s["setup_inclusive_tokens_per_second"], 1)
        self.assertEqual(s["failed"], 1)

    def test_no_price_no_cost(self):
        self.assertIsNone(summarize([request_metrics(0, [.1], .2)], 1)["estimated_cost_per_1m_output_tokens"])

    def test_invalid_cost(self):
        for value in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                summarize([], 1, hourly_cost=value)

    def test_percentile(self):
        self.assertEqual(percentile([4, 2, 1, 3], .5), 2.5)
        self.assertIsNone(percentile([], .95))

    def test_paired_ci(self):
        result = paired_bootstrap_delta([10, 20, 30], [5, 15, 25])
        self.assertEqual(result["ci95"], [-5, -5])


class PrefixTests(unittest.TestCase):
    def test_token_prefix(self):
        self.assertEqual(common_prefix([[1, 2, 3], [1, 2, 4]]), [1, 2])
        self.assertEqual(common_prefix([[1, 2], [1, 2]]), [1])
        self.assertEqual(common_prefix([]), [])

    def test_isolated_requests(self):
        cache = PrefixSnapshot([1, 2], [[1]], "model", "tenant", "policy")
        suffix, state = cache.checkout([1, 2, 9], "model", "tenant", "policy")
        state[0].append(99)
        self.assertEqual(suffix, [9])
        self.assertEqual(cache.state, [[1]])

    def test_invalidation(self):
        cache = PrefixSnapshot([1, 2], [], "model", "tenant", "policy")
        for identity in (("new-model", "tenant", "policy"), ("model", "other-tenant", "policy"), ("model", "tenant", "new-policy")):
            with self.assertRaises(ValueError):
                cache.checkout([1, 2, 3], *identity)

    def test_prefix_mismatch(self):
        cache = PrefixSnapshot([1, 2], [], "m", "t", "p")
        for tokens in ([1, 9, 3], [1, 2], []):
            with self.assertRaises(ValueError):
                cache.checkout(tokens, "m", "t", "p")


class QualityTests(unittest.TestCase):
    def test_boundary_oracle(self):
        self.assertEqual(oracle({"category": "fashion", "days_since_delivery": 30, "unused": True})["action"], "approve_return")
        self.assertEqual(oracle({"category": "fashion", "days_since_delivery": 31, "unused": True})["action"], "deny_return")
        self.assertEqual(oracle({"category": "fashion", "days_since_delivery": 31})["action"], "escalate")

    def test_strict_json(self):
        expected = {"action": "escalate", "policy_id": "MANUAL"}
        self.assertTrue(score(json.dumps(expected), expected)["correct"])
        for text in ("not JSON", '[]', 'null', '42', '```json\n' + json.dumps(expected) + '\n```',
                     '{"action":"escalate","action":"escalate","policy_id":"MANUAL"}',
                     '{"action":["escalate"],"policy_id":"MANUAL"}',
                     json.dumps(dict(expected, explanation="extra"))):
            self.assertFalse(score(text, expected)["correct"], text)

    def test_no_labels_in_prompt(self):
        for case in cases():
            prompt = json.loads(messages(case)[1]["content"])
            self.assertNotIn("expected", prompt)
            self.assertNotIn("policy_id", prompt)
        self.assertEqual(len(cases()), 52)
        self.assertEqual(len({c["id"] for c in cases()}), 52)

    def test_gate_rejects_critical_regression(self):
        baseline = [dict(id=c["id"], tag=c["tag"], score=score(json.dumps(c["expected"]), c["expected"])) for c in cases()]
        candidate = json.loads(json.dumps(baseline))
        candidate[-1]["score"]["correct"] = False
        self.assertFalse(gate(baseline, candidate)["passed"])
        self.assertEqual(gate(baseline, candidate)["regressions"], [baseline[-1]["id"]])

    def test_gate_rejects_bad_baseline(self):
        rows = [dict(id="a", tag="boundary", score={"correct": False, "valid_json_schema": True})]
        self.assertFalse(gate(rows, rows)["passed"])

    def test_gate_rejects_missing_cases(self):
        with self.assertRaises(ValueError):
            gate([], [])


class PipelineTests(unittest.TestCase):
    def test_demo_report_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            comparison = demo(directory)
            self.assertEqual(comparison["evidence"], "SIMULATED_TEST_FIXTURE")
            self.assertIn("NOT ELIGIBLE", (Path(directory) / "interview_draft.txt").read_text())
            self.assertIn("SIMULATED", (Path(directory) / "report.html").read_text())
            path = Path(directory) / "int4-round-00.json"
            data = json.loads(path.read_text())
            data["source_revision"] = "different"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_runs(directory)

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "keep.txt").write_text("existing evidence")
            with self.assertRaises(ValueError):
                demo(directory)


if __name__ == "__main__":
    unittest.main()
