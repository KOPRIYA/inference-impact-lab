import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inference_lab.runtime import MLXRunner
from inference_lab.server_eval import evaluate_server


class AdapterTests(unittest.TestCase):
    def engine(self, responses):
        engine = MLXRunner.__new__(MLXRunner)
        engine.mx = SimpleNamespace(synchronize=lambda: None)
        engine.make_cache = lambda model: []
        engine.model = object()
        engine.tokenizer = SimpleNamespace(eos_token_ids={99})
        engine.sampler = None
        engine.stream = lambda *a, **kw: iter(responses)
        return engine

    def test_stop_eos_is_not_a_text_token(self):
        responses = [SimpleNamespace(text="hello", token=1, finish_reason=None),
                     SimpleNamespace(text="", token=99, finish_reason="stop")]
        result = self.engine(responses).generate([11, 12], 10)
        self.assertEqual(result["output_tokens"], 1)
        self.assertEqual(result["token_ids"], [1])
        self.assertEqual(result["text"], "hello")
        self.assertIsNone(result["tpot_ms"])

    def test_length_final_token_is_counted(self):
        responses = [SimpleNamespace(text="one", token=1, finish_reason=None),
                     SimpleNamespace(text=" two", token=2, finish_reason="length")]
        result = self.engine(responses).generate([11, 12], 2)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["text"], "one two")

    def test_quality_endpoint_request_contract(self):
        calls = []
        def mock_open(request, timeout):
            body = json.loads(request.data)
            self.assertFalse(body["stream"])
            self.assertEqual(body["temperature"], 0)
            self.assertNotIn("expected", body["messages"][-1]["content"])
            calls.append(body)
            return io.BytesIO(json.dumps({"choices": [{"message": {
                "content": '{"action":"escalate","policy_id":"MANUAL"}'}}]}).encode())
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(base_url="http://localhost:8000/v1", model="test-model",
                                   out=str(Path(directory) / "quality.json"))
            with patch("urllib.request.urlopen", mock_open), patch("builtins.print"):
                evaluate_server(args)
            saved = json.loads(Path(args.out).read_text())
            self.assertEqual(len(calls), 52)
            self.assertEqual(saved["summary"]["n"], 52)
            self.assertLess(saved["summary"]["accuracy"], 1)

    def test_endpoint_errors_fail_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(base_url="http://localhost:8000/v1", model="test-model",
                                   out=str(Path(directory) / "quality.json"))
            with patch("urllib.request.urlopen", side_effect=TimeoutError()), patch("builtins.print"):
                evaluate_server(args)
            saved = json.loads(Path(args.out).read_text())
            self.assertEqual(saved["summary"]["accuracy"], 0)
            self.assertTrue(all(r["error"] == "TimeoutError" for r in saved["quality"]))
