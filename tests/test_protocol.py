import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inference_lab.evidence import bundle, verify, review_text
from inference_lab.demo import demo
from inference_lab.load_test import run_load, workload
from inference_lab.provenance import provenance
from inference_lab.validation import check_freeze
from inference_lab.workload import cases, messages, oracle, workload_hash


class ProtocolTests(unittest.TestCase):
    def test_splits_disjoint_and_holdout_size(self):
        dev={json.dumps(c['order'],sort_keys=True) for c in cases()}
        held={json.dumps(c['order'],sort_keys=True) for c in cases('heldout')}
        self.assertEqual(len(held),64)
        self.assertFalse(dev & held)
        self.assertEqual(len({c['id'] for c in cases('heldout')}),64)

    def test_format_instructions_do_not_conflict(self):
        case=cases()[0]
        perf=messages(case,True)[0]['content']
        quality=messages(case)[0]['content']
        self.assertNotIn('Return ONLY one JSON',perf)
        self.assertIn('Do not emit JSON',perf)
        self.assertNotIn('Explain the applicable',quality)
        self.assertNotIn('Catalogue reference',quality)

    def test_new_invalid_inputs_escalate(self):
        for category in ([],{},None,42):
            self.assertEqual(oracle({'category':category})['action'],'escalate')
        self.assertEqual(oracle({'category':'beauty','days_since_delivery':3,'sealed':False}),
                         {'action':'deny_return','policy_id':'BEAUTY_14'})

    def test_freeze_rejects_failed_or_changed_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'freeze.json'
            data=dict(gate={'passed':True},split='development',workload_hash=workload_hash(),provenance=provenance())
            p.write_text(json.dumps(data));check_freeze(p)
            data['provenance']['source_sha256']='changed';p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):check_freeze(p)
            data['gate']['passed']=False;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):check_freeze(p)

    def test_bundle_tampering_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);demo(root/'run');bundle(root/'run',root/'export')
            verify(root/'export')
            p=root/'export/raw/fp16-round-00.json';p.write_text('{}')
            with self.assertRaises(ValueError):verify(root/'export')

    def test_bundle_rejects_private_paths(self):
        with self.assertRaises(ValueError):review_text('/Users/example/private.txt')

    def test_offered_reuse_is_not_actual_hits(self):
        items=workload(20,.5)
        self.assertEqual(sum(reuse for _,reuse in items),10)
        with self.assertRaises(ValueError):workload(10,float('nan'))
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(requests=8,reuse=.5,concurrency=4,out=str(Path(d)/'load.json'),model='test')
            def fake_request(args,item,submitted):
                return dict(error=None,completion_tokens=10,first_content_ms=2,offered_shared_prefix=item[1])
            with patch('inference_lab.load_test.request',side_effect=fake_request):run_load(args)
            saved=json.loads(Path(args.out).read_text())
            self.assertEqual(saved['successful'],8)
            self.assertIsNone(saved['actual_cache_hit_rate'])
            self.assertNotIn('itl_ms',saved)
