import io
import json
import unittest
from unittest.mock import patch

from scripts.check_server import check


class ServerCheckTests(unittest.TestCase):
    def test_checks_listing_and_generation_with_alias(self):
        calls=[]
        def respond(request,timeout):
            calls.append(request)
            payload=({'data':[{'id':'local-model'}]} if len(calls)==1 else
                     {'choices':[{'message':{'content':'ready'}}]})
            return io.BytesIO(json.dumps(payload).encode())
        with patch('urllib.request.urlopen',side_effect=respond), patch('builtins.print'):
            check('http://localhost:8000/v1','default_model')
        self.assertTrue(calls[0].full_url.endswith('/v1/models'))
        self.assertEqual(json.loads(calls[1].data)['model'],'default_model')
        self.assertTrue(calls[1].full_url.endswith('/v1/chat/completions'))

    def test_empty_listing_fails_before_generation(self):
        with patch('urllib.request.urlopen',return_value=io.BytesIO(b'{"data":[]}')) as call:
            with self.assertRaises(ValueError): check('http://localhost:8000/v1','default_model')
        self.assertEqual(call.call_count,1)

    def test_empty_completion_fails(self):
        responses=[io.BytesIO(b'{"data":[{"id":"test"}]}'),
                   io.BytesIO(b'{"choices":[{"message":{"content":""}}]}')]
        with patch('urllib.request.urlopen',side_effect=responses):
            with self.assertRaises(ValueError): check('http://localhost:8000/v1','default_model')
