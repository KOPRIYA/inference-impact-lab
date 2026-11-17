"""Check model discovery and a real completion without requiring MLX in the client."""
import argparse
import json
import os
import urllib.error
import urllib.request


def check(base_url, model):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('INFERENCE_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['INFERENCE_API_KEY']
    root = base_url.rstrip('/')
    with urllib.request.urlopen(urllib.request.Request(root+'/models', headers=headers), timeout=10) as response:
        listing = json.load(response)
    if not listing.get('data'):
        raise ValueError('Server returned an empty model list')
    payload = {'model': model, 'messages': [{'role':'user','content':'Reply with the single word ready.'}],
               'max_tokens':16, 'temperature':0, 'stream':False}
    request = urllib.request.Request(root+'/chat/completions', data=json.dumps(payload).encode(),
                                     headers=headers, method='POST')
    with urllib.request.urlopen(request, timeout=120) as response:
        completion = json.load(response)
    content = completion['choices'][0]['message']['content']
    if not isinstance(content,str) or not content.strip():
        raise ValueError('Server returned no text')
    print('Model discovery and real completion succeeded. Model alias:', model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='default_model')
    args = parser.parse_args()
    try:
        check(args.base_url,args.model)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print('Server check failed (%s). Start scripts/serve_local.zsh in another Terminal; verify BASE_URL and MODEL_ID.' % type(exc).__name__)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
