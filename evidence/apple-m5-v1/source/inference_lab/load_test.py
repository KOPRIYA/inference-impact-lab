"""Concurrent HTTP streaming load; chunk intervals are never labeled token ITL."""
import concurrent.futures
import json
import math
import os
import random
import time
import urllib.request
import uuid
from pathlib import Path

from .metrics import distribution
from .provenance import provenance
from .validation import save_new
from .workload import cases, messages, workload_hash


def workload(count, reuse):
    if not math.isfinite(reuse) or not 0 <= reuse <= 1:
        raise ValueError('--reuse must be between zero and one')
    shared = round(count*reuse)
    flags = [True]*shared + [False]*(count-shared)
    random.Random(73).shuffle(flags)
    suite = cases()
    result = []
    for i,reused in enumerate(flags):
        prompt = messages(suite[i % len(suite)], performance=True)
        # Unique content before the policy prevents reuse of its expensive prefix.
        if not reused:
            prompt[0]['content'] = 'Request namespace %08d.\n' % i + prompt[0]['content']
        result.append((prompt,reused))
    return result


def request(args, item, submitted):
    prompt,reused = item
    started = time.perf_counter()
    arrivals, usage, error = [], None, None
    headers = {'Content-Type':'application/json'}
    if os.environ.get('INFERENCE_API_KEY'):
        headers['Authorization'] = 'Bearer '+os.environ['INFERENCE_API_KEY']
    payload = dict(model=args.model, messages=prompt, temperature=0, max_tokens=128,
                   stream=True, stream_options={'include_usage':True})
    req = urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',
                                 data=json.dumps(payload).encode(), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req,timeout=120) as response:
            for raw in response:
                line=raw.decode().strip()
                if not line.startswith('data:'): continue
                data=line[5:].strip()
                if data=='[DONE]': break
                event=json.loads(data)
                if event.get('usage'): usage=event['usage'].get('completion_tokens')
                if any(c.get('delta',{}).get('content') for c in event.get('choices',[])):
                    arrivals.append(time.perf_counter())
        if not arrivals: error='EmptyOutput'
    except Exception as exc:
        error=type(exc).__name__  # do not persist URLs, authorization or server error bodies
    end=time.perf_counter()
    return dict(offered_shared_prefix=reused, error=error,
                client_queue_ms=(started-submitted)*1000,
                first_content_ms=(arrivals[0]-started)*1000 if arrivals else None,
                first_content_including_queue_ms=(arrivals[0]-submitted)*1000 if arrivals else None,
                chunk_intervals_ms=[(b-a)*1000 for a,b in zip(arrivals,arrivals[1:])],
                completion_tokens=usage, elapsed_ms=(end-started)*1000)


def run_load(args):
    if Path(args.out).exists(): raise ValueError('Choose a new load-test output')
    inputs=workload(args.requests,args.reuse)
    # Avoid warming later experiments with an earlier run's identical prefix.
    namespace=uuid.uuid4().hex
    for prompt,_ in inputs:
        prompt[0]['content'] = 'Experiment namespace '+namespace+'.\n'+prompt[0]['content']
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures=[pool.submit(request,args,item,time.perf_counter()) for item in inputs]
        rows=[f.result() for f in futures]
    duration=time.perf_counter()-start
    good=[r for r in rows if not r['error']]
    token_count=sum(r['completion_tokens'] for r in good) if good and all(type(r['completion_tokens']) is int for r in good) else None
    save_new(args.out,dict(evidence='measured_http_load', model=args.model, concurrency=args.concurrency,
        offered_shared_prefix_fraction=sum(r['offered_shared_prefix'] for r in rows)/len(rows),
        actual_cache_hit_rate=None, warmup='none; first shared request may be a miss', experiment_namespace=namespace,
        duration_seconds=duration, successful=len(good), failed=len(rows)-len(good),
        successful_requests_per_second=len(good)/duration,
        output_tokens_per_second=token_count/duration if token_count is not None else None,
        first_content_ms=distribution([r['first_content_ms'] for r in good]),
        limitation='HTTP content chunks are not model tokens; no token-level ITL claim. Client queue is separate. Record server revision, hardware, cache metrics and startup configuration independently.',
        requests=rows, provenance=provenance(), workload_hash=workload_hash()))
    if len(good)!=len(rows): raise RuntimeError('Load test had failures; evidence saved')
