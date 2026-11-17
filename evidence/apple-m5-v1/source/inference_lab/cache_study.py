"""Controlled mixed cache hits at concurrency one; not a cache eviction simulation."""
import random
import time
from pathlib import Path

from .runtime import MLXRunner, environment
from .metrics import summarize
from .provenance import provenance
from .validation import save_new
from .workload import cases, workload_hash


def run_study(args):
    if Path(args.out).exists():
        raise ValueError('Choose a new cache-study output file')
    conditions = [(precision, hit_count, rnd) for precision in ('fp16','int4')
                  for hit_count in (0,6,12) for rnd in range(3)]
    random.Random(73).shuffle(conditions)
    results = []
    for precision, hit_count, rnd in conditions:
        engine = MLXRunner(args.models, precision)
        # Three cases per category, shared prefix; misses deliberately bypass the cache.
        suite = [c for c in cases() if c['id'].endswith(('-00','-04','-10')) and not c['id'].startswith('unknown')]
        prompts = [engine.tokenize(c, performance=True) for c in suite]
        engine.generate(prompts[0], 32, fixed_length=True)
        snap, setup = engine.build_prefix(prompts, 'performance') if hit_count else (None,0)
        flags = [True]*hit_count + [False]*(12-hit_count)
        random.Random(73+rnd).shuffle(flags)
        engine.mx.reset_peak_memory()
        start = time.perf_counter()
        rows = []
        for c,prompt,hit in zip(suite,prompts,flags):
            row = engine.generate(prompt, 128, snap if hit else None, fixed_length=True)
            row.update(id=c['id'], cache_hit=hit)
            rows.append(row)
        wall = time.perf_counter()-start
        results.append(dict(precision=precision, round=rnd, requested_hit_fraction=hit_count/12,
                            model_identity=engine.model_identity, requests=rows,
                            summary=summarize(rows, wall, setup_seconds=setup),
                            peak_allocated_bytes=engine.mx.get_peak_memory()))
        del snap, engine
        print('Finished cache condition', precision, hit_count, rnd, flush=True)
    save_new(args.out,dict(evidence='measured_cache_bypass_experiment', concurrency=1,
             limitation='Controlled reuse/bypass, not measured production hit rate, eviction or concurrent serving',
             conditions=results, environment=environment(), provenance=provenance(), workload_hash=workload_hash()))
