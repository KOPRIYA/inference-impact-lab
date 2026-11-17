"""Development selection, frozen held-out evaluation, and native integration checks."""
import json
from pathlib import Path

from .provenance import provenance
from .quality import gate, quality_summary, score
from .runtime import MLXRunner, environment
from .workload import cases, workload_hash


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def evaluate(engine, split, cached=False):
    suite = cases(split)
    prompts = [engine.tokenize(c) for c in suite]
    snap = engine.build_prefix(prompts, 'quality')[0] if cached else None
    rows = []
    for case, prompt in zip(suite, prompts):
        result = engine.generate(prompt, 96, snap, namespace='quality')
        rows.append(dict(id=case['id'], tag=case['tag'], expected=case['expected'],
                         text=result['text'], token_ids=result['token_ids'],
                         finish_reason=result['finish_reason'],
                         score=score(result['text'], case['expected'])))
    return rows


def check_freeze(path):
    frozen = json.loads(Path(path).read_text())
    if not frozen['gate']['passed'] or frozen['split'] != 'development':
        raise ValueError('A passing development baseline is required')
    if frozen['workload_hash'] != workload_hash() or frozen['provenance']['source_sha256'] != provenance()['source_sha256']:
        raise ValueError('Source/workload changed after baseline selection; create a new freeze')
    return frozen


def assess(args):
    if Path(args.out).exists():
        raise ValueError('Choose a new assessment output file')
    if args.split == 'heldout':
        if not args.freeze:
            raise ValueError('Held-out assessment requires --freeze from a passing development run')
        frozen = check_freeze(args.freeze)
    else:
        frozen = None
    engine = MLXRunner(args.models, 'fp16')
    if frozen and frozen['model_identity'] != engine.model_identity:
        raise ValueError('Model changed after selection')
    rows = evaluate(engine, args.split)
    result = dict(evidence='measured_local_quality', split=args.split, quality=rows,
                  summary=quality_summary(rows), gate=gate(rows, rows),
                  model_identity=engine.model_identity, source_model=engine.manifest['source_model'],
                  source_revision=engine.manifest['source_revision'], workload_hash=workload_hash(),
                  environment=environment(), provenance=provenance())
    save_new(args.out, result)
    print(json.dumps(result['summary']), flush=True)
    if not result['gate']['passed']:
        raise RuntimeError('Baseline quality gate failed; evidence saved, further workflow stopped')


def native_check(args):
    if Path(args.out).exists():
        raise ValueError('Choose a new native-check output file')
    checks = []
    for precision in ('fp16', 'int4'):
        try:
            engine = MLXRunner(args.models, precision)
        except (ImportError, RuntimeError) as exc:
            save_new(args.out, dict(evidence='native_mlx_integration', passed=False,
                     status='blocked', error_type=type(exc).__name__, checks=checks,
                     environment=environment(), provenance=provenance()))
            raise RuntimeError('Native GPU check blocked; failure record saved: ' + str(exc)) from exc
        suite = cases()[:3]
        prompts = [engine.tokenize(c) for c in suite]
        snap, seconds = engine.build_prefix(prompts, 'quality')
        first = engine.generate(prompts[0], 96, snap, namespace='quality')
        engine.generate(prompts[1], 96, snap, namespace='quality')
        repeated = engine.generate(prompts[0], 96, snap, namespace='quality')
        plain = engine.generate(prompts[0], 96)
        perf = engine.generate(engine.tokenize(suite[0], performance=True), 16, fixed_length=True)
        checks.append(dict(precision=precision, model_identity=engine.model_identity,
            cache_isolation=first['token_ids'] == repeated['token_ids'],
            cached_uncached_tokens_equal=first['token_ids'] == plain['token_ids'],
            natural_eos=first['finish_reason'] == 'stop', fixed_length=perf['output_tokens'] == 16,
            monotonic_offsets=all(a <= b for a,b in zip(perf['token_offsets_ms'],perf['token_offsets_ms'][1:])),
            prefix_tokens=len(snap.tokens), prefix_seconds=seconds,
            peak_allocated_bytes=engine.mx.get_peak_memory()))
        del engine, snap
    passed = all(all(row[k] for k in ('cache_isolation','cached_uncached_tokens_equal','natural_eos','fixed_length','monotonic_offsets')) for row in checks)
    save_new(args.out, dict(evidence='native_mlx_integration', checks=checks, passed=passed,
                           environment=environment(), provenance=provenance()))
    if not passed:
        raise RuntimeError('Native check failed; inspect saved evidence before benchmarking')
    print('Native integration checks passed', flush=True)
