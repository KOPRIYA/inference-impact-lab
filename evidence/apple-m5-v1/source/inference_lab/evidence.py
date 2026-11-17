"""Allowlisted, hash-verifiable evidence export. Never includes model weights or credentials."""
import hashlib
import json
import re
from pathlib import Path

from .provenance import ROOT, provenance, source_files
from .report import read_runs


def digest(data):
    return hashlib.sha256(data).hexdigest()


def review_text(text):
    patterns = [r'/Users/[^\s"\']+', r'/home/[^\s"\']+',
                r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                r'\bAKIA[0-9A-Z]{16}\b', r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
                r'\bhf_[A-Za-z0-9]{25,}\b']
    if any(re.search(p,text) for p in patterns):
        raise ValueError('Export contains a private path or credential pattern; review before publication')


def bundle(directory, destination, validation_dir=None):
    directory, destination = Path(directory),Path(destination)
    if destination.exists(): raise ValueError('Evidence destination must be new')
    grouped=read_runs(directory)
    runs=[r for group in grouped.values() for r in group]
    files={}
    for path in sorted(directory.glob('*-round-*.json')):
        if path.is_symlink(): raise ValueError('Symlinks are not allowed')
        text=path.read_text();review_text(text)
        files['raw/'+path.name]=text.encode()
    for name in ('report.md','report.html','comparison.json','evidence_summary.txt'):
        path=directory/name
        if path.exists():
            text=path.read_text();review_text(text);files['reports/'+name]=text.encode()
    if validation_dir:
        for name in ('native.json', 'freeze.json', 'heldout.json', 'cache-study.json'):
            path=Path(validation_dir)/name
            if not path.is_file() or path.is_symlink():
                raise ValueError('Missing or unsafe validation evidence: '+name)
            text=path.read_text();review_text(text)
            data=json.loads(text)
            if data.get('provenance',{}).get('source_sha256') != runs[0].get('provenance',{}).get('source_sha256'):
                raise ValueError('Validation evidence runner fingerprint differs')
            files['validation/'+name]=text.encode()
    for path in (ROOT/'docs').glob('*.md'):
        files['source/docs/'+path.name]=path.read_bytes()
    for path in source_files():
        files['source/'+str(path.relative_to(ROOT))]=path.read_bytes()
    for name in ('README.md','LICENSE','CONTRIBUTING.md'):
        if (ROOT/name).exists():files['source/'+name]=(ROOT/name).read_bytes()
    env=runs[0]['environment']
    # Exact direct runtime versions, explicitly not a transitive dependency lock.
    requirements='\n'.join('%s==%s'%(k,v) for k,v in sorted(env.get('packages',{}).items()) if v)+'\n'
    files['runtime-versions.txt']=requirements.encode()
    source=provenance()
    captured=[r.get('provenance',{}).get('source_sha256') for r in runs]
    matches=all(h==source['source_sha256'] for h in captured)
    metadata=dict(evidence=runs[0]['evidence'], source_matches_recorded_runs=matches,
                  raw_runner_fingerprints=captured, bundled_source=source,
                  review='Allowlisted synthetic benchmark raw data; private-path/credential pattern scan passed. No weights or environment secrets exported.',
                  reproduction='Install source with .[mac], use recorded model revision, prepare weights, run matching config. runtime-versions.txt pins recorded direct packages only, not all transitive dependencies.',
                  limitation=None if matches else 'Legacy runs did not record this runner fingerprint. Bundled source is a review snapshot, not proof of the exact original runner. Do not claim bit-for-bit reproduction.')
    files['metadata.json']=(json.dumps(metadata,indent=2)+'\n').encode()
    destination.mkdir(parents=True)
    for name,data in files.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    (destination/'manifest.json').write_text(json.dumps({'sha256':{n:digest(d) for n,d in files.items()}},indent=2)+'\n')
    verify(destination)


def verify(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())['sha256']
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual != set(manifest)|{'manifest.json'}: raise ValueError('Bundle file set differs from manifest')
    for name,expected in manifest.items():
        path=directory/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError('Unsafe bundle path')
        if digest(path.read_bytes())!=expected: raise ValueError('Bundle checksum mismatch: '+name)
    print('Verified evidence bundle:', directory)
