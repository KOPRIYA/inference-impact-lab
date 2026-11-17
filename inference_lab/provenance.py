"""Portable runner fingerprints; no absolute paths or Git remote credentials."""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def source_files():
    paths = [ROOT / 'pyproject.toml']
    for directory in ('inference_lab', 'scripts', 'tests'):
        paths.extend((ROOT / directory).glob('*.py'))
    paths.extend((ROOT / 'scripts').glob('*.zsh'))
    return sorted(paths)


def provenance():
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
    import json
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {'source_sha256': digest, 'files': files, 'git_commit': commit, 'git_dirty': dirty}
