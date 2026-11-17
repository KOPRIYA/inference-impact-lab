"""MLX integration. Imported lazily so offline validation needs no ML dependencies."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

from .metrics import request_metrics
from .prefix import PrefixSnapshot, common_prefix
from .workload import messages, workload_hash


def environment():
    result = {"python": platform.python_version(), "os": platform.platform(),
              "machine": platform.machine(), "packages": {}}
    for pkg in ("mlx", "mlx-lm", "transformers", "huggingface-hub"):
        try:
            result["packages"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][pkg] = None
    if platform.system() == "Darwin":
        try:
            data = json.loads(subprocess.check_output(
                ["/usr/sbin/system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"],
                timeout=30))
            # Whitelist fields; never retain host serial numbers or UUIDs.
            hardware = data["SPHardwareDataType"][0]
            result["hardware"] = {k: hardware.get(k) for k in ("chip_type", "physical_memory", "machine_model")}
            result["gpus"] = [{k: gpu.get(k) for k in ("sppci_model", "sppci_cores")}
                              for gpu in data.get("SPDisplaysDataType", [])]
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            result["hardware_error"] = type(exc).__name__
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_models(root, repo, revision):
    from huggingface_hub import HfApi, snapshot_download
    from mlx_lm import convert
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Model directory must be new or empty; choose a new --models path")
    root.mkdir(parents=True, exist_ok=True)
    sha = HfApi().model_info(repo, revision=revision).sha
    snapshot = snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    convert(snapshot, mlx_path=str(root / "fp16"), dtype="float16")
    # Quantize the very same FP16 weights rather than comparing unrelated Hub exports.
    convert(str(root / "fp16"), mlx_path=str(root / "int4"), quantize=True,
            q_bits=4, q_group_size=64)
    manifest = {"source_model": repo, "source_revision": sha,
                "quantization": {"weight_bits": 4, "group_size": 64, "method": "MLX affine weight quantization"},
                "environment": environment(), "files": {}}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(root))] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared FP16 and INT4 from revision", sha, flush=True)


class MLXRunner:
    def __init__(self, model_root, precision):
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        if not mx.metal.is_available():
            raise RuntimeError("This backend requires an Apple Silicon Metal GPU")
        mx.set_default_device(mx.gpu)
        self.mx, self.stream = mx, stream_generate
        self.make_cache, self.sampler = make_prompt_cache, make_sampler(temp=0)
        self.root = Path(model_root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        # Verify saved artifacts, outside the timing window.
        relevant = {p: info for p, info in self.manifest["files"].items() if p.startswith(precision + "/")}
        if not relevant:
            raise ValueError("Missing model hashes in manifest")
        for name, info in relevant.items():
            if file_hash(self.root / name) != info["sha256"]:
                raise ValueError("Model hash mismatch: " + name)
        self.model_identity = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
        start = time.perf_counter()
        self.model, self.tokenizer = load(str(self.root / precision))
        mx.eval(self.model.parameters())
        mx.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.model_bytes_on_disk = sum(v["bytes"] for v in relevant.values())

    def tokenize(self, case, performance=False):
        return list(self.tokenizer.apply_chat_template(messages(case, performance),
                    tokenize=True, add_generation_prompt=True))

    def build_prefix(self, prompts, namespace):
        mx = self.mx
        tokens = common_prefix(prompts)
        if not tokens:
            raise ValueError("No common prefix found")
        state = self.make_cache(self.model)
        start = time.perf_counter()
        for offset in range(0, len(tokens), 256):
            self.model(mx.array(tokens[offset:offset + 256])[None], cache=state)
            mx.eval([c.state for c in state])
        mx.synchronize()
        seconds = time.perf_counter() - start
        return PrefixSnapshot(tokens, state, self.model_identity, namespace, workload_hash()), seconds

    def generate(self, tokens, max_tokens, snapshot=None, namespace="performance", fixed_length=False):
        mx = self.mx
        mx.synchronize()
        start = time.perf_counter()
        if snapshot:
            prompt, cache = snapshot.checkout(tokens, self.model_identity, namespace, workload_hash())
        else:
            prompt, cache = tokens, self.make_cache(self.model)
        eos = set(self.tokenizer.eos_token_ids)
        processors = None
        if fixed_length:
            # Only the performance workload masks EOS; quality uses normal stopping.
            def mask_eos(history, logits):
                ids = mx.arange(logits.shape[-1])
                mask = mx.zeros(logits.shape[-1], dtype=mx.bool_)
                for token_id in eos:
                    mask = mask | (ids == token_id)
                return mx.where(mask, -float("inf"), logits)
            processors = [mask_eos]
        token_times, token_ids, pieces = [], [], []
        reason = None
        for response in self.stream(self.model, self.tokenizer, prompt=prompt,
                max_tokens=max_tokens, sampler=self.sampler, prompt_cache=cache,
                logits_processors=processors, prefill_step_size=256):
            now = time.perf_counter()
            pieces.append(response.text)
            # MLX emits a final response for EOS: exclude EOS from text-token metrics.
            if response.token not in eos:
                token_times.append(now)
                token_ids.append(int(response.token))
            reason = response.finish_reason
        mx.synchronize()
        end = time.perf_counter()
        if fixed_length and len(token_ids) != max_tokens:
            raise RuntimeError("Fixed output budget was not honored: %s/%s" % (len(token_ids), max_tokens))
        return dict(request_metrics(start, token_times, end), text="".join(pieces),
                    token_ids=token_ids, token_offsets_ms=[(t - start) * 1000 for t in token_times],
                    prompt_tokens=len(tokens), uncached_prompt_tokens=len(prompt),
                    finish_reason=reason)
