"""Weight sync into the running vLLM server: a worker extension that loads the trainer's merged tensors in place, and
the client calls the loop makes (VLLM_SERVER_DEV_MODE=1 for /collective_rpc).

- Worker side: `vllm serve ... --worker-extension-cls inference.grpo.vllm_sync.WeightSyncExtension` with the repo's
  ARC3-Inference on PYTHONPATH. `grpo_load_weights(path)` reads a safetensors file of checkpoint-named tensors
  (train_grpo's sync/step_<k>.safetensors: attention q/k/v/o and GDN projections) and hands them to the model's own
  `load_weights`, so each parameter's weight_loader takes its tensor-parallel shard and fills its slice of a fused
  projection, in place: the captured CUDA graphs keep their buffers. It does not use vLLM's layerwise reload, which
  finalizes a fused layer that received only some of its tensors with uninitialized memory in the rest; the adapted
  layers are plain BF16 linear layers with nothing to post-process. It returns per worker the tensors read, the
  parameters written, and a checksum of the first one, so the loop can see every rank took the update.
- Client side, between waves: /pause?mode=wait, /collective_rpc, /reset_prefix_cache (cached KV and GDN states
  belong to the old weights; retried while blocks are held), /update_weight_version, /resume.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any


class WeightSyncExtension:
    """Mixed into vLLM's GPU worker (--worker-extension-cls); `self.model_runner` is the worker's."""

    def grpo_load_weights(self, path: str) -> dict[str, Any]:
        import torch
        from safetensors import safe_open

        try:
            from vllm.model_executor.model_loader.mtp_validation import disable_mtp_completeness_check
        except ImportError:  # older vLLM: no completeness check on partial loads
            from contextlib import nullcontext as disable_mtp_completeness_check

        runner = self.model_runner
        model = runner.get_model() if hasattr(runner, "get_model") else runner.model
        started = time.time()
        names, written, first = [], set(), None
        with disable_mtp_completeness_check(), safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if first is None:
                    first = (name, float(tensor.float().sum()))
                loaded = model.load_weights([(name, tensor)])
                if loaded:
                    written.update(loaded)
                names.append(name)
        torch.cuda.synchronize()
        return dict(tensors=len(names), parameters_written=len(written), seconds=round(time.time() - started, 2),
                    first=first[0] if first else None, first_sum=first[1] if first else None)


def _post(base: str, path: str, body: dict[str, Any] | None = None, timeout: float = 900) -> dict[str, Any]:
    data = json.dumps(body or {}).encode()
    request = urllib.request.Request(f"{base.rstrip('/')}{path}", data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def server_root(base_url: str) -> str:
    """http://host:port from an OpenAI base URL (.../v1)."""
    return base_url[:-3] if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/")


def sync_weights(base_url: str, path: str, version: str, reset_tries: int = 30) -> dict[str, Any]:
    """Load `path` into every worker between waves. Raises if a worker loaded nothing or the cache never reset."""
    root = server_root(base_url)
    started = time.time()
    _post(root, "/pause?mode=wait&clear_cache=false")
    try:
        rpc = _post(root, "/collective_rpc", {"method": "grpo_load_weights", "args": [path], "timeout": 900})
        results = rpc.get("results") or []
        if not results or any(not isinstance(r, dict) or not r.get("tensors") for r in results):
            raise RuntimeError(f"weight load failed on a worker: {results}")
        if len({r.get("first_sum") for r in results}) != 1:
            raise RuntimeError(f"workers read different files: {results}")
        for _ in range(reset_tries):
            if _post(root, "/reset_prefix_cache").get("success"):
                break
            time.sleep(2)
        else:
            raise RuntimeError("prefix cache did not reset")
        _post(root, "/update_weight_version", {"new_version": version})
    finally:
        _post(root, "/resume")
    return dict(workers=results, seconds=round(time.time() - started, 1), version=version)
