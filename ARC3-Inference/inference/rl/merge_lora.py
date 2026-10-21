"""Merge a train_rft LoRA adapter into a safetensors checkpoint: W' = W + (alpha / r) B A.

The adapter covers BF16 tensors only (attention q/k/v/o and GDN projections; INT4 experts, PLE and
everything else are untouched), so the merged checkpoint keeps the base's config, quantization and
kernels: serving loads it like the base. Only shards holding adapted tensors are rewritten (each to a
temporary file, then renamed over the original), so the merge needs one shard of free disk.

Merged weights are stored in the base dtype (BF16). An update far below BF16's resolution (~0.4% of a
weight) partly rounds away: the report gives, per tensor, the share of the update that survives
(||bf16(W + dW) - W|| / ||dW||) and its direction (cosine with dW), plus ||dW|| / ||W||.

    python -m inference.rl.merge_lora --checkpoint /workspace/w4a16 --adapter train/adapter_final [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def adapter_updates(adapter: Path) -> dict[str, torch.Tensor]:
    """checkpoint-style module name (e.g. model.language_model.layers.0.self_attn.q_proj) -> FP32 delta W."""
    config = json.loads((adapter / "adapter_config.json").read_text())
    scaling = config["lora_alpha"] / config["r"]
    weights = load_file(adapter / "adapter_model.safetensors")
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, value in weights.items():
        for part in ("lora_A", "lora_B"):
            marker = f".{part}."
            if marker in key:
                pairs[key.split(marker)[0]][part] = value
    out = {}
    for name, ab in pairs.items():
        module = name.removeprefix("base_model.model.")
        out[module] = scaling * (ab["lora_B"].float() @ ab["lora_A"].float())
    return out


def resolve(modules: list[str], weight_keys: list[str]) -> dict[str, str]:
    """adapter module -> checkpoint weight key: an exact `<module>.weight`, else the one key ending in it."""
    keys = set(weight_keys)
    out = {}
    for module in modules:
        exact = f"{module}.weight"
        if exact in keys:
            out[module] = exact
            continue
        tail = module.split(".", 1)[1] if "." in module else module  # e.g. drop a leading "model."
        hits = [k for k in weight_keys if k.endswith(f"{tail}.weight")]
        if len(hits) != 1:
            raise KeyError(f"{module}: {len(hits)} checkpoint keys match {hits[:3]}")
        out[module] = hits[0]
    return out


def merge(checkpoint: Path, adapter: Path, dry_run: bool = False) -> dict:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
    if index is None:
        with safe_open(checkpoint / "model.safetensors", "pt") as f:
            index = {k: "model.safetensors" for k in f.keys()}
    updates = adapter_updates(adapter)
    targets = resolve(sorted(updates), sorted(index))
    by_shard: dict[str, list[str]] = defaultdict(list)
    for module, key in targets.items():
        by_shard[index[key]].append(module)
    stats = []
    for shard, modules in sorted(by_shard.items()):
        path = checkpoint / shard
        with safe_open(path, "pt") as f:
            metadata = f.metadata()
        tensors = load_file(path)
        for module in modules:
            key = targets[module]
            w = tensors[key]
            delta = updates[module]
            if tuple(delta.shape) != tuple(w.shape):
                raise ValueError(f"{key}: weight {tuple(w.shape)}, update {tuple(delta.shape)}")
            merged = (w.float() + delta).to(w.dtype)
            kept = merged.float() - w.float()
            dn = float(delta.norm())
            stats.append(dict(key=key, shard=shard, dtype=str(w.dtype), update_rel=dn / max(float(w.float().norm()), 1e-30),
                              kept=float(kept.norm()) / max(dn, 1e-30),
                              cosine=float((kept.flatten() @ delta.flatten()) / max(float(kept.norm()) * dn, 1e-30))))
            tensors[key] = merged
        if not dry_run:
            tmp = path.with_name(path.name + ".merging")
            save_file(tensors, tmp, metadata=metadata)
            os.replace(tmp, path)
        print(f"{'checked' if dry_run else 'merged'} {len(modules)} tensors in {shard}", flush=True)
    summary = dict(adapter=str(adapter), tensors=len(stats), shards=len(by_shard), dry_run=dry_run,
                   update_rel_median=statistics.median(s["update_rel"] for s in stats),
                   kept_median=statistics.median(s["kept"] for s in stats), kept_min=min(s["kept"] for s in stats),
                   cosine_median=statistics.median(s["cosine"] for s in stats), cosine_min=min(s["cosine"] for s in stats))
    if not dry_run:
        (checkpoint / "MERGED_LORA.json").write_text(json.dumps(dict(summary, per_tensor=stats), indent=1))
    return dict(summary, per_tensor=stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="report what the merge keeps, write nothing")
    args = parser.parse_args(argv)
    result = merge(args.checkpoint, args.adapter, args.dry_run)
    print(json.dumps({k: v for k, v in result.items() if k != "per_tensor"}, indent=1))
    worst = sorted(result["per_tensor"], key=lambda s: s["kept"])[:5]
    for s in worst:
        print(f"  least kept: {s['key']} kept {s['kept']:.3f} cosine {s['cosine']:.4f} update/weight {s['update_rel']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
