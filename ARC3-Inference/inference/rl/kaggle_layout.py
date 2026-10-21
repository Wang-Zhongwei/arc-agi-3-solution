"""Put a (merged) HF checkpoint in the Kaggle layout serving loads, so it uploads as a drop-in model.

The base Kaggle model (dfranzen/intel-qwen3.8-flash-next-w4a16-autoround) is Intel's HF checkpoint with
the 102 GB PLE shard (model-00016-of-00017.safetensors) split into 22 model-ple-* files, listed with a
sha256 each in ple_reshard_manifest.json; every other file is the same. Training never touches the PLE,
so instead of resharding locally (another 102 GB, past a 250 GB volume) this deletes the HF PLE shard,
fetches the base's 22 PLE files one at a time (each checked against the manifest), and installs the
base's index and manifest. Reruns skip files already in place.

    KAGGLEHUB_CACHE=/workspace/kcache python -m inference.rl.kaggle_layout --checkpoint /workspace/w4a16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Callable

BASE = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
MANIFEST = "ple_reshard_manifest.json"
INDEX = "model.safetensors.index.json"


def kaggle_fetch(handle: str, name: str) -> Path:
    import kagglehub  # only on the pod

    return Path(kagglehub.model_download(handle, path=name))


def sha256(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def check_index(ours: dict[str, str], theirs: dict[str, str], manifest: dict) -> None:
    """The two layouts must hold the same tensors, differing only where ours points at the PLE shard."""
    if set(ours) != set(theirs):
        raise ValueError(f"tensor sets differ: {len(set(ours) ^ set(theirs))} keys")
    in_ple = {k for s in manifest["shards"] for k in s["tensors"]}
    if any("ngram_embedding." not in k for k in in_ple):
        raise ValueError("the manifest moves tensors other than the PLE n-gram tables")
    for key in ours:
        if ours[key] == theirs[key]:
            continue
        if ours[key] != manifest["source_file"] or key not in in_ple:
            raise ValueError(f"{key}: ours in {ours[key]}, Kaggle's in {theirs[key]}")


def install(checkpoint: Path, handle: str = BASE, fetch: Callable[[str, str], Path] = kaggle_fetch) -> dict:
    manifest = json.loads(fetch(handle, MANIFEST).read_text())
    theirs = json.loads(fetch(handle, INDEX).read_text())
    ours = json.loads((checkpoint / INDEX).read_text())
    check_index(ours["weight_map"], theirs["weight_map"], manifest)
    source = checkpoint / manifest["source_file"]
    if source.exists():
        if source.stat().st_size != manifest["source_size_bytes"]:
            raise ValueError(f"{source.name}: {source.stat().st_size} bytes, the manifest's source has "
                             f"{manifest['source_size_bytes']}")
        source.unlink()
        print(f"removed {source.name}", flush=True)
    fetched = 0
    for shard in manifest["shards"]:
        out = checkpoint / shard["file"]
        if out.exists() and out.stat().st_size == shard["size_bytes"] and sha256(out) == shard["sha256"]:
            continue
        got = fetch(handle, shard["file"])
        digest = sha256(got)
        if digest != shard["sha256"]:
            raise ValueError(f"{shard['file']}: sha256 {digest}, manifest {shard['sha256']}")
        shutil.move(got, out)  # a rename when the cache is on the same volume
        fetched += 1
        print(f"fetched {shard['file']}", flush=True)
    (checkpoint / INDEX).write_text(json.dumps(theirs, indent=2))
    (checkpoint / MANIFEST).write_text(json.dumps(manifest, indent=2))
    missing = sorted(set(theirs["weight_map"].values()) - {p.name for p in checkpoint.glob("*.safetensors")})
    if missing:
        raise ValueError(f"index names missing files: {missing}")
    return dict(ple_files=len(manifest["shards"]), fetched=fetched)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--handle", default=BASE, help="the Kaggle model whose layout (and PLE files) to use")
    args = parser.parse_args(argv)
    print(json.dumps(install(args.checkpoint, args.handle)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
