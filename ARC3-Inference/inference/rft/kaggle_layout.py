"""Put a (merged) HF checkpoint in the Kaggle layout serving loads, without its PLE table.

The base Kaggle model (dfranzen/intel-qwen3.8-flash-next-w4a16-autoround) is Intel's HF checkpoint with
the 102 GB PLE shard (model-00016-of-00017.safetensors) split into 22 model-ple-* files, listed with a
sha256 each in ple_reshard_manifest.json; every other file is the same. Training never touches the PLE,
so a published model is partial (user, 2026-10-21: no PLE download, no PLE upload): this installs the
base's index and manifest (two small files) and moves any local PLE file out of the checkpoint folder
(to ple_aside/ beside it), so only the merged non-PLE files (~79 GB) upload. Serving takes the 22 PLE
files from the base model. --with-ple instead fetches them from the base (each checked against the
manifest) for a self-contained copy. Reruns skip what is in place.

    KAGGLEHUB_CACHE=/workspace/kcache python -m inference.rft.kaggle_layout --checkpoint /workspace/w4a16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def install(checkpoint: Path, handle: str = BASE, fetch: Callable[[str, str], Path] = kaggle_fetch,
            ple: bool = False) -> dict:
    manifest = json.loads(fetch(handle, MANIFEST).read_text())
    theirs = json.loads(fetch(handle, INDEX).read_text())
    ours = json.loads((checkpoint / INDEX).read_text())
    check_index(ours["weight_map"], theirs["weight_map"], manifest)
    source = checkpoint / manifest["source_file"]
    ple_files = [s["file"] for s in manifest["shards"]]
    if not ple:
        aside = checkpoint.parent / "ple_aside"
        for name in [manifest["source_file"], *ple_files]:
            if (checkpoint / name).exists():
                aside.mkdir(exist_ok=True)
                os.replace(checkpoint / name, aside / name)
                print(f"moved {name} to {aside}", flush=True)
        (checkpoint / INDEX).write_text(json.dumps(theirs, indent=2))
        (checkpoint / MANIFEST).write_text(json.dumps(manifest, indent=2))
        missing = set(theirs["weight_map"].values()) - {p.name for p in checkpoint.glob("*.safetensors")}
        if missing != set(ple_files):
            raise ValueError(f"index names missing non-PLE files: {sorted(missing - set(ple_files))}")
        return dict(ple_files=len(ple_files), fetched=0, partial=True)
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
    return dict(ple_files=len(manifest["shards"]), fetched=fetched, partial=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--handle", default=BASE, help="the Kaggle model whose layout (and PLE files) to use")
    parser.add_argument("--with-ple", action="store_true", help="fetch the 22 PLE files too (a self-contained copy)")
    args = parser.parse_args(argv)
    print(json.dumps(install(args.checkpoint, args.handle, ple=args.with_ple)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
