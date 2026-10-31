"""Build a self-contained vLLM draft directory from lordhansolo's checkpoint and a fine-tuned MTP head (stdlib only).

mtp_train.export_draft keeps lordhansolo's carry-model shards next to the fine-tuned tensors. Those shards still hold
the original non-expert mtp.* tensors, and the runtime's loader filters tensors by name (mtp.*), not by the index's
file mapping, so both copies would load. This directory holds each draft tensor once:
  mtp-experts-nvfp4.safetensors   lordhansolo's NVFP4 experts, unchanged (hard link or copy)
  mtp-finetuned-bf16.safetensors  the fine-tuned non-expert mtp.* tensors
  draft-shared-bf16.safetensors   lm_head.weight and embed_tokens, copied byte for byte from the carry shards
plus lordhansolo's config, tokenizer and template files and an index of exactly those tensors.

  python3 build_tuned_draft.py --lord <lord_mtp> --tuned <run>/draft/mtp-finetuned-bf16.safetensors --out <dir>
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

SHARED = ("lm_head.weight", "model.language_model.embed_tokens.weight")
EXPERTS = "mtp-experts-nvfp4.safetensors"
TUNED = "mtp-finetuned-bf16.safetensors"
SHARED_FILE = "draft-shared-bf16.safetensors"
CHUNK = 64 << 20


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def write_subset(sources: dict[str, Path], out: Path) -> None:
    """Copy the named tensors (name -> source file) into one safetensors file, bytes unchanged."""
    headers = {p: read_header(p) for p in set(sources.values())}
    layout, offset = {}, 0
    for name in sorted(sources):
        info = headers[sources[name]][0][name]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        layout[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps({"__metadata__": {"format": "pt"}, **layout}, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for name in sorted(sources):
            header, base = headers[sources[name]]
            begin, end = header[name]["data_offsets"]
            with sources[name].open("rb") as src:
                src.seek(base + begin)
                left = end - begin
                while left:
                    chunk = src.read(min(CHUNK, left))
                    assert chunk, f"{sources[name]} ends inside {name}"
                    f.write(chunk)
                    left -= len(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def build(lord: Path, tuned: Path, out: Path) -> dict:
    weight_map = json.loads((lord / "model.safetensors.index.json").read_text())["weight_map"]
    kept = {n: s for n, s in weight_map.items() if n.startswith("mtp.") or n in SHARED}
    tuned_header, _ = read_header(tuned)
    experts = {n for n, s in kept.items() if s == EXPERTS}
    non_expert = set(kept) - experts - set(SHARED)
    if set(tuned_header) != non_expert:
        raise SystemExit(f"tuned tensors differ from lordhansolo's non-expert mtp.*: "
                         f"missing {sorted(non_expert - set(tuned_header))}, extra {sorted(set(tuned_header) - non_expert)}")
    lord_headers = {s: read_header(lord / s)[0] for s in set(kept.values())}
    for name, info in tuned_header.items():
        ref = lord_headers[kept[name]][name]
        if (info["dtype"], info["shape"]) != (ref["dtype"], ref["shape"]):
            raise SystemExit(f"{name}: tuned {info['dtype']} {info['shape']} vs lordhansolo {ref['dtype']} {ref['shape']}")
    missing = [n for n in SHARED if n not in kept]
    if missing:
        raise SystemExit(f"{lord} lacks {missing}")

    out.mkdir(parents=True, exist_ok=False)
    for entry in sorted(lord.iterdir()):
        if entry.is_file() and not entry.name.endswith(".safetensors") and entry.name not in (
                "model.safetensors.index.json", "download.log"):
            shutil.copy2(entry, out / entry.name)
    for src, name in ((lord / EXPERTS, EXPERTS), (tuned, TUNED)):
        try:
            os.link(src, out / name)
        except OSError:
            shutil.copy2(src, out / name)
    write_subset({n: lord / kept[n] for n in SHARED}, out / SHARED_FILE)
    new_map = {n: (EXPERTS if n in experts else SHARED_FILE if n in SHARED else TUNED) for n in kept}
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": new_map}, indent=1))
    for name in SHARED:  # the copied shared tensors keep lordhansolo's bytes
        ref, _ = read_header(lord / kept[name])
        got, _ = read_header(out / SHARED_FILE)
        assert (got[name]["dtype"], got[name]["shape"]) == (ref[name]["dtype"], ref[name]["shape"])
    summary = {
        "tensors": {"experts": len(experts), "tuned": len(tuned_header), "shared": len(SHARED)},
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(out.iterdir())},
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lord", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.lord, args.tuned, args.out), indent=1))
