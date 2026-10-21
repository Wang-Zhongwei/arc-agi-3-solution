"""Serve qwen4_exp's PLE n-gram table from the checkpoint files instead of host RAM.

The table (102.5 GB in BF16, stored as `...ngram_embedding.shard_N.weight` rows) is
frozen and only ever gathered row by row. Loading it makes transformers hold it in
anonymous memory (and peak at ~233 GB during load), which a container OOM-kills when
offloaded activations are added. Here the n-gram module gets a parameter-less table
before loading, so transformers skips the shards without reading them, and after
loading the table memory-maps them: rows come from the page cache, which the kernel
can evict under pressure instead of killing the process. The values are identical.

    install()                      # before building or loading the model
    model = ...from_pretrained(path)
    attach(model, path)            # map the shards and ask the kernel to read them ahead
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

SHARD_KEY = "ngram_embedding.shard_"
FP8_FILE = "model-fp8-mtp-ple.safetensors"  # nvidia/Qwen3.8-Flash-Next-NVFP4: PLE (FP8, one scale) and MTP
FP8_MAX = 448.0  # largest finite float8_e4m3fn
# Which table the lookup reads (set_fp8):
# - None: the checkpoint's BF16 shards;
# - "row": the BF16 rows rounded to FP8 E4M3 with a scale per row (amax / 448) and back (a simulation);
# - "nvidia": nvidia/Qwen3.8-Flash-Next-NVFP4's FP8 table (model-fp8-mtp-ple.safetensors, one scale for the
#   whole table), mapped by attach(fp8_path=...): half the page cache of the BF16 one.
# last_rel_err: the looked-up rows' relative difference from the BF16 rows, when both are mapped.
_FP8 = {"mode": None, "last_rel_err": None}
_LOOKUP = {"seconds": 0.0}  # time the main thread spent gathering rows (prefetch threads are not counted)


def lookup_seconds() -> float:
    return _LOOKUP["seconds"]


def set_fp8(mode: str | None) -> None:
    if mode not in (None, "row", "nvidia"):
        raise ValueError(f"PLE FP8 mode {mode!r}: None, 'row' or 'nvidia'")
    _FP8["mode"], _FP8["last_rel_err"] = mode, None


def fp8_rows(rows: torch.Tensor) -> torch.Tensor:
    """BF16 rows through per-row FP8 E4M3 and back."""
    x = rows.float()
    scale = (x.abs().amax(-1, keepdim=True) / FP8_MAX).clamp_min(1e-30)
    return ((x / scale).to(torch.float8_e4m3fn).float() * scale).to(rows.dtype)


class MappedTable(nn.Module):
    """Row lookup over memory-mapped shards; stands in for the n-gram nn.Embedding."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.shards: list[torch.Tensor] = []
        self.starts: torch.Tensor | None = None
        self.fp8_shards: list[torch.Tensor] = []  # uint8 views of F8_E4M3 rows
        self.fp8_starts: torch.Tensor | None = None
        self.fp8_scale: float = 1.0
        self._device_probe = torch.empty(0, device="cpu")  # the n-gram forward reads `.weight.device`

    @property
    def weight(self):
        return self._device_probe

    @staticmethod
    def _gather(shards, starts, flat):
        """Rows `flat` from shards whose first rows are `starts`, in the order of `flat`."""
        shard = torch.bucketize(flat, starts[1:], right=True)
        order = torch.argsort(shard, stable=True)
        counts = torch.bincount(shard, minlength=len(shards)).tolist()
        pieces = torch.split(flat[order], counts)
        rows = torch.cat([shards[s].index_select(0, piece - starts[s]) for s, piece in enumerate(pieces) if piece.numel()])
        out = torch.empty_like(rows)
        out[order] = rows
        return out

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        try:
            return self._forward(ids)
        finally:
            if threading.current_thread() is threading.main_thread():
                _LOOKUP["seconds"] += time.perf_counter() - start

    def _forward(self, ids: torch.Tensor) -> torch.Tensor:
        mode = _FP8["mode"]
        if mode == "nvidia" and not self.fp8_shards:
            raise RuntimeError("PLE FP8 mode 'nvidia' needs attach(..., fp8_path=...)")
        if mode != "nvidia" and not self.shards:
            raise RuntimeError("PLE shards are not attached; call ple_mmap.attach(model, path) after loading")
        flat = ids.reshape(-1).to("cpu")
        reference = self._gather(self.shards, self.starts, flat) if self.shards else None
        if mode == "nvidia":
            raw = self._gather(self.fp8_shards, self.fp8_starts, flat)
            rows = (raw.view(torch.float8_e4m3fn).float() * self.fp8_scale).to(torch.bfloat16)
        elif mode == "row":
            rows = fp8_rows(reference)
        else:
            rows = reference
        if mode and reference is not None:
            _FP8["last_rel_err"] = float((rows.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-30))
        return rows.view(*ids.shape, self.embedding_dim)


def install() -> None:
    """Give every Qwen4ExpTextNGramEmbedding a MappedTable instead of its nn.Embedding."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextNGramEmbedding
    if hasattr(cls, "_hf_init"):
        return
    cls._hf_init = cls.__init__

    def __init__(self, *args, **kwargs):
        # from_pretrained builds on the meta device, so the original table is never allocated
        cls._hf_init(self, *args, **kwargs)
        table = self.ngram_embedding
        self.ngram_embedding = MappedTable(table.num_embeddings, table.embedding_dim)

    cls.__init__ = __init__
    # the table is no longer a parameter, so nothing is left to keep off the device map
    for obj in vars(m).values():
        if isinstance(obj, type) and getattr(obj, "_no_placement_params", None):
            obj._hf_no_placement_params = obj._no_placement_params
            obj._no_placement_params = None


def uninstall() -> None:
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextNGramEmbedding
    if hasattr(cls, "_hf_init"):
        cls.__init__ = cls._hf_init
        del cls._hf_init
    for obj in vars(m).values():
        if isinstance(obj, type) and "_hf_no_placement_params" in vars(obj):
            obj._no_placement_params = obj._hf_no_placement_params
            del obj._hf_no_placement_params


def _tensor_locations(path: Path, files: list[str] | None = None) -> dict[str, tuple[Path, int, list[int], str]]:
    """key -> (file, absolute byte offset, shape, dtype) for every n-gram tensor (shards, scale) in the checkpoint."""
    if files is None:
        index = path / "model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    out = {}
    for name in files:
        file = path / name
        with open(file, "rb") as fh:
            size = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(size))
        for key, meta in header.items():
            if "ngram_embedding." in key and key != "__metadata__":
                out[key] = (file, 8 + size + meta["data_offsets"][0], meta["shape"], meta["dtype"])
    return out


def _map_shards(locations, layer: str, dtype: str, np_dtype, prefetch_pages: bool):
    keys = sorted((k for k in locations if f".layers.{layer}.ple." in k and SHARD_KEY in k),
                  key=lambda k: int(k.split(SHARD_KEY)[1].split(".")[0]))
    shards, nbytes = [], 0
    for key in keys:
        file, offset, shape, kind = locations[key]
        assert kind == dtype, f"{key}: dtype {kind}, expected {dtype}"
        raw = np.memmap(file, dtype=np_dtype, mode="r", offset=offset, shape=tuple(shape))
        shards.append(torch.from_numpy(raw))  # read-only view, no copy
        nbytes += raw.nbytes
        if prefetch_pages and hasattr(os, "posix_fadvise"):  # let the kernel read ahead into the page cache
            with open(file, "rb") as fh:
                os.posix_fadvise(fh.fileno(), offset, raw.nbytes, os.POSIX_FADV_WILLNEED)
    return shards, torch.tensor([0] + [s.shape[0] for s in shards]).cumsum(0), nbytes


def attach(model, path, prefetch_pages: bool = True, fp8_path=None, bf16: bool = True) -> dict:
    """Memory-map the checkpoint's BF16 n-gram shards (unless `bf16` is False) and, with `fp8_path` (a directory
    holding nvidia's model-fp8-mtp-ple.safetensors), its FP8 shards and scale into each MappedTable."""
    path = Path(path)
    locations = _tensor_locations(path) if bf16 else {}
    fp8_locations = _tensor_locations(Path(fp8_path), [FP8_FILE]) if fp8_path else {}
    tables = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, MappedTable)]
    mapped_bytes = 0
    for name, table in tables:
        # e.g. "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        layer = name.split(".layers.")[1].split(".")[0]
        if bf16:
            shards, rows, nbytes = _map_shards(locations, layer, "BF16", np.uint16, prefetch_pages)
            assert int(rows[-1]) >= table.num_embeddings - 128, f"{name}: {int(rows[-1])} rows for {table.num_embeddings}"
            table.shards, table.starts = [s.view(torch.bfloat16) for s in shards], rows
            mapped_bytes += nbytes
        if fp8_path:
            shards, rows, nbytes = _map_shards(fp8_locations, layer, "F8_E4M3", np.uint8, prefetch_pages)
            assert int(rows[-1]) >= table.num_embeddings - 128, f"{name} FP8: {int(rows[-1])} rows for {table.num_embeddings}"
            scale_key = next(k for k in fp8_locations if f".layers.{layer}.ple." in k and k.endswith("weight_scale"))
            file, offset, shape, kind = fp8_locations[scale_key]
            assert kind == "BF16" and int(np.prod(shape)) == 1, f"{scale_key}: {kind} {shape}"
            raw = np.fromfile(file, dtype=np.uint16, count=1, offset=offset)
            table.fp8_scale = float(torch.from_numpy(raw).view(torch.bfloat16)[0])
            table.fp8_shards, table.fp8_starts = shards, rows
            mapped_bytes += nbytes
    return dict(tables=len(tables), shards=sum(len(t.shards) for _, t in tables),
                fp8_shards=sum(len(t.fp8_shards) for _, t in tables),
                fp8_scale=[t.fp8_scale for _, t in tables if t.fp8_shards], mapped_gb=round(mapped_bytes / 2**30, 1))
