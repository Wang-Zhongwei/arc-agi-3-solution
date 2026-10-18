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
from pathlib import Path

import numpy as np
import torch
from torch import nn

SHARD_KEY = "ngram_embedding.shard_"


class MappedTable(nn.Module):
    """Row lookup over memory-mapped shards; stands in for the n-gram nn.Embedding."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.shards: list[torch.Tensor] = []
        self.starts: torch.Tensor | None = None
        self._device_probe = torch.empty(0, device="cpu")  # the n-gram forward reads `.weight.device`

    @property
    def weight(self):
        return self._device_probe

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if not self.shards:
            raise RuntimeError("PLE shards are not attached; call ple_mmap.attach(model, path) after loading")
        flat = ids.reshape(-1).to("cpu")
        shard = torch.bucketize(flat, self.starts[1:], right=True)
        order = torch.argsort(shard, stable=True)
        counts = torch.bincount(shard, minlength=len(self.shards)).tolist()
        pieces = torch.split(flat[order], counts)
        rows = torch.cat([self.shards[s].index_select(0, piece - self.starts[s])
                          for s, piece in enumerate(pieces) if piece.numel()])
        out = torch.empty_like(rows)
        out[order] = rows
        return out.view(*ids.shape, self.embedding_dim)


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


def _tensor_locations(path: Path) -> dict[str, tuple[Path, int, list[int], str]]:
    """key -> (file, absolute byte offset, shape, dtype) for every n-gram shard in the checkpoint."""
    index = path / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    out = {}
    for name in files:
        file = path / name
        with open(file, "rb") as fh:
            size = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(size))
        for key, meta in header.items():
            if SHARD_KEY in key:
                out[key] = (file, 8 + size + meta["data_offsets"][0], meta["shape"], meta["dtype"])
    return out


def attach(model, path, prefetch_pages: bool = True) -> dict:
    """Memory-map the checkpoint's n-gram shards into each MappedTable."""
    path = Path(path)
    locations = _tensor_locations(path)
    tables = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, MappedTable)]
    mapped_bytes = 0
    for name, table in tables:
        # e.g. "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        layer = name.split(".layers.")[1].split(".")[0]
        keys = sorted((k for k in locations if f".layers.{layer}.ple." in k),
                      key=lambda k: int(k.split(SHARD_KEY)[1].split(".")[0]))
        shards = []
        for key in keys:
            file, offset, shape, dtype = locations[key]
            assert dtype == "BF16", f"{key}: unsupported dtype {dtype}"
            raw = np.memmap(file, dtype=np.uint16, mode="r", offset=offset, shape=tuple(shape))
            shards.append(torch.from_numpy(raw).view(torch.bfloat16))  # read-only view, no copy
            mapped_bytes += raw.nbytes
            if prefetch_pages and hasattr(os, "posix_fadvise"):  # let the kernel read ahead into the page cache
                with open(file, "rb") as fh:
                    os.posix_fadvise(fh.fileno(), offset, raw.nbytes, os.POSIX_FADV_WILLNEED)
        rows = torch.tensor([0] + [s.shape[0] for s in shards]).cumsum(0)
        assert int(rows[-1]) >= table.num_embeddings - 128, f"{name}: {int(rows[-1])} rows for {table.num_embeddings}"
        table.shards, table.starts = shards, rows
    return dict(tables=len(tables), shards=sum(len(t.shards) for _, t in tables), mapped_gb=round(mapped_bytes / 2**30, 1))
