"""Checkpoint views and a shard prefetch window for the Pennyroyal launcher.

The launcher serves the read-only Kaggle model mounts through views: directories
of symlinks plus a rewritten config/index, so nothing is copied.

- `build_target_view` swaps a checkpoint's BF16 PLE table for a matching FP8 one
  (RadixArk's `model-plefp8-*` shards and their global `weight_scale`). Every
  GPU-resident tensor stays the original checkpoint's.
- `build_ple_dtype_view` serves a checkpoint as is but sets its declared PLE
  dtype, e.g. BF16 for lordhansolo's shards, whose config says FP8.
- `build_mtp_draft_view` keeps only the shards holding the embedded MTP head
  (plus the embedding and LM head), so the NEXTN draft load does not read the
  whole target checkpoint a second time.
- `ShardWindowPrefetcher` is lordhansolo's GPU shard prefetcher (Milestone 2
  notebook v293) ported to SGLang: 4 threads keep a bounded 8 GiB window of
  shards ahead of the single-stream safetensors loader in the page cache,
  following the loader's progress bar in the server log.
"""
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import struct
import threading
import time
from pathlib import Path

INDEX_NAME = "model.safetensors.index.json"
PLE_TABLE_MARKER = ".ple.ple_embedding.ngram_embedding."
FP8_DTYPE = "F8_E4M3"
DRAFT_EXTRA_TENSORS = ("lm_head.weight", "model.language_model.embed_tokens.weight")


def read_index(root) -> dict:
    return json.loads((Path(root) / INDEX_NAME).read_text())


def read_safetensors_header(path) -> dict:
    """Tensor name -> {dtype, shape, ...}, read from the header only."""
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


def _headers_for(root, weight_map: dict, names) -> dict:
    """Header entries of `names`, reading each shard's header once."""
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(weight_map[name], []).append(name)
    entries = {}
    for file_name, file_names in by_file.items():
        header = read_safetensors_header(Path(root) / file_name)
        for name in file_names:
            entries[name] = header[name]
    return entries


def _link(dest: Path, src: Path) -> None:
    """Idempotent symlink; refuses to replace an unrelated existing file."""
    src = Path(src).resolve()
    if dest.exists() or dest.is_symlink():
        if not dest.is_symlink() or dest.resolve() != src:
            raise RuntimeError(f"Unexpected existing view file: {dest}")
        return
    dest.symlink_to(src)


def _write_once(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_text() != text:
            raise RuntimeError(f"Unexpected existing view file: {path}")
        return
    path.write_text(text)


def _materialize_view(work_root, label: str, sources: dict, config: dict, weight_map: dict, extra_files: dict) -> Path:
    """Create `<work_root>/<label>-<hash>` holding config, index and shard links.

    sources maps each shard file name to the directory holding it; extra_files maps
    other file names (tokenizer, templates) to their source paths.
    """
    identity = json.dumps(
        {"sources": {k: str(Path(v).resolve()) for k, v in sorted(sources.items())},
         "extra": {k: str(Path(v).resolve()) for k, v in sorted(extra_files.items())},
         "config": config, "weight_map_sha": hashlib.sha256(json.dumps(weight_map, sort_keys=True).encode()).hexdigest()},
        sort_keys=True,
    )
    view = Path(work_root) / f"{label}-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    view.mkdir(parents=True, exist_ok=True)
    _write_once(view / "config.json", json.dumps(config, indent=2) + "\n")
    index = {"metadata": {"view_of": sorted({str(Path(v).resolve()) for v in sources.values()})}, "weight_map": weight_map}
    _write_once(view / INDEX_NAME, json.dumps(index, indent=1, sort_keys=True) + "\n")
    for name, root in sources.items():
        if not (Path(root) / name).is_file():
            raise FileNotFoundError(f"Missing shard for view: {Path(root) / name}")
        _link(view / name, Path(root) / name)
    for name, src in extra_files.items():
        _link(view / name, Path(src))
    return view


def _extra_files(root) -> dict:
    """Non-weight files that travel with a checkpoint (tokenizer, templates, ...)."""
    skip = {"config.json", INDEX_NAME}
    return {
        p.name: p for p in sorted(Path(root).iterdir())
        if p.is_file() and p.name not in skip and not p.name.endswith(".safetensors")
    }


def build_target_view(target, ple_fp8, work_root) -> Path:
    """`target` with its BF16 PLE table replaced by the FP8 table from `ple_fp8`.

    Both checkpoints must name the PLE shards identically with equal shapes; the
    FP8 side adds one `weight_scale` per PLE layer. Shards that held only BF16
    PLE rows drop out of the view, so neither the loader nor a prefetcher reads them.
    """
    target, ple_fp8 = Path(target), Path(ple_fp8)
    target_map = read_index(target)["weight_map"]
    fp8_map = read_index(ple_fp8)["weight_map"]
    target_ple = {k for k in target_map if PLE_TABLE_MARKER in k}
    fp8_ple = {k for k in fp8_map if PLE_TABLE_MARKER in k}
    if not target_ple:
        raise RuntimeError(f"No PLE table tensors in {target}")
    missing = sorted(target_ple - fp8_ple)
    if missing:
        raise RuntimeError(f"FP8 PLE checkpoint lacks {len(missing)} PLE tensors, e.g. {missing[:3]}")
    extra = sorted(fp8_ple - target_ple)
    if any(not k.endswith(".weight_scale") for k in extra):
        raise RuntimeError(f"Unexpected extra FP8 PLE tensors: {extra[:5]}")
    # Headers only: shapes must match 1:1, and the FP8 side must really be FP8.
    target_headers = _headers_for(target, target_map, sorted(target_ple))
    fp8_headers = _headers_for(ple_fp8, fp8_map, sorted(fp8_ple))
    for name in sorted(target_ple):
        if target_headers[name]["shape"] != fp8_headers[name]["shape"]:
            raise RuntimeError(f"PLE shape mismatch for {name}: {target_headers[name]['shape']} vs {fp8_headers[name]['shape']}")
        if fp8_headers[name]["dtype"] != FP8_DTYPE:
            raise RuntimeError(f"{name} in {ple_fp8} is {fp8_headers[name]['dtype']}, expected {FP8_DTYPE}")
    for name in extra:
        if fp8_headers[name]["shape"] not in ([1], []):
            raise RuntimeError(f"Unexpected PLE scale shape for {name}: {fp8_headers[name]['shape']}")

    weight_map = dict(target_map)
    for name in sorted(fp8_ple):
        weight_map[name] = fp8_map[name]
    sources = {}
    for name, file_name in weight_map.items():
        root = ple_fp8 if name in fp8_ple else target
        if sources.setdefault(file_name, root) != root:
            raise RuntimeError(f"Shard name {file_name} exists in both checkpoints")

    config = json.loads((target / "config.json").read_text())
    text_config = config.get("text_config", config)
    text_config["ple_embedding_dtype"] = "float8_e4m3fn"
    view = _materialize_view(work_root, "target-view", sources, config, weight_map, _extra_files(target))
    dropped = sorted(set(target_map.values()) - set(weight_map.values()))
    print(f"Target view: {view} ({len(sources)} shards; FP8 PLE from {ple_fp8}; "
          f"{len(dropped)} BF16-only PLE shards left out)")
    return view


def build_ple_dtype_view(target, work_root, ple_dtype: str) -> Path:
    """`target` unchanged except for `text_config.ple_embedding_dtype`.

    lordhansolo's mixed checkpoint ships BF16 PLE shards but declares
    `ple_embedding_dtype: float8_e4m3fn`, so Pennyroyal would allocate an FP8 host
    table and downcast the shards. `ple_dtype="bfloat16"` keeps his BF16 table, as
    his vLLM run did.
    """
    target = Path(target)
    target_map = read_index(target)["weight_map"]
    ple_tensors = sorted(k for k in target_map if PLE_TABLE_MARKER in k)
    if not ple_tensors:
        raise RuntimeError(f"No PLE table tensors in {target}")
    shard_dtypes = {h["dtype"] for h in _headers_for(target, target_map, ple_tensors).values()}
    if ple_dtype == "bfloat16" and shard_dtypes != {"BF16"}:
        raise RuntimeError(f"PLE shards in {target} are {sorted(shard_dtypes)}, not BF16")
    config = json.loads((target / "config.json").read_text())
    config.get("text_config", config)["ple_embedding_dtype"] = ple_dtype
    sources = {f: target for f in set(target_map.values())}
    view = _materialize_view(work_root, "target-view", sources, config, target_map, _extra_files(target))
    print(f"Target view: {view} ({len(sources)} shards; PLE kept {ple_dtype}, {len(ple_tensors)} tensors)")
    return view


def has_embedded_mtp(root) -> bool:
    return any(k.startswith("mtp.") for k in read_index(root)["weight_map"])


def build_mtp_draft_view(target, work_root) -> Path:
    """Only the shards with `mtp.*` tensors, the embedding and the LM head.

    SGLang's NEXTN draft reads every shard of its model path and keeps the MTP
    tensors, so pointing it at the whole target re-reads the full checkpoint.
    """
    target = Path(target)
    target_map = read_index(target)["weight_map"]
    keep = {f for k, f in target_map.items() if k.startswith("mtp.") or k in DRAFT_EXTRA_TENSORS}
    if not any(k.startswith("mtp.") for k in target_map):
        raise RuntimeError(f"No embedded MTP tensors in {target}")
    weight_map = {k: f for k, f in target_map.items() if f in keep}
    config = json.loads((target / "config.json").read_text())
    sources = {f: target for f in keep}
    view = _materialize_view(work_root, "draft-view", sources, config, weight_map, _extra_files(target))
    gib = sum((target / f).stat().st_size for f in keep) / 1024**3
    print(f"MTP draft view: {view} ({len(keep)}/{len(set(target_map.values()))} shards, {gib:.2f} GiB)")
    return view


def loader_shard_order(model_path) -> list[Path]:
    """Shards in the order SGLang's safetensors loader reads them.

    `_prepare_weights` globs `*.safetensors` in the model directory and keeps the
    files the index references, preserving glob order; the iterator does not sort.
    """
    model_path = str(model_path)
    indexed = {os.path.join(model_path, f) for f in read_index(model_path)["weight_map"].values()}
    return [Path(p) for p in glob.glob(os.path.join(model_path, "*.safetensors")) if p in indexed]


def read_file_into_page_cache(path: Path, stop_event: threading.Event, block_bytes: int) -> dict:
    started_at = time.monotonic()
    buffer = memoryview(bytearray(block_bytes))
    with open(path, "rb", buffering=0) as handle:
        while not (stopped := stop_event.is_set()) and handle.readinto(buffer):
            pass
    return {"started_at": started_at, "finished_at": time.monotonic(), "stopped": stopped}


def drop_file_from_page_cache(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError) as error:
        print(f"Shard prefetch could not evict {path.name}: {error!r}", flush=True)


class ShardWindowPrefetcher(threading.Thread):
    """Keep shards a bounded window ahead of SGLang's safetensors loader.

    Each phase is one load (target, then draft) and follows the progress bar whose
    total equals its shard count (`| N/<total> [`, one line per shard). Nothing is
    read before a phase's first bar line, so the window never competes with the
    pinned PLE allocation that precedes the load. Shards are submitted in loader
    order while the bytes submitted and not yet evicted fit the window; the shard
    being loaded is always submitted. A passed shard has its read stopped and,
    once the read ends, its pages evicted, so re-read pages cannot push the
    window's unread pages out. Writes a per-shard timing report when done.
    """

    def __init__(self, phases: list[tuple[str, list[Path]]], server_log, report_path, *,
                 log_offset=0, threads=4, window_bytes=8 * 1024**3, block_bytes=16 * 1024**2,
                 poll_seconds=0.5, deadline_seconds=1800.0):
        super().__init__(name="shard-window-prefetch", daemon=True)
        totals = [len(shards) for _, shards in phases]
        if len(set(totals)) != len(totals):
            raise ValueError(f"Phases need distinct shard counts to tell their progress bars apart: {totals}")
        self.phases = phases
        self.server_log = Path(server_log)
        self.report_path = Path(report_path)
        self.log_offset = log_offset
        self.threads = threads
        self.window_bytes = window_bytes
        self.block_bytes = block_bytes
        self.poll_seconds = poll_seconds
        self.deadline_seconds = deadline_seconds
        self.abort = threading.Event()
        self.report: dict = {"threads": threads, "window_gib": window_bytes / 1024**3, "phases": []}

    def run(self) -> None:
        start = time.monotonic()
        while not self.server_log.exists():
            if self.abort.wait(self.poll_seconds):
                return
        with open(self.server_log, encoding="utf-8", errors="replace") as log, \
                concurrent.futures.ThreadPoolExecutor(self.threads, thread_name_prefix="shard-prefetch") as pool:
            log.seek(self.log_offset)
            pending = ""
            for label, shards in self.phases:
                state = _PhaseState(label, shards, self)
                while not state.done() and not self.abort.is_set():
                    if time.monotonic() - start > self.deadline_seconds:
                        self.abort.set()
                        break
                    complete, _, pending = (pending + log.read()).rpartition("\n")
                    state.observe(complete)
                    state.step(pool)
                    if not state.done():
                        self.abort.wait(self.poll_seconds)
                state.finish()
                self.report["phases"].append(state.describe())
                if self.abort.is_set():
                    break
        self.report["aborted"] = self.abort.is_set()
        self.report_path.write_text(json.dumps(self.report, indent=1))

    def stop(self) -> None:
        self.abort.set()


class _PhaseState:
    def __init__(self, label: str, shards: list[Path], owner: ShardWindowPrefetcher):
        self.label, self.shards, self.owner = label, shards, owner
        self.sizes = [p.stat().st_size for p in shards]
        self.pattern = re.compile(rf"\| (\d+)/{len(shards)} \[")
        self.count = -1
        self.reached_at: dict[int, float] = {}
        self.futures: dict[int, concurrent.futures.Future] = {}
        self.stops: dict[int, threading.Event] = {}
        self.submitted = self.dropped = 0

    def done(self) -> bool:
        return self.count >= len(self.shards) and self.dropped >= len(self.shards)

    def observe(self, text: str) -> None:
        for match in self.pattern.finditer(text):
            self.count = max(self.count, int(match.group(1)))
        now = time.monotonic()
        for n in range(len(self.reached_at), self.count + 1):
            self.reached_at[n] = now

    def step(self, pool) -> None:
        if self.count < 0:
            return
        passed = min(self.count, len(self.shards))
        for i in range(self.dropped, passed):
            if i in self.stops:
                self.stops[i].set()
                self.futures[i].cancel()
        while self.dropped < passed:
            future = self.futures.get(self.dropped)
            if future is not None and not future.done():
                break
            drop_file_from_page_cache(self.shards[self.dropped])
            self.dropped += 1
        self.submitted = max(self.submitted, passed)
        ahead = sum(self.sizes[self.dropped:self.submitted])
        while self.submitted < len(self.shards) and (
            self.submitted == passed or ahead + self.sizes[self.submitted] <= self.owner.window_bytes
        ):
            i = self.submitted
            self.stops[i] = threading.Event()
            self.futures[i] = pool.submit(read_file_into_page_cache, self.shards[i], self.stops[i], self.owner.block_bytes)
            ahead += self.sizes[i]
            self.submitted += 1

    def finish(self) -> None:
        for event in self.stops.values():
            event.set()

    def describe(self) -> dict:
        t0 = self.reached_at.get(0)
        rows = []
        for i, path in enumerate(self.shards):
            row = {"name": path.name, "gib": round(self.sizes[i] / 1024**3, 3)}
            if t0 is not None and i in self.reached_at:
                row["loader_reached"] = round(self.reached_at[i] - t0, 1)
            if t0 is not None and i + 1 in self.reached_at:
                row["loader_passed"] = round(self.reached_at[i + 1] - t0, 1)
            future = self.futures.get(i)
            if future is None or future.cancelled() or not future.done():
                row["prefetch"] = "not started" if future is None or future.cancelled() else "running"
            elif future.exception() is not None:
                row["prefetch"] = f"failed: {future.exception()!r}"
            else:
                read = future.result()
                row["prefetch"] = "stopped" if read["stopped"] else "complete"
                if t0 is not None:
                    row["prefetch_started"] = round(read["started_at"] - t0, 1)
                    row["prefetch_finished"] = round(read["finished_at"] - t0, 1)
            rows.append(row)
        return {
            "phase": self.label,
            "shards": len(self.shards),
            "gib": round(sum(self.sizes) / 1024**3, 2),
            "loader_seconds": round(self.reached_at[len(self.shards)] - t0, 1)
            if t0 is not None and len(self.shards) in self.reached_at else None,
            "complete_before_loader_reached": sum(
                r.get("prefetch") == "complete" and "loader_reached" in r
                and r["prefetch_finished"] <= r["loader_reached"] for r in rows
            ),
            "rows": rows,
        }

