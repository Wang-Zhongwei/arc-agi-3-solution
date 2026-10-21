"""Offload the tensors saved for backward into host buffers that are reused across steps.

`torch.autograd.graph.save_on_cpu(pin_memory=False)` allocates a fresh host tensor for every
saved tensor, so every step writes into new pages. On the US-NC-1 secure H200 the first-touch
page faults limited GPU -> host copies to 1.7 GiB/s against 16.5 GiB/s into a reused buffer:
~35 s of a 58 s forward at 64K tokens. `HostArena` keeps large host segments allocated
between steps and hands them out bump-style; `offload_to_host` packs into them and also

- offloads a tensor saved twice (the same storage, offset, shape and version) only once;
- keeps the last `keep` bytes of a step on the GPU when there is room: backward needs them
  first, and the earlier ones have time to come back;
- with `async_copies` and pinned segments, copies on a side stream: the forward does not wait
  for its device -> host copies, and backward prefetches the tensors it will need next.

Segments are pinned only as whole 16 GiB blocks allocated once. Pinning every saved tensor
separately got the container OOM-killed: the pinned allocator rounds each block up to a power
of two and caches it (see AGENTS.md).

`max_gb` caps the arena (`reserve()` allocates it all up front); what does not fit stays on the
device. Uncapped, the first training run's arena grew to ~157 GiB on a 126K-token sample and never
shrank, which pushed the memory-mapped PLE table (102.5 GB) out of a 251 GB pod's page cache: every
step then re-read it from disk while the GPU waited.
"""

from __future__ import annotations

import contextlib
import time
import weakref

import torch

ALIGN = 64  # bytes; every view starts aligned for any dtype
MIN_BYTES = 1 << 20  # smaller saved tensors stay where they are


class HostArena:
    """Bump allocator over host segments that stay allocated between steps.

    `reset()` at the start of a step makes all segments free again, so a step must not
    run while tensors packed by the previous one are still needed."""

    def __init__(self, segment_gb: float = 16, pin: bool = False, max_gb: float | None = None):
        self.segment_bytes = int(segment_gb * 2**30)
        self.pin = pin
        self.max_bytes = None if max_gb is None else int(max_gb * 2**30)
        self.segments: list[torch.Tensor] = []
        self.stream = None  # side stream of async copies, shared by all steps
        self.bytes_per_token: dict[str, float] = {}  # saved bytes per token, per layout of saved tensors
        self.stats: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()  # copies of a step that failed half-way may still be running
        self.segment, self.offset, self.used = 0, 0, 0

    @property
    def capacity_gb(self) -> float:
        return sum(s.numel() for s in self.segments) / 2**30

    @property
    def pinned_gb(self) -> float:
        return sum(s.numel() for s in self.segments if s.is_pinned()) / 2**30

    def reserve(self) -> float:
        """Allocate the capped arena's segments now (pinning costs ~0.7 s per GiB, once); returns seconds."""
        start = time.perf_counter()
        while self.max_bytes is not None and self.capacity_gb * 2**30 < self.max_bytes:
            size = min(self.segment_bytes, self.max_bytes - int(self.capacity_gb * 2**30))
            self.segments.append(self._new_segment(size))
        return time.perf_counter() - start

    def _new_segment(self, size: int) -> torch.Tensor:
        segment = None
        if self.pin and torch.cuda.is_available():
            with contextlib.suppress(RuntimeError):  # pinning can fail (memory lock limits): fall back
                segment = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        if segment is None:
            segment = torch.empty(size, dtype=torch.uint8)
            segment.zero_()  # fault the pages in now, on all CPU threads, rather than inside a GPU copy
        return segment

    def take(self, nbytes: int) -> torch.Tensor | None:
        """A uint8 host view of `nbytes`, from the current segment, the next one that fits, or a new one;
        None when a new one would pass `max_gb`."""
        while self.segment < len(self.segments):
            segment = self.segments[self.segment]
            if self.offset + nbytes <= segment.numel():
                view = segment[self.offset:self.offset + nbytes]
                self.offset += -(-nbytes // ALIGN) * ALIGN
                self.used += nbytes
                return view
            self.segment, self.offset = self.segment + 1, 0
        size = max(self.segment_bytes, nbytes)
        if self.max_bytes is not None:
            size = max(min(size, self.max_bytes - int(self.capacity_gb * 2**30)), 0)
            if size < nbytes:
                return None
        self.segments.append(self._new_segment(size))
        self.segment, self.offset = len(self.segments) - 1, 0
        return self.take(nbytes)


def _dense_strides(tensor: torch.Tensor) -> tuple[int, ...] | None:
    """The strides of `tensor` if its elements fill tensor.numel() slots without gaps or overlaps
    (any dimension order), else None. Such tensors keep their layout on the host, as .cpu() does:
    a contiguous copy would make backward kernels take another path and round differently."""
    expected = 1
    for stride, size in sorted((st, n) for st, n in zip(tensor.stride(), tensor.shape) if n != 1):
        if stride != expected:
            return None
        expected *= size
    return tuple(tensor.stride())


class _Saved:
    """One offloaded tensor: its host copy, the event of its last copy, and a device copy once fetched."""
    __slots__ = ("device", "host", "event", "device_copy", "uses", "index")


@contextlib.contextmanager
def offload_to_host(arena: HostArena, keep_bytes: float = 0, async_copies: bool = False, prefetch: int = 2,
                    layout: str = "", tokens: int | None = None, include_cpu: bool = False,
                    min_bytes: int = MIN_BYTES):
    """Like save_on_cpu(pin_memory=False), but packs into `arena`, which is reset on entry.

    `keep_bytes` of the step's last saved tensors stay on the device; this needs the step's total,
    learnt per `layout` from an earlier step with the same `tokens` scaling (the first step of a
    layout offloads everything). Host tensors, parameters and tensors under `min_bytes` are kept
    as they are; `include_cpu` copies host tensors too (tests)."""
    arena.reset()
    stats = arena.stats = dict(d2h_gb=0.0, h2d_gb=0.0, d2h_sync_s=0.0, h2d_sync_s=0.0, kept_gb=0.0, overflow_gb=0.0,
                               deduplicated=0)
    known = arena.bytes_per_token.get(layout)
    budget = max(known * tokens - keep_bytes, 0) if known is not None and tokens and keep_bytes else float("inf")
    use_stream = async_copies and arena.pin and torch.cuda.is_available()
    if use_stream and arena.stream is None:
        arena.stream = torch.cuda.Stream()
    stream = arena.stream if use_stream else None
    records: list[_Saved] = []
    by_key: dict[tuple, tuple[weakref.ref, _Saved | None]] = {}
    totals = dict(offloaded=0, unique=0)

    def fetch(record: _Saved) -> None:
        if record.device.type == "cpu":
            record.device_copy = record.host
        elif stream is not None and record.host.is_pinned():
            device_copy = torch.empty(record.host.shape, dtype=record.host.dtype, device=record.device)
            stream.wait_stream(torch.cuda.current_stream(record.device))
            with torch.cuda.stream(stream):
                device_copy.copy_(record.host, non_blocking=True)
            device_copy.record_stream(stream)
            record.event = stream.record_event()
            record.device_copy = device_copy
        else:
            start = time.perf_counter()
            record.device_copy = record.host.to(record.device)
            stats["h2d_sync_s"] += time.perf_counter() - start
            record.event = None
        stats["h2d_gb"] += record.host.numel() * record.host.element_size() / 2**30

    def pack(tensor):
        nbytes = tensor.numel() * tensor.element_size()
        if ((tensor.device.type == "cpu" and not include_cpu) or tensor.layout != torch.strided
                or nbytes < min_bytes or isinstance(tensor, torch.nn.Parameter)):
            return tensor
        # the same view of the same live tensor, saved again (e.g. by two checkpoints): one copy. Keyed
        # by the base tensor object, held weakly: once it is freed its memory can hold a new tensor
        # with the same address, shape and version
        base = tensor if tensor._base is None else tensor._base
        key = (id(base), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
               tensor._version)
        hit = by_key.get(key)
        if hit is not None and hit[0]() is base:
            stats["deduplicated"] += 1
            record = hit[1]
            if record is None:  # kept on the device the first time
                return tensor
            record.uses += 1
            return record
        totals["unique"] += nbytes
        if totals["offloaded"] + nbytes > budget:  # the rest of the step stays on the device
            stats["kept_gb"] += nbytes / 2**30
            by_key[key] = (weakref.ref(base), None)  # counted once, like an offloaded tensor
            return tensor
        flat = arena.take(nbytes)
        if flat is None:  # the capped arena is full: the rest of the step stays on the device
            stats["overflow_gb"] += nbytes / 2**30
            by_key[key] = (weakref.ref(base), None)
            return tensor
        flat = flat.view(tensor.dtype)
        record = _Saved()
        record.device, record.uses, record.index, record.device_copy = tensor.device, 1, len(records), None
        strides = _dense_strides(tensor)
        record.host = flat.as_strided(tensor.shape, strides) if strides else flat.view(tensor.shape)
        if stream is not None and tensor.is_cuda and record.host.is_pinned():
            stream.wait_stream(torch.cuda.current_stream(tensor.device))
            with torch.cuda.stream(stream):
                record.host.copy_(tensor, non_blocking=True)
            tensor.record_stream(stream)  # its memory is not reused before the copy is done
            record.event = stream.record_event()
        else:
            start = time.perf_counter()
            record.host.copy_(tensor)
            stats["d2h_sync_s"] += time.perf_counter() - start
            record.event = None
        stats["d2h_gb"] += nbytes / 2**30
        totals["offloaded"] += nbytes
        records.append(record)
        by_key[key] = (weakref.ref(base), record)
        return record

    def unpack(packed):
        if isinstance(packed, torch.Tensor):
            return packed
        record = packed
        if record.device_copy is None:
            fetch(record)
        if record.event is not None:
            torch.cuda.current_stream(record.device).wait_event(record.event)
        tensor = record.device_copy
        record.uses -= 1
        if record.uses <= 0:
            record.device_copy = None
        # backward unpacks roughly in reverse pack order: bring the previous ones in now
        for other in records[max(record.index - prefetch, 0):record.index][::-1]:
            if other.device_copy is None and other.uses > 0:
                fetch(other)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield arena
    if tokens:
        arena.bytes_per_token[layout] = totals["unique"] / tokens
