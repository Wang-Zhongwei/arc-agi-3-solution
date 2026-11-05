"""One line a minute with the CPU KV tier's counters, printed by the scheduler process. It only reads the manager."""
import dataclasses
import time

REPORT_SECONDS = 60

_loads = 0
_load_blocks = 0
_store_blocks = 0
_load_started: dict[str, float] = {}
_ready_ms: list[float] = []
_last_report = time.monotonic()


def on_step(manager, load_gpu_blocks: list, load_req_ids: list, store_gpu_blocks: list) -> None:
    global _loads, _load_blocks, _store_blocks, _last_report
    now = time.monotonic()
    _loads += len(load_req_ids)
    _load_blocks += len(load_gpu_blocks)
    _store_blocks += len(store_gpu_blocks)
    for req_id in load_req_ids:
        _load_started[req_id] = now
    if now - _last_report < REPORT_SECONDS:
        return
    _last_report = now
    ready = sorted(_ready_ms)
    _ready_ms.clear()
    p50 = ready[len(ready) // 2] if ready else 0.0
    p95 = ready[min(len(ready) - 1, int(0.95 * len(ready)))] if ready else 0.0
    print(
        f'KV_OFFLOAD_STATS loads={_loads} load_blocks={_load_blocks} stored_blocks={_store_blocks} '
        f'free_cpu_blocks={manager.cpu_block_pool.get_num_free_blocks()} '
        f'load_ready_ms_p50={p50:.0f} load_ready_ms_p95={p95:.0f} load_ready_n={len(ready)} '
        f'boundary={dataclasses.asdict(manager.boundary_store_stats)}',
        flush=True,
    )


def on_load_done(req_id: str) -> None:
    started = _load_started.pop(req_id, None)
    if started is not None:
        _ready_ms.append(1000 * (time.monotonic() - started))
