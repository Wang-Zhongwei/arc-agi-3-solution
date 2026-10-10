"""Check bounded parallel reads and cleanup without loading a GPU runtime."""

import io
import json
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from inference.framework.lordhansolo import kaggle as lord


@pytest.fixture
def watchdog(tmp_path, monkeypatch):
    config = tmp_path / "watchdog.json"
    config.write_text(json.dumps({
        "command": [], "server_log": str(tmp_path / "server.log"),
        "server_pid": str(tmp_path / "server.pid"), "server_ready": str(tmp_path / "ready"),
        "prefetch_threads": 8,
    }))
    monkeypatch.setattr("sys.argv", ["watchdog.py", str(config)])
    namespace = {}
    exec(lord._DUCK_VLLM_WATCHDOG_SCRIPT.split("if len(sys.argv) > 2", 1)[0], namespace)
    namespace["GPU_SHARD_PREFETCH_BLOCK_BYTES"] = 16
    return namespace


def prefetcher(watchdog, tmp_path, names, sizes):
    shards = []
    for name, size in zip(names, sizes, strict=True):
        path = tmp_path / name
        path.write_bytes(bytes(size))
        shards.append(path)
    return watchdog["GpuShardPrefetcher"](shards, tmp_path / "server.log", tmp_path / "report.json")


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        future = Future()
        self.calls.append((function, args, future))
        return future


@pytest.mark.parametrize("size", [0, 1, 16, 17, 127, 128, 129, 151])
def test_ranges_cover_a_shard_once_including_short_tail(watchdog, tmp_path, size):
    reader = prefetcher(watchdog, tmp_path, ["model-1.safetensors"], [size])
    executor = RecordingExecutor()
    reader._submit_window(executor, 0)
    ranges = [(args[-2], args[-1]) for _, args, _ in executor.calls]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == size
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert len(ranges) <= 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [pool.submit(function, *args) for function, args, _ in executor.calls]
        assert all(not future.result()["stopped"] for future in results)


def test_read_stays_in_its_range_and_honors_cancellation(watchdog):
    class TrackingBuffer(io.BytesIO):
        def __exit__(self, *args):
            pass

    buffer = TrackingBuffer(bytes(64))
    path = SimpleNamespace(open=lambda *args, **kwargs: buffer)
    stop = watchdog["threading"].Event()
    assert not watchdog["read_file_into_page_cache"](path, stop, 3, 20)["stopped"]
    assert buffer.tell() == 20
    stop.set()
    assert watchdog["read_file_into_page_cache"](path, stop, 3, 20)["stopped"]
    assert buffer.tell() == 3
    stop.clear()
    watchdog["GPU_SHARD_PREFETCH_ABORT"].set()
    assert watchdog["read_file_into_page_cache"](path, stop, 3, 20)["stopped"]
    watchdog["GPU_SHARD_PREFETCH_ABORT"].clear()
    with pytest.raises(OSError, match="shrank"):
        watchdog["read_file_into_page_cache"](path, stop, 60, 70)


def test_window_respects_loader_order_and_both_ple_names(watchdog, tmp_path):
    watchdog["drop_file_from_page_cache"] = lambda path: None
    reader = prefetcher(watchdog, tmp_path, [
        "model-1.safetensors", "model-2.safetensors", "ple-1.safetensors",
        "model-ple-1.safetensors", "model_extra_tensors.safetensors",
    ], [32] * 5)
    executor = RecordingExecutor()
    reader._submit_window(executor, 0)
    assert set(reader._futures) == {0, 1}
    reader._stop_passed_shards(2)
    reader._drop_passed_shards()
    reader._submit_window(executor, 2)
    assert set(reader._futures) == {0, 1}  # Do not warm extra tensors across the PLE gap.
    reader._submit_window(executor, 3)
    assert set(reader._futures) == {0, 1, 4}
    assert reader._gpu_shard_indices == [0, 1, 4]


def test_byte_window_limits_admission(watchdog, tmp_path):
    watchdog["GPU_SHARD_PREFETCH_WINDOW_BYTES"] = 48
    reader = prefetcher(watchdog, tmp_path, ["model-1", "model-2"], [32, 32])
    reader._submit_window(RecordingExecutor(), 0)
    assert set(reader._futures) == {0}


def test_passed_shard_waits_for_every_reader_before_eviction(watchdog, tmp_path):
    reader = prefetcher(watchdog, tmp_path, ["model-1"], [32])
    executor = RecordingExecutor()
    reader._submit_window(executor, 0)
    running, queued = reader._futures[0]
    running.set_running_or_notify_cancel()
    dropped = []
    watchdog["drop_file_from_page_cache"] = dropped.append
    reader._record_progress(1)
    reader._stop_passed_shards(1)
    assert queued.cancelled()
    assert reader._stop_events[0].is_set()
    reader._drop_passed_shards()
    assert dropped == []
    running.set_result({"started_at": 0, "finished_at": 1, "stopped": True})
    reader._drop_passed_shards()
    assert dropped == reader._shards
    assert reader._describe_shard(0, 0)["prefetch"] == "stopped"


def test_report_aggregates_ranges_and_records_failures(watchdog, tmp_path):
    reader = prefetcher(watchdog, tmp_path, ["model-1"], [32])
    reader._submit_window(RecordingExecutor(), 0)
    reader._progress_reached_at = {0: 5, 1: 10}
    futures = reader._futures[0]
    futures[0].set_result({"started_at": 5, "finished_at": 7, "stopped": False})
    futures[1].set_result({"started_at": 6, "finished_at": 8, "stopped": False})
    reader._write_report()
    report = json.loads(reader._report_path.read_text())
    assert report["reader_mode"] == "parallel_ranges"
    assert report["shards"][0]["prefetch_started"] == 0
    assert report["shards"][0]["prefetch_finished"] == 3
    failed = Future()
    failed.set_exception(OSError("unreadable"))
    reader._futures[0] = [failed]
    assert "unreadable" in reader._describe_shard(0, 5)["prefetch"]


def test_progress_uses_only_this_start_and_handles_partial_lines(watchdog):
    log = io.StringIO("| 38/38 [previous load]\n| 0/38 [current load]\n| 1/38 [partial")
    log.seek(len("| 38/38 [previous load]\n"))
    follower = watchdog["LoaderProgressFollower"](log, 38)
    assert follower.update_loaded_shard_count() == 0
    position = log.tell()
    log.write("]\n| 2/2 [draft]\n")
    log.seek(position)
    assert follower.update_loaded_shard_count() == 1


def test_prefetch_setting_reaches_rendered_watchdog(monkeypatch, tmp_path):
    monkeypatch.setenv("TAAF_KAGGLE_WORKING_DIR", str(tmp_path))
    script = lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(prefetch_threads=3))
    source = script.split("\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
    namespace = {}
    exec(source.split("print(f'vLLM runtime dataset path:", 1)[0], namespace)
    assert namespace["build_watchdog_config"]([])["prefetch_threads"] == 3
    with pytest.raises(ValueError, match="prefetch_threads"):
        lord.duck_kaggle_setup_command(lord.DuckKaggleVllmConfig(prefetch_threads=0))
