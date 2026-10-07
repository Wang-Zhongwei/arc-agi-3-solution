"""Checkpoint views and the shard prefetch window used by the Pennyroyal launcher."""

import json
import struct
import threading
import time

import pytest

from inference.framework import pennyroyal_views as views

PLE = "model.language_model.layers.2.ple.ple_embedding"


def write_shard(path, tensors):
    """A safetensors file whose header lists `tensors` ({name: (dtype, shape)})."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = 1
        for dim in shape:
            size *= dim
        nbytes = size * (1 if dtype == "F8_E4M3" else 2)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def write_checkpoint(root, shards, config):
    root.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for file_name, tensors in shards.items():
        write_shard(root / file_name, tensors)
        weight_map.update({name: file_name for name in tensors})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text("{}")
    return root


@pytest.fixture
def checkpoints(tmp_path):
    config = {"text_config": {"ple_embedding_dtype": "bfloat16"}, "quantization_config": {"quant_method": "compressed-tensors"}}
    target = write_checkpoint(tmp_path / "lord", {
        "carry-00001.safetensors": {
            f"{PLE}.layer_multipliers": ("BF16", [4]),
            "model.language_model.layers.0.mlp.gate.weight": ("BF16", [2, 2]),
        },
        "carry-00012.safetensors": {
            "lm_head.weight": ("BF16", [8, 2]),
            "model.language_model.embed_tokens.weight": ("BF16", [8, 2]),
            "mtp.fc_hidden.weight": ("BF16", [2, 2]),
        },
        "ct-experts-layer00.safetensors": {"model.language_model.layers.0.mlp.experts.0.up_proj.weight_packed": ("BF16", [2, 2])},
        "mtp-experts.safetensors": {"mtp.layers.0.mlp.experts.0.up_proj.weight_packed": ("BF16", [2, 2])},
        "ple-bf16-00.safetensors": {f"{PLE}.ngram_embedding.shard_{i}.weight": ("BF16", [3, 4]) for i in range(2)},
        "ple-bf16-01.safetensors": {f"{PLE}.ngram_embedding.shard_{i}.weight": ("BF16", [3, 4]) for i in range(2, 4)},
    }, config)
    fp8 = {f"{PLE}.ngram_embedding.shard_{i}.weight": ("F8_E4M3", [3, 4]) for i in range(4)}
    fp8[f"{PLE}.ngram_embedding.weight_scale"] = ("BF16", [1])
    radix = write_checkpoint(tmp_path / "radix", {
        "model-plefp8-00000.safetensors": fp8,
        "model-bf16-00001.safetensors": {"model.language_model.layers.0.mlp.gate.weight": ("BF16", [2, 2])},
    }, config)
    return target, radix, tmp_path / "work"


def test_target_view_swaps_only_the_ple_table(checkpoints):
    target, radix, work = checkpoints
    view = views.build_target_view(target, radix, work)
    weight_map = views.read_index(view)["weight_map"]
    assert weight_map[f"{PLE}.ngram_embedding.shard_3.weight"] == "model-plefp8-00000.safetensors"
    assert weight_map[f"{PLE}.ngram_embedding.weight_scale"] == "model-plefp8-00000.safetensors"
    assert weight_map[f"{PLE}.layer_multipliers"] == "carry-00001.safetensors"
    assert weight_map["model.language_model.layers.0.mlp.gate.weight"] == "carry-00001.safetensors"
    assert not (view / "ple-bf16-00.safetensors").exists()
    assert (view / "model-plefp8-00000.safetensors").resolve() == (radix / "model-plefp8-00000.safetensors").resolve()
    assert not (view / "model-bf16-00001.safetensors").exists()
    assert (view / "tokenizer.json").is_symlink()
    assert json.loads((view / "config.json").read_text())["text_config"]["ple_embedding_dtype"] == "float8_e4m3fn"
    assert views.build_target_view(target, radix, work) == view


def test_target_view_rejects_mismatched_ple_shapes(checkpoints, tmp_path):
    target, _, work = checkpoints
    bad = {f"{PLE}.ngram_embedding.shard_{i}.weight": ("F8_E4M3", [3, 5]) for i in range(4)}
    radix = write_checkpoint(tmp_path / "bad", {"model-plefp8-00000.safetensors": bad}, {})
    with pytest.raises(RuntimeError, match="shape mismatch"):
        views.build_target_view(target, radix, work)


def test_target_view_rejects_non_fp8_table(checkpoints, tmp_path):
    target, _, work = checkpoints
    bf16 = {f"{PLE}.ngram_embedding.shard_{i}.weight": ("BF16", [3, 4]) for i in range(4)}
    radix = write_checkpoint(tmp_path / "bf16", {"model-plefp8-00000.safetensors": bf16}, {})
    with pytest.raises(RuntimeError, match="expected F8_E4M3"):
        views.build_target_view(target, radix, work)


def test_ple_dtype_view_keeps_every_shard_and_rewrites_only_the_dtype(checkpoints):
    target, _, work = checkpoints
    config = json.loads((target / "config.json").read_text())
    config["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"  # lordhansolo's hint
    (target / "config.json").write_text(json.dumps(config))
    view = views.build_ple_dtype_view(target, work, "bfloat16")
    assert views.read_index(view)["weight_map"] == views.read_index(target)["weight_map"]
    for shard in set(views.read_index(target)["weight_map"].values()):
        assert (view / shard).resolve() == (target / shard).resolve()
    served = json.loads((view / "config.json").read_text())
    assert served["text_config"]["ple_embedding_dtype"] == "bfloat16"
    assert served["quantization_config"] == config["quantization_config"]
    assert views.build_ple_dtype_view(target, work, "bfloat16") == view


def test_ple_dtype_view_refuses_bfloat16_for_fp8_shards(checkpoints):
    _, radix, work = checkpoints
    with pytest.raises(RuntimeError, match="not BF16"):
        views.build_ple_dtype_view(radix, work, "bfloat16")


def test_draft_view_keeps_mtp_embedding_and_head_shards(checkpoints):
    target, radix, work = checkpoints
    view = views.build_mtp_draft_view(views.build_target_view(target, radix, work), work)
    files = set(views.read_index(view)["weight_map"].values())
    assert files == {"carry-00012.safetensors", "mtp-experts.safetensors"}
    assert views.has_embedded_mtp(target)
    assert {p.name for p in views.loader_shard_order(view)} == files


def test_loader_order_follows_glob_filtered_by_index(checkpoints, monkeypatch):
    target, radix, work = checkpoints
    view = views.build_target_view(target, radix, work)
    names = ["ple-bf16-00.safetensors", "model-plefp8-00000.safetensors", "carry-00001.safetensors"]
    monkeypatch.setattr(views.glob, "glob", lambda pattern: [str(view / n) for n in names])
    assert [p.name for p in views.loader_shard_order(view)] == names[1:]


def test_prefetcher_follows_each_phase_progress_bar(tmp_path):
    shards = []
    for i in range(3):
        shard = tmp_path / f"s{i}.safetensors"
        shard.write_bytes(b"x" * 1024)
        shards.append(shard)
    log = tmp_path / "serve.log"
    log.write_text("previous run | 2/3 [\n")
    offset = log.stat().st_size
    report = tmp_path / "report.json"
    prefetcher = views.ShardWindowPrefetcher(
        [("target", shards), ("draft", shards[:1])], log, report,
        log_offset=offset, window_bytes=2048, block_bytes=256, poll_seconds=0.01, deadline_seconds=30)
    prefetcher.start()
    bar = "Loading safetensors checkpoint shards: {pct}% Completed | {n}/{total} [00:01<00:00]\n"
    with log.open("a") as f:
        for n in range(4):
            f.write(bar.format(pct=n * 33, n=n, total=3))
            f.flush()
            time.sleep(0.05)
        for n in range(2):
            f.write(bar.format(pct=n * 100, n=n, total=1))
            f.flush()
            time.sleep(0.05)
    prefetcher.join(timeout=10)
    assert not prefetcher.is_alive()
    data = json.loads(report.read_text())
    assert data["aborted"] is False
    target, draft = data["phases"]
    assert [r["name"] for r in target["rows"]] == ["s0.safetensors", "s1.safetensors", "s2.safetensors"]
    assert all(r["prefetch"] in ("complete", "stopped") for r in target["rows"] + draft["rows"])
    assert target["loader_seconds"] is not None and draft["shards"] == 1


def test_prefetcher_stops_without_progress(tmp_path):
    shard = tmp_path / "s0.safetensors"
    shard.write_bytes(b"x")
    log = tmp_path / "serve.log"
    log.write_text("")
    report = tmp_path / "report.json"
    prefetcher = views.ShardWindowPrefetcher([("target", [shard])], log, report, poll_seconds=0.01)
    prefetcher.start()
    threading.Timer(0.1, prefetcher.stop).start()
    prefetcher.join(timeout=5)
    data = json.loads(report.read_text())
    assert data["aborted"] is True
    assert data["phases"][0]["rows"][0]["prefetch"] == "not started"


def test_prefetcher_requires_distinct_phase_sizes(tmp_path):
    with pytest.raises(ValueError):
        views.ShardWindowPrefetcher([("a", [tmp_path]), ("b", [tmp_path])], tmp_path / "log", tmp_path / "r")
