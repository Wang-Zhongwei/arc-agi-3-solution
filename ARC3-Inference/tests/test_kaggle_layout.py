"""inference/rl/kaggle_layout.py: a partial checkpoint (base index, no PLE) by default; --with-ple swaps in the base's checked PLE files."""

import json
import shutil

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file  # noqa: E402

from inference.rl import kaggle_layout  # noqa: E402

PLE = [f"model.language_model.layers.{i}.ple.ple_embedding.ngram_embedding.shard_0.weight" for i in range(3)]


def setup(tmp_path):
    ckpt, remote = tmp_path / "ckpt", tmp_path / "remote"
    ckpt.mkdir(), remote.mkdir()
    save_file({"a.weight": torch.ones(4)}, ckpt / "model-00001-of-00002.safetensors")
    ple = {k: torch.full((8,), float(i)) for i, k in enumerate(PLE)}
    save_file(ple, ckpt / "model-00002-of-00002.safetensors")
    ours = {"a.weight": "model-00001-of-00002.safetensors", **{k: "model-00002-of-00002.safetensors" for k in PLE}}
    (ckpt / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": ours}))
    shards = [("model-ple-00001-of-00002.safetensors", PLE[:2]), ("model-ple-00002-of-00002.safetensors", PLE[2:])]
    for name, keys in shards:
        save_file({k: ple[k] for k in keys}, remote / name)
    manifest = dict(source_file="model-00002-of-00002.safetensors",
                    source_size_bytes=(ckpt / "model-00002-of-00002.safetensors").stat().st_size,
                    shards=[dict(file=n, size_bytes=(remote / n).stat().st_size,
                                 sha256=kaggle_layout.sha256(remote / n), tensors=k) for n, k in shards])
    (remote / kaggle_layout.MANIFEST).write_text(json.dumps(manifest))
    theirs = {**ours, **{k: n for n, keys in shards for k in keys}}
    (remote / kaggle_layout.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": theirs}))
    calls = []

    def fetch(handle, name):  # like kagglehub: a fresh copy in a cache folder
        calls.append(name)
        (tmp_path / "cache").mkdir(exist_ok=True)
        return shutil.copy(remote / name, tmp_path / "cache" / name) and tmp_path / "cache" / name

    return ckpt, remote, fetch, calls


def test_install_swaps_the_ple_shard(tmp_path):
    ckpt, remote, fetch, calls = setup(tmp_path)
    assert kaggle_layout.install(ckpt, fetch=fetch, ple=True) == dict(ple_files=2, fetched=2, partial=False)
    assert not (ckpt / "model-00002-of-00002.safetensors").exists()
    assert json.loads((ckpt / "model.safetensors.index.json").read_text()) == json.loads((remote / kaggle_layout.INDEX).read_text())
    got = {**load_file(ckpt / "model-ple-00001-of-00002.safetensors"), **load_file(ckpt / "model-ple-00002-of-00002.safetensors")}
    assert sorted(got) == PLE and float(got[PLE[2]][0]) == 2.0
    calls.clear()
    assert kaggle_layout.install(ckpt, fetch=fetch, ple=True)["fetched"] == 0  # a rerun keeps what is in place
    assert calls == [kaggle_layout.MANIFEST, kaggle_layout.INDEX]


def test_install_refuses_a_bad_download_or_another_layout(tmp_path):
    ckpt, remote, fetch, _ = setup(tmp_path)
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    index["weight_map"]["a.weight"] = "elsewhere.safetensors"
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="a.weight"):
        kaggle_layout.install(ckpt, fetch=fetch, ple=True)
    assert (ckpt / "model-00002-of-00002.safetensors").exists()  # nothing deleted before the check
    index["weight_map"]["a.weight"] = "model-00001-of-00002.safetensors"
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(index))
    save_file({PLE[2]: torch.zeros(8)}, remote / "model-ple-00002-of-00002.safetensors")  # corrupt remote file
    with pytest.raises(ValueError, match="sha256"):
        kaggle_layout.install(ckpt, fetch=fetch, ple=True)


def test_partial_by_default_fetches_no_ple(tmp_path):
    """No PLE download, no PLE in the upload folder: the HF PLE shard moves to ple_aside/, the base's index goes in."""
    ckpt, remote, fetch, calls = setup(tmp_path)
    assert kaggle_layout.install(ckpt, fetch=fetch) == dict(ple_files=2, fetched=0, partial=True)
    assert calls == [kaggle_layout.MANIFEST, kaggle_layout.INDEX]
    assert sorted(p.name for p in ckpt.glob("*.safetensors")) == ["model-00001-of-00002.safetensors"]
    assert (tmp_path / "ple_aside" / "model-00002-of-00002.safetensors").exists()
    assert json.loads((ckpt / "model.safetensors.index.json").read_text()) == json.loads((remote / kaggle_layout.INDEX).read_text())
    assert kaggle_layout.install(ckpt, fetch=fetch)["partial"]  # a rerun (base index already in place) passes
    (ckpt / "model-00001-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="non-PLE"):
        kaggle_layout.install(ckpt, fetch=fetch)
