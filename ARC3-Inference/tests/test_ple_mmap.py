"""inference/rl/ple_mmap.py serves the PLE n-gram table from the checkpoint files with identical results.

A tiny qwen4_exp saved in the real checkpoint's shard layout is loaded twice: normally,
and with the table memory-mapped (transformers then skips the shards without reading
them). Logits and gradients match, and the mapped model holds no table parameter.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")

from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig  # noqa: E402

from inference.rl import ple_mmap  # noqa: E402
from test_qsa import TEXT, VISION  # noqa: E402


def test_mapped_table_matches_the_loaded_table(tmp_path):
    torch.manual_seed(0)
    config = Qwen4ExpConfig(text_config=TEXT, vision_config=VISION)
    AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16).save_pretrained(tmp_path)
    ids = torch.randint(0, 1000, (1, 64))

    def run(model):
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        logits.pow(2).mean().backward()
        return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    reference = AutoModelForImageTextToText.from_pretrained(tmp_path, dtype=torch.bfloat16)
    ref_logits, ref_grads = run(reference)
    assert any("ngram_embedding.weight" in n for n, _ in reference.named_parameters())

    ple_mmap.install()
    try:
        mapped = AutoModelForImageTextToText.from_pretrained(tmp_path, dtype=torch.bfloat16)
        info = ple_mmap.attach(mapped, tmp_path)
        logits, grads = run(mapped)
    finally:
        ple_mmap.uninstall()
    assert info["tables"] == 1 and info["shards"] == TEXT["split_ngram_parts"]
    assert not any("ngram_embedding" in n for n, _ in mapped.named_parameters())
    torch.testing.assert_close(logits, ref_logits, rtol=0, atol=0)
    # the table is frozen in training; the loaded model only differs by also giving it a gradient
    ref_grads = {n: g for n, g in ref_grads.items() if "ngram_embedding" not in n}
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=0, atol=0, msg=name)


def test_fp8_rows_round_each_row_to_e4m3_with_its_own_scale():
    g = torch.Generator().manual_seed(0)
    rows = (torch.randn(64, 256, generator=g) * torch.logspace(-3, 1, 64)[:, None]).to(torch.bfloat16)
    q = ple_mmap.fp8_rows(rows)
    assert q.dtype == rows.dtype and q.shape == rows.shape
    # relative error per row is set by E4M3's 3 mantissa bits (half-ulp 6.25%), whatever the row's magnitude
    rel = (q.float() - rows.float()).norm(dim=-1) / rows.float().norm(dim=-1)
    assert float(rel.max()) < 0.04 and float(rel.min()) > 0.005
    # the largest element of each row is exactly representable (it maps to 448)
    top = rows.float().abs().argmax(-1)
    torch.testing.assert_close(q.float().gather(1, top[:, None]), rows.float().gather(1, top[:, None]), rtol=1e-2, atol=0)
    with pytest.raises(ValueError):
        ple_mmap.set_fp8("tensor")


def test_nvidia_fp8_table_is_read_as_fp8_times_its_scale(tmp_path):
    """attach(fp8_path=...) maps a table in nvidia's layout (F8_E4M3 shards, one BF16 weight_scale) beside the BF16
    one; set_fp8("nvidia") reads it, set_fp8(None) still reads the BF16 table unchanged."""
    from safetensors.torch import load_file, save_file

    torch.manual_seed(0)
    config = Qwen4ExpConfig(text_config=TEXT, vision_config=VISION)
    AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16).save_pretrained(tmp_path)
    shards = {}
    for f in tmp_path.glob("*.safetensors"):
        shards.update({k: v for k, v in load_file(f).items() if ple_mmap.SHARD_KEY in k})
    amax = max(float(v.float().abs().max()) for v in shards.values())
    scale = torch.tensor([amax / ple_mmap.FP8_MAX], dtype=torch.bfloat16)
    fp8 = {k: (v.float() / float(scale)).to(torch.float8_e4m3fn) for k, v in shards.items()}
    prefix = next(iter(fp8)).split(ple_mmap.SHARD_KEY)[0]
    fp8[prefix + "ngram_embedding.weight_scale"] = scale
    nv = tmp_path / "nvidia"
    nv.mkdir()
    save_file(fp8, nv / ple_mmap.FP8_FILE)

    ple_mmap.install()
    try:
        model = AutoModelForImageTextToText.from_pretrained(tmp_path, dtype=torch.bfloat16)
        info = ple_mmap.attach(model, tmp_path, fp8_path=nv)
    finally:
        ple_mmap.uninstall()
    table = next(m for m in model.modules() if isinstance(m, ple_mmap.MappedTable))
    assert info["fp8_shards"] == info["shards"] and info["fp8_scale"] == [float(scale)]
    ids = torch.randint(0, table.num_embeddings, (3, 40))
    flat = torch.cat([v for _, v in sorted(shards.items(), key=lambda kv: int(kv[0].split(ple_mmap.SHARD_KEY)[1].split(".")[0]))])
    try:
        ple_mmap.set_fp8(None)
        torch.testing.assert_close(table(ids), flat[ids], rtol=0, atol=0)
        ple_mmap.set_fp8("nvidia")
        got = table(ids)
        fp8_flat = torch.cat([fp8[k] for k in sorted((k for k in fp8 if ple_mmap.SHARD_KEY in k),
                                                    key=lambda k: int(k.split(ple_mmap.SHARD_KEY)[1].split(".")[0]))])
        torch.testing.assert_close(got, (fp8_flat[ids].float() * float(scale)).to(torch.bfloat16), rtol=0, atol=0)
        assert 0 < ple_mmap._FP8["last_rel_err"] < 0.1
    finally:
        ple_mmap.set_fp8(None)
