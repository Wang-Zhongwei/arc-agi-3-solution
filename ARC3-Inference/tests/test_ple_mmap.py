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
