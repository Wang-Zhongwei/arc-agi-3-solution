"""inference/rl/merge_lora.py: a merged checkpoint gives what base + adapter gives, and only adapted tensors change."""

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
pytest.importorskip("transformers.models.qwen4_exp")

from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402

from inference.rl import merge_lora, smoke_lora  # noqa: E402
from test_layer_patches import tiny  # noqa: E402


def test_merged_checkpoint_matches_base_plus_adapter(tmp_path):
    base = tiny().to(torch.bfloat16)
    base.save_pretrained(tmp_path / "ckpt", max_shard_size="200KB")  # several shards
    before = {}
    for f in (tmp_path / "ckpt").glob("*.safetensors"):
        before.update(load_file(f))
    model = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=smoke_lora.ATTENTION_GDN))
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)  # large enough to survive BF16 rounding
    model.save_pretrained(tmp_path / "adapter")
    modules = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    result = merge_lora.merge(tmp_path / "ckpt", tmp_path / "adapter")
    assert result["tensors"] == modules > 10 and result["kept_min"] > 0.9 and result["cosine_min"] > 0.99
    after = {}
    for f in (tmp_path / "ckpt").glob("*.safetensors"):
        after.update(load_file(f))
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed == {s["key"] for s in result["per_tensor"]}  # nothing else touched
    assert all(".self_attn." in k or ".linear_attn." in k for k in changed)
    # the standard answer: peft's own merge of the same adapter
    reference = model.merge_and_unload().state_dict()
    for key in changed:
        torch.testing.assert_close(after[key].float(), reference[key].float(), rtol=1e-2, atol=1e-2, msg=key)
    merged = AutoModelForImageTextToText.from_pretrained(tmp_path / "ckpt", dtype=torch.bfloat16)
    ids = torch.randint(0, 1000, (1, 40), generator=torch.Generator().manual_seed(1))
    peft_merged = tiny().to(torch.bfloat16)
    peft_merged.load_state_dict(reference)
    with torch.no_grad():
        got = merged(input_ids=ids, use_cache=False).logits.float()
        want = peft_merged(input_ids=ids, use_cache=False).logits.float()
    assert float((got - want).norm() / want.norm()) < 1e-2
    assert not list((tmp_path / "ckpt").glob("*.merging"))


def test_dry_run_writes_nothing(tmp_path):
    base = tiny().to(torch.bfloat16)
    base.save_pretrained(tmp_path / "ckpt")
    model = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=smoke_lora.ATTENTION_GDN))
    model.save_pretrained(tmp_path / "adapter")
    stamp = {f.name: f.stat().st_mtime_ns for f in (tmp_path / "ckpt").iterdir()}
    result = merge_lora.merge(tmp_path / "ckpt", tmp_path / "adapter", dry_run=True)
    assert result["tensors"] == sum(1 for n, _ in model.named_modules() if n.endswith("lora_A")) and {f.name: f.stat().st_mtime_ns for f in (tmp_path / "ckpt").iterdir()} == stamp
