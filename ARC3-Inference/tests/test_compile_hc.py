"""Compiled hyper-connections and MoE combine and the channels-last GDN convolution (layer_patches
compile_hc, compile_moe, fast_conv; CUDA only) are as accurate as the eager BF16 decoder layer.

Both round differently in BF16 (torch.compile fuses elementwise chains and rounds once), so each is
compared with the same layer in FP32: the compiled layer's error may be at most 1.5x eager's. Every
token routes to all experts here: with top-3 of 8, one token whose 3rd and 4th experts were 1.2e-5
apart switched experts under compile (not under eager) and moved the MoE output by 5%; top-k ties
say nothing about accuracy."""

import copy

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("compile_hc runs on CUDA only", allow_module_level=True)

from inference.rft import layer_patches  # noqa: E402
from test_qsa import TEXT, VISION  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as m  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig  # noqa: E402


def tiny():
    """test_layer_patches' tiny model with continuous routing (every token uses all 4 experts)."""
    torch.manual_seed(0)
    config = Qwen4ExpConfig(text_config=dict(TEXT, num_experts=4, num_experts_per_tok=4), vision_config=VISION)
    config._attn_implementation = "sdpa"
    return AutoModelForImageTextToText.from_config(config, dtype=torch.float32)


def run(layer, x, pos):
    x = x.clone().requires_grad_(True)
    y = layer(x, pos, attention_mask=None)
    (y.float() ** 2).sum().backward()
    grads = {n: p.grad.float().clone() for n, p in layer.named_parameters() if p.grad is not None}
    layer.zero_grad(set_to_none=True)
    return {"out": y.detach().float(), "x": x.grad.float(), **grads}


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize("fast_gdn", [False, True], ids=["plain", "fast_gdn"])
def test_compiled_layer_is_as_accurate_as_eager(fast_gdn):
    """[plain]: compile_hc, compile_moe, fast_conv; [fast_gdn] adds fla's kernels with grouped value heads (on
    Hopper their backward runs on tilelang): pod/make_jobs.py switches off only the features whose case failed."""
    model = tiny().to("cuda", torch.bfloat16).train()
    layer_patches.patch_for_training(model, token_chunk=64)
    layer = next(x for x in model.modules() if isinstance(x, m.Qwen4ExpTextDecoderLayer)
                 and x.layer_type == "linear_attention")
    with torch.no_grad():
        for p in (*layer.attn_hyper_connection.parameters(), *layer.mlp_hyper_connection.parameters()):
            p.normal_(0, 0.1)
    n = 200
    width = layer.attn_hyper_connection.hc_count * layer.attn_hyper_connection.hidden_size
    x = torch.randn(1, n, width, device="cuda", dtype=torch.bfloat16)
    rotary = model.model.language_model.rotary_emb
    pos = rotary(x, torch.arange(n, device="cuda")[None, None].expand(3, 1, n))

    fp32 = copy.deepcopy(layer).float()
    reference = run(fp32, x.float(), tuple(p.float() for p in pos))
    eager = run(layer, x, pos)
    layer_patches.configure(model, compile_hc=True, compile_moe=True)
    layer_patches._FAST_CONV["on"] = True
    layer_patches._FAST_GDN["on"] = fast_gdn  # fused conv + split, grouped value heads, fused gated norm (fla)
    try:
        compiled = run(layer, x, pos)
    finally:
        layer_patches.configure(model, compile_hc=False, compile_moe=False)
        layer_patches._FAST_CONV["on"] = False
        layer_patches._FAST_GDN["on"] = False
    assert compiled.keys() == reference.keys()
    for name in reference:
        assert rel(compiled[name], reference[name]) <= 1.5 * rel(eager[name], reference[name]) + 1e-3, \
            (name, rel(compiled[name], reference[name]), rel(eager[name], reference[name]))


def test_compiled_lora_matches_peft():
    """fast_lora compiled on CUDA against peft's forward: same math, rounding may differ by fused ops."""
    peft = pytest.importorskip("peft")
    from inference.rft.smoke_lora import ATTENTION_GDN

    def run(fast):
        model = tiny().to("cuda", torch.bfloat16).train()
        peft.inject_adapter_in_model(peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                                     target_modules=ATTENTION_GDN), model)
        torch.manual_seed(1)
        for name, p in model.named_parameters():
            p.requires_grad_("lora_" in name)
            if "lora_" in name:
                p.data = torch.randn_like(p, dtype=torch.float32) * 0.05
        layer_patches.patch_for_training(model, token_chunk=64, fast_lora=fast)
        ids = torch.randint(0, 1000, (1, 150), generator=torch.Generator().manual_seed(7)).cuda()
        try:
            logits = model(input_ids=ids, use_cache=False).logits
            logits.float().pow(2).mean().backward()
        finally:
            layer_patches.unpatch(model)
        return logits.detach().float(), {n: p.grad.float() for n, p in model.named_parameters() if p.grad is not None}

    reference, ref_grads = run(False)
    ours, grads = run(True)
    assert rel(ours, reference) < 1e-3
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        assert rel(grads[name], ref_grads[name]) < 1e-2, (name, rel(grads[name], ref_grads[name]))
