"""inference/rl/layer_patches.py rewrites qwen4_exp layers for long training sequences without changing results.

On a tiny random qwen4_exp: grouped experts match the per-expert loop, the token-chunked
decoder layer matches transformers' layer (outputs and gradients) with chunks smaller than
the sequence, and the INT4 dequantization matches the GPTQ packing it reverses.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")

from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as m  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig  # noqa: E402

from inference.rl import layer_patches  # noqa: E402
from test_qsa import TEXT, VISION  # noqa: E402


def tiny():
    torch.manual_seed(0)
    config = Qwen4ExpConfig(text_config=dict(TEXT, num_experts=8, num_experts_per_tok=3), vision_config=VISION)
    config._attn_implementation = "sdpa"
    return AutoModelForImageTextToText.from_config(config, dtype=torch.float32)


def test_grouped_experts_match_the_loop():
    model = tiny()
    experts = next(x for x in model.modules() if isinstance(x, m.Qwen4ExpTextExperts))
    experts._rft_expert_tile = 3  # several tiles, one partial
    x = torch.randn(40, experts.hidden_dim)
    index = torch.stack([torch.randperm(8)[:3] for _ in range(40)])
    weights = torch.softmax(torch.randn(40, 3), -1)
    reference = m.Qwen4ExpTextExperts.forward.__wrapped__(experts, x, index, weights) \
        if hasattr(m.Qwen4ExpTextExperts.forward, "__wrapped__") else None
    if reference is None:
        experts.config._experts_implementation = "eager"
        reference = experts(x, index, weights)
    ours = layer_patches.grouped_experts_forward(experts, x, index, weights)
    torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-6)


def test_chunked_layers_match_transformers():
    model = tiny()
    ids = torch.randint(0, 1000, (1, 50))

    def run():
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, use_cache=False).logits
        logits.float().pow(2).mean().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return logits.detach(), grads

    layer_patches.unpatch()
    reference, ref_grads = run()
    layer_patches.patch_for_training(model, token_chunk=16, expert_tile=3)
    try:
        ours, grads = run()
    finally:
        layer_patches.unpatch()
    torch.testing.assert_close(ours, reference, rtol=1e-4, atol=1e-5)
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-4, atol=1e-6, msg=name)


def pack_int4(q, zeros, group_size):
    """GPTQ packing (the inverse of dequant_int4): q [in, out] in 0..15, zeros [groups, out] stored minus 1."""
    in_features, out_features = q.shape
    qweight = torch.zeros(in_features // 8, out_features, dtype=torch.int64)
    for j in range(8):
        qweight |= q[j::8].long() << (4 * j)
    stored = zeros - 1
    qzeros = torch.zeros(stored.shape[0], out_features // 8, dtype=torch.int64)
    for j in range(8):
        qzeros |= stored[:, j::8].long() << (4 * j)
    as_int32 = lambda t: torch.where(t >= 2**31, t - 2**32, t).to(torch.int32)
    return as_int32(qweight), as_int32(qzeros)


def test_dequant_int4_reverses_the_packing():
    torch.manual_seed(1)
    in_features, out_features, group = 256, 64, 128
    q = torch.randint(0, 16, (in_features, out_features))
    zeros = torch.randint(1, 16, (in_features // group, out_features))
    scales = torch.rand(in_features // group, out_features) + 0.1
    qweight, qzeros = pack_int4(q, zeros, group)
    expected = scales.repeat_interleave(group, 0) * (q - zeros.repeat_interleave(group, 0))
    got = layer_patches.dequant_int4(qweight, scales, qzeros, group, dtype=torch.float32)
    torch.testing.assert_close(got, expected.float())
