"""inference/rl/qsa.py computes the same QSA as transformers' reference implementation.

The reference loops over every query and builds a dense [N, N] mask, which
runs out of memory at 32K tokens; qsa.py replaces it for training. These tests
compare both on a tiny random qwen4_exp with the checkpoint's attention layout
(partial interleaved M-RoPE, GQA, gated output) and a budget small enough that
the selection is sparse: the selected tokens, the attention output, the full
model's logits with an image, and the gradients.
"""

import inspect
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")

from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as m  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig  # noqa: E402

from inference.rl import qsa  # noqa: E402

TEXT = dict(
    num_hidden_layers=4, hidden_size=64, hc_lowrank=16, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
    moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
    ple_layer_ids=[2], ngram_vocab_size_base=1000, ple_embed_dim=64, split_ngram_parts=4,
    # 8 index heads make a block score of exactly 0 (all heads negative under the relu) rare, so no
    # tie at the topk cut-off makes the two implementations pick different blocks
    indexer_head_dim=16, indexer_n_heads=8, indexer_kv_heads=1, indexer_budget=16, indexer_compress_ratio=4,
    mtp_num_hidden_layers=0, eos_token_id=248044,  # PLE needs an eos id; the checkpoint's value
    # the checkpoint rotates 25% of each head with interleaved M-RoPE
    rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25,
                         mrope_section=[2, 1, 1], mrope_interleaved=True),
)
VISION = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)


def tiny_model():
    torch.manual_seed(0)
    config = Qwen4ExpConfig(text_config=TEXT, vision_config=VISION)
    config._attn_implementation = "sdpa"
    return AutoModelForImageTextToText.from_config(config, dtype=torch.float32).eval(), config


def sample(config, n_text=300, grid=(1, 4, 4)):
    """Text tokens around one synthetic image: input_ids, mm_token_type_ids, pixel_values, image_grid_thw."""
    g = torch.Generator().manual_seed(1)
    image_tokens = math.prod(grid) // 4  # 2x2 spatial merge
    text = torch.randint(0, 1000, (n_text,), generator=g)
    half = n_text // 2
    ids = torch.cat([text[:half], torch.tensor([config.vision_start_token_id]),
                     torch.full((image_tokens,), config.image_token_id), torch.tensor([config.vision_end_token_id]),
                     text[half:]])
    types = (ids == config.image_token_id).long()
    pixels = torch.randn(math.prod(grid), 3 * 2 * 16 * 16, generator=g)
    return dict(input_ids=ids[None], mm_token_type_ids=types[None], pixel_values=pixels,
                image_grid_thw=torch.tensor([grid]))


def capture_attention_inputs(model, inputs):
    """(module, hidden_states, position_embeddings, attention_mask) of the first QSA layer."""
    captured = {}
    layer = next(mod for mod in model.modules() if isinstance(mod, m.Qwen4ExpTextAttention))
    signature = inspect.signature(m.Qwen4ExpTextAttention.forward)

    def hook(module, args, kwargs):
        if not captured:
            bound = signature.bind(module, *args, **kwargs).arguments
            captured.update(module=module, hidden=bound["hidden_states"], pos=bound["position_embeddings"],
                            mask=bound["attention_mask"])

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        model(**inputs, use_cache=False)
    handle.remove()
    return captured


def to_mask(indices, n):
    mask = torch.zeros(n, n + 1, dtype=torch.bool)
    mask.scatter_(1, torch.where(indices >= 0, indices, n), True)
    return mask[:, :n]


def equal_up_to_ties(reference, ours, scores, ratio=4):
    """Same tail tokens, and the selected blocks have the same scores (only tied blocks may differ)."""
    for i in (reference != ours).any(-1).nonzero().flatten().tolist():
        complete = (i + 1) // ratio
        tail = slice(complete * ratio, i + 1)
        if not torch.equal(reference[i, tail], ours[i, tail]):
            return False
        starts = torch.arange(complete) * ratio
        picked = lambda mask: scores[i, :complete][mask[i, starts]].sort().values
        if not torch.equal(picked(reference), picked(ours)):
            return False
    return True


@pytest.fixture(scope="module")
def setup():
    qsa.unpatch_qsa_attention()
    model, config = tiny_model()
    inputs = sample(config)
    return model, config, inputs, capture_attention_inputs(model, inputs)


def test_selection_matches_reference(setup):
    _, _, _, c = setup
    n = c["hidden"].shape[1]
    with torch.no_grad():
        reference = c["module"].indexer(c["hidden"], c["pos"], c["mask"], None)[0, 0]
        cos, sin = c["pos"]
        ours = qsa.select_tokens(c["module"].indexer, c["hidden"], cos, sin, chunk_size=64)
    assert torch.equal(to_mask(ours, n), reference)
    assert reference.sum(-1).max() < n // 2  # the budget really is binding


def test_attention_output_matches_reference(setup):
    _, _, _, c = setup
    with torch.no_grad():
        hf_forward = getattr(m.Qwen4ExpTextAttention, "_hf_forward", m.Qwen4ExpTextAttention.forward)
        reference, _ = hf_forward(c["module"], c["hidden"], c["pos"], c["mask"])
        c["module"]._qsa_chunk_size = 64
        ours, _ = qsa.qsa_forward(c["module"], c["hidden"], c["pos"], c["mask"])
    torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-5)


def test_model_logits_and_gradients_match_reference(setup):
    model, _, inputs, _ = setup
    target = inputs["input_ids"][0, 1:]

    def run():
        model.zero_grad(set_to_none=True)
        logits = model(**inputs, use_cache=False).logits[0, :-1]
        torch.nn.functional.cross_entropy(logits, target).backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()
                 if p.grad is not None and "self_attn" in n and "indexer" not in n}
        return logits.detach(), grads

    qsa.unpatch_qsa_attention()
    ref_logits, ref_grads = run()
    qsa.patch_qsa_attention(chunk_size=64)
    try:
        logits, grads = run()
    finally:
        qsa.unpatch_qsa_attention()
    torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-5)
    assert grads.keys() == ref_grads.keys() and grads
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-4, atol=1e-6, msg=name)


def test_selection_at_the_real_budget():
    """Real indexer sizes and budget (512 blocks of 4) on 4,100 tokens, past where topk binds."""
    torch.manual_seed(2)
    config = Qwen4ExpConfig(text_config=dict(TEXT, hidden_size=64, indexer_head_dim=128, indexer_n_heads=4,
                                             indexer_budget=2048)).text_config
    indexer = m.Qwen4ExpTextQSAIndexer(config, layer_idx=3).eval()
    n, rotary = 4100, 64
    hidden = torch.randn(1, n, config.hidden_size)
    angles = torch.arange(n)[:, None] * (1e4 ** (-torch.arange(rotary // 2) / (rotary // 2)))[None, :]
    cos = torch.cat([angles.cos()] * 2, -1)[None]
    sin = torch.cat([angles.sin()] * 2, -1)[None]
    causal = torch.ones(n, n, dtype=torch.bool).tril()[None, None]
    with torch.no_grad():
        reference = indexer(hidden, (cos, sin), causal, None)[0, 0]
        ours = to_mask(qsa.select_tokens(indexer, hidden, cos, sin, chunk_size=1000), n)
        q, block_keys = qsa.indexer_queries_and_keys(indexer, hidden, cos, sin)
        scores = torch.relu(torch.einsum("nhd,bd->nhb", q, block_keys)).sum(1) / math.sqrt(q.shape[-1])
    assert equal_up_to_ties(reference, ours, scores)
    assert int(reference[-1].sum()) == 2048 + (n % 4)


def test_fp32_kv_gradients_are_closer_to_float64():
    """With many queries selecting the same keys, BF16 sums of their gradient contributions drift;
    gathering from FP32 copies keeps the sums in FP32 and lands closer to a float64 reference."""
    g = torch.Generator().manual_seed(0)
    n, c, hkv, h, d, s, chunks = 64, 256, 2, 8, 32, 48, 16  # each key is picked by ~3,000 queries
    q = torch.randn(c * chunks, h, d, generator=g)
    k = torch.randn(n, hkv, d, generator=g)
    v = torch.randn(n, hkv, d, generator=g)
    idx = torch.stack([torch.randperm(n, generator=g)[:s] for _ in range(c * chunks)])
    idx[:, -5:] = -1  # padding slots
    upstream = torch.randn(c * chunks, h, d, generator=g)

    def grads(dtype, source_dtype):
        qq = q.to(dtype)
        kk = k.to(dtype).to(source_dtype).requires_grad_(True)
        vv = v.to(dtype).to(source_dtype).requires_grad_(True)
        total = 0
        for i in range(chunks):  # the gradients of k and v add up over chunks, as in qsa_forward
            rows = slice(i * c, (i + 1) * c)
            out = qsa._attend_chunk(qq[rows], kk, vv, idx[rows], d ** -0.5)
            total = total + (out.to(upstream.dtype) * upstream[rows]).sum()
        total.backward()
        return kk.grad.double(), vv.grad.double()

    reference = grads(torch.float64, torch.float64)
    bf16 = grads(torch.bfloat16, torch.bfloat16)
    fp32 = grads(torch.bfloat16, torch.float32)
    error = lambda pair: max(float((a - b).norm() / b.norm()) for a, b in zip(pair, reference))
    # what is left in the FP32 path is the BF16 rounding of each contribution (0.44% here vs 0.69%)
    assert error(fp32) < 0.8 * error(bf16)

