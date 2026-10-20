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


@pytest.mark.parametrize("cache", [True, False])
def test_grouped_experts_match_the_loop(cache):
    """Cached: all experts in one grouped GEMM; not cached: tiles of experts, one partial."""
    model = tiny()
    experts = next(x for x in model.modules() if isinstance(x, m.Qwen4ExpTextExperts))
    experts._rft_expert_tile = 3  # several tiles, one partial
    experts._rft_cache_experts = cache
    layer_patches.clear_expert_cache()
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


def test_parts_recompute_matches_and_runs_the_experts_twice():
    """With gradient checkpointing, "layer" recomputes the token chunks inside an already recomputed
    layer (three forwards); "parts" checkpoints the mixer instead and gets the same gradients."""
    model = tiny()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ids = torch.randint(0, 1000, (1, 50))
    calls = []
    for module in model.modules():
        if isinstance(module, m.Qwen4ExpTextExperts):
            module.register_forward_hook(lambda *_: calls.append(1))

    def run(recompute):
        layer_patches.patch_for_training(model, token_chunk=16, expert_tile=3, recompute=recompute)
        assert layer_patches.current_recompute(model) == recompute
        model.zero_grad(set_to_none=True)
        calls.clear()
        logits = model(input_ids=ids, use_cache=False).logits
        logits.float().pow(2).mean().backward()
        return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, len(calls)

    try:
        reference, ref_grads, layer_calls = run("layer")
        ours, grads, parts_calls = run("parts")
        run("layer")  # switching back restores transformers' layer checkpoint
        assert any(getattr(x, "gradient_checkpointing", False) for x in model.modules()
                   if isinstance(x, m.Qwen4ExpTextDecoderLayer))
    finally:
        layer_patches.unpatch()
    chunks_per_layer = 4  # 50 tokens in chunks of 16
    layers = TEXT["num_hidden_layers"]
    # "layer": forward, the layer's recompute (which stops early, before the last chunk: nothing after
    # it is saved) and each chunk's own recompute. "parts": forward and each chunk's recompute
    assert layer_calls == (3 * chunks_per_layer - 1) * layers and parts_calls == 2 * chunks_per_layer * layers
    torch.testing.assert_close(ours, reference, rtol=0, atol=0)
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=0, atol=0, msg=name)



def test_hyper_connection_functions_match_the_modules():
    """The pure functions compile_hc compiles compute exactly what Qwen4ExpTextGatedResidual does."""
    model = tiny().to(torch.bfloat16)
    layer = next(x for x in model.modules() if isinstance(x, m.Qwen4ExpTextDecoderLayer))
    attn_hc, mlp_hc = layer.attn_hyper_connection, layer.mlp_hyper_connection
    kw = dict(eps=attn_hc.hc_norm.eps, hc=attn_hc.hc_count)
    with torch.no_grad():
        for p in (*attn_hc.parameters(), *mlp_hc.parameters()):
            p.normal_(0, 0.1)  # norm weights start at 0: make them count
    x = torch.randn(2, 7, attn_hc.hc_count * attn_hc.hidden_size, dtype=torch.bfloat16)
    mixer_out = torch.randn(2, 7, attn_hc.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        mixed, _, weights = attn_hc(x)
        ours = layer_patches.hc_before_mixer(x, *layer_patches._hc_args(attn_hc), **kw)
        assert torch.equal(ours[0], mixed) and torch.equal(ours[1], weights)
        h = x + (mixer_out.unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)
        mlp_in, _, mlp_weights = mlp_hc(h)
        h2, ours_in, ours_weights = layer_patches.hc_after_mixer(x, mixer_out, weights,
                                                                 *layer_patches._hc_args(mlp_hc), **kw)
        assert torch.equal(h2, h) and torch.equal(ours_in, mlp_in) and torch.equal(ours_weights, mlp_weights)


def test_short_conv_matches_conv1d():
    """The channels-last GDN short convolution computes transformers' F.conv1d fallback."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as mm

    reference_fn = getattr(mm, "_rft_hf_causal_conv1d_fn", mm.causal_conv1d_fn)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 37, 24, generator=g)  # [B, T, C], as the projection produces it
    weight = torch.randn(24, 4, generator=g)
    reference = reference_fn(x.transpose(1, 2), weight, None, activation="silu").transpose(1, 2)
    ours = layer_patches.short_conv(x, weight, None, silu=True)
    torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-6)


def test_loop_grouped_mm_matches_per_group_matmuls():
    """The dX-only grouped GEMM (GPUs without torch._grouped_mm, frozen experts) equals per-group matmuls,
    forward and input gradient, including an empty group."""
    g = torch.Generator().manual_seed(0)
    offsets = torch.tensor([3, 3, 8, 10], dtype=torch.int32)  # group 1 is empty
    x = torch.randn(10, 6, generator=g)
    w = torch.randn(4, 6, 7, generator=g)  # frozen: no gradient wanted
    upstream = torch.randn(10, 7, generator=g)
    starts = [0, *offsets[:-1].tolist()]

    xr = x.clone().requires_grad_(True)
    reference = torch.cat([xr[s:e] @ w[i] for i, (s, e) in enumerate(zip(starts, offsets.tolist()))])
    (reference * upstream).sum().backward()
    xo = x.clone().requires_grad_(True)
    ours = layer_patches._LoopGroupedMM.apply(xo, w, offsets)
    (ours * upstream).sum().backward()
    assert torch.equal(ours, reference)
    assert torch.equal(xo.grad, xr.grad)


def test_unpatch_restores_transformers():
    """unpatch(model) (the job server's reference mode) runs exactly transformers' layers and experts again."""
    model = tiny().eval()
    ids = torch.randint(0, 1000, (1, 40), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        before = model(input_ids=ids, use_cache=False).logits
    layer_patches.patch_for_training(model, token_chunk=16)
    layer_patches.unpatch(model)
    experts = next(x for x in model.modules() if isinstance(x, m.Qwen4ExpTextExperts))
    assert experts.config._experts_implementation != layer_patches.EXPERTS_IMPLEMENTATION
    with torch.no_grad():
        after = model(input_ids=ids, use_cache=False).logits
    assert torch.equal(before, after)


def test_ngram_lookup_runs_once_per_step():
    """With gradient checkpointing the PLE block is recomputed in backward; its n-gram lookup (a host-side
    gather of the memory-mapped table) is reused instead of repeated, with identical logits and gradients."""
    ids = torch.randint(0, 1000, (1, 50), generator=torch.Generator().manual_seed(5))

    def run(patched):
        model = tiny().train()
        for module in model.modules():  # frozen, as in LoRA training
            if isinstance(module, m.Qwen4ExpTextNGramEmbedding):
                module.ngram_embedding.weight.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        layer_patches.patch_for_training(model, token_chunk=16, recompute="parts")
        if not patched:  # transformers' lookup every time, through _hf_forward so it is counted
            m.Qwen4ExpTextNGramEmbedding.forward = lambda self, *a, **k: type(self)._hf_forward(self, *a, **k)
        calls = []
        hf = m.Qwen4ExpTextNGramEmbedding._hf_forward
        m.Qwen4ExpTextNGramEmbedding._hf_forward = lambda *a, **k: calls.append(1) or hf(*a, **k)
        try:
            logits = model(input_ids=ids, use_cache=False).logits
            logits.float().pow(2).mean().backward()
        finally:
            m.Qwen4ExpTextNGramEmbedding._hf_forward = hf
            layer_patches.unpatch(model)
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return logits.detach(), grads, len(calls)

    ref_logits, ref_grads, ref_calls = run(patched=False)
    logits, grads, calls = run(patched=True)
    ple_layers = sum(1 for x in tiny().modules() if isinstance(x, m.Qwen4ExpTextNGramEmbedding))
    assert ref_calls == 2 * ple_layers and calls == ple_layers
    assert torch.equal(logits, ref_logits)
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        assert torch.equal(grads[name], ref_grads[name]), name


def test_fast_lora_matches_peft(monkeypatch):
    """fast_lora's update (eager on CPU; compiled on CUDA) is peft's op for op: identical logits and gradients
    (LoRA, base and input), with an FP32 adapter on a BF16 model as in training."""
    peft = pytest.importorskip("peft")
    calls = []
    lora_add = layer_patches.lora_add
    monkeypatch.setattr(layer_patches, "lora_add", lambda *a, **k: calls.append(1) or lora_add(*a, **k))
    from inference.rl.smoke_lora import ATTENTION_GDN

    def run(fast):
        model = tiny().to(torch.bfloat16).train()
        peft.inject_adapter_in_model(peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                                     target_modules=ATTENTION_GDN), model)
        torch.manual_seed(1)
        for name, p in model.named_parameters():
            p.requires_grad_("lora_" in name)
            if "lora_" in name:
                p.data = (torch.randn_like(p, dtype=torch.float32) * 0.05)  # FP32 adapters, B non-zero
        layer_patches.patch_for_training(model, token_chunk=16, fast_lora=fast)
        ids = torch.randint(0, 1000, (1, 40), generator=torch.Generator().manual_seed(7))
        try:
            logits = model(input_ids=ids, use_cache=False).logits
            logits.float().pow(2).mean().backward()
        finally:
            layer_patches.unpatch(model)
        return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    reference, ref_grads = run(False)
    assert not calls
    ours, grads = run(True)
    modules = 3 * 5 + 1 * 4  # 3 GDN layers x 5 LoRA projections + 1 QSA layer x 4 (no checkpointing: one call each)
    assert len(calls) == modules  # every LoRA module took the fused path
    assert torch.equal(ours, reference)
    assert grads.keys() == ref_grads.keys() and len(grads) > 10
    for name in grads:
        assert torch.equal(grads[name], ref_grads[name]), name


def test_gdn_pure_functions_match_the_module():
    """fast_gdn's conv_split and gated_rms_norm compute transformers' convolution + split and RMSNormGated."""
    model = tiny().to(torch.bfloat16)
    gdn = next(x for x in model.modules() if isinstance(x, m.Qwen4ExpTextGatedDeltaNet))
    g = torch.Generator().manual_seed(0)
    n = 37
    mixed = torch.randn(1, n, gdn.conv_dim, generator=g).to(torch.bfloat16)
    with torch.no_grad():
        conv = layer_patches.short_conv(mixed, gdn.conv1d.weight.squeeze(1), gdn.conv1d.bias, silu=True)
        q, k, v = layer_patches.conv_split(mixed, gdn.conv1d.weight.squeeze(1), gdn.conv1d.bias, gdn.key_dim,
                                           gdn.num_k_heads, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim)
        rq, rk, rv = torch.split(conv, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
        assert torch.equal(q, rq.reshape(1, n, -1, gdn.head_k_dim)) and q.is_contiguous()
        assert torch.equal(k, rk.reshape(1, n, -1, gdn.head_k_dim)) and k.is_contiguous()
        assert torch.equal(v, rv.reshape(1, n, -1, gdn.head_v_dim)) and v.is_contiguous()
        gdn.norm.weight.normal_(1, 0.1)
        x = torch.randn(n * gdn.num_v_heads, gdn.head_v_dim, generator=g).to(torch.bfloat16)
        gate = torch.randn(n * gdn.num_v_heads, gdn.head_v_dim, generator=g).to(torch.bfloat16)
        ours = layer_patches.gated_rms_norm(x, gate, gdn.norm.weight, gdn.norm.variance_epsilon,
                                            gdn.norm.activation == "sigmoid")
        assert torch.equal(ours, gdn.norm(x, gate))


def test_chunked_ple_matches_the_module():
    """The PLE layer with its per-token part in token chunks: same output and gradients as transformers'."""
    model = tiny()
    ple = next(x for x in model.modules() if isinstance(x, m.Qwen4ExpTextPLELayer))
    g = torch.Generator().manual_seed(2)
    n = 45
    width = ple.hc_count * ple.hidden_size
    ids = torch.randint(0, 1000, (1, n), generator=g)
    x = torch.randn(1, n, width, generator=g)
    upstream = torch.randn(1, n, width, generator=g)

    def run(chunked):
        h = x.clone().requires_grad_(True)
        ple.zero_grad(set_to_none=True)
        if chunked:
            ple._rft_token_chunk = 16  # 3 chunks, the last partial
            out = layer_patches.ple_forward(ple, h, ids, None)
        else:  # transformers' forward, whether or not patch_for_training replaced it
            hf = getattr(m.Qwen4ExpTextPLELayer, "_hf_forward", m.Qwen4ExpTextPLELayer.forward)
            out = hf(ple, h, ids, None)
        (out * upstream).sum().backward()
        return out.detach(), h.grad, {k: p.grad.clone() for k, p in ple.named_parameters() if p.grad is not None}

    ref = run(False)
    ours = run(True)
    assert torch.equal(ours[0], ref[0])
    print("dh max rel", float((ours[1] - ref[1]).abs().max() / ref[1].abs().max()))
    torch.testing.assert_close(ours[1], ref[1], rtol=1e-5, atol=1e-6)  # per-chunk reductions: float reassociation
    assert ours[2].keys() == ref[2].keys()
    for name in ref[2]:
        torch.testing.assert_close(ours[2][name], ref[2][name], rtol=1e-6, atol=1e-7, msg=name)


def test_precomputed_ngram_rows_replace_the_lookup():
    """precompute_ngram (train_rft's prefetch thread) gives the rows the forward would gather; handed over with
    provide_ngram, a training step does no lookup of its own and its logits and gradients are bit-identical.
    A no-grad forward (validation) takes them too."""
    ids = torch.randint(0, 1000, (1, 50), generator=torch.Generator().manual_seed(6))
    model = tiny().train()
    for module in model.modules():
        if isinstance(module, m.Qwen4ExpTextNGramEmbedding):
            module.ngram_embedding.weight.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    layer_patches.patch_for_training(model, token_chunk=16, recompute="parts")
    calls = []
    hf = m.Qwen4ExpTextNGramEmbedding._hf_forward
    try:
        def run(rows):
            model.zero_grad(set_to_none=True)
            layer_patches.provide_ngram(model, rows)
            logits = model(input_ids=ids, use_cache=False).logits
            logits.float().pow(2).mean().backward()
            return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        ref_logits, ref_grads = run(None)
        rows = layer_patches.precompute_ngram(model, ids)
        assert len(rows) >= 1 and all(r.device.type == "cpu" for r in rows)
        m.Qwen4ExpTextNGramEmbedding._hf_forward = lambda *a, **k: calls.append(1) or hf(*a, **k)
        logits, grads = run(rows)
        assert calls == []
        assert torch.equal(logits, ref_logits) and grads.keys() == ref_grads.keys()
        for name in grads:
            assert torch.equal(grads[name], ref_grads[name]), name
        with torch.no_grad():
            layer_patches.provide_ngram(model, rows)
            assert torch.equal(model(input_ids=ids, use_cache=False).logits, ref_logits)
        assert calls == []
    finally:
        m.Qwen4ExpTextNGramEmbedding._hf_forward = hf
        layer_patches.unpatch(model)
