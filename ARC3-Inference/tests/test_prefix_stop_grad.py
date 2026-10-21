"""Prefix stop-grad (layer_patches.set_prefix) equals running the suffix on a cache of the prefix.

The standard answer is transformers' own cache path on the unpatched tiny model: the prefix under no_grad
with use_cache, then the suffix with gradients from that cache. With set_prefix our patched layers run the
whole sequence, the prefix without gradient; logits of the suffix and every gradient must agree. PLE is
frozen, as in LoRA training (it runs on the whole sequence and has no prefix-free path of its own)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")

from transformers.models.qwen4_exp import modeling_qwen4_exp as m  # noqa: E402

from inference.rft import layer_patches, qsa  # noqa: E402
from test_layer_patches import tiny  # noqa: E402


def test_gdn_mixer_continues_its_state():
    model = tiny()
    gdn = next(x for x in model.modules() if isinstance(x, m.Qwen4ExpTextGatedDeltaNet))
    x = torch.randn(1, 50, gdn.hidden_size, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        full = gdn(x)
        assert torch.equal(layer_patches.gdn_mixer(gdn, x), full)
        out_p, state = layer_patches.gdn_mixer(gdn, x[:, :30], return_state=True)
        out_s = layer_patches.gdn_mixer(gdn, x[:, 30:], state=state)
    torch.testing.assert_close(torch.cat([out_p, out_s], dim=1), full, rtol=1e-5, atol=1e-6)
    # the suffix's checkpoint holds the state until backward: it must not keep the prefix's projection alive
    conv = state["conv"]
    assert conv.untyped_storage().nbytes() == conv.numel() * conv.element_size()


def assign_recurrent_states(cache):
    """transformers' linear-attention cache copies a new recurrent state into the old tensor (a static address
    for CUDA graphs); the suffix's computation saved that tensor for backward, so the reference could not
    backpropagate. Same values, a new tensor instead."""
    for layer in getattr(cache, "layers", []):
        if hasattr(layer, "recurrent_states"):
            def update(recurrent_states, state_idx=0, _layer=layer, **kwargs):
                if not _layer.is_recurrent_states_initialized[state_idx]:
                    _layer.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
                _layer.recurrent_states[state_idx] = recurrent_states
                return recurrent_states
            layer.update_recurrent_state = update
    return cache


def test_prefix_stop_grad_matches_the_cache_path():
    model = tiny().train()
    for module in model.modules():
        if isinstance(module, m.Qwen4ExpTextPLELayer):
            module.requires_grad_(False)
    ids = torch.randint(0, 1000, (1, 60), generator=torch.Generator().manual_seed(1))
    p = 37
    target = ids[0, p + 1:]

    def loss_of(logits):  # logits for positions p.. predict tokens p+1..
        return torch.nn.functional.cross_entropy(logits[0, :-1].float(), target)

    def grads():
        out = {n: x.grad.clone() for n, x in model.named_parameters() if x.grad is not None}
        model.zero_grad(set_to_none=True)
        return out

    # the standard answer: transformers' cache path, unpatched
    qsa.unpatch_qsa_attention()
    layer_patches.unpatch()
    with torch.no_grad():
        cache = assign_recurrent_states(model(input_ids=ids[:, :p], use_cache=True).past_key_values)
    ref_logits = model(input_ids=ids[:, p:], past_key_values=cache, use_cache=True).logits
    loss_of(ref_logits).backward()
    ref = grads()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    qsa.patch_qsa_attention(chunk_size=16)
    layer_patches.patch_for_training(model, token_chunk=16, recompute="parts")
    try:
        loss_of(model(input_ids=ids, use_cache=False).logits[:, p:]).backward()
        full = grads()  # without stop-grad: the gradient through the prefix is there
        layer_patches.set_prefix(p)
        logits = model(input_ids=ids, use_cache=False).logits[:, p:]
        loss_of(logits).backward()
        ours = grads()
    finally:
        layer_patches.set_prefix(0)
        layer_patches.unpatch(model)
        qsa.unpatch_qsa_attention()
    torch.testing.assert_close(logits.detach(), ref_logits.detach(), rtol=1e-4, atol=1e-5)
    assert ours.keys() == ref.keys() and len(ours) > 20
    differs = [n for n in ref if (full[n] - ref[n]).norm() > 1e-2 * ref[n].norm()]
    assert len(differs) > 10, differs  # the test would notice a stop-grad that cut nothing
    for name in ref:
        torch.testing.assert_close(ours[name], ref[name], rtol=1e-3, atol=1e-6, msg=name)
