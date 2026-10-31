"""Prefix stop-grad with the CUDA paths (fused QSA kernel with q_offset/key_start, fast_gdn's fla kernels with an
initial state, compiled per-token functions) against transformers' cache path on the tiny model in FP32.

Every token routes to all experts (test_compile_hc's tiny model): with top-3 of 8 a near-tied token switched
experts and moved one GDN layer's dt_bias/A_log gradients by 19%. What remains is fla's GDN kernel, whose TF32
dots re-round under any change of input: a 1e-7 perturbation of the unpatched model's embeddings moves those
per-head gradients by 1.5-2% (0.03% with transformers' torch GDN, where compile_hc changes nothing)."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA paths", allow_module_level=True)
pytest.importorskip("triton")
pytest.importorskip("fla")

from transformers.models.qwen4_exp import modeling_qwen4_exp as m  # noqa: E402

from inference.rft import layer_patches, qsa  # noqa: E402
from test_compile_hc import tiny  # noqa: E402
from test_prefix_stop_grad import assign_recurrent_states  # noqa: E402


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


@pytest.mark.parametrize("fast_gdn", [False, True], ids=["torch_gdn", "fast_gdn"])
def test_prefix_stop_grad_cuda_matches_the_cache_path(fast_gdn):
    model = tiny().cuda().train()
    for module in model.modules():
        if isinstance(module, m.Qwen4ExpTextPLELayer):
            module.requires_grad_(False)
    ids = torch.randint(0, 1000, (1, 300), generator=torch.Generator().manual_seed(1)).cuda()
    p = 170
    target = ids[0, p + 1:]
    loss_of = lambda logits: torch.nn.functional.cross_entropy(logits[0, :-1].float(), target)

    def grads():
        out = {n: x.grad.clone() for n, x in model.named_parameters() if x.grad is not None}
        model.zero_grad(set_to_none=True)
        return out

    qsa.unpatch_qsa_attention()
    layer_patches.unpatch()
    with torch.no_grad():
        cache = assign_recurrent_states(model(input_ids=ids[:, :p], use_cache=True).past_key_values)
    ref_logits = model(input_ids=ids[:, p:], past_key_values=cache, use_cache=True).logits
    loss_of(ref_logits).backward()
    ref = grads()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    qsa.patch_qsa_attention(chunk_size=64, fused=True, compile=True)
    layer_patches.patch_for_training(model, token_chunk=64, recompute="parts", compile_hc=True, compile_moe=True,
                                     fast_conv=True, fast_gdn=fast_gdn)
    try:
        loss_of(model(input_ids=ids, use_cache=False).logits[:, p:]).backward()
        full = grads()
        layer_patches.set_prefix(p)
        logits = model(input_ids=ids, use_cache=False).logits[:, p:]
        loss_of(logits).backward()
        ours = grads()
    finally:
        layer_patches.set_prefix(0)
        layer_patches.unpatch(model)
        qsa.patch_qsa_attention(chunk_size=64, compile=False)
        qsa.unpatch_qsa_attention()
    assert rel(logits.detach(), ref_logits.detach()) < 1e-2
    assert ours.keys() == ref.keys()
    worst = max(rel(ours[n], ref[n]) for n in ref)
    full_gap = sorted(rel(full[n], ref[n]) for n in ref)[-10]
    assert worst < 3e-2, worst  # TF32 dots in the QSA and GDN kernels
    assert full_gap > 10 * worst, (full_gap, worst)  # the stop-grad is visible above the kernels' rounding
