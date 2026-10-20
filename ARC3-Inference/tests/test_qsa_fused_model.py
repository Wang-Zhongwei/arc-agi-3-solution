"""The fused QSA path through a whole model (CUDA only): logits and attention gradients of the tiny
qwen4_exp (image included, gradient checkpointing on) against transformers' reference QSA in FP32.

The fused BF16 model must be as close to that answer as the chunked BF16 path it replaces (at most
1.5x its error), and a checkpoint's recompute must reuse the forward's selection: the indexer runs
once per QSA layer per step."""

import copy

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the QSA kernel needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from inference.rl import qsa  # noqa: E402
from test_qsa import sample, tiny_model  # noqa: E402


def run(model, inputs):
    target = inputs["input_ids"][0, 1:]
    model.zero_grad(set_to_none=True)
    logits = model(**inputs, use_cache=False).logits[0, :-1]
    torch.nn.functional.cross_entropy(logits.float(), target).backward()
    grads = {n: p.grad.float().clone() for n, p in model.named_parameters()
             if p.grad is not None and "self_attn" in n and "indexer" not in n}
    return logits.detach().float(), grads


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def test_fused_model_matches_reference(monkeypatch):
    qsa.unpatch_qsa_attention()
    model, config = tiny_model()
    inputs = {k: v.cuda() for k, v in sample(config).items()}
    model = model.cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = run(model, inputs)  # transformers' QSA, FP32

    bf16 = copy.deepcopy(model).to(torch.bfloat16)
    bf16_inputs = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in inputs.items()}
    calls = []
    select_blocks = qsa.select_blocks
    monkeypatch.setattr(qsa, "select_blocks", lambda *a, **k: calls.append(1) or select_blocks(*a, **k))
    results = {}
    for name, fused, compiled in (("chunked", False, False), ("fused", True, False), ("compiled", True, True)):
        qsa.patch_qsa_attention(chunk_size=64, fused=fused, compile=compiled)
        try:
            calls.clear()
            results[name] = run(bf16, bf16_inputs)
            results[name, "calls"] = len(calls)
        finally:
            qsa.patch_qsa_attention(chunk_size=64, compile=False)
            qsa.unpatch_qsa_attention()
    qsa_layers = sum(1 for t in config.text_config.layer_types if t != "linear_attention")
    assert results["fused", "calls"] == qsa_layers  # forward computes the selection, the recompute reuses it

    chunked = results["chunked"]
    for variant in ("fused", "compiled"):  # compiled: compile_qsa (norms + rotary, gate, indexer scores)
        ours = results[variant]
        assert rel(ours[0], reference[0]) <= 1.5 * rel(chunked[0], reference[0]) + 1e-3, variant
        assert ours[1].keys() == reference[1].keys() and ours[1]
        for name in reference[1]:
            a, b = rel(ours[1][name], reference[1][name]), rel(chunked[1][name], reference[1][name])
            assert a <= 1.5 * b + 1e-3, (variant, name, a, b)
