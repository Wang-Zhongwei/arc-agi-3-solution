"""inference/rl/host_offload.py offloads saved tensors into reused host buffers without changing results.

The arena hands the same memory back after a reset and grows only when a step needs more;
gradients of a tiny checkpointed qwen4_exp match a run without offload, and so do those of a
small network on an accelerator, where the real device -> host copies happen.
"""

import contextlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")

from torch.utils.checkpoint import checkpoint  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig  # noqa: E402

from inference.rl.host_offload import ALIGN, HostArena, offload_to_host  # noqa: E402
from test_qsa import TEXT, VISION  # noqa: E402

MiB = 2**20


def test_arena_reuses_memory_after_reset():
    arena = HostArena(segment_gb=4 / 1024)  # 4 MiB segments
    first = [arena.take(n).data_ptr() for n in (MiB, 3, MiB)]
    assert len(arena.segments) == 1 and first[1] % ALIGN == 0
    arena.reset()
    assert [arena.take(n).data_ptr() for n in (MiB, 3, MiB)] == first
    arena.take(3 * MiB)  # does not fit in what is left of segment 0: a second segment
    big = arena.take(10 * MiB)  # larger than a segment: a segment of its own
    assert [s.numel() for s in arena.segments] == [4 * MiB, 4 * MiB, 10 * MiB]
    arena.reset()
    arena.take(3 * MiB), arena.take(3 * MiB)
    assert arena.take(10 * MiB).data_ptr() == big.data_ptr() and len(arena.segments) == 3


def tiny():
    torch.manual_seed(0)
    config = Qwen4ExpConfig(text_config=dict(TEXT, num_experts=8, num_experts_per_tok=3), vision_config=VISION)
    config._attn_implementation = "sdpa"
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def test_checkpointed_model_gradients_match_without_offload():
    model = tiny()
    ids = torch.randint(0, 1000, (1, 50))

    def run(context):
        model.zero_grad(set_to_none=True)
        with context:
            logits = model(input_ids=ids, use_cache=False).logits
        logits.float().pow(2).mean().backward()
        return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    reference, ref_grads = run(torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t))
    arena = HostArena(segment_gb=1 / 1024)
    ptrs = []
    for _ in range(2):  # the second step reuses the first one's buffers
        logits, grads = run(offload_to_host(arena, include_cpu=True, min_bytes=0))
        ptrs.append([s.data_ptr() for s in arena.segments])
        torch.testing.assert_close(logits, reference, rtol=0, atol=0)
        assert grads.keys() == ref_grads.keys()
        for name, grad in grads.items():
            torch.testing.assert_close(grad, ref_grads[name], rtol=0, atol=0, msg=name)
    assert arena.used > 0 and ptrs[0] == ptrs[1]


accelerator = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(accelerator is None, reason="needs a GPU for device -> host copies")
def test_device_tensors_round_trip_through_host_buffers():
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(torch.nn.Linear(64, 64) for _ in range(4)).to(accelerator, torch.bfloat16)
    x = torch.randn(256, 64, device=accelerator, dtype=torch.bfloat16)

    def run(context):
        layers.zero_grad(set_to_none=True)
        with context:
            h = x
            for layer in layers:
                h = checkpoint(lambda t, layer=layer: torch.tanh(layer(t)), h, use_reentrant=False)
        h.float().pow(2).mean().backward()
        return [p.grad.cpu() for p in layers.parameters()]

    reference = run(torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t))
    arena = HostArena(segment_gb=1 / 1024)
    ptrs = []
    for _ in range(2):
        grads = run(offload_to_host(arena, min_bytes=0))
        ptrs.append([s.data_ptr() for s in arena.segments])
        for grad, ref in zip(grads, reference):
            torch.testing.assert_close(grad, ref, rtol=0, atol=0)
    assert arena.used >= 4 * 256 * 64 * 2 and ptrs[0] == ptrs[1]


def test_tensors_saved_twice_are_offloaded_once():
    x = torch.randn(64, 64, requires_grad=True)
    y = torch.randn(64, 64)
    arena = HostArena(segment_gb=1 / 1024)
    with offload_to_host(arena, include_cpu=True, min_bytes=0):
        out = (x * y).sum() + (x * y * 2).sum()  # both products save y
    assert arena.stats["deduplicated"] == 1 and arena.stats["d2h_gb"] * 2**30 == 64 * 64 * 4
    out.backward()
    torch.testing.assert_close(x.grad, 3 * y)


def test_the_last_bytes_stay_on_the_device_once_the_total_is_known():
    """The first step of a layout offloads everything and learns its size; later steps keep the
    last `keep_bytes` where they are and still produce the same gradients."""
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(torch.nn.Linear(256, 256) for _ in range(6))
    x = torch.randn(512, 256)

    def run(**kwargs):
        layers.zero_grad(set_to_none=True)
        with offload_to_host(arena, include_cpu=True, min_bytes=0, layout="mlp", tokens=512, **kwargs):
            h = x
            for layer in layers:
                h = checkpoint(lambda t, layer=layer: torch.tanh(layer(t)), h, use_reentrant=False)
        h.pow(2).mean().backward()
        return [p.grad.clone() for p in layers.parameters()]

    arena = HostArena(segment_gb=1 / 1024)
    reference = run()
    everything = arena.stats["d2h_gb"]
    assert arena.stats["kept_gb"] == 0 and arena.bytes_per_token["mlp"] * 512 == everything * 2**30
    grads = run(keep_bytes=everything * 2**30 / 2)
    assert 0 < arena.stats["kept_gb"] and arena.stats["d2h_gb"] <= everything / 2 + 1e-9
    for grad, ref in zip(grads, reference):
        torch.testing.assert_close(grad, ref, rtol=0, atol=0)


def test_a_freed_tensor_is_not_mistaken_for_a_new_one_at_the_same_address():
    """Packing lets the original tensor go; a new tensor can then reuse its memory."""
    torch.manual_seed(0)
    x = torch.randn(256, 256, requires_grad=True)
    arena = HostArena(segment_gb=1 / 1024)
    with offload_to_host(arena, include_cpu=True, min_bytes=0):
        h = x
        for _ in range(4):
            h = torch.tanh(h)  # each tanh saves its output; the previous one is freed and its memory reused
    h.sum().backward()
    reference = x.detach().clone().requires_grad_(True)
    r = reference
    for _ in range(4):
        r = torch.tanh(r)
    r.sum().backward()
    assert arena.stats["deduplicated"] == 0
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams and pinned memory")
def test_async_copies_and_prefetch_are_exact_on_cuda():
    """Pinned segments, copies on a side stream, prefetch in backward and the device budget only
    move data: the gradients are bit-identical to a run without offload."""
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(torch.nn.Linear(1024, 1024) for _ in range(8)).cuda().to(torch.bfloat16)
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)

    def run(context):
        layers.zero_grad(set_to_none=True)
        with context:
            h = x
            for layer in layers:
                h = checkpoint(lambda t, layer=layer: torch.relu(layer(t)), h, use_reentrant=False)
                h = h * 1.5  # a tensor saved outside the checkpoints too
        h.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        return [p.grad.clone() for p in layers.parameters()]

    reference = run(contextlib.nullcontext())
    arena = HostArena(segment_gb=0.25, pin=True)
    offload = lambda **kw: offload_to_host(arena, async_copies=True, min_bytes=0, layout="t", tokens=4096, **kw)
    runs = [run(offload()), run(offload()), run(offload(keep_bytes=2**30 * arena.stats["d2h_gb"] / 2))]
    assert arena.pinned_gb > 0 and arena.stats["kept_gb"] > 0 and arena.stats["d2h_sync_s"] == 0
    for grads in runs:
        for grad, ref in zip(grads, reference):
            torch.testing.assert_close(grad, ref, rtol=0, atol=0)


def test_saved_tensors_keep_their_layout():
    """A transposed (dense, non-contiguous) saved tensor comes back with the same strides, as with
    .cpu(); a sliced one with gaps comes back contiguous."""
    arena = HostArena(segment_gb=1 / 1024)
    x = torch.randn(32, 48, requires_grad=True)
    with offload_to_host(arena, include_cpu=True, min_bytes=0):
        y = x * torch.randn(48, 32).t()  # saves the transposed tensor
        z = x * torch.randn(32, 96)[:, ::2]  # saves a strided slice with gaps
    assert y.grad_fn._saved_other.stride() == (1, 32)  # unpacked from the arena
    assert z.grad_fn._saved_other.stride() == (48, 1)


def test_a_kept_tensor_saved_twice_is_counted_once():
    """The step's size, which sets the next step's budget, must not grow with what was kept."""
    torch.manual_seed(0)
    weights = [torch.randn(256, 256, requires_grad=True) for _ in range(4)]
    x = torch.randn(512, 256)
    arena = HostArena(segment_gb=1 / 1024)

    def run(**kwargs):
        with offload_to_host(arena, include_cpu=True, min_bytes=0, layout="twice", tokens=512, **kwargs):
            h = x
            for w in weights:
                h = torch.tanh(h @ w) * (h @ w).sigmoid()  # each h is saved by two matmuls
        h.sum().backward()

    run()
    size = arena.bytes_per_token["twice"]
    run(keep_bytes=size * 512 / 2)
    assert arena.stats["kept_gb"] > 0 and arena.bytes_per_token["twice"] == size

