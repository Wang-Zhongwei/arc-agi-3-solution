"""train_rft's gradient accumulation: samples run one at a time, each adding its share of the batch's trained
tokens times its own gradient, give Σ (n_i / N) ∇L_i (every trained token of the batch weighted once), and a
sample whose first attempt ran out of memory after touching the gradients leaves no trace of that attempt."""

import zlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")
peft = pytest.importorskip("peft")

from inference.rl import layer_patches, qsa, smoke_lora, train_rft  # noqa: E402
from test_layer_patches import tiny  # noqa: E402


def sample(seed: int, n: int, trained: list[tuple[int, int]]):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 1000, (1, n), generator=g)
    labels = torch.full_like(ids, -100)
    for a, b in trained:
        labels[0, a:b] = ids[0, a:b]
    return {"input_ids": ids, "labels": labels, "mm_token_type_ids": torch.zeros_like(ids)}


@pytest.fixture
def model():
    m = tiny().train()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m = peft.get_peft_model(m, peft.LoraConfig(r=4, lora_alpha=8, target_modules=smoke_lora.ATTENTION_GDN))
    with torch.no_grad():  # lora_B starts at zero: give it values so lora_A gets gradients too
        for name, p in m.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.02, generator=torch.Generator().manual_seed(zlib.crc32(name.encode())))
    qsa.patch_qsa_attention(chunk_size=16, fused=False)
    layer_patches.patch_for_training(m, token_chunk=16, recompute="parts")
    yield m
    layer_patches.unpatch(m)
    qsa.unpatch_qsa_attention()


def lora_grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}


def test_accumulated_samples_weight_every_trained_token_once(model, monkeypatch):
    samples = [sample(1, 48, [(30, 40)]), sample(2, 64, [(20, 30), (50, 64)])]
    counts = [int((s["labels"] != -100).sum()) for s in samples]
    total = sum(counts)
    each = []
    for s in samples:  # the standard answer: each sample's own mean-loss gradient, from a plain step
        smoke_lora.step(model, s, "cpu", prefix_stop_grad=True, keep_grads="ref")
        each.append(model._rft_grads["ref"])
    expected = {n: sum(c / total * g[n] for c, g in zip(counts, each)) for n in each[0]}

    real_step, calls = smoke_lora.step, []

    def failing_once(m, s, device, **kwargs):  # the first attempt dirties the gradients, then runs out of memory
        calls.append(kwargs["gpu_keep_gb"])
        if len(calls) == 1:
            for p in m.parameters():
                if p.requires_grad:
                    p.grad = torch.full_like(p, 7.0)
            raise torch.cuda.OutOfMemoryError("test")
        return real_step(m, s, device, **kwargs)

    monkeypatch.setattr(smoke_lora, "step", failing_once)
    model.zero_grad(set_to_none=True)
    results = [train_rft.run_sample(model, "cpu", s, c / total) for s, c in zip(samples, counts)]
    got = lora_grads(model)
    # the retry runs with recompute "layer": the same gradient, other tensors saved
    assert results[0]["retries"] == 1 and results[0]["recompute_mode"] == "layer" and len(calls) == 3
    assert results[1]["retries"] == 0 and results[1]["recompute_mode"] == "parts"
    assert got.keys() == expected.keys() and len(got) > 20
    # FP32 rounding differs between scaling the loss before backward and the gradients after: up to ~1e-4
    # relative in parameters whose gradient norm is ~1e-6 (16 hash seeds); a wrong weight or a lost sample is O(1)
    rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-30))
    for name in expected:
        assert rel(got[name], expected[name]) < 1e-3, name
    flat = lambda d: torch.cat([d[n].flatten() for n in sorted(d)])
    assert rel(flat(got), flat(expected)) < 1e-4


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    lrs = [train_rft.lr_at(s, 100, 5, 0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.2) and lrs[4] == pytest.approx(1.0)
    assert max(lrs) == pytest.approx(1.0) and lrs[-1] == pytest.approx(0.1, abs=1e-3)
    assert all(a >= b for a, b in zip(lrs[4:], lrs[5:]))


def test_arena_cap_leaves_the_ple_table_its_page_cache():
    # H200 pod: 251 GB limit, 95.5 GiB PLE table, 14 GiB process -> 251 - 95.5 - 14 - 12 - 10
    assert train_rft.arena_cap_gb(95.5, 14.0, 251.0) == 119.5
    assert train_rft.arena_cap_gb(95.5, 14.0, None) == 0.0 and train_rft.arena_cap_gb(95.5, 14.0, 100.0) == 0.0


def test_samples_whose_checkpoints_do_not_fit_the_arena_recompute_by_layer():
    labels = torch.full((1, 100), -100)
    labels[0, 61:70] = 1  # first trained token at 61: positions 60.. are after the prefix
    sample = {"input_ids": torch.zeros(1, 100, dtype=torch.long), "labels": labels}
    assert train_rft.suffix_tokens(sample, True) == 40 and train_rft.suffix_tokens(sample, False) == 100
    cap = 100 * 2**30
    assert train_rft.choose_recompute(40_000, cap, {}, True) == "parts"  # 40K x 2.4 MB = 96 GB
    assert train_rft.choose_recompute(50_000, cap, {}, True) == "layer"  # 120 GB
    assert train_rft.choose_recompute(50_000, cap, {"parts+prefix": 2.0e6}, True) == "parts"  # measured: 100 GB
    assert train_rft.choose_recompute(10**6, None, {}, True) == "parts"  # no cap: never "layer"
