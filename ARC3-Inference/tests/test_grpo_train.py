"""The GRPO trainer's pieces on a tiny random qwen4_exp (inference/grpo/train_grpo.py): a step through train_rft's
machinery gives the gradient of the objective computed directly; two ranks' summed gradients equal one rank's;
the merged export carries checkpoint names and is unbiased; the probe sees the adapter; image tokens are checked."""

import zlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")
peft = pytest.importorskip("peft")

from inference.grpo import episodes, train_grpo  # noqa: E402
from inference.grpo.loss import _kept_logprob, grpo_objective  # noqa: E402
from inference.rft import layer_patches, qsa, smoke_lora, train_rft  # noqa: E402
from test_layer_patches import tiny  # noqa: E402

SETTINGS = SimpleNamespace(temperature=0.7, top_k=0, top_p=1.0, eps_low=0.2, eps_high=0.28, ratio_min=0.5,
                           ratio_max=2.0)


def lora_model():
    m = tiny().train()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m = peft.get_peft_model(m, peft.LoraConfig(r=4, lora_alpha=8, target_modules=smoke_lora.ATTENTION_GDN))
    with torch.no_grad():  # lora_B starts at zero: give it values so the adapter changes the model
        for name, p in m.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.02, generator=torch.Generator().manual_seed(zlib.crc32(name.encode())))
    return m


@pytest.fixture
def model():
    m = lora_model()
    qsa.patch_qsa_attention(chunk_size=16, fused=False)
    layer_patches.patch_for_training(m, token_chunk=16, recompute="parts")
    yield m
    layer_patches.unpatch(m)
    qsa.unpatch_qsa_attention()


def episode(seed: int, n: int, spans: list[tuple[int, int]]) -> episodes.Episode:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 1000, (n,), generator=g).tolist()
    targets = [p for a, b in spans for p in range(a, b)]
    ep = episodes.Episode(stem=f"ls20-x_p{seed}", game="ls20", info={}, requests=[], ids=ids, targets=targets)
    return ep


def direct_loss(model, ep, behavior, advantage, norm):
    base = model.get_base_model()
    ids = torch.tensor(ep.ids)[None]
    targets = torch.tensor(ep.targets)
    hidden = base.model(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids), use_cache=False).last_hidden_state[0]
    logp = _kept_logprob(base.lm_head(hidden[targets - 1]), ids[0, targets], SETTINGS.temperature, SETTINGS.top_k,
                         SETTINGS.top_p)
    loss, _ = grpo_objective(logp, behavior, torch.isfinite(logp), advantage)
    return loss / norm, logp.detach()


def lora_grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}


def test_step_gradient_equals_the_direct_objective(model):
    eps = [episode(1, 48, [(20, 30), (40, 48)]), episode(2, 64, [(10, 25)])]
    advantages = [0.6, -0.4]
    norm = float(sum(len(e.targets) for e in eps))
    model.zero_grad(set_to_none=True)
    behaviors = []
    for ep, a in zip(eps, advantages):  # the reference: one plain forward over the whole sequence, no patches' help
        with torch.no_grad():
            _, logp = direct_loss(model, ep, torch.zeros(len(ep.targets)), a, norm)
        behavior = logp + 0.05 * torch.randn(len(ep.targets), generator=torch.Generator().manual_seed(len(ep.ids)))
        behaviors.append(behavior)
        loss, _ = direct_loss(model, ep, behavior, a, norm)
        loss.backward()
    expected = lora_grads(model)
    model.zero_grad(set_to_none=True)
    for ep, a, behavior in zip(eps, advantages, behaviors):
        ep.behavior = behavior.tolist()
        inputs = train_grpo.build_inputs(ep)
        result = train_rft.run_sample(model, "cpu", inputs, 1.0, False,
                                      loss_fn=train_grpo.make_loss_fn(ep.behavior, a, norm, SETTINGS))
        assert "error" not in result and result["loss_stats"]["tokens"] == len(ep.targets)
    got = lora_grads(model)
    assert got.keys() == expected.keys() and len(got) > 20
    flat = lambda d: torch.cat([d[n].flatten() for n in sorted(d)])  # noqa: E731
    rel = float((flat(got) - flat(expected)).norm() / flat(expected).norm())
    assert rel < 1e-4, rel


def _rank_worker(rank, world, path, init):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    torch.manual_seed(0)
    m = lora_model()
    qsa.patch_qsa_attention(chunk_size=16, fused=False)
    layer_patches.patch_for_training(m, token_chunk=16, recompute="parts")
    eps = [episode(1, 48, [(20, 30)]), episode(2, 64, [(10, 25)])]
    norm = float(sum(len(e.targets) for e in eps))
    mine = [eps[rank]] if world == 2 else eps
    m.zero_grad(set_to_none=True)
    for ep in mine:
        ep.behavior = [-3.0] * len(ep.targets)
        train_rft.run_sample(m, "cpu", train_grpo.build_inputs(ep), 1.0, False,
                             loss_fn=train_grpo.make_loss_fn(ep.behavior, 0.5, norm, SETTINGS))
    train_grpo.sum_gradients(train_grpo.trainable(m))
    torch.save({n: p.grad for n, p in m.named_parameters() if p.requires_grad}, f"{path}_{world}_{rank}.pt")
    dist.destroy_process_group()


def test_two_ranks_sum_to_the_gradient_of_one(tmp_path):
    import torch.multiprocessing as mp

    path = str(tmp_path / "grads")
    mp.spawn(_rank_worker, args=(2, path, f"file://{tmp_path}/init2"), nprocs=2, join=True)
    mp.spawn(_rank_worker, args=(1, path, f"file://{tmp_path}/init1"), nprocs=1, join=True)
    one = torch.load(f"{path}_1_0.pt")
    for rank in (0, 1):
        two = torch.load(f"{path}_2_{rank}.pt")
        for name in one:
            torch.testing.assert_close(two[name], one[name], rtol=1e-4, atol=1e-7)


def test_merged_export_has_checkpoint_names_and_is_unbiased():
    model = lora_model()
    keys = sorted(k for k in tiny().state_dict())  # the tiny base's own names stand in for the checkpoint index
    first, stats = train_grpo.merged_weights(model, keys, seed=1)
    again, _ = train_grpo.merged_weights(model, keys, seed=1)
    # 3 GDN layers x 5 projections + 1 attention layer x 4 in the tiny model
    assert first.keys() <= set(keys) and len(first) == stats["tensors"] == 19
    assert all(torch.equal(first[k], again[k]) for k in first)  # seeded
    modules = dict(train_grpo.lora_modules(model))
    name = next(n for n in sorted(modules) if n.endswith("q_proj"))
    module = modules[name]
    exact = module.base_layer.weight.float() + module.scaling["default"] * (
        module.lora_B["default"].weight.float() @ module.lora_A["default"].weight.float())
    key = next(k for k in first if k.endswith(name.split(".", 1)[1] + ".weight") or k == name + ".weight")
    mean = sum(train_grpo.merged_weights(model, keys, seed=s)[0][key].double() for s in range(40)) / 40
    nearest_error = float((exact.bfloat16().double() - exact.double()).abs().mean())
    assert float((mean - exact.double()).abs().mean()) < nearest_error  # averaging stochastic roundings converges


def test_probe_sees_the_adapter():
    model = lora_model()
    ids = torch.randint(0, 1000, (40,), generator=torch.Generator().manual_seed(5)).tolist()
    out = train_grpo.probe_logprobs(model, "cpu", ids, 20)
    with model.disable_adapter(), torch.no_grad():
        base = model.get_base_model()
        x = torch.tensor(ids)[None]
        logits = base.lm_head(base.model(input_ids=x, mm_token_type_ids=torch.zeros_like(x),
                                         use_cache=False).last_hidden_state[0, 19:-1]).float()
        direct = logits.log_softmax(-1).gather(1, x[0, 20:, None]).squeeze(1)
    torch.testing.assert_close(torch.tensor(out["base"]), direct, rtol=0, atol=1e-5)
    assert max(abs(a - b) for a, b in zip(out["current"], out["base"])) > 1e-4


def test_image_tokens_must_match_the_processor():
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    processor = SimpleNamespace(image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2,
                                                                      temporal_patch_size=2))
    image_path = None
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        image_path = f"{d}/0000.png"
        Image.new("RGB", (768, 768), (1, 2, 3)).save(image_path)
        pad = train_grpo.IMAGE_TOKEN
        good = episodes.Episode(stem="x", game="x", info={}, requests=[], ids=[1, 2] + [pad] * 576 + [3],
                                targets=[2], images=[image_path])
        inputs = train_grpo.build_inputs(good, processor)
        assert inputs["image_grid_thw"].tolist() == [[1, 48, 48]] and int(inputs["mm_token_type_ids"].sum()) == 576
        bad = episodes.Episode(stem="y", game="x", info={}, requests=[], ids=[1, 2] + [pad] * 575 + [3],
                               targets=[2], images=[image_path])
        with pytest.raises(ValueError, match="image tokens"):
            train_grpo.build_inputs(bad, processor)
