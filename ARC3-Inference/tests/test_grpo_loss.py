"""GRPO's per-token pieces (inference/grpo/loss.py): the sampler's processed logprob, the clipped objective,
stochastic BF16 rounding and Muon."""

import pytest

torch = pytest.importorskip("torch")

from inference.grpo.loss import (  # noqa: E402
    Muon,
    _kept_logprob,
    grpo_objective,
    processed_logprobs,
    stochastic_bf16,
)


def vllm_reference(logits, k, p, temperature):
    """vLLM's apply_top_k_top_p (sort ascending, mask below the k-th value, mask ascending cumulative <= 1 - p,
    keep the largest), then log_softmax: the processed logprobs over the whole vocabulary."""
    x = logits.float() / temperature
    s, order = x.sort(dim=-1)
    s = s.masked_fill(s < s[:, -k:-k + 1], float("-inf"))
    cumulative = s.softmax(-1).cumsum(-1)
    drop = cumulative <= 1 - p
    drop[:, -1] = False
    s = s.masked_fill(drop, float("-inf"))
    return torch.full_like(x, float("-inf")).scatter(1, order, s).log_softmax(-1)


@pytest.mark.parametrize("seed,forced_ties", [(0, True), (1, False), (2, False)])
def test_processed_logprob_matches_vllm_on_sampled_tokens(seed, forced_ties):
    """BF16 logits tie often (vLLM keeps every tie at the k-th value); tokens are sampled from vLLM's processed
    distribution, as rollouts are, so every one of them must be kept and get vLLM's logprob."""
    torch.manual_seed(seed)
    logits = (torch.randn(400, 5000) * 3).bfloat16().float()
    if forced_ties:
        logits[:20, 100:110] = logits[:20].topk(20).values[:, -1:]
    reference = vllm_reference(logits, 20, 0.95, 0.7)
    tokens = torch.multinomial(reference.exp(), 1).squeeze(1)
    ours = _kept_logprob(logits, tokens, 0.7, 20, 0.95)
    expected = reference.gather(1, tokens[:, None]).squeeze(1)
    assert torch.isfinite(ours).all()
    torch.testing.assert_close(ours, expected, rtol=0, atol=1e-5)


def test_chunked_checkpointed_logprobs_equal_the_direct_ones_with_gradients():
    torch.manual_seed(3)
    head = torch.nn.Linear(32, 3000, bias=False)
    hidden = torch.randn(150, 32, requires_grad=True)
    with torch.no_grad():
        tokens = torch.multinomial(vllm_reference(head(hidden), 20, 0.95, 0.7).exp(), 1).squeeze(1)
    chunked, kept = processed_logprobs(head, hidden, tokens, chunk=32)
    direct = _kept_logprob(head(hidden), tokens, 0.7, 20, 0.95)
    assert kept.all()
    torch.testing.assert_close(chunked, direct)
    g1 = torch.autograd.grad(chunked.sum(), hidden)[0]
    g2 = torch.autograd.grad(direct.sum(), hidden)[0]
    torch.testing.assert_close(g1, g2)


def test_objective_and_its_gradient():
    torch.manual_seed(4)
    logp = (torch.randn(200) - 2).requires_grad_()
    behavior = logp.detach() + torch.randn(200) * 0.3
    kept = torch.ones(200, dtype=torch.bool)
    kept[:5] = False
    for advantage in (0.7, -0.7):
        loss, stats = grpo_objective(logp, behavior, kept, advantage)
        grad = torch.autograd.grad(loss, logp)[0]
        ratio = (logp.detach() - behavior).exp()
        use = kept & (ratio >= 0.5) & (ratio <= 2.0)
        # min(r A, clip(r) A): the gradient flows where the unclipped term is the smaller one
        unclipped_smaller = ratio * advantage <= ratio.clamp(0.8, 1.28) * advantage
        expected = torch.where(use & unclipped_smaller, -advantage * ratio, torch.zeros(200))
        torch.testing.assert_close(grad, expected)
        assert stats["outside_kept"] == 5 and stats["used"] == int(use.sum())


def test_stochastic_rounding_is_unbiased_and_exact_on_bf16_values():
    x = torch.full((400_000,), 0.0213) + 1e-6  # nearest-BF16 rounding loses all of the 1e-6
    generator = torch.Generator().manual_seed(0)
    rounded = stochastic_bf16(x, generator).double()
    assert abs(float(rounded.mean()) - float(x.double()[0])) < 2e-7
    assert abs(float(stochastic_bf16(-x, generator).double().mean()) + float(x.double()[0])) < 2e-7
    exact = torch.randn(1000).bfloat16().float()
    assert torch.equal(stochastic_bf16(exact, generator).float(), exact)


def test_muon_update_has_adamw_like_rms():
    p = torch.nn.Parameter(torch.randn(16, 64))
    p.grad = torch.randn(16, 64)
    before = p.detach().clone()
    Muon([p], lr=1e-3).step()
    rms = float((p.detach() - before).pow(2).mean().sqrt())
    assert 1.5e-4 < rms < 2.5e-4  # 0.2 x lr, up to Newton-Schulz's approximation
