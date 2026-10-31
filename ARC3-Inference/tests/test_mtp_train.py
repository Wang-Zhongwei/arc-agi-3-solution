"""mtp_train's pieces on CPU: NVFP4 decode, the target's sampling distribution, the soft loss and its acceptance,
the draft-chain rates, and the head's checkpoint names on a tiny Qwen4Exp config."""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")

from inference.rft import mtp_train  # noqa: E402

LORD_NON_EXPERT = {
    "mtp.fc_embedding.weight", "mtp.fc_hidden.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight", "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    *(f"mtp.layers.0.{hc}_hyper_connection.{w}.weight" for hc in ("attn", "mlp")
      for w in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")),
    *(f"mtp.layers.0.self_attn.{w}.weight" for w in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")),
    *(f"mtp.layers.0.self_attn.indexer.{w}.weight" for w in ("index_qk_proj", "q_layernorm", "k_layernorm")),
    "mtp.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.shared_expert_gate.weight",
    *(f"mtp.layers.0.mlp.shared_expert.{w}.weight" for w in ("gate_proj", "up_proj", "down_proj")),
}


def tiny_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    return Qwen4ExpTextConfig(
        hidden_size=64, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, hc_count=4, hc_lowrank=8, num_experts=8,
        num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32, vocab_size=128,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=4,
        ple_layer_ids=[], mtp_num_hidden_layers=0)


def test_dequant_nvfp4_low_nibble_first_and_scales():
    packed = torch.tensor([[0x21, 0xF9]], dtype=torch.uint8)  # codes 1, 2, 9, 15 -> 0.5, 1.0, -0.5, -6.0
    packed = packed.repeat(1, 8)  # 32 inputs: two scale groups of 16
    scale = torch.tensor([[2.0, 4.0]]).to(torch.float8_e4m3fn)
    w = mtp_train.dequant_nvfp4(packed, scale, torch.tensor([2.0]))
    assert w.shape == (1, 32)
    assert w[0, :4].tolist() == [0.5, 1.0, -0.5, -6.0]  # x scale 2 / global 2
    assert w[0, 16:20].tolist() == [1.0, 2.0, -1.0, -12.0]  # x scale 4 / global 2


def test_target_distribution_is_tempered_top_k_top_p():
    head = torch.nn.Linear(4, 6, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(6, 4))
    hidden = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    ids, probs = mtp_train.target_distribution(head, hidden, temperature=0.5, top_k=3, top_p=0.95)
    assert ids[0].tolist() == [0, 1, 2]
    tempered = torch.tensor([6.0, 4.0, 2.0]).softmax(0)  # logits / 0.5, top 3
    kept = tempered.cumsum(0) - tempered < 0.95  # the third token starts past 0.95
    expected = torch.where(kept, tempered, torch.zeros(3))
    assert torch.allclose(probs[0], expected / expected.sum())
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-6)


def test_soft_cross_entropy_gradient_and_acceptance():
    head = torch.nn.Linear(4, 5, bias=False)
    hidden = torch.randn(3, 4, requires_grad=True)
    ids = torch.tensor([[0, 1], [2, 3], [4, 0]])
    probs = torch.tensor([[0.7, 0.3], [1.0, 0.0], [0.5, 0.5]])
    loss, accepted = mtp_train.soft_cross_entropy(head, hidden, ids, probs, None, chunk=2)
    logq = torch.log_softmax(head(hidden).float(), -1)
    reference = -(probs * logq.gather(-1, ids)).sum()
    assert torch.allclose(loss, reference, atol=1e-5)
    loss.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    draft = head(hidden).argmax(-1)
    expected = (probs * (ids == draft[:, None])).sum(-1)
    assert torch.allclose(accepted, expected)


def test_chain_rates_multiply_along_the_anchor_chain():
    rows = torch.tensor([2, 3, 4, 7])
    accepted = [torch.tensor([0.5, 0.8, 0.9, 1.0]), torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.tensor([1.0, 0.2, 0.4, 1.0])]
    out = mtp_train.chain_rates(accepted, rows, n=10)
    # only anchor 2 has rows 2, 3, 4 trained: a1(2)=0.5, a2(3)=0.5, a3(4)=0.4
    assert out["anchors"] == 1
    assert [round(r, 6) for r in out["rates"]] == [0.5, 0.25, 0.1]
    assert math.isclose(out["length"], 1.85)


def test_head_names_are_lords_checkpoint_names_and_it_runs():
    from inference.rft.qsa import patch_qsa_attention
    patch_qsa_attention(256, fused=False)
    config = tiny_config()
    head = mtp_train.MTPHead(config)
    names = set(mtp_train.checkpoint_tensor_names(head).values())
    assert names == LORD_NON_EXPERT
    assert {n for n, _ in head.named_parameters() if not mtp_train.trainable(n)} == {
        "layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.down_proj",
        "layers.0.self_attn.indexer.index_qk_proj.weight", "layers.0.self_attn.indexer.q_layernorm.weight",
        "layers.0.self_attn.indexer.k_layernorm.weight"}
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.02)
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    rotary = m.Qwen4ExpTextRotaryEmbedding(config)
    n = 12
    embeds, hidden = torch.randn(1, n, 64), torch.randn(1, n, 256)
    position_embeddings = rotary(embeds, torch.arange(n).view(1, 1, -1).expand(3, 1, -1))
    out, h2 = head(hidden, embeds, position_embeddings)
    assert out.shape == (1, n, 64) and h2.shape == (1, n, 256)
    out.sum().backward()
    assert head.fc_hidden.weight.grad is not None


def test_game_round_robin_draws_each_game_in_turn():
    samples = [{"game": g} for g in ["a"] * 5 + ["b"] * 2 + ["c"] * 4]
    order = mtp_train.game_round_robin(samples, 7, seed=0)
    assert len(order) == len(set(order)) == 7
    games = [samples[i]["game"] for i in order]
    assert sorted(games[:3]) == ["a", "b", "c"] and sorted(games[3:6]) == ["a", "b", "c"]
    assert games[6] in ("a", "c")  # b is exhausted after two rounds
    assert mtp_train.game_round_robin(samples, 7, seed=0) == order


def test_wandb_metrics_map_the_log_records():
    eval_row = dict(event="eval", step=100, length=2.9, rows=5, rates=[0.8, 0.6, 0.5], loss=[1.0, 1.2, 1.4])
    assert mtp_train.wandb_metrics(eval_row) == {"eval/step": 100, "eval/length": 2.9, "eval/rows": 5,
                                                 "eval/rate1": 0.8, "eval/rate2": 0.6, "eval/rate3": 0.5,
                                                 "eval/loss1": 1.0, "eval/loss2": 1.2, "eval/loss3": 1.4}
    sample = dict(event="sample", position=3, tokens=9, rows=4, total_s=1.0, target_s=0.5, losses=[1.0],
                  chain=dict(anchors=2, rates=[0.7], length=1.7))
    assert mtp_train.wandb_metrics(sample)["sample/length"] == 1.7
    assert mtp_train.wandb_metrics(dict(event="loaded")) == {}
