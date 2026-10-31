"""dflash_train's drafter on CPU against what vLLM serves: the grouped conv and the selector's pair scores against
vLLM's own formulas (qwen3_dflash2.py, copied below), the attention mask, blocks trained together equal to blocks
drafted alone, the checkpoint's tensor names, anchors, the loss terms and the acceptance chain."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
F = torch.nn.functional

from inference.rft import dflash_train as dt  # noqa: E402

# incoai/Qwen3.8-Flash-Next-DFlash2 (a22a1744): every tensor of model.safetensors, layers.N for N in 0..4
CHECKPOINT_NAMES = {
    "candidate_selector.hidden_projection.weight", "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook", "fc.weight", "hidden_norm.weight", "norm.weight",
    *(f"layers.{n}.{name}" for n in range(5) for name in (
        "attention_conv.base_kernel", "attention_conv.kernel_projection.weight", "input_layernorm.weight",
        "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp_conv.base_kernel",
        "mlp_conv.kernel_projection.weight", "post_attention_layernorm.weight", "self_attn.k_norm.weight",
        "self_attn.k_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.q_proj.weight",
        "self_attn.v_proj.weight")),
}
REAL_CONFIG = {
    "hidden_size": 2560, "intermediate_size": 7680, "num_hidden_layers": 5, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 128, "rms_norm_eps": 1e-6, "sliding_window": 2048, "vocab_size": 248320,
    "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248077,
                      "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 15, 25, 35, 45]},
}


def tiny_config(window=64):
    return {
        "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6, "sliding_window": window, "vocab_size": 50,
        "rope_parameters": {"rope_theta": 10000},
        "dflash_config": {"block_size": 8, "conv_group_size": 4, "conv_kernel_size": 2, "mask_token_id": 49,
                          "selector_rank": 6, "selector_top_k": 5, "target_layer_ids": [1, 2]},
    }


def tiny_draft(window=64, seed=0):
    torch.manual_seed(seed)
    model = dt.DFlash2Draft(tiny_config(window), block=8)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if name.endswith("norm.weight") else 0.3 * torch.randn_like(p))
    return model


# vLLM qwen3_dflash2.py (e975732 rc3 runtime): the CPU path of the grouped conv and the selector's edge scores
def vllm_grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


def vllm_score_edges(predecessor_table, successor_table, candidate_ids, unary_logits, hidden, anchor_token_ids, top_k):
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]), dim=1)
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors * hidden[:, :, None], successors)


def test_grouped_conv_matches_vllm_prepare_and_finish():
    torch.manual_seed(0)
    conv = dt.GroupedConv(hidden=32, taps=2, group=4)
    with torch.no_grad():
        conv.base_kernel.normal_()
    x, y = torch.randn(3, 8, 32), torch.randn(3, 8, 32)
    mine0, deltas = conv.prepare(x)
    mine1 = conv.convolve(y, deltas, 1)
    flat = conv.kernel_projection(x.flatten(0, 1)).reshape(24, 2, 2, 8)  # vLLM's prepare: (rows, 2, taps, groups)
    ref0 = vllm_grouped_conv(x.flatten(0, 1), flat[:, 0], conv.base_kernel[0], 8, 8, 4, 2)
    ref1 = vllm_grouped_conv(y.flatten(0, 1), flat[:, 1], conv.base_kernel[1], 8, 8, 4, 2)
    assert torch.allclose(mine0.flatten(0, 1), ref0, atol=1e-5)
    assert torch.allclose(mine1.flatten(0, 1), ref1, atol=1e-5)


def test_pair_scores_are_vllm_edge_scores_after_the_true_previous_token():
    torch.manual_seed(0)
    selector = dt.CandidateSelector(hidden=32, vocab=50, rank=6)
    with torch.no_grad():
        selector.predecessor_codebook.normal_()
        selector.successor_codebook.normal_()
    steps, k = 7, 5
    candidates = torch.randint(0, 50, (1, steps, k))
    unary = torch.randn(1, steps, k)
    hidden = torch.randn(1, steps, 32)
    anchor = torch.tensor([7])
    edges = vllm_score_edges(selector.predecessor_codebook, selector.successor_codebook, candidates, unary,
                             selector.hidden_projection(hidden), anchor, k)  # [1, steps, previous, candidate]
    for s in range(steps):
        for a in range(k):
            previous = anchor if s == 0 else candidates[0, s - 1, a:a + 1]
            mine = unary[0, s] + selector.pair_scores(candidates[0, s:s + 1], hidden[0, s:s + 1], previous)[0]
            assert torch.allclose(mine, edges[0, s, 0 if s == 0 else a], atol=1e-5)


def test_block_mask_is_own_context_window_and_own_block():
    anchors, context, block, window = torch.tensor([5, 20]), 30, 8, 6
    q = torch.arange(16)[:, None]
    kv = torch.arange(30 + 16)[None]
    mask = dt.block_allowed(q, kv, anchors, context, block, window)
    for r in range(16):
        p, position = int(anchors[r // 8]), int(anchors[r // 8]) + 1 + r % 8
        for j in range(46):
            expected = (j <= p and position - j < window) if j < 30 else (j - 30) // 8 == r // 8
            assert bool(mask[r, j]) == expected, (r, j)


@pytest.mark.parametrize("window", [64, 6])
def test_blocks_trained_together_equal_blocks_drafted_alone(window):
    model = tiny_draft(window)
    features = torch.randn(40, 64)
    anchors = torch.tensor([3, 11, 12, 30])
    embeds = torch.randn(4, 8, 32)
    together = model(embeds, anchors, features, attention="dense")
    for i, p in enumerate(anchors.tolist()):
        alone = model(embeds[i:i + 1], anchors[i:i + 1], features, attention="dense")
        assert torch.allclose(together[i], alone[0], atol=1e-5)
        # served: the context ends at p (the target has not run past the last accepted token)
        served = model(embeds[i:i + 1], anchors[i:i + 1], features[:p + 1], attention="dense")
        assert torch.allclose(together[i], served[0], atol=1e-5)


def test_flex_attention_matches_dense():
    model = tiny_draft(6)
    features, anchors, embeds = torch.randn(40, 64), torch.tensor([3, 11, 30]), torch.randn(3, 8, 32)
    dense = model(embeds, anchors, features, attention="dense")
    try:
        flex = model(embeds, anchors, features, attention="flex")
    except Exception as exc:  # noqa: BLE001 - flex_attention needs a compiler backend on this host
        pytest.skip(f"flex_attention unavailable on CPU here: {type(exc).__name__}")
    assert torch.allclose(dense, flex, atol=1e-4)


def test_checkpoint_names_and_export_round_trip(tmp_path):
    from safetensors.torch import load_file, save_file

    with torch.device("meta"):
        real = dt.DFlash2Draft(REAL_CONFIG, block=8)
    assert set(real.state_dict()) == CHECKPOINT_NAMES
    model = tiny_draft()
    source = tmp_path / "dflash2"
    source.mkdir()
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps(tiny_config()))
    loaded = dt.load_draft(source, block=8)
    dt.export_draft(loaded, source, tmp_path / "out")
    original, exported = load_file(source / "model.safetensors"), load_file(tmp_path / "out" / "model.safetensors")
    assert original.keys() == exported.keys()
    assert all(torch.equal(original[k], exported[k]) for k in original)
    assert json.loads((tmp_path / "out" / "config.json").read_text()) == tiny_config()


def test_draw_anchors_keep_the_first_draft_trained_and_the_block_inside():
    labels = torch.full((40,), -100)
    labels[10:20] = 1
    labels[35:40] = 1
    anchors = dt.draw_anchors(labels, 100, 8, torch.Generator().manual_seed(0))
    assert anchors.tolist() == list(range(8, 18))  # x[p+2] trained; 33..37 would also need p + 8 <= 39
    few = dt.draw_anchors(labels, 3, 8, torch.Generator().manual_seed(0))
    assert len(few) == 3 and set(few.tolist()) <= set(anchors.tolist())


def test_row_terms_acceptance_is_overlap_with_the_path_distribution():
    model = tiny_draft()
    lm_head = torch.nn.Linear(32, 50, bias=False)
    hidden = torch.randn(6, 32, requires_grad=True)
    previous = torch.randint(0, 50, (6,))
    ids = torch.stack([torch.randperm(50)[:4] for _ in range(6)])
    probs = torch.softmax(torch.randn(6, 4), -1)
    unary, path, accept, greedy = dt.row_terms(model, lm_head, hidden, previous, ids, probs, temperature=0.7)
    (unary + path).sum().backward()
    assert hidden.grad is not None and model.candidate_selector.successor_codebook.grad is not None
    with torch.no_grad():
        logits = lm_head(hidden)
        candidates = logits.topk(5, -1).indices
        q = ((logits.gather(-1, candidates) + model.candidate_selector.pair_scores(candidates, hidden, previous))
             / 0.7).softmax(-1)
        pc = dt.candidate_probs(ids, probs, candidates)
        assert torch.allclose(accept, torch.minimum(pc, q).sum(-1), atol=1e-5)
        assert torch.allclose(greedy, pc.gather(-1, q.argmax(-1, keepdim=True)).squeeze(-1))
        expected_unary = -(probs * (logits / 0.7).log_softmax(-1).gather(-1, ids)).sum(-1)
        assert torch.allclose(unary, expected_unary, atol=1e-5)


def test_chain_rates_use_fully_trained_blocks():
    accept = torch.tensor([[0.5, 0.5], [1.0, 0.2], [0.9, 0.9]])
    trained = torch.tensor([[True, True], [True, True], [True, False]])
    chain = dt.chain_rates(accept, trained)
    assert chain["anchors"] == 2
    assert chain["rates"] == pytest.approx([0.75, (0.25 + 0.2) / 2])
    assert chain["length"] == pytest.approx(1 + 0.75 + 0.225)
