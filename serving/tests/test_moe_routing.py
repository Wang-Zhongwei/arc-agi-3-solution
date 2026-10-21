"""Tests for serving/runtime_patches/20-moe-routing.py (needs torch; skipped without it).

Run with any Python that has torch and pytest, from the repo root:
    python -m pytest -q serving/tests
Set SGLANG_TOPK_PY to a copy of SGLang 2.5.3's sglang/srt/layers/moe/topk.py to
also test that the patch applies to the real file.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import pytest

torch = pytest.importorskip("torch")

PATCH = Path(__file__).resolve().parents[1] / "runtime_patches" / "20-moe-routing.py"
spec = importlib.util.spec_from_file_location("moe_routing_patch", PATCH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


class StandardTopKOutput(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor


class Cfg(NamedTuple):
    top_k: int
    scoring_func: str = "softmax"
    use_grouped_topk: bool = False
    correction_bias: Optional[torch.Tensor] = None
    custom_routing_function: Optional[object] = None
    num_fused_shared_experts: int = 0
    routed_scaling_factor: Optional[float] = None
    allow_routed_experts_capture: bool = True


def renormalised_topk(hidden_states, router_logits, topk_config, **_):
    """What SGLang's default softmax top-k with renormalize=True returns."""
    probs = torch.softmax(router_logits.float(), dim=-1)
    w, ids = torch.topk(probs, topk_config.top_k, dim=-1)
    return StandardTopKOutput(w / w.sum(-1, keepdim=True), ids.to(torch.int32), router_logits)


def load_block(monkeypatch, **env):
    """Execute the appended block as it would run at the end of topk.py."""
    for key in ("SGLANG_ARC_MOE_ROUTING", "SGLANG_ARC_MOE_BASE_K", "SGLANG_ARC_MOE_MIN_K",
                "SGLANG_ARC_MOE_PARAM", "SGLANG_ARC_MOE_DRAFT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    namespace = {"StandardTopKOutput": StandardTopKOutput, "select_experts": renormalised_topk}
    exec(patch.BLOCK, namespace)
    return namespace


def logits(tokens=4, experts=64, seed=0):
    return torch.randn(tokens, experts, generator=torch.Generator().manual_seed(seed)) * 3


def test_unset_leaves_select_experts_alone(monkeypatch):
    ns = load_block(monkeypatch)
    assert ns["select_experts"] is renormalised_topk


@pytest.mark.parametrize("k", [12, 13, 16])  # the experts12/13/16-basenorm arms
def test_base_norm_keeps_the_trained_top10_weights_and_adds_the_rest_raw(monkeypatch, k):
    ns = load_block(monkeypatch, SGLANG_ARC_MOE_ROUTING="base_norm", SGLANG_ARC_MOE_BASE_K="10")
    x = logits()
    out = ns["select_experts"](None, x, Cfg(top_k=k))
    trained = renormalised_topk(None, x, Cfg(top_k=10))
    probs = torch.softmax(x, dim=-1)
    for t in range(x.shape[0]):
        by_id = dict(zip(out.topk_ids[t].tolist(), out.topk_weights[t].tolist()))
        for e, w in zip(trained.topk_ids[t].tolist(), trained.topk_weights[t].tolist()):
            assert by_id[e] == pytest.approx(w, rel=1e-5)  # trained top-10 unchanged
        base = probs[t, trained.topk_ids[t].long()].sum().item()
        extra = [e for e in by_id if e not in set(trained.topk_ids[t].tolist())]
        assert len(extra) == k - 10
        for e in extra:
            assert by_id[e] == pytest.approx(probs[t, e].item() / base, rel=1e-5)
        assert sum(by_id.values()) > 1.0


def test_base_norm_at_the_trained_k_is_a_no_op(monkeypatch):
    ns = load_block(monkeypatch, SGLANG_ARC_MOE_ROUTING="base_norm", SGLANG_ARC_MOE_BASE_K="10")
    x = logits(seed=1)
    out = ns["select_experts"](None, x, Cfg(top_k=10))
    ref = renormalised_topk(None, x, Cfg(top_k=10))
    assert torch.allclose(out.topk_weights, ref.topk_weights, rtol=1e-5)


def test_nucleus_keeps_the_fewest_experts_reaching_p(monkeypatch):
    ns = load_block(monkeypatch, SGLANG_ARC_MOE_ROUTING="nucleus", SGLANG_ARC_MOE_PARAM="0.5")
    x = torch.full((1, 32), -10.0)
    x[0, :4] = torch.tensor([1.0, 0.9, 0.8, 0.0])
    out = ns["select_experts"](None, x, Cfg(top_k=8))
    kept = (out.topk_weights[0] > 0).sum().item()
    chosen = torch.softmax(x[0], -1)[torch.topk(x[0], 8).indices]
    share = chosen / chosen.sum()
    assert share[0] < 0.5 < share[0] + share[1]  # the top two are the fewest reaching 0.5
    assert kept == 2
    assert out.topk_weights[0].sum().item() == pytest.approx(1.0, rel=1e-5)


def test_temp_sharpens(monkeypatch):
    ns = load_block(monkeypatch, SGLANG_ARC_MOE_ROUTING="temp", SGLANG_ARC_MOE_PARAM="0.5")
    x = logits(seed=2)
    out = ns["select_experts"](None, x, Cfg(top_k=10))
    ref = renormalised_topk(None, x, Cfg(top_k=10))
    assert out.topk_weights.max(dim=-1).values.ge(ref.topk_weights.max(dim=-1).values - 1e-6).all()
    assert torch.allclose(out.topk_weights.sum(-1), torch.ones(x.shape[0]), rtol=1e-5)


def test_padded_rows_and_inapplicable_configs_are_untouched(monkeypatch):
    ns = load_block(monkeypatch, SGLANG_ARC_MOE_ROUTING="base_norm")
    x = logits(seed=3)

    def with_padding(h, r, c, **kw):
        out = renormalised_topk(h, r, c)
        ids = out.topk_ids.clone()
        ids[-1] = -1
        return out._replace(topk_ids=ids)

    ns["_arc_select_experts"] = with_padding
    out = ns["select_experts"](None, x, Cfg(top_k=16))
    ref = with_padding(None, x, Cfg(top_k=16))
    assert torch.equal(out.topk_weights[-1], ref.topk_weights[-1])
    ns["_arc_select_experts"] = renormalised_topk
    for cfg in (Cfg(top_k=16, scoring_func="sigmoid"), Cfg(top_k=16, use_grouped_topk=True),
                Cfg(top_k=16, correction_bias=torch.zeros(64)), Cfg(top_k=16, routed_scaling_factor=2.5),
                Cfg(top_k=16, num_fused_shared_experts=1), Cfg(top_k=16, allow_routed_experts_capture=False)):
        out = ns["select_experts"](None, x, cfg)
        assert torch.equal(out.topk_weights, renormalised_topk(None, x, cfg).topk_weights), cfg


def test_patch_appends_once(tmp_path):
    source = os.environ.get("SGLANG_TOPK_PY")
    target = tmp_path / "sglang" / "srt" / "layers" / "moe" / "topk.py"
    target.parent.mkdir(parents=True)
    target.write_text(Path(source).read_text() if source else "\n".join(patch.ANCHORS) + "\n")
    for _ in range(2):
        subprocess.run([sys.executable, str(PATCH), str(tmp_path)], check=True, capture_output=True)
    assert target.read_text().count(patch.MARK) == 1


def test_patch_refuses_an_unknown_topk_py(tmp_path):
    target = tmp_path / "sglang" / "srt" / "layers" / "moe" / "topk.py"
    target.parent.mkdir(parents=True)
    target.write_text("def something_else(): pass\n")
    result = subprocess.run([sys.executable, str(PATCH), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and "missing" in result.stderr + result.stdout
