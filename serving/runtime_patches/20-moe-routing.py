"""Runtime patch: MoE routing variants for the installed SGLang 2.5.3.

Run by the launcher as `python 20-moe-routing.py <site-packages>`. It appends a
wrapper around `select_experts` in sglang/srt/layers/moe/topk.py.
`TopK.forward_cuda` looks that name up at call time, so the wrapper sees every
standard top-k routing decision, including inside CUDA-graph capture.

The wrapper does nothing unless SGLANG_ARC_MOE_ROUTING is set in the server's
environment (an arm's "serve_env"). Then it recomputes the weights of the
experts the router already chose, from the router's own softmax probabilities:

  base_norm  w_i = p_i / (sum of the top SGLANG_ARC_MOE_BASE_K probabilities).
             With more experts per token than the model was trained with
             (num_experts_per_tok via TEXT_CONFIG_OVERRIDE), the original
             experts keep exactly their trained weights and the extra ones add
             their small raw probability on top, instead of SGLang's default
             renormalisation over all k, which shrinks every trained weight.
  nucleus    keep the fewest experts whose probability reaches
             SGLANG_ARC_MOE_PARAM of the chosen mass (at least
             SGLANG_ARC_MOE_MIN_K), zero the rest and renormalise.
  temp       w_i proportional to p_i ** (1 / SGLANG_ARC_MOE_PARAM), renormalised.

The expert ids are unchanged, so compute is the same as the configured k.
Routing is left untouched, with a one-time log line, wherever the formula does
not apply: non-softmax scoring, grouped top-k, a correction bias, a custom
routing function, fused shared experts, routed scaling, or a non-standard
top-k output format. Draft (MTP) MoE blocks are skipped unless
SGLANG_ARC_MOE_DRAFT=1.
"""
from __future__ import annotations

import sys
from pathlib import Path

MARK = "# ---- ARC MoE routing variants (serving/runtime_patches/20-moe-routing.py) ----"
ANCHORS = ("def select_experts(", "class StandardTopKOutput(NamedTuple):", "class TopKConfig:")

BLOCK = '''
# ---- ARC MoE routing variants (serving/runtime_patches/20-moe-routing.py) ----
import os as _arc_os

_ARC_MODE = _arc_os.environ.get("SGLANG_ARC_MOE_ROUTING", "").strip().lower()
_ARC_BASE_K = int(_arc_os.environ.get("SGLANG_ARC_MOE_BASE_K", "10"))
_ARC_MIN_K = int(_arc_os.environ.get("SGLANG_ARC_MOE_MIN_K", "1"))
_ARC_PARAM = float(_arc_os.environ.get("SGLANG_ARC_MOE_PARAM", "0.9"))
_ARC_DRAFT = _arc_os.environ.get("SGLANG_ARC_MOE_DRAFT", "0").strip().lower() in ("1", "true", "yes", "on")
_ARC_LOGGED = set()


def _arc_route_weights(topk_weights, topk_ids, router_logits, mode, base_k, min_k, param):
    """New weights for the experts already chosen in `topk_ids` (same shape)."""
    import torch

    padded = topk_ids < 0  # SGLang marks padded tokens' ids with -1
    ids = topk_ids.long().clamp(min=0)
    probs = torch.softmax(router_logits.float(), dim=-1)
    p = torch.gather(probs, 1, ids)
    k = p.shape[-1]
    if mode == "base_norm":
        base = torch.topk(p, min(base_k, k), dim=-1).values.sum(dim=-1, keepdim=True)
        w = p / base.clamp_min(1e-20)
    elif mode == "nucleus":
        sp, order = torch.sort(p, dim=-1, descending=True)
        total = sp.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        before = (sp.cumsum(dim=-1) - sp) / total  # mass of the larger experts
        keep_sorted = before < param
        keep_sorted[..., : max(1, min(min_k, k))] = True
        keep = torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
        w = torch.where(keep, p, torch.zeros_like(p))
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    elif mode == "temp":
        w = p.clamp_min(1e-20).pow(1.0 / param)
        w = w / w.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown SGLANG_ARC_MOE_ROUTING={mode!r}")
    w = torch.where(padded, topk_weights.float(), w)
    return w.to(topk_weights.dtype)


def _arc_applicable(out, cfg):
    return (
        isinstance(out, StandardTopKOutput)
        and getattr(cfg, "scoring_func", "softmax") == "softmax"
        and not getattr(cfg, "use_grouped_topk", False)
        and getattr(cfg, "correction_bias", None) is None
        and getattr(cfg, "custom_routing_function", None) is None
        and not getattr(cfg, "num_fused_shared_experts", 0)
        and getattr(cfg, "routed_scaling_factor", None) in (None, 1.0)
        and (_ARC_DRAFT or getattr(cfg, "allow_routed_experts_capture", True))
    )


if _ARC_MODE:
    _arc_select_experts = select_experts

    def select_experts(hidden_states, router_logits, topk_config, **kwargs):
        out = _arc_select_experts(hidden_states, router_logits, topk_config, **kwargs)
        ok = _arc_applicable(out, topk_config)
        key = (ok, topk_config.top_k, getattr(topk_config, "allow_routed_experts_capture", True))
        if key not in _ARC_LOGGED:
            _ARC_LOGGED.add(key)
            print(
                f"[arc-moe-routing] mode={_ARC_MODE} base_k={_ARC_BASE_K} param={_ARC_PARAM} "
                f"top_k={topk_config.top_k} target={key[2]} applied={ok}",
                flush=True,
            )
        if not ok:
            return out
        weights = _arc_route_weights(
            out.topk_weights, out.topk_ids, out.router_logits,
            _ARC_MODE, _ARC_BASE_K, _ARC_MIN_K, _ARC_PARAM,
        )
        return out._replace(topk_weights=weights)
'''


def main() -> None:
    target = Path(sys.argv[1]) / "sglang" / "srt" / "layers" / "moe" / "topk.py"
    text = target.read_text()
    if MARK in text:
        print(f"{target}: MoE routing variants already present")
        return
    missing = [anchor for anchor in ANCHORS if anchor not in text]
    if missing:
        raise SystemExit(f"{target}: not the SGLang 2.5.3 topk.py this patch targets; missing {missing}")
    target.write_text(text.rstrip("\n") + "\n\n" + BLOCK)
    print(f"{target}: MoE routing variants appended (active only with SGLANG_ARC_MOE_ROUTING)")


if __name__ == "__main__":
    main()
