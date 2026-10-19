"""Time one real-size qwen4_exp decoder layer (forward + backward) without the checkpoint.

Builds a GDN layer and a QSA layer with the checkpoint's dimensions and random weights
(routed experts as packed INT4, dequantized per tile like the AutoRound model), adds the
LoRA adapters the RFT run trains, applies the training patches and times what a step does
per layer: the forward under checkpointing, then backward (recompute + gradients). A 64K
step is then about 36 GDN layers + 12 QSA layers (plus PLE, lm_head and copies). Fits a
48 GB GPU at 64K; no 181 GB download, so kernels can be measured on a cheap GPU.

    PYTHONPATH=. python inference/rl/bench_layer.py --tokens 65536 --out results/bench.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# the checkpoint's text config (Intel/Qwen3.8-Flash-Next-W4A16-AutoRound config.json)
REAL = dict(
    num_hidden_layers=48, hidden_size=2560, num_attention_heads=24, num_key_value_heads=2, head_dim=256,
    num_experts=512, num_experts_per_tok=10, moe_intermediate_size=640, shared_expert_intermediate_size=640,
    linear_num_value_heads=48, linear_num_key_heads=16, linear_key_head_dim=128, linear_value_head_dim=128,
    linear_conv_kernel_dim=4, hc_count=4, hc_lowrank=320, ple_layer_ids=[2], ple_embed_dim=2560,
    indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
    norm_topk_prob=True, output_gate_type="sigmoid", hidden_act="silu", mtp_num_hidden_layers=0,
    eos_token_id=248044, ngram_vocab_size_base=1000,
    rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25,
                         mrope_section=[11, 11, 10], mrope_interleaved=True),
)
GDN_LAYER, QSA_LAYER = 0, 3  # neither has PLE (ple_layer_ids [2] puts it on layer 1)


class PackedInt4(nn.Module):
    """The attributes layer_patches reads from an AutoRound INT4 linear (GPTQ packing, groups of 128)."""

    def __init__(self, in_features, out_features, device, group_size=128):
        super().__init__()
        self.bits, self.group_size = 4, group_size
        groups = in_features // group_size
        self.register_buffer("qweight", torch.randint(-2**31, 2**31 - 1, (in_features // 8, out_features),
                                                      dtype=torch.int32, device=device))
        self.register_buffer("qzeros", torch.full((groups, out_features // 8), 0x77777777, dtype=torch.int32,
                                                  device=device))
        self.register_buffer("scales", torch.full((groups, out_features), 0.004, dtype=torch.bfloat16,
                                                  device=device))
        self.g_idx = None


class ExpertContainer(nn.Module):
    def __init__(self, hidden, inter, device):
        super().__init__()
        self.gate_proj = PackedInt4(hidden, inter, device)
        self.up_proj = PackedInt4(hidden, inter, device)
        self.down_proj = PackedInt4(inter, hidden, device)


def build_layer(config, index, device, experts="int4"):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    with torch.device(device):
        layer = m.Qwen4ExpTextDecoderLayer(config, index).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if name.endswith(("A_log", "dt_bias")):
                continue
            if p.dim() == 1:
                p.zero_() if "norm" in name else p.normal_(0, 0.02)
            else:
                p.normal_(0, 1 / math.sqrt(p.shape[-1]))
    moe = layer.mlp.experts
    if experts == "int4":  # replace the fused BF16 tensors with AutoRound-style unfused INT4 containers
        del moe.gate_up_proj, moe.down_proj
        for e in range(moe.num_experts):
            moe.add_module(str(e), ExpertContainer(config.hidden_size, config.moe_intermediate_size, device))
    return layer


class Wrap(nn.Module):  # module names as in the model ("layer.self_attn.q_proj"), so the LoRA regex matches
    def __init__(self, layer):
        super().__init__()
        self.layer = layer


def rotary(n, dim, device):
    angles = torch.arange(n, device=device)[:, None] * (1e7 ** (-torch.arange(dim // 2, device=device) / (dim // 2)))
    cos = torch.cat([angles.cos()] * 2, -1)[None].to(torch.bfloat16)
    sin = torch.cat([angles.sin()] * 2, -1)[None].to(torch.bfloat16)
    return cos, sin


@torch.no_grad()
def realistic_blocks(n, topk=512, ratio=4, chunk=256, alpha=0.55, beta=0.27, recent_blocks=1024, device="cuda"):
    """A QSA selection with the real model's statistics (64K, 10-19 overlap job): ~43% of the selected
    tokens among the last 4,096 and 16 neighbouring queries selecting ~3.4x one query's tokens.

    Random weights make neighbouring queries pick nearly independent blocks (~16x), which no kernel
    tiling over neighbours would see in training. Scores here are a per-16-query-tile noise shared by
    the tile, plus per-query noise (alpha) and a recency bonus (beta), then the model's top-k."""
    g = torch.Generator(device=device).manual_seed(0)
    blocks = n // ratio
    k = min(topk, blocks)
    out = torch.full((n, k + 1), -1, dtype=torch.int32, device=device)
    b = torch.arange(blocks, device=device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        complete = (torch.arange(start, stop, device=device) + 1) // ratio
        tiles = -(-(stop - start) // 16)
        scores = torch.rand(tiles, blocks, generator=g, device=device).repeat_interleave(16, 0)[: stop - start]
        scores += alpha * torch.rand(stop - start, blocks, generator=g, device=device)
        scores += beta * ((b[None] >= complete[:, None] - recent_blocks) & (b[None] < complete[:, None]))
        scores.masked_fill_(b[None] >= complete[:, None], float("-inf"))
        if k:
            chosen = scores.topk(k, dim=-1).indices
            valid = torch.arange(k, device=device)[None] < complete[:, None].clamp(max=k)
            out[start:stop, :k] = torch.where(valid, chosen, -1).to(torch.int32)
        out[start:stop, k] = complete.to(torch.int32)
    return out


def selection_stats(blocks, ratio=4, tile=16):
    """Share of selected tokens among the last 4,096 and the mean union of `tile` neighbours / one query."""
    n, k = blocks.shape[0], blocks.shape[1] - 1
    first = min(4 * k * ratio, n - tile)
    sel = blocks[first:, :k].long()
    pos = torch.arange(first, n, device=blocks.device)[:, None]
    recent = float(((sel * ratio > pos - 4096) & (sel >= 0)).sum() / (sel >= 0).sum().clamp_min(1))
    unions = []
    for t in range(first // tile + 1, n // tile, max(1, (n // tile - first // tile) // 64)):
        w = blocks[t * tile:(t + 1) * tile, :k]
        unions.append(torch.unique(w[w >= 0]).numel() / max(k, 1))
    return dict(recent_share=round(recent, 3), union_16=round(sum(unions) / max(len(unions), 1), 2))


def categorize(name: str) -> str:
    n = name.lower()
    rules = [("qsa kernel", ("_fwd_kernel", "_bwd_kernel", "_dkv_union_kernel")), ("fla (gdn)", ("chunk_", "fla", "tilelang", "fwd_prepare",
             "bwd_prepare", "l2norm", "solve_tril", "recompute_w_u", "kkt")),
             ("dequant", ("dequant",)), ("gemm", ("gemm", "cutlass", "xmma", "sm90", "sm80", "ampere", "cublas", "matmul",
             "grouped_mm", "wgmma", "s16816", "nvjet")),
             ("index/scatter/gather", ("index", "scatter", "gather", "sort", "radix", "topk", "bitonic")),
             ("copy/cat", ("copy", "cat", "memcpy", "memset", "fill")),
             ("reduce/norm", ("reduce", "norm", "softmax", "sum", "mean")),
             ("conv", ("conv",)), ("elementwise", ("elementwise", "vectorized", "unrolled", "triton_"))]
    for label, keys in rules:
        if any(k in n for k in keys):
            return label
    return "other"


def bench(layer, config, n, device, iters, profile, out_rows):
    layer.train()
    kind = layer.layer_type
    cos, sin = rotary(n, int(config.head_dim * 0.25), device)
    pos = (cos, sin)
    width = config.hc_count * config.hidden_size
    x0 = (torch.randn(1, n, width, device=device) * 0.5).to(torch.bfloat16)
    grad_out = (torch.randn(1, n, width, device=device) * 1e-3).to(torch.bfloat16)

    def step():
        x = x0.clone().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = layer(x, pos, attention_mask=None)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        y.backward(grad_out)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        layer.zero_grad(set_to_none=True)
        return t1 - t0, t2 - t1

    step()  # warm-up: JIT, autotune
    torch.cuda.reset_peak_memory_stats()
    times = [step() for _ in range(iters)]
    fwd, bwd = min(t[0] for t in times), min(t[1] for t in times)
    row = dict(kind=kind, tokens=n, forward_s=round(fwd, 4), backward_s=round(bwd, 4), total_s=round(fwd + bwd, 4),
               peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    if profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
        events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
        total = sum(e.self_device_time_total for e in events)
        cats: dict[str, float] = {}
        for e in events:
            c = categorize(e.key)
            cats[c] = cats.get(c, 0) + e.self_device_time_total
        row["gpu_busy_s"] = round(total / 1e6, 4)
        row["categories_s"] = {k: round(v / 1e6, 4) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}
        top = sorted(events, key=lambda e: -e.self_device_time_total)[:25]
        row["top_kernels"] = [dict(name=e.key[:110], s=round(e.self_device_time_total / 1e6, 4), calls=e.count,
                                   category=categorize(e.key)) for e in top]
    out_rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "top_kernels"}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--layers", default="gdn,qsa")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--experts", default="int4", choices=["int4", "bf16"])
    parser.add_argument("--recompute", default="parts", choices=["parts", "layer"])
    parser.add_argument("--qsa", default="fused", choices=["fused", "chunked"])
    parser.add_argument("--token-chunk", type=int, default=8192)
    parser.add_argument("--compile-hc", action="store_true")
    parser.add_argument("--compile-moe", action="store_true")
    parser.add_argument("--no-cache-experts", action="store_true")
    parser.add_argument("--fast-conv", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--selection", default="realistic", choices=["realistic", "indexer"],
                        help="realistic: run the indexer for its cost, attend over realistic_blocks")
    parser.add_argument("--union-dkv", default="on", choices=["on", "off"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from peft import LoraConfig, inject_adapter_in_model
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    from inference.rl import layer_patches, qsa
    from inference.rl.smoke_lora import ATTENTION_GDN

    device = torch.device("cuda")
    config = Qwen4ExpConfig(text_config=REAL).text_config
    config._attn_implementation = "sdpa"
    qsa.patch_qsa_attention(chunk_size=256, fused=args.qsa == "fused")
    from inference.rl import qsa_kernel
    qsa_kernel.UNION_DKV = args.union_dkv == "on"
    stats: dict = {}
    if args.selection == "realistic":
        indexer_blocks = qsa.select_blocks

        def select_blocks(indexer, hidden_states, cos, sin, chunk_size):
            real = indexer_blocks(indexer, hidden_states, cos, sin, chunk_size)  # its cost, not its choice
            ours = realistic_blocks(hidden_states.shape[1], indexer.block_topk, indexer.compress_ratio,
                                    device=hidden_states.device)
            if not stats:
                stats.update(realistic=selection_stats(ours), random_indexer=selection_stats(real))
                print(json.dumps(dict(selection=stats)), flush=True)
            return ours

        qsa.select_blocks = select_blocks
    rows: list[dict] = []
    for kind in args.layers.split(","):
        torch.manual_seed(0)
        layer = build_layer(config, GDN_LAYER if kind == "gdn" else QSA_LAYER, device, args.experts)
        wrap = Wrap(layer)
        if not args.no_lora:
            inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                               target_modules=ATTENTION_GDN), wrap)
            for name, p in wrap.named_parameters():
                p.requires_grad_("lora_" in name)
            for name, p in wrap.named_parameters():
                if "lora_" in name:
                    p.data = p.data.to(torch.bfloat16)
        else:
            for p in wrap.parameters():
                p.requires_grad_(False)
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        layer_patches.patch_for_training(wrap, token_chunk=args.token_chunk, recompute=args.recompute,
                                         compile_hc=args.compile_hc, compile_moe=args.compile_moe,
                                         cache_experts=not args.no_cache_experts, fast_conv=args.fast_conv)
        bench(layer, config, args.tokens, device, args.iters, args.profile, rows)
        del layer, wrap
        torch.cuda.empty_cache()
    by_kind = {r["kind"]: r["total_s"] for r in rows}
    others = [v for k, v in by_kind.items() if k != "linear_attention"]
    if "linear_attention" in by_kind and others:  # 36 GDN + 12 QSA ("indexed_attention") layers
        estimate = 36 * by_kind["linear_attention"] + 12 * others[0]
        print(json.dumps(dict(layers_per_step_s=round(estimate, 2),
                              tokens_per_s=round(args.tokens / estimate, 1))), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), device=torch.cuda.get_device_name(0), selection=stats, rows=rows), f,
                      indent=1)


if __name__ == "__main__":
    main()
