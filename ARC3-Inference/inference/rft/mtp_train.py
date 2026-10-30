"""Fine-tune the Qwen3.8-Flash-Next MTP draft head on the frozen target's own replies (mtp_data samples).

The target (Intel W4A16 AutoRound) runs forward only, loaded as train_rft loads it but without LoRA. A pre-hook
on its final hyper-connection mixer keeps the multi-stream state [T, 4 x 2560] that vLLM hands the drafter
(Qwen4ExpModel._mtp_hidden_buffer). The head is lordhansolo's checkpoint (Intel's MTP layer with NVFP4 routed
experts), rebuilt from transformers' own Qwen4Exp classes, as vLLM's Qwen4ExpMultiTokenPredictor runs it:

    e  = fc_embedding(norm_H(embed(x[i+1])))            target's embed_tokens
    u  = fc_hidden(norm_4H(h[i]) per stream)             h: the target's state, or the head's own for steps 2..k
    h2 = DecoderLayer(u + e on every stream)             QSA attention (1-D positions, as the drafter's), MoE
    q  = lm_head(mixer(h2))                              target's lm_head; h2 is the next step's h

Row i predicts x[i+2]. Step k (1..steps) feeds row i the head's step k-1 state of row i-1, so row i is the k-th
draft of anchor i-k+1. The rows of a step attend to the same step's rows (an approximation of the drafter, whose
step-k row attends to step-1 keys); steps are back-propagated one at a time, the next step's input detached.

Loss: per step, cross-entropy of the head against the target's own sampling distribution for that token
(temperature, top-k, top-p of the harness: 0.7, 20, 0.95), on the rows whose predicted token is a trained reply
token. Greedy drafts are accepted with that probability, so the expected acceptance of a head is measured the same
way: a_k(i) = p_target(draft_k(i)), the draft taken over the 32K draft vocabulary as served. Acceptance length of
anchor t = 1 + a1(t) + a1(t) a2(t+1) + a1(t) a2(t+1) a3(t+2); the per-position rates are the means of those
products, comparable to vLLM's "Per-position acceptance rate" (porting-advaith v6: 0.767, 0.568, 0.412).

Trained: everything of the head but its routed experts (kept bit-identical, NVFP4 on export) and the QSA indexer
(its top-k selection carries no gradient). Embedding and lm_head are the target's and stay frozen.

    python -m inference.rft.mtp_train --model W4A16 --mtp LORD_MTP --data mtp_v1 --out OUT [--eval-only]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from inference.rft import layer_patches, ple_mmap, smoke_lora, train_rft
from inference.rft.qsa import patch_qsa_attention

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
FROZEN = ("mlp.experts.", "self_attn.indexer.")
LORD_SHARDS = ("mtp-experts-nvfp4.safetensors",)


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------

def dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """compressed-tensors NVFP4: two E2M1 codes per byte (low nibble first), an FP8 scale per 16 inputs,
    W = code * scale / global_scale. Matches Intel's BF16 MTP experts to cosine 0.995 (NVFP4 rounding)."""
    table = torch.tensor(E2M1, device=packed.device)
    codes = torch.stack([packed & 15, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    return table[codes] * scale.float().repeat_interleave(16, 1) / global_scale.float()


class MTPHead(nn.Module):
    """vLLM's Qwen4ExpMultiTokenPredictor in transformers modules; parameter names are the checkpoint's minus `mtp.`."""

    def __init__(self, text_config):
        super().__init__()
        from transformers.models.qwen4_exp import modeling_qwen4_exp as m

        config = copy.deepcopy(text_config)
        config.layer_types = list(config.layer_types) + ["full_attention"]  # the head's layer, index num_hidden_layers
        self.hc, self.hidden = config.hc_count, config.hidden_size
        width = self.hc * self.hidden
        self.pre_fc_norm_embedding = m.Qwen4ExpTextRMSNorm(self.hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = m.Qwen4ExpTextRMSNorm(width, eps=config.rms_norm_eps)  # one norm over all streams
        self.fc_embedding = nn.Linear(self.hidden, self.hidden, bias=False)
        self.fc_hidden = nn.Linear(self.hidden, self.hidden, bias=False)  # shared by the streams
        self.layers = nn.ModuleList([m.Qwen4ExpTextDecoderLayer(config, config.num_hidden_layers)])
        self.hyper_connection_mixer = m.Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.config = config

    def forward(self, hidden: torch.Tensor, embeds: torch.Tensor, position_embeddings) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden [1, n, 4H] (row i: the state at i), embeds [1, n, H] (row i: x[i+1]) -> (mixer output [1, n, H], h2)."""
        e = self.fc_embedding(self.pre_fc_norm_embedding(embeds))
        u = self.fc_hidden(self.pre_fc_norm_hidden(hidden).unflatten(-1, (self.hc, self.hidden))).flatten(-2)
        h2 = self.layers[0](u + e.repeat(1, 1, self.hc), position_embeddings=position_embeddings, attention_mask=None)
        return self.hyper_connection_mixer(h2), h2


def checkpoint_tensor_names(head: MTPHead) -> dict[str, str]:
    """Head parameter name -> checkpoint name, for everything but the fused experts."""
    return {name: f"mtp.{name}" for name, _ in head.named_parameters() if "mlp.experts." not in name}


def load_head(head: MTPHead, mtp_dir: Path) -> dict[str, Any]:
    """Fill the head from lordhansolo's draft checkpoint (BF16 tensors + NVFP4 experts, dequantized to BF16)."""
    from safetensors import safe_open

    weight_map = json.loads((mtp_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = checkpoint_tensor_names(head)
    params = dict(head.named_parameters())
    missing = [c for c in names.values() if c not in weight_map]
    if missing:
        raise RuntimeError(f"{mtp_dir} lacks {missing[:5]} ({len(missing)} tensors)")
    handles: dict[str, Any] = {}

    def tensor(name: str) -> torch.Tensor:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(str(mtp_dir / shard), "pt")
        return handles[shard].get_tensor(name)

    with torch.no_grad():
        for name, ckpt in names.items():
            value = tensor(ckpt)
            if value.shape != params[name].shape:
                raise RuntimeError(f"{ckpt}: checkpoint {tuple(value.shape)} vs head {tuple(params[name].shape)}")
            params[name].copy_(value)
        experts = head.layers[0].mlp.experts
        for e in range(experts.num_experts):
            prefix = f"mtp.layers.0.mlp.experts.{e}."
            proj = {p: dequant_nvfp4(tensor(prefix + p + ".weight_packed"), tensor(prefix + p + ".weight_scale"),
                                     tensor(prefix + p + ".weight_global_scale"))
                    for p in ("gate_proj", "up_proj", "down_proj")}
            experts.gate_up_proj[e].copy_(torch.cat([proj["gate_proj"], proj["up_proj"]]))  # chunk(2): gate, then up
            experts.down_proj[e].copy_(proj["down_proj"])
    return dict(tensors=len(names), experts=experts.num_experts, shards=sorted(handles))


def trainable(name: str) -> bool:
    return not any(part in name for part in FROZEN)


# ---------------------------------------------------------------------------
# Target distribution, loss and acceptance
# ---------------------------------------------------------------------------

@torch.no_grad()
def target_distribution(lm_head, hidden: torch.Tensor, temperature: float, top_k: int, top_p: float,
                        chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """The harness's sampling distribution at each row ([n, H] hidden): (token ids [n, k], probabilities [n, k]).
    Temperature, then top-k, then top-p (the smallest prefix reaching top_p), renormalized; as vLLM samples."""
    ids, probs = [], []
    for start in range(0, hidden.shape[0], chunk):
        logits = lm_head(hidden[start:start + chunk]).float() / temperature
        values, index = logits.topk(top_k, dim=-1)
        p = values.softmax(-1)
        keep = (p.cumsum(-1) - p) < top_p
        p = torch.where(keep, p, torch.zeros_like(p))
        probs.append(p / p.sum(-1, keepdim=True))
        ids.append(index)
    return torch.cat(ids), torch.cat(probs)


def soft_cross_entropy(lm_head, hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
                       draft_rows: torch.Tensor | None, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed -sum_j p_j log q(ids_j) over rows (full-vocabulary softmax, chunks checkpointed), and the acceptance
    p_target(argmax q) of each row's greedy draft over `draft_rows` (the draft vocabulary), without gradient."""
    def piece(h, i, p):
        logits = lm_head(h).float()
        logq = logits.gather(-1, i) - logits.logsumexp(-1, keepdim=True)
        return -(p * logq).sum()

    total = hidden.new_zeros((), dtype=torch.float32)
    accepted = []
    for start in range(0, hidden.shape[0], chunk):
        h, i, p = hidden[start:start + chunk], ids[start:start + chunk], probs[start:start + chunk]
        total = total + (checkpoint(piece, h, i, p, use_reentrant=False) if torch.is_grad_enabled() else piece(h, i, p))
        with torch.no_grad():
            logits = lm_head(h.detach()).float()
            if draft_rows is not None:
                draft = draft_rows[logits[:, draft_rows].argmax(-1)]
            else:
                draft = logits.argmax(-1)
            accepted.append((p * (i == draft[:, None])).sum(-1))
    return total, torch.cat(accepted)


def chain_rates(accepted: list[torch.Tensor], rows: torch.Tensor, n: int) -> dict[str, Any]:
    """Per-position acceptance rates and length over anchors t whose rows t..t+k-1 are all trained rows.
    accepted[k][j]: step k+1's acceptance at rows[j]."""
    steps = len(accepted)
    full = [torch.zeros(n, device=rows.device) for _ in range(steps)]
    valid = torch.zeros(n, dtype=torch.bool, device=rows.device)
    valid[rows] = True
    for k in range(steps):
        full[k][rows] = accepted[k]
    anchors = valid.clone()
    for k in range(1, steps):
        anchors[:-k] &= valid[k:]
        anchors[-k:] = False
    t = torch.nonzero(anchors).squeeze(1)
    if t.numel() == 0:
        return dict(anchors=0)
    product = torch.ones_like(t, dtype=torch.float32)
    rates = []
    for k in range(steps):
        product = product * full[k][t + k]
        rates.append(float(product.mean()))
    return dict(anchors=int(t.numel()), rates=rates, length=1 + sum(rates))


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------

class Target:
    """The frozen target with a hook that keeps the final mixer's input (the drafter's hidden state)."""

    def __init__(self, model, mixer):
        self.model, self.captured = model, {}
        mixer.register_forward_pre_hook(lambda module, args: self.captured.__setitem__("hidden", args[0]))

    @torch.no_grad()
    def __call__(self, inputs: dict[str, torch.Tensor], ngram) -> tuple[torch.Tensor, torch.Tensor]:
        layer_patches.provide_ngram(self.model, ngram)
        out = self.model.model(**inputs, use_cache=False).last_hidden_state
        hidden = self.captured.pop("hidden")
        layer_patches.provide_ngram(self.model, None)
        return hidden, out


def run_sample(target: Target, head: MTPHead, lm_head, embed, rotary, item: dict[str, Any], device, args,
               draft_rows, train: bool, loss_scale: float = 1.0) -> dict[str, Any]:
    """Target forward, then the head's steps over all rows; loss and acceptance on the trained rows."""
    labels = item["labels"][0].to(device)
    input_ids = item["input_ids"].to(device)
    inputs = {k: v.to(device) for k, v in item.items() if k != "labels" and not k.startswith("_")}
    started = time.perf_counter()
    hidden, last = target(inputs, item.get("_ngram"))
    target_s = time.perf_counter() - started
    n = input_ids.shape[1] - 1  # rows 0..n-1: (state i, token i+1) -> token i+2
    rows = torch.nonzero(labels[2:] != -100).squeeze(1)  # rows whose predicted token is a trained reply token
    rows = rows[rows < n]
    ids, probs = target_distribution(lm_head, last[0, rows + 1], args.temperature, args.top_k, args.top_p)
    embeds = embed(input_ids[:, 1:])
    if args.positions == "index":
        position_ids = torch.arange(n, device=device).view(1, 1, -1).expand(3, 1, -1)
    else:  # the target's M-RoPE positions, as transformers computes them (calibration check)
        position_ids = inputs_position_ids(target.model, inputs)[:, :, :n]
    position_embeddings = rotary(embeds, position_ids)
    state = hidden[:, :n]
    losses, accepted = [], []
    for step in range(args.steps):
        with torch.set_grad_enabled(train):
            out, h2 = head(state, embeds, position_embeddings)
            loss, acc = soft_cross_entropy(lm_head, out[0, rows], ids, probs, draft_rows)
            if train:
                weight = args.step_weights[step] * loss_scale / max(1, rows.numel())
                (loss * weight).backward()
        losses.append(float(loss) / max(1, rows.numel()))
        accepted.append(acc)
        # row i of the next step: this step's state of row i-1 (row 0 keeps the target's, never trained)
        state = torch.cat([hidden[:, :1], h2[:, :-1].detach()], dim=1)
        del out, h2, loss
    chain = chain_rates(accepted, rows, n)
    return dict(tokens=int(input_ids.shape[1]), rows=int(rows.numel()), losses=[round(x, 5) for x in losses],
                step1_accept=round(float(accepted[0].mean()), 5) if rows.numel() else None, chain=chain,
                target_s=round(target_s, 2), total_s=round(time.perf_counter() - started, 2))


def inputs_position_ids(model, inputs):
    """transformers' M-RoPE position ids for the sample: [3, 1, T] (text axes of get_rope_index)."""
    base = model.model
    position_ids, _ = base.get_rope_index(inputs["input_ids"], mm_token_type_ids=inputs.get("mm_token_type_ids"),
                                          image_grid_thw=inputs.get("image_grid_thw"))
    return position_ids[-3:] if position_ids.shape[0] == 4 else position_ids


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_draft(head: MTPHead, mtp_dir: Path, out: Path) -> None:
    """A draft directory for vLLM: lordhansolo's files, with every non-expert mtp.* tensor taken from the head in
    mtp-finetuned-bf16.safetensors. The NVFP4 experts, lm_head and embed_tokens shards are hard-linked unchanged
    (copied when the link fails), so the directory works outside the container that wrote it."""
    import shutil

    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    names = checkpoint_tensor_names(head)
    params = dict(head.named_parameters())
    tensors = {ckpt: params[name].detach().to(torch.bfloat16).contiguous().cpu() for name, ckpt in names.items()}
    save_file(tensors, out / "mtp-finetuned-bf16.safetensors", metadata={"format": "pt"})
    index = json.loads((mtp_dir / "model.safetensors.index.json").read_text())
    for ckpt in tensors:
        index["weight_map"][ckpt] = "mtp-finetuned-bf16.safetensors"
    (out / "model.safetensors.index.json").write_text(json.dumps(index))
    for entry in mtp_dir.iterdir():
        target = out / entry.name
        if entry.name in ("model.safetensors.index.json", "download.log") or target.exists():
            continue
        try:
            os.link(entry, target)
        except OSError:
            shutil.copy2(entry, target)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_target(args):
    """train_rft.load_model without LoRA: frozen, patched for long forwards (QSA kernel, grouped INT4 experts)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    patch_qsa_attention(256, fused=args.qsa_kernel)
    ple_mmap.install()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        device_map=smoke_lora.gpu_device_map(args.model),
                                                        attn_implementation="sdpa")
    ple = ple_mmap.attach(model, args.model)
    model.config.use_cache = False
    model.requires_grad_(False)
    model.train()  # the patched layers' token chunks; dropout is 0
    patches = layer_patches.patch_for_training(model, args.token_chunk, 128, "layer", cache_experts=True)
    check = layer_patches.check_experts(model)
    if check["max_rel_error"] > 0.05:
        raise RuntimeError(f"grouped experts disagree with AutoRound's: {check}")
    return model, processor, dict(patches=patches, experts_check=check, ple=ple)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the target checkpoint (Intel W4A16 AutoRound)")
    parser.add_argument("--mtp", type=Path, required=True, help="lordhansolo's draft checkpoint (mtp_nvfp4 files)")
    parser.add_argument("--data", type=Path, required=True, help="mtp_data output: train.jsonl, eval.jsonl, images/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draft-vocab", type=Path, help="configs/draft_vocab_32k.json: drafts as served (default: full)")
    parser.add_argument("--steps", type=int, default=3, help="draft steps trained and measured (served: 3)")
    parser.add_argument("--step-weights", default="1.0,1.0,1.0")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--positions", choices=("index", "mrope"), default="index",
                        help="the head's RoPE positions: the drafter's 1-D positions, or the target's M-RoPE ones")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accumulate", type=int, default=1, help="samples per optimizer step")
    parser.add_argument("--train-samples", type=int, default=600, help="samples drawn from train.jsonl (one epoch)")
    parser.add_argument("--eval-samples", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=100, help="optimizer steps")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--max-hours", type=float, default=20.0)
    parser.add_argument("--token-chunk", type=int, default=8192)
    parser.add_argument("--no-qsa-kernel", dest="qsa_kernel", action="store_false")
    parser.add_argument("--eval-only", action="store_true", help="measure the head as loaded and stop")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.step_weights = [float(w) for w in args.step_weights.split(",")][:args.steps]
    if len(args.step_weights) < args.steps:
        raise SystemExit("--step-weights needs one weight per step")
    args.out.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out / "train_log.jsonl", "a")

    def log(**record):
        record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(json.dumps(record), flush=True)

    started = time.time()
    model, processor, info = load_target(args)
    device = model.get_input_embeddings().weight.device
    base = model.model  # Qwen4ExpModel
    language = base.language_model
    target = Target(model, language.hyper_connection_mixer)
    head = MTPHead(model.config.get_text_config()).to(torch.bfloat16)
    loaded = load_head(head, args.mtp)
    head.to(device)
    head.train()
    for module in head.modules():
        if type(module).__name__ == "Qwen4ExpTextExperts":
            module.config._experts_implementation = layer_patches.EXPERTS_IMPLEMENTATION
            module._rft_cache_experts = False  # the head's BF16 experts: views, not the target's dequant cache
    for name, p in head.named_parameters():
        p.requires_grad_(trainable(name))
    params = [(n, p) for n, p in head.named_parameters() if p.requires_grad]
    draft_rows = None
    if args.draft_vocab:
        ranges = json.loads(args.draft_vocab.read_text())["token_ranges"]  # half-open, as vLLM reads them
        draft_rows = torch.cat([torch.arange(a, b) for a, b in ranges]).unique().to(device)
    log(event="loaded", load_s=round(time.time() - started, 1), head=loaded, trainable=sum(p.numel() for _, p in params),
        frozen=sum(p.numel() for _, p in head.named_parameters() if not p.requires_grad), target=info["experts_check"],
        draft_vocab=int(draft_rows.numel()) if draft_rows is not None else None, args={k: str(v) for k, v in vars(args).items()})

    def prepare(sample):
        item = train_rft.tokenize_sample(processor, sample, args.data)
        if item["input_ids"].shape[1] > args.max_tokens:
            return None
        item["_ngram"] = layer_patches.precompute_ngram(model, item["input_ids"])
        return item

    lm_head, embed, rotary = model.lm_head, language.embed_tokens, language.rotary_emb
    eval_samples = train_rft.load_samples(args.data / "eval.jsonl")
    random.Random(args.seed).shuffle(eval_samples)
    eval_samples = eval_samples[:args.eval_samples]

    def evaluate(tag):
        totals, chains = {"rows": 0, "loss": [0.0] * args.steps}, []
        for sample in eval_samples:
            try:
                item = prepare(sample)
                if item is None:
                    continue
                r = run_sample(target, head, lm_head, embed, rotary, item, device, args, draft_rows, train=False)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            totals["rows"] += r["rows"]
            for k in range(args.steps):
                totals["loss"][k] += r["losses"][k] * r["rows"]
            if r["chain"].get("anchors"):
                chains.append(r["chain"])
        anchors = sum(c["anchors"] for c in chains)
        rates = [sum(c["rates"][k] * c["anchors"] for c in chains) / max(1, anchors) for k in range(args.steps)]
        log(event="eval", tag=tag, samples=len(chains), rows=totals["rows"], anchors=anchors,
            loss=[round(x / max(1, totals["rows"]), 5) for x in totals["loss"]],
            rates=[round(x, 4) for x in rates], length=round(1 + sum(rates), 4))

    if args.eval_only:
        evaluate("as loaded")
        return 0

    masters = [p.detach().float().clone().requires_grad_() for _, p in params]
    optimizer = torch.optim.AdamW(masters, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    train_samples = train_rft.load_samples(args.data / "train.jsonl")
    order = list(range(len(train_samples)))
    random.Random(args.seed).shuffle(order)
    order = order[:args.train_samples]
    total_steps = math.ceil(len(order) / args.accumulate)
    state = dict(position=0, step=0)
    latest = args.out / "latest"
    if args.resume and (latest / "state.json").exists():
        from safetensors.torch import load_file
        weights = load_file(latest / "head.safetensors")
        with torch.no_grad():
            for (name, p), m in zip(params, masters):
                m.copy_(weights[name].float())
                p.copy_(m)
        optimizer.load_state_dict(torch.load(latest / "optimizer.pt", map_location=device))
        state = json.loads((latest / "state.json").read_text())
        log(event="resumed", **state)

    def save(name):
        from safetensors.torch import save_file
        target_dir = args.out / name
        target_dir.mkdir(parents=True, exist_ok=True)
        save_file({n: p.detach().cpu().contiguous() for n, p in params}, target_dir / "head.safetensors")
        torch.save(optimizer.state_dict(), target_dir / "optimizer.pt")
        (target_dir / "state.json").write_text(json.dumps(state))
        log(event="saved", name=name, step=state["step"])

    if state["step"] == 0:
        evaluate("step 0")
    prefetch = train_rft.Prefetch(lambda i: prepare(train_samples[order[i]]))
    if state["position"] < len(order):
        prefetch.start(state["position"])
    pending = 0
    while state["position"] < len(order):
        if time.time() - started > args.max_hours * 3600:
            log(event="time limit", **state)
            break
        try:
            item = prefetch.get()
        except RuntimeError as exc:
            item = None
            log(event="skip", position=state["position"], error=str(exc)[-300:])
        index = state["position"]
        state["position"] += 1
        if state["position"] < len(order):
            prefetch.start(state["position"])
        if item is None:
            continue
        try:
            r = run_sample(target, head, lm_head, embed, rotary, item, device, args, draft_rows, train=True,
                           loss_scale=1.0 / args.accumulate)
        except torch.OutOfMemoryError:
            for _, p in params:
                p.grad = None
            torch.cuda.empty_cache()
            log(event="oom", position=index, tokens=int(item["input_ids"].shape[1]))
            continue
        pending += 1
        sample = train_samples[order[index]]
        log(event="sample", position=index, id=sample["id"], **r)
        if pending < args.accumulate and state["position"] < len(order):
            continue
        for (_, p), m in zip(params, masters):
            m.grad = p.grad.float() if p.grad is not None else torch.zeros_like(m)
        norm = torch.nn.utils.clip_grad_norm_(masters, args.grad_clip)
        lr = args.lr * train_rft.lr_at(state["step"], total_steps, args.warmup, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for (_, p), m in zip(params, masters):
                p.copy_(m)
                p.grad = None
        pending = 0
        state["step"] += 1
        log(event="step", step=state["step"], lr=lr, grad_norm=round(float(norm), 4))
        if state["step"] % args.eval_every == 0:
            evaluate(f"step {state['step']}")
        if state["step"] % args.save_every == 0:
            save("latest")
    save("latest")
    evaluate(f"final step {state['step']}")
    export_draft(head, args.mtp, args.out / "draft")
    log(event="exported", path=str(args.out / "draft"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - logged for the job's output
        traceback.print_exc()
        sys.exit(1)
