"""LoRA smoke test: what an RFT step costs on the deployed checkpoint.

Loads the checkpoint (routed experts stay INT4 and frozen), attaches LoRA to
the attention and GDN projections, and runs forward + backward on real packed
samples rebuilt from a transcript at several sequence lengths. Reports load
memory per GPU, peak memory and step time per length, and whether gradients
reach the LoRA weights through the quantized experts.

Images are synthetic boards of the deployed size (640x640, 402 tokens each):
their token count and vision cost matter here, not their content.

    python -m inference.rl.smoke_lora --model /models/w4a16 --transcript game.txt \\
        --tools tools.json --lengths 32768,65536,100000 --out smoke.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inference.rl.context_rebuild import estimate_tokens, rebuild

ATTENTION_GDN = r".*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))$"
SHARED_EXPERT = r"|.*\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$"
ASSISTANT = "<|im_start|>assistant\n<think>\n"
END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"


def board_image(seed: int) -> Image.Image:
    """A 64x64 grid of palette colors upscaled 10x, like a rendered board."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(16, 3), dtype=np.uint8)
    grid = rng.integers(0, 16, size=(64, 64))
    pixels = palette[grid].repeat(10, axis=0).repeat(10, axis=1)
    return Image.fromarray(pixels, "RGB")


def to_chat(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Rebuilt messages in the processor's format; image parts become PIL boards."""
    chat, images = [], []
    for m in messages:
        role = m["role"]
        if role == "system":
            chat.append({"role": "system", "content": m["content"]})
        elif role == "user":
            parts = []
            for p in m["content"]:
                if p["type"] == "image_url":
                    images.append(board_image(len(images)))
                    parts.append({"type": "image"})
                else:
                    parts.append({"type": "text", "text": p["text"]})
            chat.append({"role": "user", "content": parts})
        elif role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                args = c["function"].get("arguments") or "{}"
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"code": args}
                calls.append({"type": "function", "function": {"name": c["function"]["name"], "arguments": args}})
            chat.append({"role": "assistant", "content": m.get("content") or "",
                         "reasoning_content": m.get("reasoning") or "", "tool_calls": calls})
        else:
            chat.append({"role": "tool", "content": m["content"]})
    return chat, images


def pick_request(game, length: int):
    """The non-uncertain request whose estimated prompt is largest but under `length`."""
    best = None
    for r in game.requests:
        e = estimate_tokens(r.messages)
        if not r.uncertain and e <= length and (best is None or e > best[0]):
            best = (e, r)
    return best[1] if best else None


def tokenize(processor, text: str, images: list[Image.Image], loss_replies: int):
    """input_ids, labels (-100 outside the last `loss_replies` assistant replies) and vision inputs.

    The text is cut at reply boundaries, which fall on special tokens, and each
    piece is tokenized with its own images, so the mask needs no offset mapping."""
    starts = [m.end() for m in re.finditer(re.escape(ASSISTANT), text)]
    spans = []
    for s in starts[-loss_replies:]:
        e = text.find(END, s)
        spans.append((s, e + len(END)))
    pieces, cursor = [], 0
    for s, e in spans:
        pieces += [(text[cursor:s], False), (text[s:e], True)]
        cursor = e
    pieces.append((text[cursor:], False))
    ids, labels, pixel_values, grids, used = [], [], [], [], 0
    for piece, train in pieces:
        if not piece:
            continue
        n = piece.count(IMAGE_PAD)
        batch = processor(text=[piece], images=images[used:used + n] or None, return_tensors="pt", add_special_tokens=False)
        used += n
        piece_ids = batch["input_ids"][0]
        ids.append(piece_ids)
        labels.append(piece_ids.clone() if train else torch.full_like(piece_ids, -100))
        if n:
            pixel_values.append(batch["pixel_values"])
            grids.append(batch["image_grid_thw"])
    out = {"input_ids": torch.cat(ids)[None], "labels": torch.cat(labels)[None]}
    if pixel_values:
        out["pixel_values"] = torch.cat(pixel_values)
        out["image_grid_thw"] = torch.cat(grids)
    return out


def memory_gb() -> dict[str, float]:
    return {f"cuda:{i}": round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}


def peak_gb() -> dict[str, float]:
    return {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}


def step(model, sample, device) -> dict[str, Any]:
    labels = sample["labels"][0].to(device)
    targets = torch.nonzero(labels[1:] != -100).squeeze(1) + 1  # positions whose token is trained
    inputs = {k: v.to(device) for k, v in sample.items() if k != "labels"}
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # logits only where a target follows: full [length, 248K] logits would not fit
    out = model(**inputs, logits_to_keep=targets - 1, use_cache=False)
    loss = F.cross_entropy(out.logits[0].float(), labels[targets])
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm = float(torch.sqrt(sum((g.float() ** 2).sum().to("cpu") for g in grads))) if grads else 0.0
    model.zero_grad(set_to_none=True)
    return dict(tokens=int(sample["input_ids"].shape[1]), loss_tokens=int(targets.numel()),
                images=int(sample["image_grid_thw"].shape[0]) if "image_grid_thw" in sample else 0,
                loss=round(float(loss), 4), forward_s=round(t1 - t0, 2), backward_s=round(t2 - t1, 2),
                tokens_per_s=round(sample["input_ids"].shape[1] / (t2 - t0), 1),
                lora_params_with_grad=len(grads), lora_grad_norm=norm, peak_gb=peak_gb())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True, help="tool schema as sent by the harness (JSON list)")
    parser.add_argument("--lengths", default="32768,65536,100000")
    parser.add_argument("--repeats", type=int, default=2, help="steps per length; the first includes warm-up")
    parser.add_argument("--loss-replies", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--shared-expert", action="store_true", help="also put LoRA on the shared expert MLP")
    parser.add_argument("--backend", default="auto", help="AutoRound kernel backend (auto, torch, tritonv2, ...)")
    parser.add_argument("--attn", default="sdpa", help="attn_implementation")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("smoke_lora.json"))
    args = parser.parse_args(argv)

    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    report: dict[str, Any] = dict(args={k: str(v) for k, v in vars(args).items()}, env=dict(
        python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    for package in ("peft", "auto_round", "fla", "causal_conv1d", "flash_attn"):
        try:
            report["env"][package] = getattr(__import__(package), "__version__", "installed")
        except ImportError:
            report["env"][package] = None

    def save():
        args.out.write_text(json.dumps(report, indent=1))

    processor = AutoProcessor.from_pretrained(args.model)
    tools = json.loads(args.tools.read_text())
    game = rebuild(args.transcript.read_text(encoding="utf-8"))
    samples = {}
    for length in sorted(int(x) for x in args.lengths.split(",")):
        request = pick_request(game, length)
        if request is None:
            continue
        chat, images = to_chat(request.messages + [request.reply])
        text = processor.apply_chat_template(chat, tools=tools, tokenize=False, preserve_thinking=True)
        samples[length] = tokenize(processor, text, images, args.loss_replies)
    report["samples"] = {k: int(v["input_ids"].shape[1]) for k, v in samples.items()}
    print("samples (target length -> tokens):", report["samples"], flush=True)

    t0 = time.perf_counter()
    kwargs: dict[str, Any] = dict(dtype=torch.bfloat16, device_map="auto", attn_implementation=args.attn)
    if args.backend != "auto":
        from transformers import AutoRoundConfig
        kwargs["quantization_config"] = AutoRoundConfig(backend=args.backend)
    model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
    report["load"] = dict(seconds=round(time.perf_counter() - t0, 1), memory_gb=memory_gb())
    kinds = {}
    for name, module in model.named_modules():
        for key, pattern in (("attention", r"self_attn\.q_proj$"), ("gdn", r"linear_attn\.in_proj_qkv$"),
                             ("routed_expert", r"mlp\.experts(\.\d+\.down_proj)?$"), ("indexer", r"indexer$")):
            if key not in kinds and re.search(pattern, name):
                kinds[key] = f"{name}: {type(module).__module__}.{type(module).__name__}"
    report["modules"] = kinds
    print("loaded", report["load"], json.dumps(kinds, indent=1), flush=True)
    save()

    model.config.use_cache = False
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                      target_modules=ATTENTION_GDN + (SHARED_EXPERT if args.shared_expert else ""))
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["lora"] = dict(trainable_params=trainable,
                          modules=sum(1 for n, _ in model.named_modules() if n.endswith("lora_A")))
    print("lora", report["lora"], flush=True)
    device = model.get_input_embeddings().weight.device

    report["steps"] = {}
    for length, sample in samples.items():
        results = []
        for _ in range(args.repeats):
            try:
                results.append(step(model, sample, device))
                print(length, results[-1], flush=True)
            except torch.cuda.OutOfMemoryError as exc:
                results.append(dict(error="oom", detail=str(exc)[:500], peak_gb=peak_gb()))
                print(length, "OOM", flush=True)
                break
            except Exception as exc:  # record it and keep the earlier numbers
                results.append(dict(error=type(exc).__name__, detail=traceback.format_exc()[-3000:]))
                print(length, "ERROR", traceback.format_exc(), flush=True)
                break
        report["steps"][str(length)] = results
        save()
        torch.cuda.empty_cache()
        if results and "error" in results[-1]:
            break

    # one optimizer step: the AdamW state is part of the training footprint
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    for p in params:
        p.grad = torch.zeros_like(p)
    optimizer.step()
    report["after_optimizer_step_gb"] = memory_gb()
    save()
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
