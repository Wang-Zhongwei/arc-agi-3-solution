"""RFT training: LoRA on the selected level clears (rft_data), one sequence per sample.

Loads the W4A16 checkpoint with the patches smoke_lora measured (fused QSA, compiled
hyper-connections and MoE, fast LoRA and GDN, parts recompute, async offload), then runs
`epochs` passes over train.jsonl in a shuffled order, `batch` samples per optimizer step.

- Loss: cross-entropy on each sample's trained replies; in a batch every trained token
  counts once (a sample's mean loss is weighted by its share of the batch's tokens).
- Prefix stop-grad: the context before a sample's first trained token runs forward only.
- AdamW, linear warm-up then cosine decay to `min_lr_ratio`, gradient clipping.
- Memory: checkpoints kept on the GPU only within `gpu_margin_gb` of the predicted peak;
  an out-of-memory sample is retried with nothing kept, then with layer recompute, then
  skipped (its partial gradients are taken back out first).
- Saves: the adapter after every epoch (`adapter_epoch<N>`) and `latest` (adapter,
  optimizer, position) every `save_every` steps; `--resume` continues from `latest`.
- Validation: mean loss on val.jsonl (forward only) before training and after each epoch.

    python -m inference.rl.train_rft --model /workspace/w4a16 --data /workspace/data --out /workspace/train
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from inference.rl import layer_patches, ple_mmap, smoke_lora
from inference.rl.qsa import patch_qsa_attention

PEAK_BASE_GB, PEAK_PER_1K_GB = 69.0, 0.35  # H200, prefix stop-grad, nothing kept: 92 GB at 64.9K, 112 GB at 122.9K


def load_samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path)] if path.exists() else []


def tokenize_sample(processor, sample: dict[str, Any], data: Path) -> dict[str, torch.Tensor]:
    messages = sample["messages"]
    chat, images = smoke_lora.to_chat(messages, load_image=lambda p: Image.open(data / p["image_url"]["url"]).convert("RGB"))
    text = processor.apply_chat_template(chat, tools=sample["tools"], tokenize=False, preserve_thinking=True)
    assistants = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if text.count(smoke_lora.ASSISTANT) != len(assistants):
        raise ValueError(f"{sample['id']}: {text.count(smoke_lora.ASSISTANT)} reply markers for {len(assistants)} replies")
    replies = [assistants.index(i) for i in sample["train"]]
    return smoke_lora.tokenize(processor, text, images, 0, replies=replies)


class Prefetch:
    """Tokenizes the next batch on a thread while the GPU runs the current one."""

    def __init__(self, fn):
        self.fn, self.thread, self.result = fn, None, None

    def start(self, *args):
        def run():
            try:
                self.result = ("ok", self.fn(*args))
            except Exception:  # noqa: BLE001 - reported by get()
                self.result = ("error", traceback.format_exc())
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def get(self):
        self.thread.join()
        kind, value = self.result
        if kind == "error":
            raise RuntimeError(value)
        return value


def keep_gb(tokens: int, margin: float) -> float:
    total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
    return round(max(0.0, total - margin - (PEAK_BASE_GB + PEAK_PER_1K_GB * tokens / 1000)), 1)


def lr_at(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def run_sample(model, device, sample, scale: float, margin: float, prefix_stop_grad: bool = True) -> dict[str, Any]:
    """Forward + backward of one sample, adding scale x its gradient; OOM fallbacks as in the docstring."""
    params = [p for p in model.parameters() if p.requires_grad]
    saved = [None if p.grad is None else p.grad.detach().clone() for p in params]
    tokens = int(sample["input_ids"].shape[1])
    attempts = [("parts", keep_gb(tokens, margin)), ("parts", 0.0), ("layer", 0.0)]
    start_mode = layer_patches.current_recompute(model)
    errors = []
    for mode, keep in attempts:
        if layer_patches.current_recompute(model) != mode:
            layer_patches.configure(model, recompute=mode)
        oom = False
        try:
            cuda = torch.cuda.is_available()  # CPU: the tests
            result = smoke_lora.step(model, sample, device, offload=cuda, async_copies=cuda, gpu_keep_gb=keep,
                                     prefix_stop_grad=prefix_stop_grad, accumulate=True, loss_scale=scale)
            result.update(recompute_mode=mode, gpu_keep_gb=keep, retries=len(errors))
            break
        except torch.cuda.OutOfMemoryError as exc:
            errors.append(f"{mode}/{keep}: {str(exc)[:120]}")
            oom = True
        except smoke_lora.HostMemoryError as exc:
            errors.append(f"host: {exc}")
            result = None
            break
        if oom:  # outside the except clause: its traceback holds the failed step's tensors
            for p, g in zip(params, saved):
                p.grad = None if g is None else g.clone()
            gc.collect()
            torch.cuda.empty_cache()
    else:
        result = None
    if layer_patches.current_recompute(model) != start_mode:
        layer_patches.configure(model, recompute=start_mode)
    if result is None:
        for p, g in zip(params, saved):
            p.grad = g
        return dict(error="; ".join(errors) or "failed", tokens=tokens)
    if errors:
        result["errors"] = errors
    return result


@torch.no_grad()
def evaluate(model, device, samples: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    total = count = 0.0
    model.train()  # the patched layers' token chunks; dropout is 0
    for sample in samples:
        labels = sample["labels"][0].to(device)
        targets = torch.nonzero(labels[1:] != -100).squeeze(1) + 1
        inputs = {k: v.to(device) for k, v in sample.items() if k != "labels"}
        hidden = base.model(**inputs, use_cache=False).last_hidden_state[0, targets - 1]
        loss = smoke_lora.chunked_cross_entropy(base.lm_head, hidden, labels[targets])
        total += float(loss) * targets.numel()
        count += targets.numel()
        del hidden, loss
    return dict(loss=round(total / max(count, 1), 5), tokens=int(count), samples=len(samples))


def load_model(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    patch_qsa_attention(256, fused=args.qsa_kernel, compile=args.compile_qsa and args.qsa_kernel)
    ple_mmap.install()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        device_map=smoke_lora.gpu_device_map(args.model),
                                                        attn_implementation="sdpa")
    ple_mmap.attach(model, args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                                             target_modules=smoke_lora.ATTENTION_GDN))
    if args.resume and (args.out / "latest" / "adapter_model.safetensors").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(args.out / "latest" / "adapter_model.safetensors"))
    device = model.get_input_embeddings().weight.device
    layers = args.compile_layers  # pod/make_jobs.py: compile_layer's test covers these four
    patches = layer_patches.patch_for_training(model, args.token_chunk, 128, "parts", compile_hc=layers,
                                               compile_moe=layers, cache_experts=True, fast_conv=layers,
                                               fast_lora=args.fast_lora, fast_gdn=layers and args.fast_gdn)
    check = layer_patches.check_experts(model)
    if check["max_rel_error"] > 0.05:
        raise RuntimeError(f"grouped experts disagree with AutoRound's: {check}")
    return model, processor, device, dict(patches=patches, experts_check=check)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True, help="rft_data output: train.jsonl, val.jsonl, images/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4, help="samples per optimizer step (~80K trained tokens)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=float, default=0.05, help="share of the steps")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=131072, help="longer samples are skipped")
    parser.add_argument("--gpu-margin-gb", type=float, default=24.0)
    parser.add_argument("--token-chunk", type=int, default=8192)
    parser.add_argument("--save-every", type=int, default=5, help="optimizer steps between `latest` saves")
    parser.add_argument("--max-hours", type=float, default=0.0, help="stop and save after this long (0: no limit)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    # the measured path (H200, 10-20) has every feature on; train_pod.sh switches off those whose CUDA test failed
    parser.add_argument("--no-qsa-kernel", dest="qsa_kernel", action="store_false")
    parser.add_argument("--no-compile-qsa", dest="compile_qsa", action="store_false")
    parser.add_argument("--no-compile-layers", dest="compile_layers", action="store_false",
                        help="without compile_hc, compile_moe, fast_conv, fast_gdn")
    parser.add_argument("--no-fast-lora", dest="fast_lora", action="store_false")
    parser.add_argument("--no-fast-gdn", dest="fast_gdn", action="store_false")
    parser.add_argument("--no-prefix-stop-grad", dest="prefix_stop_grad", action="store_false")
    parser.add_argument("--limit", type=int, default=0, help="use only this many train samples (dry runs)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    started = time.time()

    def log(**row):
        row = dict(t=round(time.time() - started, 1), **row)
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    train = load_samples(args.data / "train.jsonl")
    val = load_samples(args.data / "val.jsonl")
    if args.limit:
        train = train[:args.limit]
    model, processor, device, info = load_model(args)
    log(event="loaded", train=len(train), val=len(val), args={k: str(v) for k, v in vars(args).items()}, **info)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    rng = random.Random(args.seed)
    orders = []
    for _ in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        orders.append(order)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, round(args.warmup * total_steps))
    state = dict(step=0)
    if args.resume and (args.out / "latest" / "state.json").exists():
        state = json.loads((args.out / "latest" / "state.json").read_text())
        optimizer.load_state_dict(torch.load(args.out / "latest" / "optimizer.pt", map_location=device))
        log(event="resumed", step=state["step"])

    def save(name: str, with_state: bool):
        target = args.out / name
        model.save_pretrained(target)
        if with_state:
            torch.save(optimizer.state_dict(), target / "optimizer.pt")
            (target / "state.json").write_text(json.dumps(state))
        log(event="saved", name=name, step=state["step"])

    def tokenize_batch(indices):
        out = []
        for i in indices:
            try:
                out.append((i, tokenize_sample(processor, train[i], args.data)))
            except Exception as exc:  # noqa: BLE001 - a bad sample is skipped, not fatal
                out.append((i, f"{type(exc).__name__}: {str(exc)[:300]}"))
        return out

    val_tokens = []
    for sample in val:
        try:
            tokenized = tokenize_sample(processor, sample, args.data)
            if tokenized["input_ids"].shape[1] <= args.max_tokens:
                val_tokens.append(tokenized)
        except Exception as exc:  # noqa: BLE001
            log(event="val_skip", id=sample["id"], error=str(exc)[:300])
    if val_tokens and state["step"] == 0:
        log(event="val", epoch=0, **evaluate(model, device, val_tokens))

    batches = [(e, b, orders[e][b * args.batch:(b + 1) * args.batch]) for e in range(args.epochs)
               for b in range(steps_per_epoch)]
    prefetch = Prefetch(tokenize_batch)
    if state["step"] < len(batches):
        prefetch.start(batches[state["step"]][2])
    stopped = False
    while state["step"] < len(batches):
        epoch, _, indices = batches[state["step"]]
        tokenized = prefetch.get()
        if state["step"] + 1 < len(batches):
            prefetch.start(batches[state["step"] + 1][2])
        ready = []
        for i, item in tokenized:
            if isinstance(item, str):
                log(event="skip", id=train[i]["id"], reason=item)
            elif item["input_ids"].shape[1] > args.max_tokens:
                log(event="skip", id=train[i]["id"], reason=f"{item['input_ids'].shape[1]} tokens")
            else:
                ready.append((i, item))
        trained = sum(int((item["labels"] != -100).sum()) for _, item in ready)
        optimizer.zero_grad(set_to_none=True)
        t0, losses, done_tokens = time.time(), [], 0
        for i, item in ready:
            n = int((item["labels"] != -100).sum())
            result = run_sample(model, device, item, n / trained, args.gpu_margin_gb, args.prefix_stop_grad)
            row = {k: result.get(k) for k in ("tokens", "loss_tokens", "prefix_tokens", "loss", "forward_s", "backward_s",
                                              "tokens_per_s", "peak_gb", "recompute_mode", "gpu_keep_gb", "error", "errors")}
            log(event="sample", step=state["step"], epoch=epoch, id=train[i]["id"], **row)
            if "error" not in result:
                losses.append((result["loss"], n))
                done_tokens += n
        if not done_tokens and state["step"] == 0:  # a systematic failure: stop before paying for every batch
            log(event="abort", reason="no sample of the first batch ran")
            return 2
        if done_tokens:
            if done_tokens != trained:  # skipped samples: rescale to the tokens that did run
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(trained / done_tokens)
            norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
            scale = lr_at(state["step"], total_steps, warmup, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            optimizer.step()
            log(event="step", step=state["step"], epoch=epoch, lr=args.lr * scale, grad_norm=round(norm, 5),
                loss=round(sum(l * n for l, n in losses) / done_tokens, 5), trained_tokens=done_tokens,
                samples=len(losses), seconds=round(time.time() - t0, 1))
        state["step"] += 1
        epoch_end = state["step"] % steps_per_epoch == 0
        if epoch_end:
            save(f"adapter_epoch{epoch + 1}", with_state=False)
            if val_tokens:
                log(event="val", epoch=epoch + 1, **evaluate(model, device, val_tokens))
        if state["step"] % args.save_every == 0 or epoch_end:
            save("latest", with_state=True)
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            log(event="time_limit", step=state["step"])
            save("latest", with_state=True)
            stopped = True
            break
    if not stopped:
        save("adapter_final", with_state=False)
        (args.out / "TRAIN_DONE").write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
