"""RFT training: LoRA on the selected level clears (rft_data), one sequence per sample.

Loads the W4A16 checkpoint with the patches smoke_lora measured (fused QSA, compiled
hyper-connections and MoE, fast LoRA and GDN, parts recompute, async offload), then runs
`epochs` passes over train.jsonl in a shuffled order, `batch` samples per optimizer step.

- Loss: cross-entropy on each sample's trained replies; in a batch every trained token
  counts once (a sample's mean loss is weighted by its share of the batch's tokens).
- Prefix stop-grad: the context before a sample's first trained token runs forward only.
- AdamW, linear warm-up then cosine decay to `min_lr_ratio`, gradient clipping.
- Host memory: the checkpoints go to a pinned host arena of a fixed size, reserved at start
  (`arena_cap_gb`): the container's limit minus the PLE table (it must stay in the page cache:
  an arena that grew to ~157 GiB evicted it in the first run and the GPU idled on disk reads),
  the process, the prefetched batches and a margin. A sample uses recompute "parts" when its
  checkpoints fit the arena, else "layer" (~2.4x smaller); nothing is kept on the GPU on purpose
  (a kept view holds its whole base tensor: 19 GB kept ran out of memory at a 95 GB peak). An
  out-of-memory sample is retried with "layer", then skipped (its partial gradients are taken
  back out first).
- Prefetch: the next batch is tokenized and its PLE rows gathered on a thread while the GPU
  runs the current one.
- Saves: the adapter after every epoch (`adapter_epoch<N>`) and `latest` (adapter,
  optimizer, position) every `save_every` steps; `--resume` continues from `latest`.
- Validation: mean loss on val.jsonl (forward only) before training and after each epoch.
- wandb (`--wandb-project`, key from WANDB_API_KEY): every log row also goes to wandb (wandb_log); the
  run id is kept in OUT/wandb_id.txt so a restart from `latest` continues the same run.

    python -m inference.rft.train_rft --model /workspace/w4a16 --data /workspace/data --out /workspace/train
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

from inference.rft import host_offload, layer_patches, ple_mmap, smoke_lora
from inference.rft.qsa import patch_qsa_attention

# Saved bytes per token after the prefix, until a step of the layout has measured them: parts ~2.3 MB (the first
# run's arena, 144 GiB for a 68.5K-token suffix, rounded up), layer ~1.2 MB (48 layers x 4 x 2,560 BF16 inputs =
# 0.98 MB, plus a margin)
DEFAULT_BYTES_PER_TOKEN = {"parts": 2.4e6, "layer": 1.2e6}
PREFETCH_RESERVE_GB = 12  # two tokenized batches of 4 (images ~0.5-1 GB, PLE rows up to 0.65 GB per sample)
HOST_MARGIN_GB = 10  # other page cache (images, libraries), CUDA host allocations, allocator slack


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


def arena_cap_gb(ple_gb: float, process_gb: float, limit_gb: float | None) -> float:
    """Host arena size: what the container can give without pushing the PLE table out of the page cache."""
    if limit_gb is None:
        return 0.0
    return round(max(0.0, limit_gb - ple_gb - process_gb - PREFETCH_RESERVE_GB - HOST_MARGIN_GB), 1)


def suffix_tokens(sample, prefix_stop_grad: bool) -> int:
    """Tokens whose checkpoints are saved: from the position predicting the first trained token (prefix stop-grad)."""
    tokens = int(sample["input_ids"].shape[1])
    if not prefix_stop_grad:
        return tokens
    trained = torch.nonzero(sample["labels"][0][1:] != -100)
    return tokens - (int(trained.min()) if trained.numel() else tokens)


def choose_recompute(saved_tokens: int, arena_bytes: float | None, bytes_per_token: dict[str, float],
                     prefix_stop_grad: bool) -> str:
    """"parts" when the sample's checkpoints fit the arena, else "layer" (an uncapped arena: always "parts")."""
    if arena_bytes is None:
        return "parts"
    key = "parts+prefix" if prefix_stop_grad else "parts"
    per_token = bytes_per_token.get(key) or DEFAULT_BYTES_PER_TOKEN["parts"]
    return "parts" if saved_tokens * per_token <= arena_bytes else "layer"


def lr_at(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def run_sample(model, device, sample, scale: float, prefix_stop_grad: bool = True, loss_fn=None) -> dict[str, Any]:
    """Forward + backward of one sample, adding scale x its gradient; recompute mode and OOM fallback as in the
    docstring. `loss_fn` replaces the cross-entropy (smoke_lora.step)."""
    params = [p for p in model.parameters() if p.requires_grad]
    saved = [None if p.grad is None else p.grad.detach().clone() for p in params]
    tokens = int(sample["input_ids"].shape[1])
    arena = getattr(model, "_rft_host_arena", None)
    saved_tokens = suffix_tokens(sample, prefix_stop_grad)
    mode = choose_recompute(saved_tokens, arena.max_bytes if arena is not None else None,
                            arena.bytes_per_token if arena is not None else {}, prefix_stop_grad)
    attempts = [mode] + (["layer"] if mode != "layer" else [])
    start_mode = layer_patches.current_recompute(model)
    errors = []
    for mode in attempts:
        if layer_patches.current_recompute(model) != mode:
            layer_patches.configure(model, recompute=mode)
        oom = False
        try:
            cuda = torch.cuda.is_available()  # CPU: the tests
            result = smoke_lora.step(model, sample, device, offload=cuda, async_copies=cuda, gpu_keep_gb=0.0,
                                     prefix_stop_grad=prefix_stop_grad, accumulate=True, loss_scale=scale,
                                     loss_fn=loss_fn)
            result.update(recompute_mode=mode, retries=len(errors), saved_tokens=saved_tokens)
            break
        except torch.cuda.OutOfMemoryError as exc:
            where = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
                                for f in traceback.extract_tb(exc.__traceback__)[-4:][::-1])
            errors.append(f"{mode}: {str(exc)[:100]} at {where}")
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
        return dict(error="; ".join(errors) or "failed", tokens=tokens, saved_tokens=saved_tokens)
    if errors:
        result["errors"] = errors
    return result


@torch.no_grad()
def evaluate(model, device, samples: list[dict[str, Any]], prepare) -> dict[str, Any]:
    """Mean loss over the trained tokens of `samples`, tokenized one at a time by `prepare` (kept tokenized for the
    whole run, 16 samples held ~10-16 GB of image tensors in host memory)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    total = count = 0.0
    skipped = 0
    model.train()  # the patched layers' token chunks; dropout is 0
    for raw in samples:
        sample = prepare(raw)
        if sample is None:
            skipped += 1
            continue
        labels = sample["labels"][0].to(device)
        targets = torch.nonzero(labels[1:] != -100).squeeze(1) + 1
        inputs = {k: v.to(device) for k, v in sample.items() if k != "labels" and not k.startswith("_")}
        layer_patches.provide_ngram(model, sample.get("_ngram"))
        hidden = base.model(**inputs, use_cache=False).last_hidden_state[0, targets - 1]
        loss = smoke_lora.chunked_cross_entropy(base.lm_head, hidden, labels[targets])
        total += float(loss) * targets.numel()
        count += targets.numel()
        del hidden, loss, sample, inputs
    layer_patches.provide_ngram(model, None)
    return dict(loss=round(total / max(count, 1), 5), tokens=int(count), samples=len(samples) - skipped)


def load_model(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    patch_qsa_attention(256, fused=args.qsa_kernel, compile=args.compile_qsa and args.qsa_kernel)
    ple_mmap.install()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                                                        device_map=smoke_lora.gpu_device_map(args.model),
                                                        attn_implementation="sdpa")
    ple = ple_mmap.attach(model, args.model)
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
    return model, processor, device, dict(patches=patches, experts_check=check, ple=ple)


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
    parser.add_argument("--arena-gb", type=float, help="host arena size (default: from the container's limit)")
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
    parser.add_argument("--val-samples", type=int, default=16, help="validation samples, spread over val.jsonl")
    parser.add_argument("--wandb-project", help="also log to this wandb project (key: WANDB_API_KEY)")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    started = time.time()

    wandb_logger = None
    if args.wandb_project:
        try:
            from inference.rft import wandb_log
            id_file = args.out / "wandb_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else None
            if run_id is None:
                run_id = wandb_log.new_run_id()
                id_file.write_text(run_id)
            run = wandb_log.start(args.wandb_project, args.wandb_run_name, run_id,
                                  {k: str(v) for k, v in vars(args).items()}, args.wandb_entity)
            wandb_logger = wandb_log.WandbLogger(run)
        except Exception as exc:  # noqa: BLE001 - monitoring must never stop training
            print(f"wandb disabled: {type(exc).__name__}: {exc}", flush=True)

    def log(**row):
        row = dict(t=round(time.time() - started, 1), **row)
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if wandb_logger is not None:
            try:
                wandb_logger.event(row)
            except Exception as exc:  # noqa: BLE001
                print(f"wandb log failed: {exc}", flush=True)

    train = load_samples(args.data / "train.jsonl")
    val = load_samples(args.data / "val.jsonl")
    if args.limit:
        train = train[:args.limit]
    if args.val_samples and len(val) > args.val_samples:  # every k-th: all held-out trajectories, ~5 min per pass
        val = val[::len(val) // args.val_samples][:args.val_samples]
    model, processor, device, info = load_model(args)
    log(event="loaded", train=len(train), val=len(val), args={k: str(v) for k, v in vars(args).items()}, **info)
    if torch.cuda.is_available():
        memory, process_gb = smoke_lora.cgroup_memory_gb(), smoke_lora.host_memory_gb().get("rss", 15.0)
        cap = args.arena_gb if args.arena_gb is not None else arena_cap_gb(info["ple"]["mapped_gb"], process_gb,
                                                                             memory.get("limit_gb"))
        arena = model._rft_host_arena = host_offload.HostArena(pin=True, max_gb=cap)
        log(event="arena", cap_gb=cap, limit_gb=memory.get("limit_gb"), ple_gb=info["ple"]["mapped_gb"],
            process_gb=process_gb, prefetch_reserve_gb=PREFETCH_RESERVE_GB, margin_gb=HOST_MARGIN_GB,
            reserve_s=round(arena.reserve(), 1), cgroup=smoke_lora.cgroup_memory_gb())

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

    def prepare(sample):
        """Tokens, images and PLE rows of one sample (on the prefetch thread), with what it cost."""
        start, faults = time.perf_counter(), smoke_lora._thread_major_faults()
        item = tokenize_sample(processor, sample, args.data)
        tokenized = time.perf_counter()
        if item["input_ids"].shape[1] <= args.max_tokens:
            item["_ngram"] = layer_patches.precompute_ngram(model, item["input_ids"])
        item["_prefetch"] = dict(tokenize_s=round(tokenized - start, 2), ple_s=round(time.perf_counter() - tokenized, 2),
                                 major_faults=smoke_lora._thread_major_faults() - faults)
        return item

    def tokenize_batch(indices):
        out = []
        for i in indices:
            try:
                out.append((i, prepare(train[i])))
            except Exception as exc:  # noqa: BLE001 - a bad sample is skipped, not fatal
                out.append((i, f"{type(exc).__name__}: {str(exc)[:300]}"))
        return out

    def prepare_val(sample):
        try:
            item = prepare(sample)
        except Exception as exc:  # noqa: BLE001
            log(event="val_skip", id=sample["id"], error=str(exc)[:300])
            return None
        return item if item["input_ids"].shape[1] <= args.max_tokens else None

    if val and state["step"] == 0:
        log(event="val", epoch=0, **evaluate(model, device, val, prepare_val))

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
            result = run_sample(model, device, item, n / trained, args.prefix_stop_grad)
            row = {k: result.get(k) for k in ("tokens", "loss_tokens", "prefix_tokens", "saved_tokens", "loss",
                                              "forward_s", "backward_s", "tokens_per_s", "peak_gb", "recompute_mode",
                                              "error", "errors", "arena_used_gb", "major_faults", "ple_lookup_s",
                                              "graphs_compiled", "cgroup", "wall_start", "wall_end")}
            stats = result.get("offload_stats") or {}
            row.update(overflow_gb=stats.get("overflow_gb"), d2h_gb=stats.get("d2h_gb"), prefetch=item.get("_prefetch"))
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
            if val:
                log(event="val", epoch=epoch + 1, **evaluate(model, device, val, prepare_val))
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
    if wandb_logger is not None:
        try:
            wandb_logger.run.finish()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
