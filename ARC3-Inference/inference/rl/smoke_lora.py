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
import contextlib
import gc
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
from torch.utils.checkpoint import checkpoint

from inference.rl import host_offload, layer_patches, ple_mmap
from inference.rl.context_rebuild import estimate_tokens, rebuild
from inference.rl.qsa import patch_qsa_attention

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
    ids, labels, types, pixel_values, grids, used = [], [], [], [], [], 0
    for piece, train in pieces:
        if not piece:
            continue
        n = piece.count(IMAGE_PAD)
        batch = processor(text=[piece], images=images[used:used + n] or None, return_tensors="pt", add_special_tokens=False)
        used += n
        piece_ids = batch["input_ids"][0]
        ids.append(piece_ids)
        # marks image tokens for the multimodal RoPE; the model refuses images without it
        types.append(batch["mm_token_type_ids"][0])
        labels.append(piece_ids.clone() if train else torch.full_like(piece_ids, -100))
        if n:
            pixel_values.append(batch["pixel_values"])
            grids.append(batch["image_grid_thw"])
    out = {"input_ids": torch.cat(ids)[None], "labels": torch.cat(labels)[None],
           "mm_token_type_ids": torch.cat(types)[None]}
    if pixel_values:
        out["pixel_values"] = torch.cat(pixel_values)
        out["image_grid_thw"] = torch.cat(grids)
    return out


def gpu_device_map(model_path: str) -> dict[str, Any]:
    """Everything on cuda:0 except the model's `_no_placement_params` (the ~95 GiB PLE
    n-gram tables), which stay on CPU where the n-gram lookup then runs.

    device_map="auto" cannot be used: it sizes the routed experts as BF16 (~242 GB)
    before AutoRound swaps them for INT4, and so offloads all of them to CPU."""
    from transformers import AutoConfig, AutoModelForImageTextToText

    with torch.device("meta"):
        skeleton = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(model_path))
    skip = tuple(getattr(skeleton, "_no_placement_params", None) or ())
    device_map: dict[str, Any] = {}

    def place(module, prefix):
        full = lambda name: f"{prefix}.{name}" if prefix else name
        if not any(full(n).endswith(skip) for n, _ in module.named_parameters()) if skip else True:
            device_map[prefix] = 0
            return
        for name, _ in [*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)]:
            if not full(name).endswith(skip):
                device_map[full(name)] = 0
        for name, child in module.named_children():
            place(child, full(name))

    place(skeleton, "")
    if not torch.cuda.is_available():  # CPU dry runs of this script
        device_map = {key: "cpu" for key in device_map}
    return device_map


def memory_gb() -> dict[str, float]:
    return {f"cuda:{i}": round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}


def peak_gb() -> dict[str, float]:
    return {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def chunked_cross_entropy(lm_head, hidden, targets, chunk: int = 2048):
    """Mean cross-entropy of lm_head(hidden) without holding all [K, 248K] logits at once."""
    def piece(h, t):
        return F.cross_entropy(lm_head(h).float(), t, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk):
        total = total + checkpoint(piece, hidden[start:start + chunk], targets[start:start + chunk],
                                   use_reentrant=False)
    return total / targets.numel()


def host_memory_gb() -> dict[str, float]:
    """Resident and peak resident host memory of this process (Linux), in GiB."""
    out = {}
    with contextlib.suppress(OSError):
        for line in open("/proc/self/status"):
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":")
                out["rss" if key == "VmRSS" else "rss_peak"] = round(int(value.split()[0]) / 2**20, 1)
    return out


CGROUP_MEMORY = (  # (limit file, stat file, anonymous-memory key)
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.stat", "anon"),  # cgroup v2
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.stat", "total_rss"),  # v1 (some secure-cloud hosts)
)


def host_memory_available_gb() -> float | None:
    """Memory this container can still allocate (cgroup limit minus anonymous memory), in GiB.
    Page cache, including the memory-mapped PLE table, is reclaimable and not counted."""
    for limit_file, stat_file, anon_key in CGROUP_MEMORY:
        try:
            limit = open(limit_file).read().strip()
            anon = next(int(line.split()[1]) for line in open(stat_file) if line.startswith(anon_key + " "))
        except (OSError, StopIteration, ValueError):
            continue
        if limit == "max" or int(limit) >= 2**60:  # unlimited
            return None
        return (int(limit) - anon) / 2**30
    return None


class HostMemoryError(RuntimeError):
    pass


class LayerTimeline:
    """CUDA events at each decoder layer's forward and backward, for one step (`timeline` in step).

    Forward: a layer's pre- and post-hook. Backward: the gradient reaching the layer's output (its
    backward starts) and leaving through its input (done). Whatever the layers do not account for
    (embeddings, vision, lm_head and loss, offload waits, Python between layers) is the rest."""

    def __init__(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.layers = [x for x in base.modules() if type(x).__name__ == "Qwen4ExpTextDecoderLayer"]
        self.events: dict[tuple[str, int, str], torch.cuda.Event] = {}
        self.forward_phase = True
        self.handles = []
        for i, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._pre(i)))
            self.handles.append(layer.register_forward_hook(self._post(i)))

    def _event(self, key):
        if key not in self.events:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.events[key] = e

    def _pre(self, i):
        def hook(module, args):
            if self.forward_phase:
                self._event(("fwd", i, "start"))
                h = args[0] if args else None
                if torch.is_tensor(h) and h.requires_grad:
                    h.register_hook(lambda g: self._event(("bwd", i, "end")))
        return hook

    def _post(self, i):
        def hook(module, args, out):
            if self.forward_phase:
                self._event(("fwd", i, "end"))
                y = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(y) and y.requires_grad:
                    y.register_hook(lambda g: self._event(("bwd", i, "start")))
        return hook

    def close(self) -> dict[str, Any]:
        for h in self.handles:
            h.remove()
        torch.cuda.synchronize()

        def span(kind, i):
            a, b = self.events.get((kind, i, "start")), self.events.get((kind, i, "end"))
            return a.elapsed_time(b) / 1000 if a is not None and b is not None else None

        fwd = [span("fwd", i) for i in range(len(self.layers))]
        bwd = [span("bwd", i) for i in range(len(self.layers))]
        kinds = [getattr(x, "layer_type", "?") for x in self.layers]
        by_kind: dict[str, dict[str, float]] = {}
        for k, f, b in zip(kinds, fwd, bwd):
            row = by_kind.setdefault(k, dict(layers=0, forward_s=0.0, backward_s=0.0))
            row["layers"] += 1
            row["forward_s"] += f or 0.0
            row["backward_s"] += b or 0.0
        return dict(forward_layers_s=round(sum(f or 0 for f in fwd), 3), backward_layers_s=round(sum(b or 0 for b in bwd), 3),
                    by_kind={k: {x: round(v, 3) for x, v in r.items()} for k, r in by_kind.items()},
                    forward_s=[round(f, 4) if f else None for f in fwd], backward_s=[round(b, 4) if b else None for b in bwd])


def step(model, sample, device, offload: bool = False, profile: bool = False,
         reuse_buffers: bool = True, keep_grads: str | None = None, gpu_keep_gb: float = 0.0,
         async_copies: bool = False, timeline: bool = False, prefix_stop_grad: bool = False,
         memory_snapshot: str | None = None) -> dict[str, Any]:
    """One forward + backward. `offload` keeps the tensors saved for backward (with layer
    checkpointing: each layer's [N, 4 x 2560] input) in host memory instead of on the GPU,
    in host buffers kept on the model and reused by later steps unless `reuse_buffers` is off.
    With reused buffers, the step's last `gpu_keep_gb` stay on the GPU (once a step of the same
    recompute mode has measured the total), and `async_copies` pins the buffers and copies on a
    side stream, prefetching in backward. `memory_snapshot` (a path prefix) records CUDA allocations and
    dumps them at the end of the forward (`_fwd.pickle`) or at an out-of-memory error (`_oom.pickle`)."""
    model.zero_grad(set_to_none=True)  # a layer_check leaves its last variant's gradients on its layers
    # from_pretrained returns an eval-mode model, and transformers checkpoints layers only in
    # training mode: without this every activation is kept (OOM on the GPU, or on the host
    # with offload). Dropout is 0 everywhere, so this changes nothing else.
    model.train()
    tokens = int(sample["input_ids"].shape[1])
    # what each layer saves depends on the recompute mode, and with prefix stop-grad the prefix saves nothing
    layout = layer_patches.current_recompute(model) + ("+prefix" if prefix_stop_grad else "")
    arena = None
    if offload and reuse_buffers:
        arena = getattr(model, "_rft_host_arena", None)  # on the model: survives the job server's reloads
        if arena is None or (async_copies and not arena.pin):  # pinned segments are allocated pinned
            model._rft_host_arena = None
            arena = model._rft_host_arena = host_offload.HostArena(pin=async_copies)
    if offload:
        # each layer's checkpointed input (tokens x 4 x 2560 bf16), plus the mixer's input and output
        # with recompute "parts"; measured once a step of this layout has run
        per_token = (arena.bytes_per_token.get(layout) if arena is not None else None) \
            or 4 * 2560 * 2 * 48 * (1.5 if layout == "parts" else 1.0)
        need = tokens * per_token / 2**30 * 1.15 + 8 - (gpu_keep_gb if arena is not None else 0)
        available = host_memory_available_gb()
        if available is not None and arena is not None:
            available += arena.capacity_gb  # already counted as used, and free again for this step
        if available is not None and need > available:
            raise HostMemoryError(f"offload needs ~{need:.0f} GiB of host memory, {available:.0f} GiB available")
    labels = sample["labels"][0].to(device)
    targets = torch.nonzero(labels[1:] != -100).squeeze(1) + 1  # positions whose token is trained
    inputs = {k: v.to(device) for k, v in sample.items() if k != "labels"}
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    memory = {}
    gb = lambda: round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0
    if memory_snapshot and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(max_entries=200_000)
    _sync()
    memory["start_gb"] = gb()
    t0 = time.perf_counter()
    # pageable, not pinned: the pinned host allocator rounds every block up to a power of two and
    # keeps it cached, so 32K then 64K steps held ~144 GB besides the 102.5 GB PLE table and the
    # cgroup OOM-killed the process. Fresh pageable memory is slow to fill (first-touch page
    # faults), hence the reused arena.
    if arena is not None:
        saved = host_offload.offload_to_host(arena, keep_bytes=gpu_keep_gb * 2**30, async_copies=async_copies,
                                             layout=layout, tokens=tokens)
    elif offload:
        saved = torch.autograd.graph.save_on_cpu(pin_memory=False)
    else:
        saved = contextlib.nullcontext()
    activities = [torch.profiler.ProfilerActivity.CPU] + (
        [torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    profiler = (torch.profiler.profile(activities=activities, profile_memory=True) if profile
                else contextlib.nullcontext())
    layer_timeline = LayerTimeline(model) if timeline and torch.cuda.is_available() else None
    # prefix stop-grad: positions before the one predicting the first trained token run without gradient
    prefix = int(targets.min()) - 1 if prefix_stop_grad and targets.numel() else 0
    layer_patches.set_prefix(prefix)
    failed = True
    try:
        with profiler:
            with saved:
                hidden = base.model(**inputs, use_cache=False).last_hidden_state[0, targets - 1]
                loss = chunked_cross_entropy(base.lm_head, hidden, labels[targets])
            _sync()
            t1 = time.perf_counter()
            memory["after_forward_gb"] = gb()
            if memory_snapshot and torch.cuda.is_available():
                torch.cuda.memory._dump_snapshot(f"{memory_snapshot}_fwd.pickle")
            if layer_timeline is not None:
                layer_timeline.forward_phase = False
            loss.backward()
            _sync()
            t2 = time.perf_counter()
            memory["after_backward_gb"] = gb()
        failed = False
    except torch.cuda.OutOfMemoryError:
        if memory_snapshot:
            torch.cuda.memory._dump_snapshot(f"{memory_snapshot}_oom.pickle")
        raise
    finally:
        layer_patches.set_prefix(0)
        if memory_snapshot and torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(enabled=None)
        if failed and layer_timeline is not None:  # its hooks would stay on the layers for every later step
            for handle in layer_timeline.handles:
                handle.remove()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm = float(torch.sqrt(sum((g.float() ** 2).sum().to("cpu") for g in grads))) if grads else 0.0
    if keep_grads:  # for compare_grads: ~0.1 GB of host memory per kept step at r = 16
        if getattr(model, "_rft_grads", None) is None:
            model._rft_grads = {}
        model._rft_grads[keep_grads] = {name: p.grad.detach().float().cpu() for name, p in model.named_parameters()
                                        if p.requires_grad and p.grad is not None}
    model.zero_grad(set_to_none=True)
    loss_value = float(loss.detach())
    del loss, hidden
    memory["end_gb"] = gb()
    extra = {}
    if layer_timeline is not None:
        extra["timeline"] = layer_timeline.close()
        extra["timeline"]["forward_other_s"] = round(t1 - t0 - extra["timeline"]["forward_layers_s"], 3)
        extra["timeline"]["backward_other_s"] = round(t2 - t1 - extra["timeline"]["backward_layers_s"], 3)
    if profile:
        sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        averages = profiler.key_averages()
        extra["profile"] = averages.table(sort_by=sort, row_limit=30, max_name_column_width=70)
        if torch.cuda.is_available():
            extra["profile_memory"] = averages.table(sort_by="self_cuda_memory_usage", row_limit=25,
                                                     max_name_column_width=70)
    return dict(extra, tokens=int(sample["input_ids"].shape[1]), loss_tokens=int(targets.numel()), offload=offload,
                images=int(sample["image_grid_thw"].shape[0]) if "image_grid_thw" in sample else 0,
                loss=round(loss_value, 4), forward_s=round(t1 - t0, 2), backward_s=round(t2 - t1, 2),
                tokens_per_s=round(sample["input_ids"].shape[1] / (t2 - t0), 1),
                lora_params_with_grad=len(grads), lora_grad_norm=norm, kept_grads=keep_grads, prefix_tokens=prefix,
                peak_gb=peak_gb(), memory_gb=memory, host_gb=host_memory_gb(),
                recompute=layout, host_buffers_gb=round(arena.capacity_gb, 1) if arena is not None else None,
                host_pinned_gb=round(arena.pinned_gb, 1) if arena is not None else None,
                offload_stats={k: round(v, 3) for k, v in arena.stats.items()} if arena is not None else None)


def build_samples(processor, tools, game, specs, loss_replies: int) -> dict[int, dict[str, torch.Tensor]]:
    """One packed sample per distinct target length in `specs` (e.g. "65536o")."""
    samples = {}
    for length in sorted({int(x.rstrip("o")) for x in specs}):
        request = pick_request(game, length)
        if request is None:
            continue
        chat, images = to_chat(request.messages + [request.reply])
        text = processor.apply_chat_template(chat, tools=tools, tokenize=False, preserve_thinking=True)
        samples[length] = tokenize(processor, text, images, loss_replies)
    return samples


def current_token_chunk(model) -> int | None:
    for module in model.modules():
        if hasattr(module, "_rft_token_chunk"):
            return module._rft_token_chunk
    return None


GPU_MARGIN_GB = 8  # left free when gpu_keep_gb is "auto": fragmentation and the peak's variation


def auto_keep_gb(result: dict[str, Any]) -> float:
    """Checkpoint memory a repeat of this step can keep on the GPU: what its peak left free."""
    total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
    peak = max((result.get("peak_gb") or {}).values(), default=0.0) - (result.get("gpu_keep_gb") or 0.0)
    return round(max(total - GPU_MARGIN_GB - peak, 0.0), 1)


def run_specs(model, device, samples, specs, repeats: int, save=lambda steps: None,
              profile: bool = False, reuse_buffers: bool = True, keep_grads: str | None = None,
              gpu_keep_gb: float | str = 0.0, async_copies: bool = False, timeline: bool = False,
              prefix_stop_grad: bool = False, memory_snapshot: str | None = None) -> dict[str, list]:
    """Run each spec ("32768" or "32768o" = with offload) `repeats` times; errors are recorded, not raised.
    `keep_grads` keeps each step's LoRA gradients as "<keep_grads>/<spec>/<repeat>" for compare_grads.
    `gpu_keep_gb="auto"`: the first repeat offloads everything, later ones keep what its peak left free."""
    import gc

    # hand the caching allocator's free blocks back to the driver: Triton kernels (fla's GDN backward autotune)
    # allocate outside it and ran out of memory on an A100 with 10 GB cached but unused
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    steps: dict[str, list] = {}
    for spec in specs:
        length, offload = int(spec.rstrip("o")), spec.endswith("o")
        if length not in samples:
            continue
        results = []
        keep = 0.0 if gpu_keep_gb == "auto" else float(gpu_keep_gb)
        for repeat in range(repeats):
            start_chunk = current_token_chunk(model)
            start_recompute = layer_patches.current_recompute(model)
            retries = 2
            while True:
                oom = False
                try:
                    result = step(model, samples[length], device, offload=offload, profile=profile,
                                  reuse_buffers=reuse_buffers,
                                  keep_grads=f"{keep_grads}/{spec}/{repeat}" if keep_grads else None,
                                  gpu_keep_gb=keep, async_copies=async_copies, timeline=timeline,
                                  prefix_stop_grad=prefix_stop_grad,
                                  memory_snapshot=f"{memory_snapshot}_{spec}_{repeat}" if memory_snapshot else None)
                    result["token_chunk"] = current_token_chunk(model)
                    result["gpu_keep_gb"] = keep
                    if gpu_keep_gb == "auto" and offload:
                        keep = auto_keep_gb(result)
                    results.append(result)
                    print(spec, results[-1], flush=True)
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    where = [f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
                             for f in traceback.extract_tb(exc.__traceback__)[-6:]]
                    results.append(dict(error="oom", detail=str(exc)[:400], where=where, peak_gb=peak_gb(),
                                        token_chunk=current_token_chunk(model),
                                        recompute=layer_patches.current_recompute(model),
                                        memory=torch.cuda.memory_summary(abbreviated=True)[-4000:]
                                        if torch.cuda.is_available() else None))
                    print(spec, "OOM", where, str(exc)[:160], flush=True)
                    oom = True
                except HostMemoryError as exc:
                    results.append(dict(error="host_memory", detail=str(exc)))
                    print(spec, "SKIPPED", exc, flush=True)
                    break
                except Exception:  # record it and keep the other numbers
                    results.append(dict(error="exception", detail=traceback.format_exc()[-3000:]))
                    print(spec, "ERROR", traceback.format_exc(), flush=True)
                    break
                if oom:
                    # out of the except clause: its traceback held the failed step's frames, and with them its
                    # activations and graph (the retry OOMed with the first attempt's 92 GiB still allocated)
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    chunk = current_token_chunk(model)
                    if layer_patches.current_recompute(model) == "parts":
                        # parts OOMed at 123K on a 4.69 GiB (N x 10,240 FP32) allocation that smaller
                        # token chunks do not shrink; layer checkpoints fit that request (129 GiB)
                        layer_patches.configure(model, recompute="layer")
                        print(spec, "retrying with recompute layer", flush=True)
                        continue
                    if retries and chunk and chunk > 1024:  # smaller token chunks need less memory
                        retries -= 1
                        layer_patches.configure(model, token_chunk=chunk // 2)
                        print(spec, "retrying with token_chunk", chunk // 2, flush=True)
                        continue
                    break
            if start_chunk:
                layer_patches.configure(model, token_chunk=start_chunk)
            if layer_patches.current_recompute(model) != start_recompute:
                layer_patches.configure(model, recompute=start_recompute)
            if results and "error" in results[-1]:
                break
        steps[spec] = results
        save(steps)
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    return steps


def compare_grads(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Agreement of two kept LoRA gradient sets, `b` taken as the reference: cosine similarity and
    relative difference over all parameters, per mixer kind, and the worst parameters. Parameters
    whose gradient is zero in both (lora_A while lora_B is still zero) are left out."""
    names = [n for n in sorted(a.keys() & b.keys()) if a[n].norm() > 0 or b[n].norm() > 0]

    def agreement(subset):
        x = torch.cat([a[n].flatten() for n in subset]).double()
        y = torch.cat([b[n].flatten() for n in subset]).double()
        return dict(params=len(subset), norm=round(float(x.norm()), 5), reference_norm=round(float(y.norm()), 5),
                    cosine=round(float(x @ y / (x.norm() * y.norm()).clamp_min(1e-30)), 6),
                    rel_diff=round(float((x - y).norm() / y.norm().clamp_min(1e-30)), 5))

    per_param = []
    for n in names:
        x, y = a[n].flatten().double(), b[n].flatten().double()
        per_param.append((float((x - y).norm() / y.norm().clamp_min(1e-30)),
                          float(x @ y / (x.norm() * y.norm()).clamp_min(1e-30)), n))
    per_param.sort(reverse=True)
    kinds = {kind: [n for n in names if kind in n] for kind in ("self_attn", "linear_attn")}
    rel = sorted(r for r, _, _ in per_param)
    return dict(all=agreement(names), **{kind: agreement(subset) for kind, subset in kinds.items() if subset},
                median_param_rel_diff=round(rel[len(rel) // 2], 5) if rel else None,
                min_param_cosine=round(min(c for _, c, _ in per_param), 6) if per_param else None,
                worst=[dict(param=n, rel_diff=round(r, 5), cosine=round(c, 6)) for r, c, n in per_param[:5]])


def check_layers(model, device, sample, layers, variants: dict[str, Any], qsa_module, patches) -> dict[str, Any]:
    """Real-size check of single decoder layers against the standard answer, on real inputs.

    End-to-end gradients cannot tell a correct rewrite from a wrong one here: top-10-of-512 routing and
    the next layers' top-k selections turn rounding-level differences into other experts and blocks, and
    any two implementations (even BF16 atomics in a different order) agree to only ~0.8-0.87 cosine
    after 48 layers. One layer on one input keeps those flips to a few tokens, and they are counted.

    Captures the inputs of `layers` in one no-grad forward of `sample`, then runs each layer under every
    configuration in `variants` (name -> dict(reference=bool, qsa_fused, compile_hc, ...); "reference"
    is transformers' layer, QSA attention and AutoRound's experts with select_blocks' selection) with
    the same input and upstream gradient, and compares output, input gradient and the layer's LoRA
    gradients with the reference, plus the number of tokens routed to other experts."""
    import inspect

    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = [x for x in base.modules() if isinstance(x, m.Qwen4ExpTextDecoderLayer)]
    signature = inspect.signature(m.Qwen4ExpTextDecoderLayer.forward)
    captured: dict[int, dict[str, Any]] = {}

    def capture(index):
        def hook(module, args, kwargs):
            bound = signature.bind(module, *args, **kwargs).arguments
            bound.pop("self", None)
            captured[index] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in bound.items()}
        return hook

    handles = [decoder[i].register_forward_pre_hook(capture(i), with_kwargs=True) for i in layers]
    model.eval()
    with torch.no_grad():
        base.model(**{k: v.to(device) for k, v in sample.items() if k != "labels"}, use_cache=False)
    for h in handles:
        h.remove()
    model.train()

    routes: list[torch.Tensor] = []
    phase = {"forward": False}
    router_forward = m.Qwen4ExpTextTopKRouter.forward

    def recording_router(self, hidden_states):
        out = router_forward(self, hidden_states)
        if phase["forward"]:
            routes.append(out[2].detach().sort(dim=-1).values)
        return out

    def configure(spec):
        qsa_module.unpatch_reference_indexer()
        if spec.get("reference"):
            qsa_module.patch_qsa_attention(256, compile=False)  # the reference selection is never compiled
            qsa_module.unpatch_qsa_attention()
            qsa_module.patch_reference_indexer()
            patches.unpatch(model)
        else:
            qsa_module.patch_qsa_attention(256, False, fused=spec.get("qsa_fused", True),
                                           compile=spec.get("compile_qsa", False))
            patches.patch_for_training(model, spec.get("token_chunk", 8192), 128, "parts",
                                       compile_hc=spec.get("compile_hc", False),
                                       compile_moe=spec.get("compile_moe", False),
                                       cache_experts=spec.get("cache_experts", True),
                                       fast_conv=spec.get("fast_conv", False),
                                       fast_lora=spec.get("fast_lora", False),
                                       fast_gdn=spec.get("fast_gdn", False))

    def rel(a, b):
        return round(float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)), 6)

    def cos(a, b):
        a, b = a.double().flatten(), b.double().flatten()
        return round(float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30)), 6)

    m.Qwen4ExpTextTopKRouter.forward = recording_router
    out: dict[str, Any] = {}
    try:
        for i in layers:
            layer, kwargs = decoder[i], dict(captured[i])
            hidden = kwargs.pop("hidden_states")
            lora = {n: p for n, p in layer.named_parameters() if "lora_" in n and p.requires_grad}
            g = torch.Generator(device=hidden.device).manual_seed(i)
            results = {}
            for name, spec in variants.items():
                configure(spec)
                h = hidden.clone().requires_grad_(True)
                layer.zero_grad(set_to_none=True)
                routes.clear()
                phase["forward"] = True
                y = type(layer).forward(layer, h, **kwargs)  # not __call__: no layer checkpoint
                phase["forward"] = False
                upstream = torch.randn(y.shape, generator=g.manual_seed(i), device=y.device, dtype=torch.float32)
                (y.float() * upstream).sum().backward()
                results[name] = dict(out=y.detach().float(), dh=h.grad.float(),
                                     lora={n: p.grad.detach().float().clone() for n, p in lora.items()
                                           if p.grad is not None},
                                     routes=torch.cat(routes) if routes else None)
                del y, h
            row: dict[str, Any] = dict(kind=layer.layer_type, tokens=int(hidden.shape[1]))
            ref = results["reference"]
            for name, r in results.items():
                if name == "reference":
                    continue
                names = sorted(r["lora"].keys() & ref["lora"].keys())
                a = torch.cat([r["lora"][n].flatten() for n in names]) if names else torch.zeros(1)
                b = torch.cat([ref["lora"][n].flatten() for n in names]) if names else torch.zeros(1)
                flips = None
                if r["routes"] is not None and ref["routes"] is not None and r["routes"].shape == ref["routes"].shape:
                    flips = int((r["routes"] != ref["routes"]).any(-1).sum())
                row[name] = dict(out_rel=rel(r["out"], ref["out"]), dh_rel=rel(r["dh"], ref["dh"]),
                                 dh_cos=cos(r["dh"], ref["dh"]), lora_rel=rel(a, b), lora_cos=cos(a, b),
                                 lora_params=len(names), routing_flips=flips)
            out[str(i)] = row
            print("layer check", i, json.dumps(row), flush=True)
    finally:
        m.Qwen4ExpTextTopKRouter.forward = router_forward
        qsa_module.unpatch_reference_indexer()
    return out


@torch.no_grad()
def qsa_overlap(model, device, sample, windows=(4, 16, 64, 256), per_window: int = 256,
                recent: int = 4096) -> dict[str, Any]:
    """How much the token selections of neighbouring queries overlap, per QSA layer.

    For `per_window` random windows of w consecutive queries (all far enough in that the selection
    is sparse), the union of their selected tokens relative to one query's selection: 1 means
    they all attend to the same tokens, w that they share none. Also the share of selected
    tokens among the `recent` tokens before each query. Decides how a QSA kernel should tile
    queries: a small union lets a tile of queries load each key once."""
    from inference.rl import qsa as qsa_module
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    attentions = [(name, mod) for name, mod in base.named_modules() if isinstance(mod, m.Qwen4ExpTextAttention)]
    out: dict[str, Any] = {}
    generator = torch.Generator().manual_seed(0)

    def hook(name):
        def measure(module, args, kwargs):
            hidden, position_embeddings = args[0], args[1]
            n = hidden.shape[1]
            cos, sin = (x[:, -n:, :] for x in position_embeddings)
            selected = qsa_module.select_tokens(module.indexer, hidden, cos, sin, 256)  # [N, S], -1 = none
            width = selected.shape[1]
            first = min(4 * width, n - 1)  # earlier queries see few blocks and take nearly all of them
            row = dict(selection=width)
            positions = torch.arange(n, device=selected.device)[:, None]
            valid = selected >= 0
            near = (selected > positions - recent) & valid
            row["recent_share"] = round(float(near[first:].sum() / valid[first:].sum().clamp_min(1)), 4)
            for w in windows:
                if n - w <= first:
                    continue
                starts = torch.randint(first, n - w, (per_window,), generator=generator).tolist()
                ratios = []
                for start in starts:
                    window = selected[start:start + w]
                    union = torch.unique(window[window >= 0]).numel()
                    ratios.append(union / max(int((window >= 0).sum(dim=1).float().mean()), 1))
                ratios = torch.tensor(ratios)
                row[f"union_{w}"] = dict(mean=round(float(ratios.mean()), 2), median=round(float(ratios.median()), 2),
                                         p90=round(float(ratios.quantile(0.9)), 2))
            out[name] = row
        return measure

    handles = [mod.register_forward_pre_hook(hook(name), with_kwargs=True) for name, mod in attentions]
    try:
        model.eval()
        base.model(**{k: v.to(device) for k, v in sample.items() if k != "labels"}, use_cache=False)
    finally:
        for h in handles:
            h.remove()
        model.train()
    return out


def profile_layers(model, device, sample, layers=(0, 1, 3), iters: int = 2) -> dict[str, Any]:
    """Time single decoder layers on the inputs they get in a real forward.

    Captures each layer's inputs during one no-grad forward of `sample`, then per layer:
    a no-grad forward (the first pass under checkpointing) and a forward + backward with
    gradients (the recompute and backward), timed with CUDA events, plus one torch-profiler
    pass over the latter (a single layer has few enough events to profile)."""
    import inspect

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = [m for m in base.modules() if type(m).__name__ == "Qwen4ExpTextDecoderLayer"]
    signature = inspect.signature(type(decoder[0]).forward)
    captured: dict[int, dict[str, Any]] = {}

    def hook(index):
        def capture(module, args, kwargs):
            bound = signature.bind(module, *args, **kwargs).arguments
            bound.pop("self", None)
            captured[index] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in bound.items()}
        return capture

    handles = [decoder[i].register_forward_pre_hook(hook(i), with_kwargs=True) for i in layers]
    inputs = {k: v.to(device) for k, v in sample.items() if k != "labels"}
    model.eval()
    with torch.no_grad():
        base.model(**inputs, use_cache=False)
    for h in handles:
        h.remove()
    model.train()

    def timed(fn):
        if torch.cuda.is_available():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 1000
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    out: dict[str, Any] = {}
    for i in layers:
        layer, kwargs = decoder[i], dict(captured[i])
        hidden = kwargs.pop("hidden_states")
        # call forward directly: GradientCheckpointingLayer.__call__ would checkpoint it
        forward = lambda h: type(layer).forward(layer, h, **kwargs)

        def no_grad_forward():
            with torch.no_grad():
                forward(hidden)

        def grad_forward_backward():
            h = hidden.clone().requires_grad_(True)
            y = forward(h)
            y.backward(torch.ones_like(y))
            model.zero_grad(set_to_none=True)

        no_grad_forward(), grad_forward_backward()  # warm-up: JIT and autotune
        row = dict(kind=getattr(layer, "layer_type", "?"), ple=getattr(layer, "ple", None) is not None,
                   forward_s=round(min(timed(no_grad_forward) for _ in range(iters)), 3),
                   forward_backward_s=round(min(timed(grad_forward_backward) for _ in range(iters)), 3))
        activities = [torch.profiler.ProfilerActivity.CPU] + (
            [torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        with torch.profiler.profile(activities=activities) as prof:
            grad_forward_backward()
            _sync()
        sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        row["top_ops"] = prof.key_averages().table(sort_by=sort, row_limit=15, max_name_column_width=60)
        if torch.cuda.is_available():  # GPU time by kernel category and the top kernels, as bench_layer reports them
            from inference.rl.bench_layer import categorize

            kernels = [e for e in prof.key_averages()
                       if e.self_device_time_total > 0 and str(getattr(e, "device_type", "")).endswith("CUDA")]
            cats: dict[str, float] = {}
            for e in kernels:
                cats[categorize(e.key)] = cats.get(categorize(e.key), 0.0) + e.self_device_time_total / 1e6
            row["gpu_busy_s"] = round(sum(cats.values()), 4)
            row["categories_s"] = {k: round(v, 4) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}
            row["top_kernels"] = [dict(name=e.key[:110], s=round(e.self_device_time_total / 1e6, 4), calls=e.count)
                                  for e in sorted(kernels, key=lambda e: -e.self_device_time_total)[:30]]
        out[str(i)] = row
        print("layer", i, {k: v for k, v in row.items() if k != "top_ops"}, flush=True)
    return out


def use_adapter(model, name: str, rank: int) -> dict[str, int]:
    """Switch to LoRA adapter `name` ("default" = attention + GDN, "shared_expert" adds the shared expert MLP)."""
    from peft import LoraConfig

    if name not in model.peft_config:
        targets = ATTENTION_GDN + (SHARED_EXPERT if name == "shared_expert" else "")
        model.add_adapter(name, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none",
                                           target_modules=targets))
    model.set_adapter(name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(adapter=name, trainable_params=trainable)


def serve(model, device, args, processor, tools, game) -> None:
    """Keep the loaded model and run job files dropped in args.serve until a STOP file appears.

    A job `NAME.job` is JSON: {"lengths": "32768,65536o", "repeats": 2, "qsa_chunk": 256,
    "adapter": "default" | "shared_expert", "profile": false, "transcript": "other_game.txt",
    "token_chunk": 8192, "expert_tile": 128, "reuse_host_buffers": true, "kv_grad_fp32": false,
    "keep_grads": "label", "recompute": "layer" | "parts", "gpu_keep_gb": 0, "async_copies": false,
    "qsa_fused": true, "compile_hc": false, "compile_moe": false, "fast_conv": false, "cache_experts": true,
    "fast_lora": false, "fast_gdn": false, "compile_qsa": false, "prefix_stop_grad": false, "timeline": false (per-layer forward/backward CUDA times),
    "reference": false (true: unpatched transformers QSA and layers, AutoRound experts: the standard answer),
    "reference_qsa" / "reference_layers": one half of it, "reference_selection": transformers' own tie-breaking},
    or {"compare_grads": [["label/65536o/0", "ref/65536o/0"]]}. qsa.py, layer_patches.py,
    host_offload.py and this module are reloaded before each job, so code fixes need no 5-minute
    model reload. Results go to NAME.result.json."""
    import importlib

    jobs = args.serve
    jobs.mkdir(parents=True, exist_ok=True)
    games, default_game = {}, game
    print("serving jobs from", jobs, flush=True)
    while not (jobs / "STOP").exists():
        pending = sorted(jobs.glob("*.job"))
        if not pending:
            time.sleep(2)
            continue
        job_path = pending[0]
        running = job_path.with_suffix(".running")
        job_path.rename(running)
        result: dict[str, Any] = dict(job=job_path.stem)
        try:
            job = json.loads(running.read_text())
            result["job"] = job
            qsa_module = importlib.reload(importlib.import_module("inference.rl.qsa"))
            patches = importlib.reload(importlib.import_module("inference.rl.layer_patches"))
            importlib.reload(importlib.import_module("inference.rl.host_offload"))
            module = importlib.reload(importlib.import_module("inference.rl.smoke_lora"))
            # the standard answer: transformers' QSA ("reference_qsa"), transformers' layers with AutoRound's
            # experts ("reference_layers"), or both ("reference"); the selection is always select_blocks', since
            # transformers' per-query topk breaks the ties at score 0 its own way ("reference_selection": true
            # keeps it)
            reference_qsa = job.get("reference") or job.get("reference_qsa")
            reference_layers = job.get("reference") or job.get("reference_layers")
            qsa_module.unpatch_reference_indexer()
            if reference_qsa:
                qsa_module.patch_qsa_attention(256, compile=False)  # the reference selection is never compiled
                qsa_module.unpatch_qsa_attention()
                if not job.get("reference_selection"):
                    qsa_module.patch_reference_indexer()
            else:
                qsa_module.patch_qsa_attention(job.get("qsa_chunk", args.qsa_chunk), job.get("kv_grad_fp32", False),
                                               fused=job.get("qsa_fused", not args.no_qsa_kernel),
                                               compile=job.get("compile_qsa", False))
            if reference_layers:
                patches.unpatch(model)
                result["layer_patches"] = "reference"
            if not args.no_layer_patches and not reference_layers:
                result["layer_patches"] = patches.patch_for_training(
                    model, job.get("token_chunk", args.token_chunk), job.get("expert_tile", args.expert_tile),
                    job.get("recompute", args.recompute), compile_hc=job.get("compile_hc", args.compile_hc),
                    compile_moe=job.get("compile_moe", args.compile_moe),
                    cache_experts=job.get("cache_experts", True), fast_conv=job.get("fast_conv", args.fast_conv),
                    fast_lora=job.get("fast_lora", False), fast_gdn=job.get("fast_gdn", False))
            if job.get("transcript"):
                if job["transcript"] not in games:
                    games[job["transcript"]] = rebuild(Path(job["transcript"]).read_text(encoding="utf-8"))
                game = games[job["transcript"]]
            else:
                game = default_game
            result["lora"] = module.use_adapter(model, job.get("adapter", "default"), args.rank)
            if "compare_grads" in job:  # [[label, reference_label], ...] of steps kept with keep_grads
                kept = getattr(model, "_rft_grads", None) or {}
                result["kept"] = sorted(kept)
                result["comparisons"] = [dict(a=a, b=b, **module.compare_grads(kept[a], kept[b]))
                                         if a in kept and b in kept else dict(a=a, b=b, error="not kept")
                                         for a, b in job["compare_grads"]]
                (jobs / f"{job_path.stem}.result.json").write_text(json.dumps(result, indent=1))
                running.rename(job_path.with_suffix(".done"))
                print("job", job_path.stem, "done", flush=True)
                continue
            if "qsa_overlap" in job:  # {"length": 65536, "windows": [4, 16, 64, 256], "per_window": 256}
                spec = job["qsa_overlap"]
                samples = module.build_samples(processor, tools, game, [str(spec["length"])], args.loss_replies)
                sample = next(iter(samples.values()))
                result["tokens"] = int(sample["input_ids"].shape[1])
                result["layers"] = module.qsa_overlap(model, device, sample, tuple(spec.get("windows", (4, 16, 64, 256))),
                                                      spec.get("per_window", 256))
                (jobs / f"{job_path.stem}.result.json").write_text(json.dumps(result, indent=1))
                running.rename(job_path.with_suffix(".done"))
                print("job", job_path.stem, "done", flush=True)
                continue
            if "layer_check" in job:
                spec = job["layer_check"]
                samples = module.build_samples(processor, tools, game, [str(spec["length"])], args.loss_replies)
                sample = next(iter(samples.values()))
                result["tokens"] = int(sample["input_ids"].shape[1])
                result["layers"] = module.check_layers(model, device, sample, spec.get("layers", [0, 1, 3]),
                                                       spec["variants"], qsa_module, patches)
                (jobs / f"{job_path.stem}.result.json").write_text(json.dumps(result, indent=1))
                running.rename(job_path.with_suffix(".done"))
                continue
            if "layer_profile" in job:
                spec = job["layer_profile"]
                samples = module.build_samples(processor, tools, game, [str(spec["length"])], args.loss_replies)
                sample = next(iter(samples.values()))
                result["tokens"] = int(sample["input_ids"].shape[1])
                result["layers"] = module.profile_layers(model, device, sample, spec.get("layers", [0, 1, 3]),
                                                         spec.get("iters", 2))
                (jobs / f"{job_path.stem}.result.json").write_text(json.dumps(result, indent=1))
                running.rename(job_path.with_suffix(".done"))
                print("job", job_path.stem, "done", flush=True)
                continue
            specs = [x.strip() for x in job["lengths"].split(",") if x.strip()]
            samples = module.build_samples(processor, tools, game, specs, job.get("loss_replies", args.loss_replies))
            result["samples"] = {k: int(v["input_ids"].shape[1]) for k, v in samples.items()}
            print("job", job_path.stem, job, result["lora"], flush=True)
            out = jobs / f"{job_path.stem}.result.json"
            save = lambda steps: out.write_text(json.dumps(dict(result, steps=steps), indent=1))
            result["steps"] = module.run_specs(model, device, samples, specs, job.get("repeats", args.repeats), save,
                                               profile=job.get("profile", False),
                                               reuse_buffers=job.get("reuse_host_buffers", True),
                                               keep_grads=job.get("keep_grads"),
                                               gpu_keep_gb=job.get("gpu_keep_gb", 0.0),
                                               async_copies=job.get("async_copies", False),
                                               timeline=job.get("timeline", False),
                                               prefix_stop_grad=job.get("prefix_stop_grad", False),
                                               memory_snapshot=job.get("memory_snapshot"))
        except Exception:
            result["error"] = traceback.format_exc()[-3000:]
            print("job", job_path.stem, "ERROR", result["error"], flush=True)
        (jobs / f"{job_path.stem}.result.json").write_text(json.dumps(result, indent=1))
        running.rename(job_path.with_suffix(".done"))
        print("job", job_path.stem, "done", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True, help="tool schema as sent by the harness (JSON list)")
    parser.add_argument("--lengths", default="32768,32768o,65536o,100000o",
                        help="target lengths, run in order; suffix o = offload saved activations to host memory")
    parser.add_argument("--repeats", type=int, default=2, help="steps per length; the first includes warm-up")
    parser.add_argument("--loss-replies", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--shared-expert", action="store_true", help="also put LoRA on the shared expert MLP")
    parser.add_argument("--backend", default="auto", help="AutoRound kernel backend (auto, torch, tritonv2, ...)")
    parser.add_argument("--attn", default="sdpa", help="attn_implementation")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--reference-qsa", action="store_true", help="keep transformers' QSA (dense mask, slow)")
    parser.add_argument("--qsa-chunk", type=int, default=256, help="queries per chunk in our QSA")
    parser.add_argument("--no-qsa-kernel", action="store_true", help="chunked PyTorch QSA instead of qsa_kernel")
    parser.add_argument("--compile-hc", action="store_true", help="torch.compile the hyper-connections")
    parser.add_argument("--compile-moe", action="store_true", help="torch.compile the MoE combine")
    parser.add_argument("--fast-conv", action="store_true", help="channels-last GDN short convolution (compiled)")
    parser.add_argument("--serve", type=Path, help="after loading, run job files from this directory (see serve)")
    parser.add_argument("--no-layer-patches", action="store_true",
                        help="keep transformers' decoder layers and AutoRound's experts loop")
    parser.add_argument("--token-chunk", type=int, default=8192, help="tokens per chunk in patched decoder layers")
    parser.add_argument("--expert-tile", type=int, default=128, help="experts dequantized per grouped GEMM")
    parser.add_argument("--recompute", default="layer", choices=layer_patches.RECOMPUTE_MODES,
                        help="layer: transformers' layer checkpoints (token chunks run 3x); parts: mixer and "
                             "token chunks checkpointed separately (2x)")
    parser.add_argument("--use-kernels", action="store_true", help="load hub kernels (causal-conv1d) via use_kernels")
    parser.add_argument("--ple-in-ram", action="store_true",
                        help="load the 102.5 GB PLE table into host RAM instead of memory-mapping the checkpoint")
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
    specs = [x.strip() for x in args.lengths.split(",") if x.strip()]
    samples = {} if args.serve else build_samples(processor, tools, game, specs, args.loss_replies)
    report["samples"] = {k: int(v["input_ids"].shape[1]) for k, v in samples.items()}
    print("samples (target length -> tokens):", report["samples"], flush=True)

    if not args.reference_qsa:
        patch_qsa_attention(args.qsa_chunk, fused=not args.no_qsa_kernel)
    if not args.ple_in_ram:
        ple_mmap.install()  # before the device map and the load: transformers then skips the table's shards
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = dict(dtype=torch.bfloat16, device_map=gpu_device_map(args.model),
                                  attn_implementation=args.attn)
    if args.backend != "auto":
        from transformers import AutoRoundConfig
        kwargs["quantization_config"] = AutoRoundConfig(backend=args.backend)
    if args.use_kernels:
        kwargs["use_kernels"] = True
    model = None
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
    except Exception as exc:  # hub kernels are optional
        if not kwargs.pop("use_kernels", False):
            raise
        report["use_kernels_error"] = f"{type(exc).__name__}: {exc}"[:500]
    if model is None:
        # retry outside the except block: its traceback keeps the failed model (~70 GB on the GPU) alive
        gc.collect()
        torch.cuda.empty_cache()
        print("use_kernels failed, loading without:", report["use_kernels_error"], flush=True)
        model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
    if not args.ple_in_ram:
        report["ple_mmap"] = ple_mmap.attach(model, args.model)
    report["load"] = dict(seconds=round(time.perf_counter() - t0, 1), memory_gb=memory_gb(), host_gb=host_memory_gb())
    kinds = {}
    for name, module in model.named_modules():
        for key, pattern in (("attention", r"self_attn\.q_proj$"), ("gdn", r"linear_attn\.in_proj_qkv$"),
                             ("routed_expert", r"mlp\.experts(\.\d+\.down_proj)?$"), ("indexer", r"indexer$")):
            if key not in kinds and re.search(pattern, name):
                kinds[key] = f"{name}: {type(module).__module__}.{type(module).__name__}"
    report["modules"] = kinds
    report["ple_devices"] = sorted({str(p.device) for n, p in model.named_parameters() if "ngram_embedding" in n})
    print("loaded", report["load"], json.dumps(kinds, indent=1), flush=True)
    save()

    model.config.use_cache = False
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # layers are checkpointed only in training mode
    report["checkpointed_layers"] = sum(1 for m in model.modules() if getattr(m, "gradient_checkpointing", False))
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                      target_modules=ATTENTION_GDN + (SHARED_EXPERT if args.shared_expert else ""))
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["lora"] = dict(trainable_params=trainable,
                          modules=sum(1 for n, _ in model.named_modules() if n.endswith("lora_A")))
    print("lora", report["lora"], flush=True)
    device = model.get_input_embeddings().weight.device
    if not args.no_layer_patches:
        report["layer_patches"] = layer_patches.patch_for_training(
            model, args.token_chunk, args.expert_tile, args.recompute, compile_hc=args.compile_hc,
            compile_moe=args.compile_moe, fast_conv=args.fast_conv)
        report["experts_check"] = layer_patches.check_experts(model)
        print("layer patches", report["layer_patches"], "experts check", report["experts_check"], flush=True)
        if report["experts_check"]["max_rel_error"] > 0.05:  # grouped experts disagree: keep AutoRound's
            for module in model.modules():
                if hasattr(module, "_rft_previous_implementation"):
                    module.config._experts_implementation = module._rft_previous_implementation
            report["layer_patches"]["grouped_experts"] = "disabled: check failed"
    save()

    if args.serve:
        serve(model, device, args, processor, tools, game)
        return 0

    report["steps"] = run_specs(model, device, samples, specs, args.repeats,
                                lambda steps: (report.__setitem__("steps", steps), save()))

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
