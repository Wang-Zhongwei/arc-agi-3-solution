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


def step(model, sample, device, offload: bool = False, profile: bool = False) -> dict[str, Any]:
    """One forward + backward. `offload` keeps the tensors saved for backward (with layer
    checkpointing: each layer's [N, 4 x 2560] input) in host memory instead of on the GPU."""
    # from_pretrained returns an eval-mode model, and transformers checkpoints layers only in
    # training mode: without this every activation is kept (OOM on the GPU, or on the host
    # with offload). Dropout is 0 everywhere, so this changes nothing else.
    model.train()
    labels = sample["labels"][0].to(device)
    targets = torch.nonzero(labels[1:] != -100).squeeze(1) + 1  # positions whose token is trained
    inputs = {k: v.to(device) for k, v in sample.items() if k != "labels"}
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    _sync()
    t0 = time.perf_counter()
    # pageable, not pinned: the pinned host allocator rounds every block up to a power of two and
    # keeps it cached, so 32K then 64K steps held ~144 GB besides the 102.5 GB PLE table and the
    # cgroup OOM-killed the process
    saved = torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else contextlib.nullcontext()
    activities = [torch.profiler.ProfilerActivity.CPU] + (
        [torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    profiler = (torch.profiler.profile(activities=activities, profile_memory=True) if profile
                else contextlib.nullcontext())
    with profiler:
        with saved:
            hidden = base.model(**inputs, use_cache=False).last_hidden_state[0, targets - 1]
            loss = chunked_cross_entropy(base.lm_head, hidden, labels[targets])
        _sync()
        t1 = time.perf_counter()
        loss.backward()
        _sync()
        t2 = time.perf_counter()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm = float(torch.sqrt(sum((g.float() ** 2).sum().to("cpu") for g in grads))) if grads else 0.0
    model.zero_grad(set_to_none=True)
    extra = {}
    if profile:
        sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        averages = profiler.key_averages()
        extra["profile"] = averages.table(sort_by=sort, row_limit=30, max_name_column_width=70)
        if torch.cuda.is_available():
            extra["profile_memory"] = averages.table(sort_by="self_cuda_memory_usage", row_limit=25,
                                                     max_name_column_width=70)
    return dict(extra, tokens=int(sample["input_ids"].shape[1]), loss_tokens=int(targets.numel()), offload=offload,
                images=int(sample["image_grid_thw"].shape[0]) if "image_grid_thw" in sample else 0,
                loss=round(float(loss), 4), forward_s=round(t1 - t0, 2), backward_s=round(t2 - t1, 2),
                tokens_per_s=round(sample["input_ids"].shape[1] / (t2 - t0), 1),
                lora_params_with_grad=len(grads), lora_grad_norm=norm, peak_gb=peak_gb(), host_gb=host_memory_gb())


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


def run_specs(model, device, samples, specs, repeats: int, save=lambda steps: None,
              profile: bool = False) -> dict[str, list]:
    """Run each spec ("32768" or "32768o" = with offload) `repeats` times; errors are recorded, not raised."""
    steps: dict[str, list] = {}
    for spec in specs:
        length, offload = int(spec.rstrip("o")), spec.endswith("o")
        if length not in samples:
            continue
        results = []
        for _ in range(repeats):
            try:
                results.append(step(model, samples[length], device, offload=offload, profile=profile))
                print(spec, results[-1], flush=True)
            except torch.cuda.OutOfMemoryError as exc:
                where = [f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
                         for f in traceback.extract_tb(exc.__traceback__)[-6:]]
                results.append(dict(error="oom", detail=str(exc)[:400], where=where, peak_gb=peak_gb()))
                print(spec, "OOM", where, str(exc)[:160], flush=True)
                break
            except Exception as exc:  # record it and keep the other numbers
                results.append(dict(error=type(exc).__name__, detail=traceback.format_exc()[-3000:]))
                print(spec, "ERROR", traceback.format_exc(), flush=True)
                break
        steps[spec] = results
        save(steps)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return steps


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
    "adapter": "default" | "shared_expert", "profile": false}. qsa.py and this module are reloaded before
    each job, so code fixes need no 5-minute model reload. Results go to NAME.result.json."""
    import importlib

    jobs = args.serve
    jobs.mkdir(parents=True, exist_ok=True)
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
            module = importlib.reload(importlib.import_module("inference.rl.smoke_lora"))
            qsa_module.patch_qsa_attention(job.get("qsa_chunk", args.qsa_chunk))
            result["lora"] = module.use_adapter(model, job.get("adapter", "default"), args.rank)
            specs = [x.strip() for x in job["lengths"].split(",") if x.strip()]
            samples = module.build_samples(processor, tools, game, specs, job.get("loss_replies", args.loss_replies))
            result["samples"] = {k: int(v["input_ids"].shape[1]) for k, v in samples.items()}
            print("job", job_path.stem, job, result["lora"], flush=True)
            out = jobs / f"{job_path.stem}.result.json"
            save = lambda steps: out.write_text(json.dumps(dict(result, steps=steps), indent=1))
            result["steps"] = module.run_specs(model, device, samples, specs, job.get("repeats", args.repeats), save,
                                               profile=job.get("profile", False))
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
    parser.add_argument("--serve", type=Path, help="after loading, run job files from this directory (see serve)")
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
        patch_qsa_attention(args.qsa_chunk)
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = dict(dtype=torch.bfloat16, device_map=gpu_device_map(args.model),
                                  attn_implementation=args.attn)
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
