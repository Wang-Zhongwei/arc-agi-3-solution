"""GRPO trainer: one process per training GPU, LoRA data parallel over NCCL (grpo-pipeline.md).

Loads the model as train_rft does (same patches, host arena, OOM fallback), then serves the steps inference.grpo.loop
writes, in order:

- jobs/step_<k>.json: {"step": k, "lr": ..., "total_tokens": N (trained tokens of the step over all ranks),
  "samples": {"<rank>": [{"episode": <rl jsonl>, "advantage": A, "id": ...}]}}.
- Each rank builds its samples (episodes.load: vLLM's ids, trained positions, behavior logprobs, images) and runs
  forward + backward with the GRPO objective (loss.grpo_objective over loss.processed_logprobs) divided by N, so
  the ranks' summed gradient is the gradient of the token mean over the whole step.
- The LoRA gradients are summed over the ranks, clipped, and every rank takes the same optimizer step.
- Rank 0 then saves ckpt/latest (adapter, optimizer, state) and ckpt/step_<k> (adapter), writes the merged
  attention/GDN weights, W + (alpha/r) B A rounded stochastically to BF16 under checkpoint names, to
  sync/step_<k>.safetensors, and jobs/step_<k>.done.json. Every rank writes jobs/step_<k>.rank<r>.json.

Launch one process per GPU, each seeing only its own (CUDA_VISIBLE_DEVICES), with RANK, WORLD_SIZE, MASTER_ADDR and
MASTER_PORT set:
    python -m inference.grpo.train_grpo --model /workspace/w4a16 --work /workspace/grpo [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from inference.grpo import episodes
from inference.grpo import loss as grpo_loss

IMAGE_TOKEN = episodes.IMAGE_PAD
MERGE = 2  # vision merge size: tokens per image = grid_t * grid_h * grid_w / MERGE^2


def build_inputs(ep: episodes.Episode, processor=None, max_pixels: int = 589_824,
                 min_pixels: int = 65_536) -> dict[str, torch.Tensor]:
    """Model inputs of one episode: vLLM's ids, labels at the trained positions, image tensors (the same processor
    and max_pixels as the server's --mm-processor-kwargs)."""
    ids = torch.tensor(ep.ids, dtype=torch.long)[None]
    labels = torch.full_like(ids, -100)
    targets = torch.tensor(ep.targets, dtype=torch.long)
    labels[0, targets] = ids[0, targets]
    out = {"input_ids": ids, "labels": labels, "mm_token_type_ids": (ids == IMAGE_TOKEN).long()}
    if ep.images:
        if processor is None:
            raise ValueError("an episode with images needs the processor")
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in ep.images]
        vision = processor.image_processor(images=images, return_tensors="pt",
                                           size={"shortest_edge": min_pixels, "longest_edge": max_pixels})
        expected = int((vision["image_grid_thw"].prod(-1) // (MERGE * MERGE)).sum())
        found = int((ids == IMAGE_TOKEN).sum())
        if expected != found:
            raise ValueError(f"{ep.stem}: {found} image tokens in the ids, the processor makes {expected}")
        out["pixel_values"] = vision["pixel_values"]
        out["image_grid_thw"] = vision["image_grid_thw"]
    return out


def make_loss_fn(behavior: list[float], advantage: float, norm: float, settings: SimpleNamespace):
    """smoke_lora.step's loss_fn: the step-normalized GRPO objective of one episode."""
    behavior_t = torch.tensor(behavior, dtype=torch.float32)

    def loss_fn(lm_head, hidden, target_ids):
        if hidden.shape[0] != behavior_t.numel():
            raise ValueError(f"{hidden.shape[0]} trained positions, {behavior_t.numel()} behavior logprobs")
        logp, kept = grpo_loss.processed_logprobs(lm_head, hidden, target_ids, settings.temperature,
                                                  settings.top_k, settings.top_p)
        total, stats = grpo_loss.grpo_objective(logp, behavior_t.to(logp.device), kept, advantage,
                                                settings.eps_low, settings.eps_high, settings.ratio_min,
                                                settings.ratio_max)
        return total / norm, stats

    return loss_fn


def trainable(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def sum_gradients(params: list[torch.nn.Parameter]) -> None:
    """Sum every LoRA gradient over the ranks (a rank that trained nothing contributes zeros)."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.detach().float().flatten() for p in params])
    dist.all_reduce(flat)
    offset = 0
    for p in params:
        n = p.numel()
        p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
        offset += n


def optimizer_for(params, name: str, lr: float):
    if name == "muon":
        return grpo_loss.Muon(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)


def run_rank_samples(model, device, items: list[tuple[dict[str, torch.Tensor], Any]], prefix_stop_grad: bool,
                     run_sample) -> list[dict[str, Any]]:
    """Forward + backward of this rank's samples, accumulating gradients. items: (inputs, loss_fn)."""
    rows = []
    for inputs, loss_fn in items:
        result = run_sample(model, device, inputs, 1.0, prefix_stop_grad, loss_fn=loss_fn)
        rows.append(result)
    return rows


def finish_step(model, optimizer, lr: float, clip: float) -> float:
    """Sum the ranks' gradients, clip, step. Returns the gradient norm before clipping."""
    params = trainable(model)
    sum_gradients(params)
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    norm = float(torch.nn.utils.clip_grad_norm_(params, clip))
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    return norm


def make_prepare(settings: SimpleNamespace, thinking_budget: int, max_tokens: int, processor=None,
                 max_pixels: int = 589_824, ngram_model=None):
    """prepare(spec, norm) -> (model inputs, loss_fn, episode) for one job sample; raises when it cannot train."""
    def prepare(spec: dict[str, Any], norm: float):
        ep = episodes.load(Path(spec["episode"]), thinking_budget)
        if ep.error:
            raise ValueError(ep.error)
        if len(ep.ids) > max_tokens:
            raise ValueError(f"{len(ep.ids)} tokens")
        inputs = build_inputs(ep, processor, max_pixels)
        if ngram_model is not None:  # PLE rows gathered here, on the prefetch thread
            from inference.rft import layer_patches

            inputs["_ngram"] = layer_patches.precompute_ngram(ngram_model, inputs["input_ids"])
        return inputs, make_loss_fn(ep.behavior, float(spec["advantage"]), norm, settings), ep

    return prepare


def run_job(model, device, job: dict[str, Any], rank: int, optimizer, prepare, clip: float,
            prefix_stop_grad: bool = False, on_sample=None) -> tuple[list[dict[str, Any]], list[float]]:
    """One job: for each minibatch, this rank's rollouts one at a time (their gradients add up), the ranks' gradients
    summed, one optimizer step. Every rank steps for every minibatch, also when it has none of its rollouts (it adds
    zeros). The next rollout is prepared on a thread meanwhile, across minibatch boundaries.
    Returns the samples' rows, each with its minibatch, and each minibatch's gradient norm before clipping.
    `on_sample(row, inputs, result, episode)` sees every trained sample (the CPU example prints and stops there)."""
    from inference.rft import train_rft

    # a job without minibatches (every group's rewards were equal) makes no update
    minibatches = job["minibatches"] if "minibatches" in job else [dict(total_tokens=job["total_tokens"],
                                                                         samples=job["samples"])]
    todo = [(m, spec, max(1.0, float(batch["total_tokens"])))
            for m, batch in enumerate(minibatches) for spec in batch["samples"].get(str(rank), [])]
    prefetch = train_rft.Prefetch(lambda item: prepare(item[1], item[2]))
    if todo:
        prefetch.start(todo[0])
    rows, norms, i = [], [], 0
    for m in range(len(minibatches)):
        optimizer.zero_grad(set_to_none=True)
        while i < len(todo) and todo[i][0] == m:
            spec = todo[i][1]
            try:
                inputs, loss_fn, ep = prefetch.get()
            except Exception as exc:  # noqa: BLE001 - a bad episode is skipped, the step goes on
                inputs = None
                rows.append(dict(id=spec.get("id"), minibatch=m, error=f"prepare: {str(exc)[:300]}"))
            i += 1
            if i < len(todo):
                prefetch.start(todo[i])
            if inputs is None:
                continue
            result = train_rft.run_sample(model, device, inputs, 1.0, prefix_stop_grad, loss_fn=loss_fn)
            row = {k: result.get(k) for k in ("tokens", "loss_tokens", "loss", "forward_s", "backward_s",
                                              "tokens_per_s", "peak_gb", "recompute_mode", "error", "loss_stats")}
            rows.append(dict(id=spec.get("id"), minibatch=m, advantage=spec["advantage"], **row))
            if on_sample is not None:
                on_sample(rows[-1], inputs, result, ep)
            del inputs
        norms.append(finish_step(model, optimizer, float(job["lr"]), clip))
    return rows, norms


def lora_modules(model):
    """(checkpoint-style module name, peft LoRA layer) for every adapted module."""
    from peft.tuners.lora import LoraLayer

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and "default" in getattr(module, "lora_A", {}):
            yield name.removeprefix("base_model.model."), module


@torch.no_grad()
def merged_weights(model, weight_keys: list[str], seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """{checkpoint key: BF16 W + (alpha/r) B A, rounded stochastically}, and how much of the update survived."""
    from inference.rft.merge_lora import resolve

    modules = dict(lora_modules(model))
    keys = resolve(sorted(modules), weight_keys)
    out, kept, total = {}, 0.0, 0.0
    generator = None
    for name in sorted(modules):
        module = modules[name]
        w = module.base_layer.weight
        if generator is None or generator.device != w.device:
            generator = torch.Generator(device=w.device).manual_seed(seed)
        a = module.lora_A["default"].weight.float()
        b = module.lora_B["default"].weight.float()
        delta = module.scaling["default"] * (b @ a)
        merged = grpo_loss.stochastic_bf16(w.float() + delta, generator)
        kept += float(((merged.float() - w.float()) * delta).sum())
        total += float((delta * delta).sum())
        out[keys[name]] = merged.to(w.dtype).cpu()
    return out, dict(tensors=len(out), update_kept=round(kept / total, 4) if total else None,
                     update_norm=round(total ** 0.5, 6))


@torch.no_grad()
def probe_logprobs(model, device, ids: list[int], start: int) -> dict[str, list[float]]:
    """Raw logprobs (temperature 1, full vocabulary) of ids[start:] under the current LoRA and with it disabled:
    the loop compares their difference with vLLM's before and after a sync."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inputs = torch.tensor(ids, dtype=torch.long, device=device)[None]
    targets = inputs[0, start:]
    out = {}
    model.train()  # the patched layers' token chunks; dropout is 0
    for name, context in (("current", None), ("base", "disable")):
        manager = model.disable_adapter() if context and hasattr(model, "disable_adapter") else None
        if manager is not None:
            manager.__enter__()
        try:
            hidden = base.model(input_ids=inputs, mm_token_type_ids=torch.zeros_like(inputs),
                                use_cache=False).last_hidden_state[0, start - 1:-1]
            logp, _ = grpo_loss.processed_logprobs(base.lm_head, hidden, targets, 1.0, 0, 1.0)
            out[name] = [round(float(v), 6) for v in logp.cpu()]
        finally:
            if manager is not None:
                manager.__exit__(None, None, None)
    return out


def checkpoint_keys(model_dir: Path) -> list[str]:
    index = Path(model_dir) / "model.safetensors.index.json"
    if index.exists():
        return sorted(json.loads(index.read_text())["weight_map"])
    from safetensors import safe_open

    with safe_open(Path(model_dir) / "model.safetensors", "pt") as f:
        return sorted(f.keys())


def save_state(model, optimizer, out: Path, step: int) -> None:
    tmp = out.with_name(out.name + ".tmp")
    model.save_pretrained(tmp)
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps({"step": step}))
    if out.exists():
        import shutil

        shutil.rmtree(out)
    os.replace(tmp, out)


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="jobs/, ckpt/, sync/ live here")
    parser.add_argument("--rank", type=int, default=16, dest="lora_rank")
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--eps-low", type=float, default=0.2)
    parser.add_argument("--eps-high", type=float, default=0.28)
    parser.add_argument("--ratio-min", type=float, default=0.5)
    parser.add_argument("--ratio-max", type=float, default=2.0)
    parser.add_argument("--thinking-budget", type=int, default=10240)
    parser.add_argument("--max-pixels", type=int, default=589_824, help="the server's --mm-processor-kwargs max_pixels")
    parser.add_argument("--max-tokens", type=int, default=133_000, help="longer episodes are skipped")
    parser.add_argument("--arena-gb", type=float, required=True, help="this process's pinned host arena")
    parser.add_argument("--token-chunk", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--no-qsa-kernel", dest="qsa_kernel", action="store_false")
    parser.add_argument("--no-compile-qsa", dest="compile_qsa", action="store_false")
    parser.add_argument("--no-compile-layers", dest="compile_layers", action="store_false")
    parser.add_argument("--no-fast-lora", dest="fast_lora", action="store_false")
    parser.add_argument("--no-fast-gdn", dest="fast_gdn", action="store_false")
    # off by default: the prefix's keys and values depend on the adapted projections too, so stopping its gradient
    # (RFT's saving, ~5% here: the first reply starts 5-8K tokens in) would bias the policy gradient
    parser.add_argument("--prefix-stop-grad", action="store_true")
    args = parser.parse_args(argv)

    import torch.distributed as dist

    from inference.rft import host_offload, smoke_lora, train_rft

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    jobs, ckpt, sync = args.work / "jobs", args.work / "ckpt", args.work / "sync"
    for d in (jobs, ckpt, sync):
        d.mkdir(parents=True, exist_ok=True)
    log_path = args.work / f"train_rank{rank}.jsonl"
    started = time.time()

    def log(**row):
        row = dict(t=round(time.time() - started, 1), rank=rank, **row)
        print(json.dumps(row), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda:0"))
    load_args = SimpleNamespace(model=str(args.model), qsa_kernel=args.qsa_kernel, compile_qsa=args.compile_qsa,
                                rank=args.lora_rank, resume=False, out=ckpt, compile_layers=args.compile_layers,
                                token_chunk=args.token_chunk, fast_lora=args.fast_lora, fast_gdn=args.fast_gdn)
    model, processor, device, info = train_rft.load_model(load_args)
    model._rft_host_arena = host_offload.HostArena(pin=True, max_gb=args.arena_gb)
    log(event="loaded", world=world, args={k: str(v) for k, v in vars(args).items()}, **info,
        arena_reserve_s=round(model._rft_host_arena.reserve(), 1), cgroup=smoke_lora.cgroup_memory_gb())
    params = trainable(model)
    optimizer = optimizer_for(params, args.optimizer, 1e-5)
    step = 0
    latest = ckpt / "latest"
    if args.resume and (latest / "state.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(latest / "adapter_model.safetensors"))
        optimizer.load_state_dict(torch.load(latest / "optimizer.pt", map_location=device))
        step = json.loads((latest / "state.json").read_text())["step"]
        log(event="resumed", step=step)
    weight_keys = checkpoint_keys(args.model) if rank == 0 else []
    settings = SimpleNamespace(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                               eps_low=args.eps_low, eps_high=args.eps_high, ratio_min=args.ratio_min,
                               ratio_max=args.ratio_max)

    prepare = make_prepare(settings, args.thinking_budget, args.max_tokens, processor, args.max_pixels, model)

    while True:
        job_path = jobs / f"step_{step + 1:06d}.json"
        if (jobs / "STOP").exists():
            log(event="stop", step=step)
            break
        if rank == 0:  # probes first: the loop asks between a step and the next job
            for probe in sorted(jobs.glob("probe_*.json")):
                if probe.name.count(".") > 1 or probe.with_suffix(".done.json").exists():
                    continue
                request = json.loads(probe.read_text())
                result = probe_logprobs(model, device, request["ids"], int(request["start"]))
                write_json(probe.with_suffix(".done.json"), dict(step=step, **result))
                log(event="probe", name=probe.stem, step=step, tokens=len(result["current"]))
        if not job_path.exists():
            time.sleep(args.poll_seconds)
            continue
        job = json.loads(job_path.read_text())
        step = int(job["step"])
        t0 = time.time()
        try:
            rows, norms = run_job(model, device, job, rank, optimizer, prepare, args.clip, args.prefix_stop_grad)
        except Exception:  # noqa: BLE001 - recorded for the loop, then fatal
            write_json(jobs / f"step_{step:06d}.failed.json", dict(step=step, rank=rank, error=traceback.format_exc()))
            raise
        write_json(jobs / f"step_{step:06d}.rank{rank}.json", dict(step=step, samples=rows, grad_norms=norms,
                                                                  seconds=round(time.time() - t0, 1)))
        if rank == 0:
            t1 = time.time()
            save_state(model, optimizer, latest, step)
            model.save_pretrained(ckpt / f"step_{step:06d}")
            tensors, merge_stats = merged_weights(model, weight_keys, args.seed * 1_000_003 + step)
            from safetensors.torch import save_file

            target = sync / f"step_{step:06d}.safetensors"
            save_file(tensors, str(target) + ".tmp", metadata={"step": str(step)})
            os.replace(str(target) + ".tmp", target)
            del tensors
            write_json(jobs / f"step_{step:06d}.done.json",
                       dict(step=step, grad_norm=round(max(norms, default=0.0), 6), grad_norms=[round(n, 6) for n in norms],
                            updates=len(norms), lr=job["lr"], sync=str(target),
                            train_seconds=round(t1 - t0, 1), save_seconds=round(time.time() - t1, 1), **merge_stats))
            log(event="step", step=step, grad_norms=norms, samples=len(rows), **merge_stats)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
