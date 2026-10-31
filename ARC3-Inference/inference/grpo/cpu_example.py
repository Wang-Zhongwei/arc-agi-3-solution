"""GRPO on a laptop: the pipeline's own code on a tiny random qwen4_exp, every intermediate state printed, and a
place to stop at each stage (docs/RL/grpo-pipeline.md, "CPU example").

What runs is what the pod runs, scaled down; only the server and the game are stand-ins:
- "server": a plain copy of the base model plays the part of vLLM. It samples like vLLM's sampler (temperature, top-k,
  top-p; `sample_like_vllm`) and returns the processed logprob of each sampled token. When a reply's thinking reaches
  the budget it forces `</think>` (logprob 0), as vLLM's thinking budget does, and the next prompt re-renders that
  reply with an extra token before `</think>`, as the chat template does.
- "game": each turn is one action; the action is the reply's last token, and a level is cleared when it is a multiple
  of CLEAR_EVERY. Human action counts are HUMAN_ACTIONS.
- Everything else is the pipeline: rl_record.EpisodeRecorder writes the records, the loop's build_job turns a wave
  into the trainers' job (episodes, rewards, advantages, the max-variance subset, 4 minibatches), the trainer's
  run_job trains it (per minibatch: each rollout through train_rft.run_sample, then one optimizer step),
  train_grpo.merged_weights merges with stochastic BF16 rounding, and the merged tensors are loaded into the server
  (vLLM's grpo_load_weights does the same with model.load_weights), followed by the loop's sync probe.

The schedule is the loop's: wave k is played by the server's weights theta_{k-1}; step k trains on wave k-1 in 4
minibatches, one optimizer step each; then theta_k is synced. Step 1's first minibatch is on-policy (wave 0 was played
by theta_0, the trainer is still at theta_0): its log ratios are ~0. Each later minibatch is one update further away,
and from step 2 on every minibatch trains a wave played a whole step of updates earlier.

    PYTHONPATH=.:../tufa-arc-agi-framework/src python -m inference.grpo.cpu_example [--steps 2] [--pdb]

Stop at a stage: run with --pdb (pdb at every `stage(...)` call, the stage's variables in `state`), or set breakpoints
in your editor on the `stage(...)` lines. The records and samples stay in --work (default: a temporary directory).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from inference.grpo import episodes, rl_record, train_grpo
from inference.grpo.loop import (GATE_MAX_LOG_RATIO, GATE_MAX_OUTSIDE, GATE_MIN_CHANGE_CORR, GATE_MIN_WHOLE_REPLIES,
                                 build_job, compare_probe, sample_stats)

THINK_END = episodes.THINK_END  # </think> in the real vocabulary (the tiny model keeps its 248,320 tokens)
REPLY_MARKER = 248045  # <|im_start|>: starts each turn's observation
NEWLINE = 198  # "\n": what the chat template inserts before a forced </think> on re-render
HUMAN_ACTIONS = [2, 3]  # L1, L2
CLEAR_EVERY = 5  # an action token divisible by 5 clears the level (~20% per turn for a random model)
TINY_TEXT = dict(
    num_hidden_layers=4, hidden_size=64, hc_lowrank=16, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
    moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
    ple_layer_ids=[2], ngram_vocab_size_base=1000, ple_embed_dim=64, split_ngram_parts=4,
    indexer_head_dim=16, indexer_n_heads=8, indexer_kv_heads=1, indexer_budget=16, indexer_compress_ratio=4,
    mtp_num_hidden_layers=0, eos_token_id=248044,
    rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25,
                         mrope_section=[2, 1, 1], mrope_interleaved=True),
)
TINY_VISION = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)


def stage(name: str, args, state: dict[str, Any] | None = None, **summary) -> None:
    """Print a stage's summary; with --pdb, stop here: `summary` holds the printed numbers, `state` the objects
    (episodes, inputs, results, merged tensors) to inspect."""
    print(f"\n=== {name} " + "=" * max(0, 70 - len(name)))
    for key, value in summary.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            print(f"  {key}: {value}")
        elif isinstance(value, dict) and len(json.dumps(value, default=str)) < 600:
            print(f"  {key}: {json.dumps(value, default=str)}")
    if args.pdb:
        breakpoint()  # noqa: T100 - the point of --pdb: inspect `summary` and `state`, then `c`


def tiny_base(seed: int = 0):
    """The tiny model, weights snapped to the BF16 grid like the real checkpoint's attention and GDN weights."""
    from transformers import AutoModelForImageTextToText
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    torch.manual_seed(seed)
    config = Qwen4ExpConfig(text_config=TINY_TEXT, vision_config=TINY_VISION)
    config._attn_implementation = "sdpa"
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.bfloat16().float())
    return model


def sample_like_vllm(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                     generator: torch.Generator) -> tuple[int, float]:
    """vLLM's apply_top_k_top_p, then sampling: the token and its processed logprob."""
    x = logits.float() / temperature
    s, order = x.sort()
    s = s.masked_fill(s < s[-top_k], float("-inf"))
    drop = s.softmax(-1).cumsum(-1) <= 1 - top_p
    drop[-1] = False
    logp = torch.full_like(x, float("-inf")).scatter(0, order, s.masked_fill(drop, float("-inf"))).log_softmax(-1)
    token = int(torch.multinomial(logp.exp(), 1, generator=generator))
    return token, float(logp[token])


@torch.no_grad()
def reply(server, prompt: list[int], settings, think_budget: int, generator: torch.Generator) -> tuple[list[int], list[float]]:
    """One reply of sampled tokens; its last token is the action. A random model never samples </think>, so a
    reply's thinking length is drawn: one that would pass the budget gets the sampler's forced </think> at the
    budget (logprob 0, not the policy's), as vLLM does; the others have no </think> and count whole as thinking."""
    thinking = 2 + int(torch.randint(0, think_budget + 2, (1,), generator=generator))
    length = min(thinking, think_budget + 1) + 1 + int(torch.randint(0, 3, (1,), generator=generator))
    ids, logprobs = [], []
    while len(ids) < length:
        if thinking > think_budget and len(ids) == think_budget:
            token, logprob = THINK_END, 0.0
        else:
            x = torch.tensor([prompt + ids])
            logits = server(input_ids=x, mm_token_type_ids=torch.zeros_like(x), use_cache=False).logits[0, -1]
            token, logprob = sample_like_vllm(logits, settings.temperature, settings.top_k, settings.top_p, generator)
        ids.append(token)
        logprobs.append(logprob)
    return ids, logprobs


def play_wave(server, wave: int, args, settings, work: Path, generator: torch.Generator) -> Path:
    """One wave: every game's group of rollouts, recorded by the real EpisodeRecorder into <work>/wave_<k>/rl."""
    rl_dir = work / f"wave_{wave}" / "rl"
    system = torch.randint(0, 1000, (args.system_tokens,), generator=generator).tolist()
    for game in args.games:
        for rollout in range(args.rollouts):
            run = SimpleNamespace(game_id=f"{game}-0000cafe", levels_completed=0, number_of_levels=2,
                                  actions_per_level=[0, 0], base_actions_per_level=list(HUMAN_ACTIONS),
                                  state="playing", solver_note=None)
            recorder = rl_record.EpisodeRecorder(rl_dir / f"{game}-0000cafe_p{rollout}.jsonl", args.budget,
                                                 levels_completed=lambda r=run: r.levels_completed,
                                                 action_count=lambda r=run: sum(r.actions_per_level))
            context = list(system)  # what the harness sends: system + every observation and re-rendered reply
            for turn in range(args.max_turns):
                if recorder.should_stop():
                    break
                observation = [REPLY_MARKER] + torch.randint(0, 1000, (args.observation_tokens,),
                                                             generator=generator).tolist()
                prompt = context + observation
                cap = recorder.request_cap(len(prompt))
                if cap is None:
                    break
                ids, logprobs = reply(server, prompt, settings, args.think_budget, generator)
                recorder.after_response(dict(prompt_token_ids=prompt, token_ids=ids, logprobs=logprobs), messages=[],
                                        finish_reason="tool_calls", max_tokens=cap, usage={})
                # the game: the last token is the action; a multiple of CLEAR_EVERY clears the level
                level = run.levels_completed
                run.actions_per_level[min(level, 1)] += 1
                if ids[-1] % CLEAR_EVERY == 0:
                    run.levels_completed += 1
                # the next prompt re-renders this reply; a forced </think> gets the template's newline before it
                forced = len(ids) > args.think_budget and ids[args.think_budget] == THINK_END
                rerendered = ids[:args.think_budget] + [NEWLINE] + ids[args.think_budget:] if forced else ids
                context = prompt + rerendered
            run.state = "won" if run.levels_completed >= 2 else "gave_up"
            recorder.finish(run)
    return rl_dir


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=2, help="optimizer steps (waves 0..steps)")
    parser.add_argument("--games", nargs="+", default=["ls20", "sp80"])
    parser.add_argument("--rollouts", type=int, default=4, help="rollouts per game per wave (a group)")
    parser.add_argument("--keep", type=int, default=3, help="rollouts trained per group (6 of 8 on the pod)")
    parser.add_argument("--minibatches", type=int, default=4, help="optimizer steps per wave, as on the pod")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--budget", type=int, default=200, help="context tokens per episode")
    parser.add_argument("--system-tokens", type=int, default=40)
    parser.add_argument("--observation-tokens", type=int, default=12)
    parser.add_argument("--think-budget", type=int, default=6)
    parser.add_argument("--tau", type=float, default=6.0, help="the games' base thinking tokens per action")
    parser.add_argument("--lr", type=float, default=2e-3, help="tiny-model rate, to see changes in a few steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--pdb", action="store_true", help="stop in pdb at every stage")
    args = parser.parse_args(argv)
    import peft

    from inference.rft import layer_patches, qsa, smoke_lora

    work = args.work or Path(tempfile.mkdtemp(prefix="grpo_cpu_"))
    generator = torch.Generator().manual_seed(args.seed)
    # the recorder's margins at this scale (the pod's: 256 tokens of estimate error, >= 1,024 left for a request)
    margins = rl_record.REQUEST_MARGIN, rl_record.MIN_REQUEST_TOKENS
    rl_record.REQUEST_MARGIN, rl_record.MIN_REQUEST_TOKENS = 4, 24
    settings = SimpleNamespace(temperature=0.7, top_k=20, top_p=0.95, eps_low=0.2, eps_high=0.28, ratio_min=0.5,
                               ratio_max=2.0)

    # ---- setup: the server (a plain base model, vLLM's part) and the trainer (base + LoRA, train_rft's patches)
    base = tiny_base(args.seed)
    server = copy.deepcopy(base).train()  # train(): the patched layers' code path; dropout is 0
    trainer = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                                        target_modules=smoke_lora.ATTENTION_GDN))
    trainer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.train()
    qsa.patch_qsa_attention(chunk_size=16, fused=False)
    layer_patches.patch_for_training(trainer, token_chunk=16, recompute="parts")
    # the patches replace the layer classes' forward, so the server runs them too and needs its own settings;
    # they are exact rewrites (tests/test_layer_patches.py), so the server still computes the base model
    layer_patches.patch_for_training(server, token_chunk=16, recompute="parts")
    optimizer = train_grpo.optimizer_for(train_grpo.trainable(trainer), "adamw", args.lr)
    weight_keys = sorted(server.state_dict())
    stage("setup", args, work=str(work), lora_tensors=len(train_grpo.trainable(trainer)),
          lora_params=sum(p.numel() for p in train_grpo.trainable(trainer)), server_tensors=len(weight_keys),
          state=dict(settings=vars(settings)))

    history, waves, probe, probe_result = [], {}, None, None
    try:
        for wave in range(args.steps + 1):
            # ---- 1. a wave, played by the server's weights (theta_{wave-1}; the base for wave 0)
            rl_dir = play_wave(server, wave, args, settings, work, generator)
            eps = episodes.load_run(rl_dir.parent, args.think_budget)
            waves[wave] = eps
            first = eps[0]
            stage(f"wave {wave}: recorded and built", args, episodes=len(eps),
                  errors=sum(1 for e in eps if e.error),
                  replies=sum(e.stats["replies"] for e in eps),
                  replies_whole=sum(e.stats["replies_whole"] for e in eps),
                  untrained_cut=sum(e.stats["untrained_cut"] for e in eps),
                  untrained_forced=sum(e.stats["untrained_forced"] for e in eps),
                  example=dict(stem=first.stem, tokens=len(first.ids), trained=len(first.targets),
                               levels=first.levels_completed, stop=first.info.get("stop_reason"),
                               actions=first.info.get("actions_per_level"),
                               thinking_per_level=first.stats["thinking_per_level"]),
                  state=dict(records=str(rl_dir), first_request=first.requests[0], episode=first))
            if wave == 0:
                whole = sum(e.stats["replies_whole"] for e in eps) / max(1, sum(e.stats["replies"] for e in eps))
                print(f"  gate 1, replies trained whole: {whole:.2f} (pod threshold >= {GATE_MIN_WHOLE_REPLIES}; "
                      "a reply with a forced </think> comes back re-rendered with an extra newline, as the "
                      "template does, so it is trained only up to </think>)")
                probe_ids = eps[0].ids[:120]
                probe = dict(ids=probe_ids, start=40, v0=None)
                with torch.no_grad():  # the server's logprobs on the probe before any sync (the base weights)
                    x = torch.tensor([probe_ids])
                    logits = server(input_ids=x, mm_token_type_ids=torch.zeros_like(x), use_cache=False).logits[0]
                    probe["v0"] = logits[39:-1].float().log_softmax(-1).gather(1, x[0, 40:, None]).squeeze(1).tolist()
            if wave == 0:
                continue

            # ---- 2. the job of step `wave`, built from wave - 1 by the loop's own build_job: rewards, advantages,
            #         the kept rollouts and their minibatches
            step = wave
            job_args = SimpleNamespace(thinking_budget=args.think_budget, keep=args.keep, trainers=1, lr=args.lr,
                                       minibatches=args.minibatches)
            run_dir = work / f"wave_{wave - 1}"
            job, summary, _ = build_job(job_args, step, run_dir, {game: args.tau for game in args.games})
            rewards = json.loads((run_dir / "rewards.json").read_text())
            stage(f"step {step}: the job (from wave {wave - 1})", args,
                  rewards={stem: round(r["reward"], 4) for stem, r in rewards.items()},
                  minibatches={f"minibatch {m}": {x["id"]: x["advantage"] for x in batch["samples"]["0"]}
                               for m, batch in enumerate(job["minibatches"])},
                  trained_tokens=job["total_tokens"], state=dict(job=job, reward_parts=rewards, summary=summary))
            if not job["minibatches"]:
                print("  every group's rewards were equal: no gradient this step")
                continue

            # ---- 3. the step, by the trainer's own run_job: per minibatch, each rollout alone (gradients add up),
            #         then one optimizer step
            def on_sample(row, inputs, result, episode, step=step):
                stats = row.get("loss_stats") or {}
                stage(f"step {step}: minibatch {row['minibatch']}, sample {row['id']}", args,
                      advantage=row["advantage"], loss=row["loss"], tokens=stats.get("tokens"),
                      used=stats.get("used"), clipped=stats.get("clipped"),
                      log_ratio_abs_mean=stats.get("log_ratio_abs_mean"), error=row.get("error"),
                      state=dict(inputs=inputs, result=result, episode=episode))

            prepare = train_grpo.make_prepare(settings, args.think_budget, max_tokens=10**9)
            before = {n: p.detach().clone() for n, p in trainer.named_parameters() if p.requires_grad}
            rows, norms = train_grpo.run_job(trainer, "cpu", job, 0, optimizer, prepare, 1.0, False,
                                             on_sample=on_sample)
            moved = max(float((p.detach() - before[n]).abs().max()) for n, p in trainer.named_parameters()
                        if p.requires_grad)
            by_minibatch = [sample_stats([r for r in rows if r["minibatch"] == m]) for m in range(len(norms))]
            ratios = [b.get("log_ratio_abs_mean", 0.0) for b in by_minibatch]
            stage(f"step {step}: {len(norms)} optimizer steps", args,
                  grad_norms={f"minibatch {m}": round(n, 6) for m, n in enumerate(norms)},
                  log_ratio_abs_mean={f"minibatch {m}": r for m, r in enumerate(ratios)},
                  largest_lora_change=moved,
                  note=("minibatch 0 is on-policy (wave 0 was played by the same weights); each later one is "
                        "one update further away" if step == 1 else
                        "the wave was played one wave of updates earlier: every minibatch is off-policy"),
                  state=dict(rows=rows, by_minibatch=by_minibatch))
            if step == 1:
                print(f"  gate 3, mean |log ratio| of minibatch 0 at identical weights: {ratios[0]:.2e} (pod "
                      f"threshold <= {GATE_MAX_LOG_RATIO}); gate 4 shares are in `by_minibatch` (<= {GATE_MAX_OUTSIDE})")

            # ---- 4. sync: merged, stochastically rounded tensors into the server (vLLM: grpo_load_weights)
            tensors, merge = train_grpo.merged_weights(trainer, weight_keys, seed=step)
            if step == 1:
                probe_answer = train_grpo.probe_logprobs(trainer, "cpu", probe["ids"], probe["start"])
            missing, unexpected = server.load_state_dict({k: v.float() for k, v in tensors.items()}, strict=False)
            stage(f"step {step}: sync", args, tensors=len(tensors), unexpected=len(unexpected),
                  update_kept=merge["update_kept"], update_norm=merge["update_norm"],
                  state=dict(merged=tensors, merge=merge))
            if step == 1:
                with torch.no_grad():
                    x = torch.tensor([probe["ids"]])
                    logits = server(input_ids=x, mm_token_type_ids=torch.zeros_like(x), use_cache=False).logits[0]
                    v1 = logits[39:-1].float().log_softmax(-1).gather(1, x[0, 40:, None]).squeeze(1).tolist()
                probe_result = compare_probe(probe["v0"], v1, probe_answer["base"], probe_answer["current"])
                stage("probe after sync 1", args, **probe_result, gate=f">= {GATE_MIN_CHANGE_CORR}",
                      state=dict(probe=probe, trainer=probe_answer, v1=v1))
            history.append(dict(step=step, grad_norms=norms, log_ratio_by_minibatch=ratios,
                                reward_mean=sum(r["reward"] for r in rewards.values()) / max(1, len(rewards))))
    finally:
        layer_patches.unpatch(server)
        layer_patches.unpatch(trainer)
        qsa.unpatch_qsa_attention()
        rl_record.REQUEST_MARGIN, rl_record.MIN_REQUEST_TOKENS = margins
    stage("done", args, steps=len(history), work=str(work), state=dict(history=history))
    return dict(history=history, probe=probe_result, work=str(work))


if __name__ == "__main__":
    out = main()
    for row in out["history"]:
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v) for k, v in row.items()}))
