"""The GRPO loop: waves through the harness, rewards and advantages, jobs for the trainers, weight syncs
(grpo-pipeline.md).

One step overlaps a wave with training: wave k is played by the server's weights theta_{k-1} while the trainers turn
wave k-1 into theta_k; when both are done, theta_k is loaded into the server (vllm_sync) and wave k+1 starts. Wave 0
is the base model.

- A wave is `make interactive` with the GRPO harness config (8 games x 8 rollouts, ARC3_RL_RECORD=1), its run in
  <work>/waves/local/<time>_wave_<k>/.
- After a wave: episodes (episodes.load_run), rewards (reward.episode_reward with each game's tau), advantages and
  the max-variance subset per game group, balanced over the trainers; jobs/step_<k+1>.json.
- Every step's numbers go to <work>/loop_log.jsonl (and wandb with --wandb-project); state to loop_state.json, so a
  restarted loop continues: it re-syncs the server if its weight version is not the last step's.
- --fake-trainer answers the jobs itself (no GPU): with fake_vllm, the whole loop runs on a laptop.

    python -m inference.grpo.loop --work /workspace/grpo --config configs/grpo/v1.json \\
        --environments-dir /workspace/env/environment_files --base-url http://127.0.0.1:8000/v1 --steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from inference.grpo import episodes as episodes_mod
from inference.grpo import reward as reward_mod
from inference.grpo import vllm_sync

REPO = Path(__file__).resolve().parents[2]  # ARC3-Inference


def log_row(work: Path, row: dict[str, Any], wandb_run=None) -> None:
    row = dict(time=round(time.time(), 1), **row)
    print(json.dumps(row, default=str)[:2000], flush=True)
    with open(work / "loop_log.jsonl", "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    if wandb_run is not None:
        try:
            flat = {}
            event = row.get("event", "loop")
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    flat[f"{event}/{key}"] = value
            for game, values in (row.get("games") or {}).items():  # per-game wave numbers
                for key, value in values.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        flat[f"{event}/{game}/{key}"] = value
            if flat:
                wandb_run.log(flat)
        except Exception as exc:  # noqa: BLE001 - monitoring must never stop training
            print(f"wandb log failed: {exc}", flush=True)


def load_state(work: Path) -> dict[str, Any]:
    path = work / "loop_state.json"
    return json.loads(path.read_text()) if path.exists() else {"waves_done": 0, "steps_done": 0, "synced": None}


def save_state(work: Path, state: dict[str, Any]) -> None:
    tmp = work / "loop_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, work / "loop_state.json")


def wave_dir(work: Path, wave: int) -> Path | None:
    hits = sorted((work / "waves" / "local").glob(f"*_wave_{wave:06d}"))
    return hits[-1] if hits else None


def run_wave(args, wave: int, config: Path, log_file: Path) -> subprocess.Popen:
    """Start one wave of the harness (returns the process)."""
    command = ["make", "interactive", f"CONFIG_PATH={config}", f"ENVIRONMENTS_DIR={args.environments_dir}",
               f"EXPERIMENTS_DIR={args.work / 'waves'}", f"RUN_NAME=wave_{wave:06d}",
               f"LOCAL_ANALYZER_BASE_URL={args.base_url}"]
    out = open(log_file, "w")
    return subprocess.Popen(command, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)


def summarize_wave(eps: list[episodes_mod.Episode], rewards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_game = defaultdict(list)
    for ep in eps:
        by_game[ep.game].append(ep)
    games = {}
    for game, group in sorted(by_game.items()):
        ok = [e for e in group if not e.error]
        rs = [rewards[e.stem]["reward"] for e in ok if e.stem in rewards]
        games[game] = dict(
            episodes=len(group), errors=len(group) - len(ok),
            reward_mean=round(statistics.mean(rs), 4) if rs else None,
            reward_spread=round(max(rs) - min(rs), 4) if rs else None,
            l1=sum(e.levels_completed >= 1 for e in ok), l2=sum(e.levels_completed >= 2 for e in ok),
            tokens_mean=round(statistics.mean(len(e.ids) for e in ok)) if ok else None)
    trained = sum(e.stats.get("trained", 0) for e in eps if not e.error)
    replies = sum(e.stats.get("replies", 0) for e in eps if not e.error)
    whole = sum(e.stats.get("replies_whole", 0) for e in eps if not e.error)
    reply_tokens = sum(e.stats.get("reply_tokens", 0) for e in eps if not e.error)
    return dict(games=games, episodes=len(eps), errors=sum(1 for e in eps if e.error),
                reward_mean=round(statistics.mean(r["reward"] for r in rewards.values()), 4) if rewards else None,
                replies_whole_share=round(whole / replies, 4) if replies else None,
                trained_share=round(trained / reply_tokens, 4) if reply_tokens else None,
                untrained_cut=sum(e.stats.get("untrained_cut", 0) for e in eps if not e.error),
                untrained_forced=sum(e.stats.get("untrained_forced", 0) for e in eps if not e.error),
                error_samples=[(e.stem, e.error) for e in eps if e.error][:5])


def build_job(args, step: int, run: Path, tau: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any], list]:
    """The trainers' job for one step from one wave's run, the wave's summary and its episodes.

    The job holds `args.minibatches` minibatches (reward.split_minibatches), one optimizer step each; a minibatch's
    rollouts are balanced over the trainers by tokens, and its loss is the token mean over its own trained tokens."""
    eps = episodes_mod.load_run(run, args.thinking_budget)
    rewards = {}
    for ep in eps:
        if ep.error:
            continue
        info = ep.info
        rewards[ep.stem] = reward_mod.episode_reward(
            ep.levels_completed, info.get("actions_per_level") or [], info.get("base_actions_per_level") or [],
            {int(k): v for k, v in (ep.stats.get("thinking_per_level") or {}).items()}, tau.get(ep.game))
    good = [e for e in eps if not e.error and e.stem in rewards]
    groups = defaultdict(list)
    for e in good:
        groups[e.game].append(e)
    chosen = []
    for game, members in groups.items():
        values = [rewards[e.stem]["reward"] for e in members]
        picked = reward_mod.advantages({game: values}, args.keep).get(game, [])
        for index, advantage in picked:
            chosen.append((members[index], advantage))
    lengths = {e.stem: len(e.ids) for e, _ in chosen}
    by_stem = {e.stem: (e, a) for e, a in chosen}
    minibatches, rank_tokens = [], [0] * args.trainers
    for batch in reward_mod.split_minibatches([(e.game, e.stem, a) for e, a in chosen], args.minibatches):
        ranks = reward_mod.assign({s: lengths[s] for _, s, _ in batch}, args.trainers)
        samples = {}
        for r, stems in enumerate(ranks):
            samples[str(r)] = [dict(episode=str(run / "rl" / f"{s}.jsonl"), advantage=round(by_stem[s][1], 6), id=s,
                                    reward=round(rewards[s]["reward"], 6)) for s in stems]
            rank_tokens[r] += sum(lengths[s] for s in stems)
        minibatches.append(dict(total_tokens=sum(len(by_stem[s][0].targets) for _, s, _ in batch), samples=samples))
    total = sum(len(e.targets) for e, _ in chosen)
    job = dict(step=step, lr=args.lr, total_tokens=total, minibatches=minibatches, wave_run=str(run))
    summary = summarize_wave(eps, rewards)
    summary.update(trained_episodes=len(chosen), trained_tokens=total, minibatches=len(minibatches),
                   groups_with_signal=len({e.game for e, _ in chosen}), rank_tokens=rank_tokens)
    with open(run / "rewards.json", "w") as f:
        json.dump(rewards, f, indent=1)
    return job, summary, eps


def save_probe(work: Path, probe: dict[str, Any]) -> None:
    (work / "probe.json").write_text(json.dumps(probe))


def request_probe(work: Path, ids: list[int], start: int, timeout_s: float = 1800) -> dict[str, Any]:
    """Ask trainer rank 0 for the probe's logprobs under the current LoRA and the base, and wait for them."""
    path = work / "jobs" / "probe_000001.json"
    path.write_text(json.dumps(dict(ids=ids, start=start)))
    done = path.with_suffix(".done.json")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if done.exists():
            return json.loads(done.read_text())
        time.sleep(5)
    raise TimeoutError("the trainer did not answer the probe")


def fake_trainer(work: Path, stop: threading.Event) -> None:
    """Answers every job with a done file and an empty sync file (local tests of the loop)."""
    jobs, sync = work / "jobs", work / "sync"
    sync.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        for job_path in sorted(jobs.glob("step_*.json")):
            if job_path.name.count(".") > 1:
                continue
            step = int(job_path.stem.split("_")[1])
            done = jobs / f"step_{step:06d}.done.json"
            if done.exists():
                continue
            target = sync / f"step_{step:06d}.safetensors"
            target.write_bytes(b"")
            done.write_text(json.dumps(dict(step=step, grad_norm=0.0, sync=str(target), fake=True)))
        for probe in sorted(jobs.glob("probe_*.json")):
            if probe.name.count(".") == 1 and not probe.with_suffix(".done.json").exists():
                request = json.loads(probe.read_text())
                n = len(request["ids"]) - int(request["start"])
                probe.with_suffix(".done.json").write_text(json.dumps(dict(current=[0.0] * n, base=[0.0] * n)))
        stop.wait(1.0)


PROBE_TOKENS = 2048  # scored positions of the sync probe (text before the first image)


def probe_sequence(eps: list[episodes_mod.Episode]) -> tuple[list[int], int] | None:
    """A text-only token sequence from a wave's first good episode, and the first position to score."""
    for ep in eps:
        if ep.error:
            continue
        ids = ep.ids
        cut = ids.index(episodes_mod.VISION_START) if episodes_mod.VISION_START in ids else len(ids)
        ids = ids[:min(cut, 2 * PROBE_TOKENS)]
        if len(ids) >= PROBE_TOKENS // 2:
            return ids, max(1, len(ids) - PROBE_TOKENS)
    return None


def vllm_prompt_logprobs(base_url: str, ids: list[int], start: int, model: str = "flashnext") -> list[float]:
    """vLLM's raw logprob of ids[start:] given the prefix (temperature 1, no top-k/top-p)."""
    import urllib.request

    body = dict(model=model, prompt=ids, max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, prompt_logprobs=0)
    request = urllib.request.Request(base_url.rstrip("/") + "/completions", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        out = json.loads(response.read())
    rows = out["choices"][0]["prompt_logprobs"]
    return [float(rows[i][str(ids[i])]["logprob"]) for i in range(start, len(ids))]


def compare_probe(v0: list[float], v1: list[float], t_base: list[float], t_cur: list[float]) -> dict[str, Any]:
    """How vLLM's logprobs moved over a sync against how the trainer's moved over the same step."""
    n = min(len(v0), len(v1), len(t_base), len(t_cur))
    dv = [v1[i] - v0[i] for i in range(n)]
    dt = [t_cur[i] - t_base[i] for i in range(n)]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    mv, mt = mean(dv), mean(dt)
    cov = sum((a - mv) * (b - mt) for a, b in zip(dv, dt))
    var_v, var_t = sum((a - mv) ** 2 for a in dv), sum((b - mt) ** 2 for b in dt)
    corr = cov / (var_v * var_t) ** 0.5 if var_v > 0 and var_t > 0 else None
    return dict(tokens=n, gap_before=round(mean([abs(v0[i] - t_base[i]) for i in range(n)]), 5),
                gap_after=round(mean([abs(v1[i] - t_cur[i]) for i in range(n)]), 5),
                trainer_change=round(mean([abs(x) for x in dt]), 6), vllm_change=round(mean([abs(x) for x in dv]), 6),
                change_corr=round(corr, 4) if corr is not None else None)


# Gates on the first steps: abort before paying for a broken setup (grpo-pipeline.md, Checks)
GATE_MIN_WHOLE_REPLIES = 0.5  # share of replies trained whole after wave 0 (token-exact sequences)
GATE_MAX_EPISODE_ERRORS = 0.25
GATE_MAX_LOG_RATIO = 0.1  # mean |log pi_trainer - log pi_vLLM| on wave 0 at identical weights
GATE_MAX_OUTSIDE = 0.05  # shares of trained tokens outside the trainer's kept set / ratio band
GATE_MIN_CHANGE_CORR = 0.5  # vLLM's logprob change over sync 1 against the trainer's


def wait_done(work: Path, step: int, timeout_s: float) -> dict[str, Any]:
    done, failed = work / "jobs" / f"step_{step:06d}.done.json", work / "jobs" / f"step_{step:06d}.failed.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if done.exists():
            return json.loads(done.read_text())
        if failed.exists():
            raise RuntimeError(f"trainer failed step {step}: {failed.read_text()[:2000]}")
        time.sleep(5)
    raise TimeoutError(f"step {step} not done in {timeout_s / 60:.0f} min")


def sample_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Token-weighted statistics of trained samples (the trainers' rank-file rows)."""
    stats = [r["loss_stats"] for r in rows if r.get("loss_stats")]
    n = sum(s["tokens"] for s in stats)
    out = dict(samples=len(rows), failed=sum(1 for r in rows if r.get("error")), tokens=n)
    if n:
        for key in ("log_ratio_mean", "log_ratio_abs_mean", "kl_k3", "logp_mean"):
            out[key] = round(sum(s[key] * s["tokens"] for s in stats) / n, 6)
        for key in ("used", "outside_kept", "outside_ratio", "clipped"):
            out[f"{key}_share"] = round(sum(s[key] for s in stats) / n, 5)
        tps = [r["tokens_per_s"] for r in rows if r.get("tokens_per_s")]
        out["tokens_per_s_mean"] = round(statistics.mean(tps), 1) if tps else None
    return out


def rank_stats(work: Path, step: int) -> dict[str, Any]:
    """The step's sample statistics over all rank files, overall and per minibatch (the later minibatches train
    further from the weights that played the wave: their ratios and clipping show the drift)."""
    rows = []
    for path in sorted((work / "jobs").glob(f"step_{step:06d}.rank*.json")):
        rows += json.loads(path.read_text()).get("samples", [])
    out = sample_stats(rows)
    by_minibatch = defaultdict(list)
    for r in rows:
        by_minibatch[r.get("minibatch", 0)].append(r)
    out["per_minibatch"] = [sample_stats(by_minibatch[m]) for m in sorted(by_minibatch)]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="the GRPO harness config (ARC3_RL_RECORD=1)")
    parser.add_argument("--environments-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--steps", type=int, default=50, help="optimizer steps (waves 0..steps)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--keep", type=int, default=6, help="rollouts trained per group of 8")
    parser.add_argument("--minibatches", type=int, default=4, help="optimizer steps per wave (decided 2026-10-26)")
    parser.add_argument("--trainers", type=int, default=2)
    parser.add_argument("--thinking-budget", type=int, default=10240)
    parser.add_argument("--tau", type=Path, default=reward_mod.TAU_PATH)
    parser.add_argument("--step-timeout-min", type=float, default=180)
    parser.add_argument("--max-hours", type=float, default=0, help="stop after the step that passes this (0: none)")
    parser.add_argument("--fake-trainer", action="store_true", help="local tests: no trainers, no real sync")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args(argv)
    args.work = args.work.resolve()
    args.config = args.config.resolve()
    for d in ("jobs", "sync", "waves"):
        (args.work / d).mkdir(parents=True, exist_ok=True)
    tau = reward_mod.load_tau(args.tau)
    state = load_state(args.work)
    probe = json.loads((args.work / "probe.json").read_text()) if (args.work / "probe.json").exists() else {}
    started = time.time()
    wandb_run = None
    if args.wandb_project:
        try:
            from inference.rft import wandb_log

            id_file = args.work / "wandb_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else None
            if run_id is None:
                import wandb

                run_id = wandb.util.generate_id()
                id_file.write_text(run_id)
            wandb_run = wandb_log.start(args.wandb_project, args.wandb_run_name, run_id,
                                        {k: str(v) for k, v in vars(args).items()}, args.wandb_entity)
        except Exception as exc:  # noqa: BLE001
            print(f"wandb disabled: {exc}", flush=True)
    stop = threading.Event()
    if args.fake_trainer:
        threading.Thread(target=fake_trainer, args=(args.work, stop), daemon=True).start()
    log_row(args.work, dict(event="start", state=state, tau_games=sorted(tau), args={k: str(v) for k, v in
                                                                                       vars(args).items()}), wandb_run)

    def sync(step: int) -> None:
        version = f"step_{step:06d}"
        done = json.loads((args.work / "jobs" / f"step_{step:06d}.done.json").read_text())
        if args.fake_trainer:
            result = dict(fake=True)
        else:
            result = vllm_sync.sync_weights(args.base_url, done["sync"], version)
        state["synced"] = version
        save_state(args.work, state)
        row = dict(result)
        row.update(event="sync", step=step)
        log_row(args.work, row, wandb_run)

    # a restarted loop: put the last trained weights back into a server that does not hold them
    if state["steps_done"] and not args.fake_trainer:
        try:
            with __import__("urllib.request").request.urlopen(
                    vllm_sync.server_root(args.base_url) + "/weight_info", timeout=30) as response:
                current = json.loads(response.read()).get("weight_version")
        except Exception:  # noqa: BLE001
            current = None
        if current != f"step_{state['steps_done']:06d}":
            sync(state["steps_done"])

    try:
        while True:
            wave = state["waves_done"]
            if wave > args.steps:
                break
            if (args.work / "STOP").exists():
                log_row(args.work, dict(event="stop", reason="STOP file", wave=wave), wandb_run)
                break
            # wave `wave` is played by the weights of step wave - 1 (wave 0: the base model)
            t0 = time.time()
            proc = run_wave(args, wave, args.config, args.work / "waves" / f"wave_{wave:06d}.log")
            step = wave  # trained concurrently: the step made from wave - 1
            train_result = None
            try:
                if wave >= 1 and state["steps_done"] < step:
                    train_result = wait_done(args.work, step, args.step_timeout_min * 60)
                    if step == 1 and probe.get("ids") and "trainer" not in probe:
                        # before job 2 exists, so the trainer answers with the weights of step 1
                        probe["trainer"] = request_probe(args.work, probe["ids"], probe["start"])
                        save_probe(args.work, probe)
                code = proc.wait()
            except BaseException:
                if proc.poll() is None:  # never leave a wave playing against a server nobody syncs
                    proc.terminate()
                raise
            run = wave_dir(args.work, wave)
            if code != 0 or run is None:
                raise RuntimeError(f"wave {wave} failed (exit {code}); see waves/wave_{wave:06d}.log")
            wave_seconds = round(time.time() - t0, 1)
            if train_result is not None:
                state["steps_done"] = step
                save_state(args.work, state)
                row = {k: v for k, v in train_result.items() if k != "sync"}
                stats = rank_stats(args.work, step)
                row.update(stats)
                row.update(event="train", step=step)
                log_row(args.work, row, wandb_run)
                first = (stats.get("per_minibatch") or [{}])[0]
                if step == 1 and first.get("tokens"):
                    # wave 0's first minibatch is trained at the weights that played it: pure train/rollout mismatch
                    if first["log_ratio_abs_mean"] > GATE_MAX_LOG_RATIO or \
                            first["outside_kept_share"] > GATE_MAX_OUTSIDE or first["outside_ratio_share"] > GATE_MAX_OUTSIDE:
                        raise RuntimeError(f"gate: trainer and vLLM disagree on wave 0's tokens: {first}")
            job, summary, eps = build_job(args, wave + 1, run, tau)
            if wave < args.steps:  # the next step's job, from this wave
                job_path = args.work / "jobs" / f"step_{wave + 1:06d}.json"
                tmp = job_path.with_name(job_path.name + ".tmp")
                tmp.write_text(json.dumps(job))
                os.replace(tmp, job_path)
            state["waves_done"] = wave + 1
            save_state(args.work, state)
            log_row(args.work, dict(event="wave", wave=wave, seconds=wave_seconds, weights=state.get("synced"),
                                    **summary), wandb_run)
            if wave == 0:
                if (summary["replies_whole_share"] or 0) < GATE_MIN_WHOLE_REPLIES or \
                        summary["errors"] > GATE_MAX_EPISODE_ERRORS * max(1, summary["episodes"]):
                    raise RuntimeError(f"gate: wave 0's sequences are not token-exact or episodes failed: {summary}")
                picked = probe_sequence(eps)
                if picked is not None and not args.fake_trainer:  # the base weights, on an idle server
                    probe.update(ids=picked[0], start=picked[1],
                                 v0=vllm_prompt_logprobs(args.base_url, picked[0], picked[1]))
                    save_probe(args.work, probe)
            if train_result is not None:
                sync(step)
                if step == 1 and probe.get("trainer") and "result" not in probe:
                    v1 = vllm_prompt_logprobs(args.base_url, probe["ids"], probe["start"])
                    probe["result"] = compare_probe(probe["v0"], v1, probe["trainer"]["base"],
                                                    probe["trainer"]["current"])
                    save_probe(args.work, probe)
                    log_row(args.work, dict(event="probe", step=step, **probe["result"]), wandb_run)
                    r = probe["result"]
                    if r["trainer_change"] > 1e-4 and (r["change_corr"] is None or r["change_corr"] < GATE_MIN_CHANGE_CORR):
                        raise RuntimeError(f"gate: the sync did not carry the trainer's update: {r}")
                # keep the last two sync files: ~5.4 GB each
                for old in (args.work / "sync").glob("step_*.safetensors"):
                    if int(old.stem.split("_")[1]) < step - 1:
                        old.unlink(missing_ok=True)
            if args.max_hours and time.time() - started > args.max_hours * 3600:
                log_row(args.work, dict(event="stop", reason="time", wave=wave), wandb_run)
                break
    finally:
        stop.set()
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:  # noqa: BLE001
                pass
    log_row(args.work, dict(event="end", state=state, hours=round((time.time() - started) / 3600, 2)), None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
