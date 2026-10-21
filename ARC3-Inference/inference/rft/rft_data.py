"""RFT dataset: choose the level clears to train on and write their samples.

Selection, per (game, level), over every run that kept transcripts and events:
- an attempt is a level the agent reached and gave at least one analyzer turn (games a
  run never got to are not failures); pass rate = clears / attempts;
- levels with a pass rate of `max_pass_rate` (0.8) or more are dropped as too easy;
- a clear qualifies when its actions are within the level's best `top` share (0.35) of all
  clears and its generated tokens (thinking, reply text, tool-call code) are at most the
  level's median clear: few actions first, but not bought with a lot of thinking;
- if no clear of a level qualifies, its fewest-action eligible clear is kept when it is
  within the token median;
- only clears from runs that can be rebuilt faithfully are used (`run_problem`).

Samples: the requests the agent sent while on the level, grouped by window (requests
between two trims share their prefix), one sequence per group: the last request's
context and reply, with loss on the level's non-`uncertain` replies. Groups are cut
where a sequence would pass `max_tokens` (estimated) or a reply is missing from the
later window (rolled-back turns). Images are drawn again from the run's frames with
the run's settings (rft_images); a sample with an image that cannot be drawn is
dropped. Each sample keeps its run's own tool description.

    python -m inference.rft.rft_data runs --out data/rft_v1 [--max-pass-rate 0.8 --top 0.35]
"""

from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import io
import json
import math
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from inference.rft import rft_images
from inference.rft.context_rebuild import (TOKENS_FIXED, TOKENS_PER_CHAR, message_chars, rebuild, split_sections,
                                          split_steps)

TRIM_NOTE = "Earlier conversation has been trimmed"
EXCLUDED = {  # run path fragment -> reason
    "jonathanwang2022/priority-falloff/v03": "ARC3_GAME_PRIORS: per-game hints the model will not have",
    "kaggle/lordhansolo/": "another team's fork of the harness; its rebuild was never validated",
}
GENERATED = ("THINKING", "ASSISTANT", "TOOL CALL")
MIN_PIXELS = 65536  # preprocessor_config shortest_edge: smaller images are scaled up to this many pixels


@dataclass
class Attempt:
    run: str
    stem: str  # transcript name, e.g. ft09-0d8bbf25_p0
    game: str
    level: int
    cleared: bool
    actions: int
    turns: int
    generated_tokens: float  # estimated: thinking, reply text and tool-call code


def run_problem(run: Path) -> str | None:
    """Why a run cannot give training samples, or None."""
    for fragment, reason in EXCLUDED.items():
        if fragment in str(run):
            return reason
    agent = run / "src/ARC3-Inference/inference/agent"
    if not (agent / "vision_context.py").exists():
        return "no source snapshot"
    if not (run / "intermediate_states.pkl").exists():
        return "no saved frames"
    if not (run / "taaf_setup_env.json").exists():
        return "no harness settings"
    env = json.loads((run / "taaf_setup_env.json").read_text())
    if env.get("ARC3_VIEW_TOOL") == "1":
        return "view tool (messages the transcript does not log)"
    if any(TRIM_NOTE in p.read_text(errors="replace") for p in agent.glob("*.py")):
        return "trimmed-history note (not in the transcript)"
    return None


def level_after_actions(events: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int], set[int], Counter]:
    """(action numbers, level after each, level-completing actions, cleared levels, actions per level)."""
    numbers, levels, completing, cleared, actions = [0], [1], [], set(), Counter()
    level = 1
    for e in events:
        if e.get("type") != "action":
            continue
        actions[level] += 1
        if e.get("level_completed"):
            cleared.add(level)
            completing.append(e["action_num"])
            level = e.get("level") or level + 1
        elif e.get("state") == "WIN":
            cleared.add(level)
        numbers.append(e["action_num"])
        levels.append(level)
    return numbers, levels, completing, cleared, actions


def level_at(numbers: list[int], levels: list[int], header_action: int) -> int:
    """The level a turn played: the level after its first action - 1 actions."""
    return levels[bisect.bisect_right(numbers, header_action - 1) - 1]


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path)]


def game_attempts(run: Path, stem: str) -> list[Attempt]:
    events = read_events(run / "artifacts" / f"{stem}_events.jsonl")
    numbers, levels, _, cleared, actions = level_after_actions(events)
    turns, generated = Counter(), Counter()
    for _, action, body in split_steps((run / "transcripts" / f"{stem}.txt").read_text(encoding="utf-8", errors="replace")):
        level = level_at(numbers, levels, action)
        turns[level] += 1
        generated[level] += sum(len(text) for label, text in split_sections(body) if label.startswith(GENERATED))
    game = stem.split("-")[0]
    return [Attempt(str(run), stem, game, level, level in cleared, actions[level], turns[level],
                    round(generated[level] * TOKENS_PER_CHAR))
            for level in range(1, max(levels) + 1)]


def all_attempts(root: Path, workers: int = 8) -> list[Attempt]:
    jobs = [(t.parent, f.stem) for t in sorted(root.glob("kaggle/*/*/v*/transcripts")) for f in sorted(t.glob("*.txt"))
            if (t.parent / "artifacts" / f"{f.stem}_events.jsonl").exists()]
    from multiprocessing import Pool
    with Pool(workers) as pool:
        return [a for chunk in pool.starmap(game_attempts, jobs, chunksize=8) for a in chunk]


def lower_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def select(attempts: list[Attempt], eligible: set[str], max_pass_rate: float, top: float,
           token_quantile: float = 0.5) -> tuple[list[Attempt], list[dict[str, Any]]]:
    """The clears to train on, and one report row per level."""
    by_level: dict[tuple[str, int], list[Attempt]] = defaultdict(list)
    for a in attempts:
        if a.turns >= 1 or a.cleared:
            by_level[(a.game, a.level)].append(a)
    chosen, report = [], []
    for (game, level), tries in sorted(by_level.items()):
        clears = [a for a in tries if a.cleared]
        rate = len(clears) / len(tries)
        row = dict(game=game, level=level, attempts=len(tries), clears=len(clears), pass_rate=round(rate, 3))
        if not clears or rate >= max_pass_rate:
            report.append(dict(row, kept=0))
            continue
        max_actions = lower_quantile([a.actions for a in clears], top)
        max_tokens = lower_quantile([a.generated_tokens for a in clears], token_quantile)
        candidates = [a for a in clears if a.run in eligible]
        keep = [a for a in candidates if a.actions <= max_actions and a.generated_tokens <= max_tokens]
        if not keep and candidates:
            best = min(candidates, key=lambda a: (a.actions, a.generated_tokens))
            keep = [best] if best.generated_tokens <= max_tokens else []
        keep.sort(key=lambda a: (a.actions, a.generated_tokens))
        chosen += keep
        report.append(dict(row, eligible_clears=len(candidates), max_actions=max_actions, max_generated_tokens=max_tokens,
                           kept=len(keep), median_actions=lower_quantile([a.actions for a in clears], 0.5)))
    return chosen, report


def run_tools(run: Path) -> list[dict[str, Any]]:
    """The python tool as this run's harness declared it (its descriptions changed between versions)."""
    tree = ast.parse((run / "src/ARC3-Inference/inference/agent/tool_agent.py").read_text())
    description = code = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_PYTHON_TOOL_DESCRIPTION" for t in node.targets):
            description = ast.literal_eval(node.value)
        if isinstance(node, ast.Dict):  # {"code": {"type": "string", "description": ...}}
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "code" and isinstance(value, ast.Dict):
                    for k, v in zip(value.keys, value.values):
                        if isinstance(k, ast.Constant) and k.value == "description":
                            code = ast.literal_eval(v)
    if description is None or code is None:
        raise ValueError(f"no python tool declaration in {run}")
    return [{"type": "function", "function": {"name": "python", "description": description, "parameters": {
        "type": "object", "properties": {"code": {"type": "string", "description": code}}, "required": ["code"]}}}]


def image_tokens(image) -> float:
    w, h = image.size
    return max(w * h, MIN_PIXELS) / 1024 + 2


def estimate(messages: list[dict[str, Any]], tokens_per_image: list[float]) -> float:
    chars = sum(message_chars(m)[0] for m in messages)
    return TOKENS_PER_CHAR * chars + sum(tokens_per_image) + TOKENS_FIXED


class GameFrames:
    """intermediate_states.pkl of one run, by game id (one state list per game run, in pass order)."""

    def __init__(self, run: Path):
        states = pickle.load(open(run / "intermediate_states.pkl", "rb"))
        self.games: dict[str, list] = defaultdict(list)
        for game in states:
            if game:
                self.games[game[0].raw.game_id].append(game)

    def get(self, stem: str):
        game_id, pass_index = re.match(r"(.+)_p(\d+)$", stem).groups()
        runs = self.games.get(game_id) or []
        return runs[min(int(pass_index), len(runs) - 1)] if runs else None


def trajectory_samples(attempt: Attempt, frames_cache: dict, images_dir: Path, max_tokens: float,
                       drops: Counter) -> list[dict[str, Any]]:
    run = Path(attempt.run)
    env = json.loads((run / "taaf_setup_env.json").read_text())
    scales = rft_images.Scales.from_env(env)
    events = read_events(run / "artifacts" / f"{attempt.stem}_events.jsonl")
    numbers, levels, completing, _, _ = level_after_actions(events)
    if attempt.run not in frames_cache:
        frames_cache.clear()  # one run's frames at a time (~1-2 GB as tuples)
        frames_cache[attempt.run] = GameFrames(run)
    states = frames_cache[attempt.run].get(attempt.stem)
    if states is None:
        drops["no frames for game"] += 1
        return []
    frames = rft_images.Frames.from_run(states, events)
    game = rebuild((run / "transcripts" / f"{attempt.stem}.txt").read_text(encoding="utf-8"))
    tools = run_tools(run)
    requests = [r for r in game.requests if level_at(numbers, levels, r.action) == attempt.level]
    drawn: dict[int, list] = {}  # id(user message) -> its images

    def images_of(message):
        if id(message) not in drawn:
            drawn[id(message)] = rft_images.message_images(message, message["_action"] - 1, frames, scales, completing)
        return drawn[id(message)]

    def sequence(end):
        messages = end.messages + [end.reply]
        images = [img for m in messages if m.get("role") == "user" and isinstance(m.get("content"), list)
                  and "_action" in m for img in images_of(m)]
        return messages, images

    windows: dict[int, list] = defaultdict(list)
    for r in requests:
        windows[r.history_start].append(r)
    groups = []
    for reqs in windows.values():
        group: list = []
        for r in reqs:
            messages, images = sequence(r)
            if None in images:
                labels = [label for m in messages if m.get("role") == "user" and isinstance(m.get("content"), list)
                          and "_action" in m for label in rft_images.message_labels(m)]
                bad = {re.sub(r"\d+", "N", label)[:60] for label, image in zip(labels, images) if image is None}
                for kind in bad:
                    drops[f"image not drawable: {kind}"] += 1
                if group:
                    groups.append(group)
                group = []
                continue
            size = estimate(messages, [image_tokens(i) for i in images])
            if size > max_tokens:
                drops["request over max_tokens"] += 1
                if group:
                    groups.append(group)
                group = []
                continue
            if group and not all(any(m is q.reply for m in r.messages) for q in group):
                groups.append(group)
                group = []
            group.append(r)
        if group:
            groups.append(group)
    out = []
    for g, group in enumerate(groups):
        trained = [q for q in group if not q.uncertain]
        if not trained:
            drops["group without certain requests"] += 1
            continue
        end = group[-1]
        messages, images = sequence(end)
        it = iter(images)
        written, tokens = [], []
        for m in messages:
            m = {k: v for k, v in m.items() if not k.startswith("_")}
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                parts = []
                for part in m["content"]:
                    if part.get("type") == "image_url":
                        image = next(it)
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        data = buffer.getvalue()
                        name = f"images/{hashlib.sha1(data).hexdigest()}.png"
                        target = images_dir.parent / name
                        if not target.exists():  # several workers can write the same image: write, then rename
                            temporary = target.with_suffix(f".{os.getpid()}.tmp")
                            temporary.write_bytes(data)
                            os.replace(temporary, target)
                        tokens.append(image_tokens(image))
                        part = {"type": "image_url", "image_url": {"url": name}}
                    parts.append(part)
                m["content"] = parts
            written.append(m)
        train = [i for i, m in enumerate(messages) if any(m is q.reply for q in trained)]
        prefix = estimate(messages[:train[0]], tokens[:sum(1 for m in messages[:train[0]] if m.get("role") == "user"
                                                          and isinstance(m.get("content"), list)
                                                          for p in m["content"] if p.get("type") == "image_url")])
        out.append(dict(id=f"{attempt.run.split('runs/', 1)[-1]}|{attempt.stem}|L{attempt.level}|{g}",
                        run=attempt.run.split("runs/", 1)[-1], game=attempt.game, stem=attempt.stem, level=attempt.level,
                        actions=attempt.actions, generated_tokens=attempt.generated_tokens,
                        est_tokens=round(estimate(messages, tokens)), est_prefix=round(prefix),
                        train=train, tools=tools, messages=written))
    return out


def build_run(job) -> tuple[list[dict[str, Any]], Counter]:
    """The samples of one run's chosen attempts (a worker: one run's frames in memory at a time)."""
    attempts, images_dir, max_tokens = job
    drops: Counter = Counter()
    frames_cache: dict = {}
    samples = [s for a in attempts for s in trajectory_samples(a, frames_cache, images_dir, max_tokens, drops)]
    return samples, drops


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="the runs folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-pass-rate", type=float, default=0.8)
    parser.add_argument("--top", type=float, default=0.35, help="share of a level's clears, by fewest actions")
    parser.add_argument("--token-quantile", type=float, default=0.5,
                        help="a clear may use at most this quantile of the level's generated tokens")
    parser.add_argument("--max-tokens", type=float, default=124000, help="estimated sequence cap")
    parser.add_argument("--val-levels", type=int, default=12,
                        help="validation: for this many levels, the best eligible clear that was not selected")
    parser.add_argument("--attempts-cache", type=Path, help="JSON of all_attempts, reused when present")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    if args.attempts_cache and args.attempts_cache.exists():
        attempts = [Attempt(**a) for a in json.loads(args.attempts_cache.read_text())]
    else:
        attempts = all_attempts(args.root)
        if args.attempts_cache:
            args.attempts_cache.write_text(json.dumps([asdict(a) for a in attempts]))
    runs = sorted({a.run for a in attempts})
    problems = {run: run_problem(Path(run)) for run in runs}
    eligible = {run for run, problem in problems.items() if problem is None}
    chosen, report = select(attempts, eligible, args.max_pass_rate, args.top, args.token_quantile)

    # validation: per level, the next clear by actions that was not chosen (same quality band, held out)
    rng = random.Random(args.seed)
    chosen_keys = {(a.run, a.stem, a.level) for a in chosen}
    levels = sorted({(a.game, a.level) for a in chosen})
    rng.shuffle(levels)
    val = []
    for game, level in levels:
        rest = sorted((a for a in attempts if a.game == game and a.level == level and a.cleared and a.run in eligible
                       and (a.run, a.stem, a.level) not in chosen_keys), key=lambda a: (a.actions, a.generated_tokens))
        if rest:
            val.append(rest[0])
        if len(val) >= args.val_levels:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images").mkdir(exist_ok=True)
    drops: Counter = Counter()
    from multiprocessing import Pool
    for name, attempts_set in (("train", chosen), ("val", val)):
        by_run: dict[str, list[Attempt]] = defaultdict(list)
        for a in attempts_set:
            by_run[a.run].append(a)
        jobs = [(sorted(group, key=lambda a: (a.stem, a.level)), args.out / "images", args.max_tokens)
                for _, group in sorted(by_run.items())]
        with Pool(args.workers) as pool, open(args.out / f"{name}.jsonl", "w") as f:
            n = 0
            for samples, run_drops in pool.imap(build_run, jobs):
                drops.update(run_drops)
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    n += 1
        print(name, len(attempts_set), "trajectories", n, "samples", flush=True)
    manifest = dict(args={k: str(v) for k, v in vars(args).items()}, runs={r: p or "eligible" for r, p in problems.items()},
                    levels=report, train=[asdict(a) for a in chosen], val=[asdict(a) for a in val], drops=dict(drops))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("drops", dict(drops))
    return 0


if __name__ == "__main__":
    sys.exit(main())
