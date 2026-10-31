"""Training samples from GRPO episodes (rl_record's files): the ids vLLM saw, which reply tokens to train, and the
logprobs vLLM sampled them with.

- The sequence is the last request's prompt ids plus its reply: what the model actually conditioned on, never
  re-rendered or re-tokenized.
- A reply's tokens are trained where the sequence still holds them, with the same prefix before them: reply k
  occupies [prompt_len_k, prompt_len_k + len(reply_k)) of every later prompt only while each later prompt starts
  with its predecessor + reply up to there, i.e. while every later request's recorded common prefix (`lcp`) reaches
  that far. A reply the harness rebuilt differently (whitespace, a forced `</think>`) is trained up to the first
  difference.
- Not trained either: a `</think>` the sampler forced (the reply's thinking reached the thinking budget: the token
  sits at index >= budget), and tokens vLLM gave no logprob.
- Per level, for the reward: thinking tokens (reply tokens before `</think>`, or the whole reply when it has none)
  of the requests sent while the game was on that level, and TAAF's action counts.

    python -m inference.grpo.episodes RUN_DIR [--thinking-budget 10240]   # prints a summary per episode
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

THINK_END = 248069  # </think>
VISION_START = 248053  # <|vision_start|>
IMAGE_PAD = 248056  # <|image_pad|>


@dataclass
class Episode:
    stem: str  # e.g. ls20-9607627b_p3
    game: str  # 4-letter game, e.g. ls20
    info: dict[str, Any]  # the "episode" line
    requests: list[dict[str, Any]]
    ids: list[int] = field(default_factory=list)  # the whole sequence
    targets: list[int] = field(default_factory=list)  # positions in ids whose token is trained
    behavior: list[float] = field(default_factory=list)  # vLLM's processed logprob of each target
    images: list[Path] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def levels_completed(self) -> int:
        return int(self.info.get("levels_completed") or 0)


def thinking_tokens(ids: list[int]) -> int:
    """Reply tokens before the first </think>; all of them when the reply never closed its thinking."""
    try:
        return ids.index(THINK_END)
    except ValueError:
        return len(ids)


def trained_lengths(requests: list[dict[str, Any]]) -> list[int]:
    """How many leading tokens of each reply are still in place in the last sequence (see the module docstring)."""
    n = len(requests)
    out = [0] * n
    reach = None  # min lcp over later requests
    for k in range(n - 1, -1, -1):
        r = requests[k]
        length = len(r["ids"])
        if k == n - 1:
            out[k] = length
        else:
            out[k] = max(0, min(length, reach - r["prompt_len"]))
        reach = r["lcp"] if reach is None else min(reach, r["lcp"])
    return out


def load(path: Path, thinking_budget: int = 10240) -> Episode:
    """One episode from <run>/rl/<stem>.jsonl. Episode.error says why it cannot be trained, if it cannot."""
    path = Path(path)
    rows = [json.loads(line) for line in path.open() if line.strip()]
    requests = [r for r in rows if r.get("kind") == "request"]
    infos = [r for r in rows if r.get("kind") == "episode"]
    stem = path.stem
    ep = Episode(stem=stem, game=stem[:4], info=infos[-1] if infos else {}, requests=requests)
    if not infos:
        ep.error = "no episode line (the game did not finish)"
        return ep
    if not requests:
        ep.error = "no requests"
        return ep
    bad = [r["i"] for r in requests if r.get("error")]
    if bad:
        ep.error = f"requests without ids or logprobs: {bad[:5]}"
        return ep
    last = requests[-1]
    final_prompt = ep.info.get("final_prompt_ids") or []
    if len(final_prompt) != last["prompt_len"]:
        ep.error = f"final prompt has {len(final_prompt)} ids, the last request {last['prompt_len']}"
        return ep
    ep.ids = list(final_prompt) + list(last["ids"])
    lengths = trained_lengths(requests)
    forced = cut = missing = 0
    per_level_thinking: dict[int, int] = {}
    for r, keep in zip(requests, lengths):
        ids, logprobs = r["ids"], r["logprobs"]
        think = thinking_tokens(ids)
        per_level_thinking[r["level"]] = per_level_thinking.get(r["level"], 0) + think
        cut += len(ids) - keep
        for j in range(keep):
            if ids[j] == THINK_END and j >= thinking_budget:
                forced += 1
                continue
            if logprobs[j] is None:
                missing += 1
                continue
            ep.targets.append(r["prompt_len"] + j)
            ep.behavior.append(float(logprobs[j]))
    image_dir = path.with_suffix("")
    ep.images = sorted(image_dir.glob("*.png")) if image_dir.exists() else []
    runs = sum(1 for t in ep.ids if t == VISION_START)
    if runs != len(ep.images):
        ep.error = f"{runs} images in the ids, {len(ep.images)} image files"
    reply_tokens = sum(len(r["ids"]) for r in requests)
    ep.stats = dict(tokens=len(ep.ids), reply_tokens=reply_tokens, trained=len(ep.targets),
                    untrained_cut=cut, untrained_forced=forced, untrained_missing=missing,
                    replies_whole=sum(1 for r, k in zip(requests, lengths) if k == len(r["ids"])),
                    replies=len(requests), thinking_per_level=per_level_thinking, images=len(ep.images))
    return ep


def load_run(run_dir: Path, thinking_budget: int = 10240) -> list[Episode]:
    """Every episode under <run_dir>/rl (or run_dir itself when it holds the .jsonl files)."""
    root = Path(run_dir)
    rl = root / "rl" if (root / "rl").is_dir() else root
    return [load(p, thinking_budget) for p in sorted(rl.glob("*.jsonl"))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--thinking-budget", type=int, default=10240)
    args = parser.parse_args(argv)
    for ep in load_run(args.run_dir, args.thinking_budget):
        print(json.dumps(dict(stem=ep.stem, error=ep.error, levels=ep.levels_completed,
                              stop=ep.info.get("stop_reason"), **ep.stats)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
