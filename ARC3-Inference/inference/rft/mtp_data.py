"""MTP draft-head dataset: every request of every game, split by game.

The MTP head learns to predict what the target model generates, so there is no
quality filter: every reply the agent produced is a training target, from wins and
losses alike. Samples are built exactly as rft_data builds them (requests between
two trims share their prefix and become one sequence; loss on the non-`uncertain`
replies; images drawn again from the run's frames), but over all levels of a game.

Runs:
- only runs served by the target the head drafts for (`--target`, Intel W4A16 by
  default), from the run's kernel metadata or rendered setup command;
- only runs rft_data.run_problem accepts. Game priors are allowed here: the head
  imitates the target, whatever its prompt said.
Split: the `--eval-games` are held out entirely; every other game trains.

    python -m inference.rft.mtp_data runs --out data/mtp_v1 [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from inference.rft import rft_data
from inference.rft.rft_data import ALL_LEVELS, Attempt, build_run

INTEL_TARGET = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround"
# One game from each fifth of the 25 public games sorted by mean score over the usable
# runs (2026-10-30): 92.5, 79.8, 46.0, 25.6, 5.8; together 20% of the transcript text.
EVAL_GAMES = ("ar25", "cd82", "cn04", "g50t", "sk48")
ALLOWED_PROBLEMS = ("ARC3_GAME_PRIORS",)


def run_target(run: Path) -> str | None:
    """The Kaggle model ref of the served target, without framework/variation/version."""
    meta = run / "kernel" / "kernel-metadata.json"
    if meta.is_file():
        sources = json.loads(meta.read_text()).get("model_sources") or []
        if sources:
            return "/".join(sources[0].split("/")[:2])
    setup = run / "setup_commands.json"
    if setup.is_file():
        m = re.search(r"MODEL_SOURCE_PARTS = \('([^']+)', '([^']+)'", setup.read_text())
        if m:
            return f"{m.group(1)}/{m.group(2)}"
    return None


def run_problem(run: Path, target: str) -> str | None:
    """Why a run cannot give MTP samples, or None."""
    served = run_target(run)
    if served != target:
        return f"target {served}"
    problem = rft_data.run_problem(run)
    if problem and any(allowed in problem for allowed in ALLOWED_PROBLEMS):
        return None
    return problem


def game_jobs(root: Path, target: str, eval_games: tuple[str, ...]) -> tuple[dict[str, dict[str, list[Attempt]]], dict[str, str]]:
    """Per split, per run: one whole-game Attempt per transcript; and each run's problem."""
    jobs: dict[str, dict[str, list[Attempt]]] = {"train": defaultdict(list), "eval": defaultdict(list)}
    problems: dict[str, str] = {}
    for transcripts in sorted(root.glob("kaggle/*/*/v*/transcripts")):
        run = transcripts.parent
        problem = run_problem(run, target)
        problems[str(run)] = problem or "eligible"
        if problem:
            continue
        for path in sorted(transcripts.glob("*.txt")):
            if not (run / "artifacts" / f"{path.stem}_events.jsonl").exists():
                continue
            game = path.stem.split("-")[0]
            split = "eval" if game in eval_games else "train"
            jobs[split][str(run)].append(Attempt(str(run), path.stem, game, ALL_LEVELS, False, 0, 0, 0))
    return jobs, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="the runs folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", default=INTEL_TARGET)
    parser.add_argument("--eval-games", default=",".join(EVAL_GAMES))
    parser.add_argument("--max-tokens", type=float, default=124000, help="estimated sequence cap")
    parser.add_argument("--runs", help="comma-separated run path fragments to keep (for a partial build)")
    parser.add_argument("--workers", type=int, default=4, help="each holds one run's frames (~1-2 GB)")
    args = parser.parse_args(argv)

    eval_games = tuple(g for g in args.eval_games.split(",") if g)
    jobs, problems = game_jobs(args.root, args.target, eval_games)
    if args.runs:
        keep = args.runs.split(",")
        jobs = {split: {run: a for run, a in by_run.items() if any(k in run for k in keep)} for split, by_run in jobs.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images").mkdir(exist_ok=True)
    drops: Counter = Counter()
    counts: dict[str, Counter] = {}
    from multiprocessing import Pool
    for split, by_run in jobs.items():
        work = [(attempts, args.out / "images", args.max_tokens) for _, attempts in sorted(by_run.items())]
        counts[split] = Counter()
        with Pool(args.workers) as pool, open(args.out / f"{split}.jsonl", "w") as f:
            for samples, run_drops in pool.imap(build_run, work):
                drops.update(run_drops)
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    counts[split]["samples"] += 1
                    counts[split]["est_tokens"] += sample["est_tokens"]
                    counts[split][f"game:{sample['game']}"] += 1
                if samples:
                    counts[split][f"run:{samples[0]['run']}"] += len(samples)
        print(split, dict(samples=counts[split]["samples"], est_tokens=counts[split]["est_tokens"]), flush=True)
    manifest = dict(args={k: str(v) for k, v in vars(args).items()}, eval_games=eval_games, runs=problems,
                    counts={s: dict(c) for s, c in counts.items()}, drops=dict(drops))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("drops", dict(drops))
    return 0


if __name__ == "__main__":
    sys.exit(main())
