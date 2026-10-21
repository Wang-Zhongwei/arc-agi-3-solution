"""v9 runner: play games offline against the local engine and write run artifacts.

    cd experiments && python -m v9.run --games re86 --level 1 --passes 1 --cap-minutes 40 \
        --agent random --out ../runs/<name>

    python -m v9.run --games ls20 --level 3 --passes 2 --cap-minutes 70 \
        --base-url http://127.0.0.1:30000/v1 --model Qwen/Qwen3.8-27B-FP8 --out ../runs/<name>

Outputs, in --out: benchmark.json (taaf shape, read by scripts/summarize_run.py), stdout.log
(this process's console, also printed), run_manifest.json (engine, model id, sampling, flags,
git commit), v9_trace.jsonl (one row per engine step; the model layer adds one row per turn),
and per pass <game>_p<k>.record.json (the full EnvRunRecord). The two lines the Spark scripts
grep, `[finished] ...` and `LEVEL_START_RESULT ...`, go to stdout and stdout.log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .engine import DEFAULT_ENV_DIR, REPO_ROOT, game_metadata, make_game, open_arcade, resolve_game_id
from .results import finished_line, game_run_dict, level_start_line, utc_now, write_benchmark
from .vendor.tycho.harness import run_env


class _Tee:
    """Mirror stdout to stdout.log so the run directory is self-describing."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.fh = open(path, "a", encoding="utf-8")

    def write(self, text: str) -> None:
        with self.lock:
            sys.__stdout__.write(text)
            sys.__stdout__.flush()
            self.fh.write(text)
            self.fh.flush()

    def flush(self) -> None:
        pass


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_agent(args: argparse.Namespace, game_id: str, pass_index: int, out_dir: Path):
    if args.agent == "random":
        from .agent import RandomAgent
        return RandomAgent(seed=args.seed + pass_index)
    if args.agent == "fake":
        from .agent import V9Agent
        from .session import GameSession
        from .fakemodel import FakeModel
        session = GameSession(model=FakeModel(), config=args, out_dir=out_dir, tag=f"{game_id}_p{pass_index}")
        return V9Agent(session)
    from .agent import V9Agent
    from .client import ModelClient
    from .session import GameSession
    client = ModelClient(base_url=args.base_url, model=args.model, api_key=args.api_key,
                         thinking=args.thinking, timeout=args.request_timeout)
    if getattr(args, "seconds_per_call", 0):
        client.seconds_per_call = float(args.seconds_per_call)
    if getattr(args, "max_cost_usd", 0):
        # Per pass. The harness stops asking the model once the endpoint has billed this much,
        # and says so in done_reason rather than failing.
        client.cost_limit = float(args.max_cost_usd)
    session = GameSession(model=client, config=args, out_dir=out_dir, tag=f"{game_id}_p{pass_index}")
    return V9Agent(session)


def play_one(args: argparse.Namespace, game_id: str, pass_index: int, out_dir: Path, tee: _Tee) -> dict[str, Any]:
    """One (game, pass): its own Arcade and scorecard, its own trace file."""
    started_at = utc_now()
    arcade = open_arcade(Path(args.env_dir))
    handle = make_game(arcade, game_id, start_level=args.level)
    if args.level > 1:
        tee.write(f"LEVEL_START game={game_id} level={args.level} levels_completed={handle.first_frame.levels_completed}\n")
    deadline = time.monotonic() + args.cap_minutes * 60.0 if args.cap_minutes else None
    trace_path = out_dir / f"{game_id}_p{pass_index}.steps.jsonl"
    history: list[dict[str, Any]] = []
    t0 = time.monotonic()

    def on_step(info: dict[str, Any]) -> None:
        info.pop("frame", None)
        info["reasoning"] = None if info.get("reasoning") is None else {k: v for k, v in info["reasoning"].items() if k in ("src", "turn", "mode", "prediction_id")}
        with open(trace_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(info) + "\n")
        data = {}
        if info.get("x") is not None:
            data = {"x": info["x"], "y": info["y"]}
        history.append({"action": {"id": info["action"], "data": data},
                        "generated_tokens": 0, "uncached_input_tokens": 0,
                        "wallclock_seconds": round(time.monotonic() - t0, 3)})

    agent = build_agent(args, game_id, pass_index, out_dir)
    if getattr(agent, "session", None) is not None and int(handle.start_level or 1) > 1:
        # The session starts at level 1 by default, so with --level N the first frame reads as a
        # level change and the harness records a level-1 clear that never happened -- with zero
        # actions and an empty win signature -- straight into the level log the model reads.
        agent.session.level = int(handle.start_level)
    if not args.no_budget and getattr(agent, "session", None) is not None:
        # So a committed plan can size itself against what is actually left of the level.
        agent.session.action_budget = list(handle.baselines or [])
    rec = run_env(agent, handle.env, game_id, None if args.no_budget else handle.baselines, n_levels=handle.n_levels,
                  first_frame=handle.first_frame, start_level=handle.start_level,
                  keep_trace=False, deadline=deadline, max_actions=args.max_actions, on_step=on_step)
    tokens = 0
    session = getattr(agent, "session", None)
    if session is not None:
        tokens = int(getattr(session, "generated_tokens", 0) or 0)
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            tee.write(f"session close failed for {game_id}_p{pass_index}: {exc!r}\n")
    if history and tokens:
        history[-1]["generated_tokens"] = tokens  # summarize_run sums per-action tokens; keep the total
    run = game_run_dict(rec, started_at=started_at, history=history, generated_tokens=tokens)
    (out_dir / f"{game_id}_p{pass_index}.record.json").write_text(json.dumps(asdict(rec), indent=1, default=str))
    tee.write(finished_line(run) + "\n")
    tee.write(level_start_line(run) + "\n")
    try:
        arcade.close_scorecard(handle.scorecard_id)
    except Exception:  # noqa: BLE001
        pass
    return run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", required=True, help="comma-separated ids or 4-char prefixes")
    p.add_argument("--level", type=int, default=1, help="1-indexed start level (local engine only)")
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--cap-minutes", type=float, default=70.0, help="per (game, pass) wall-clock cap; 0 = none")
    p.add_argument("--max-actions", type=int, default=None)
    p.add_argument("--no-budget", action="store_true", help="disable the official 5x-baseline per-level action cutoff")
    p.add_argument("--agent", default="v9", choices=["v9", "random", "fake"])
    p.add_argument("--base-url", default=os.environ.get("V9_BASE_URL", "http://127.0.0.1:30000/v1"))
    p.add_argument("--model", default=os.environ.get("V9_MODEL", "Qwen/Qwen3.8-27B-FP8"))
    p.add_argument("--api-key", default=os.environ.get("V9_API_KEY", "EMPTY"))
    p.add_argument("--seconds-per-call", type=float,
                   default=float(os.environ.get("V9_SECONDS_PER_CALL", 0) or 0),
                   help="trim every role's token ceiling to what the endpoint can generate in this "
                        "many seconds (0 = no trimming). A token budget is a time budget in "
                        "disguise and the exchange rate differs by two orders of magnitude between "
                        "a local GPU and a hosted one.")
    p.add_argument("--max-cost-usd", type=float, default=float(os.environ.get("V9_MAX_COST_USD", 0) or 0),
                   help="stop this pass once the endpoint has billed this much (0 = no limit); "
                        "the charge is read from the response, not estimated")
    p.add_argument("--thinking", action="store_true", help="enable_thinking=true for the decide roles (A/B)")
    p.add_argument("--merged-turn", action="store_true", help="one call per routine turn instead of two")
    p.add_argument("--no-repl", action="store_true")
    p.add_argument("--no-opening-probe", action="store_true", help="do not spend the harness's own opening probe (each action once) before the first model turn")
    p.add_argument("--model-executes", action="store_true", help="in EXECUTE, ask the model to commit the harness plan instead of committing it directly")
    p.add_argument("--memory-from", default=None, help="seed theory from an earlier run's <game>_p<k>.memory.json (per-level tests)")
    p.add_argument("--request-timeout", type=float, default=float(os.environ.get("ANALYZER_TIMEOUT", "1800")))
    p.add_argument("--env-dir", default=str(DEFAULT_ENV_DIR))
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="v9")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"--out exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / "stdout.log")
    sys.stdout = tee  # type: ignore[assignment]
    import logging
    logging.basicConfig(level=logging.WARNING, stream=tee, format="%(levelname)s %(name)s: %(message)s")

    games = [resolve_game_id(g, Path(args.env_dir)) for g in args.games.split(",") if g.strip()]
    for gid in games:
        game_metadata(gid, Path(args.env_dir))  # preflight: env files present
    manifest = {
        "v9_version": __version__, "git_commit": _git_commit(), "argv": sys.argv, "args": vars(args),
        "games": games, "engine": "arc_agi offline", "model": args.model, "base_url": args.base_url,
        "thinking": args.thinking, "started_at": utc_now().isoformat(), "host": os.uname().nodename,
    }
    if args.agent == "v9":
        from .client import served_models
        try:
            manifest["served_models"] = served_models(args.base_url, args.api_key)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"model server not reachable at {args.base_url}: {exc}")
        if args.model not in manifest["served_models"]:
            raise SystemExit(f"served models {manifest['served_models']} do not include {args.model!r}")
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    tee.write(f"v9 {__version__} games={games} level={args.level} passes={args.passes} agent={args.agent} "
              f"model={args.model} base_url={args.base_url} thinking={args.thinking} out={out_dir}\n")

    start = utc_now()
    jobs = [(gid, k) for gid in games for k in range(args.passes)]
    runs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = [pool.submit(play_one, args, gid, k, out_dir, tee) for gid, k in jobs]
        for fut in futures:
            try:
                runs.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                tee.write(f"game failed: {type(exc).__name__}: {exc}\n")
                import traceback
                tee.write(traceback.format_exc())
    end = utc_now()
    write_benchmark(out_dir, label=args.label, runs=runs, start_time=start, end_time=end,
                    n_passes=args.passes, extra={"manifest": manifest})
    manifest["finished_at"] = end.isoformat()
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    tee.write(f"done: {len(runs)} run(s) in {(end - start).total_seconds():.0f} s -> {out_dir / 'benchmark.json'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
