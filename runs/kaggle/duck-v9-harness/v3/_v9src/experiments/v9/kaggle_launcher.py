"""One-cell Kaggle entry point for v9.

The competition notebook is a single cell:

    import sys; sys.path.insert(0, "/kaggle/input/<v9-dataset>/experiments")
    import v9.kaggle_launcher as k; await k.main()

Everything here is plumbing lifted from the pinned Duck notebook (cells 3, 5, 7, 9 and the
competition-arcade open in the v8b bundle's run cell); none of Duck's agent survives. The parts
that matter and why:

- The real submission is a *rerun* with internet off. `arc-agi` is installed from the
  competition's own wheelhouse; every other dependency must already be in the image or the
  attached datasets. v9 needs only the standard library, numpy and requests-free urllib.
- In a rerun the notebook writes NO submission file: the gateway at http://gateway:8001/ records
  play and scores it server-side. `submission.parquet` is written only on the interactive Save &
  Run path, because Kaggle demands an output artifact there.
- Baselines are hidden in competition mode, so there is no per-level action budget; the number
  of levels comes from the first frame's `win_levels`.
- The benchmark must yield and must survive cancellation: on the soft deadline every game is
  stopped, its scorecard closed, and the exception re-raised.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

WORKING = Path("/kaggle/working")
COMPETITION = "arc-prize-2026-arc-agi-3"
GATEWAY = "http://gateway:8001/"
NOTEBOOK_BUDGET_S = 32400.0
SOFT_RESERVE_S = 600.0
PER_GAME_S = 7920.0
CONCURRENCY = 28


def log(message: str) -> None:
    print(f"[v9 {time.strftime('%H:%M:%S')}] {message}", flush=True)


# --- cell 3: environment ------------------------------------------------------------------------


def set_environment() -> bool:
    """Flags the runtime reads before anything else. Returns True in a real submission rerun."""
    true_submission = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if true_submission else "0"
    os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1"
    # A RESET must reset the level, not the whole game. Duck sets this too; arc_agi reads it at
    # reset time, and the value only becomes safe to set after the arcade's own first reset.
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", GATEWAY)
    WORKING.mkdir(parents=True, exist_ok=True)
    log(f"environment set; true_submission={true_submission}")
    return true_submission


def describe_mounts(root: Path = Path("/kaggle/input"), depth: int = 2) -> list[str]:
    """What is actually mounted, two levels deep. Logged before anything needs it.

    A run that dies on `ModuleNotFoundError: No module named \'arc_agi\'` ten minutes into a GPU
    session should not leave you guessing where the competition went. It costs one directory walk.
    """
    if not root.is_dir():
        return []
    out: list[str] = []
    base = str(root).count("/")
    for dirpath, dirnames, _files in os.walk(root):
        if dirpath.count("/") - base >= depth:
            dirnames[:] = []
        out.append(dirpath)
        if len(out) > 60:
            break
    return sorted(out)


def find_wheelhouse() -> Optional[Path]:
    """The competition's `arc_agi_3_wheels`, wherever it is mounted."""
    fixed = Path(f"/kaggle/input/competitions/{COMPETITION}/arc_agi_3_wheels")
    if fixed.is_dir():
        return fixed
    root = Path("/kaggle/input")
    if root.is_dir():
        for path in sorted(root.rglob("arc_agi_3_wheels")):
            if path.is_dir():
                return path
    return None


def install_arc_runtime() -> None:
    """Install `arc-agi` from the competition wheelhouse. No internet in a rerun, so --no-index."""
    wheels = find_wheelhouse()
    if wheels is None:
        # Loudly, with the tree: this is the failure that cost a whole GPU session. The
        # competition data was not mounted where the pinned notebook reads it, the launcher
        # shrugged and carried on, and ten minutes later `import arc_agi` ended the run.
        log(f"NO WHEELHOUSE FOUND. arc_agi will not be importable unless the image ships it. "
            f"mounts: {describe_mounts()}")
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheels), "arc-agi"],
                   check=True)
    log("arc-agi installed from the competition wheelhouse")


def find_bundle(marker: str = "taaf-kaggle-bundle.json") -> Optional[Path]:
    """Locate an attached source bundle by its marker file and put its sources on sys.path."""
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    for path in sorted(root.rglob(marker)):
        bundle = path.parent
        src = bundle / "src"
        if src.is_dir():
            for repo in sorted(src.iterdir()):
                for candidate in (repo / "src", repo):
                    if candidate.is_dir() and str(candidate) not in sys.path:
                        sys.path.insert(0, str(candidate))
        log(f"bundle at {bundle}")
        return bundle
    return None


def run_setup_commands(bundle: Optional[Path]) -> None:
    """Run the bundle's own setup commands (this is what starts the model server)."""
    if bundle is None:
        log("no bundle: skipping setup commands (a server must already be serving)")
        return
    spec = bundle / "setup_commands.json"
    if not spec.is_file():
        log("bundle has no setup_commands.json")
        return
    env_file = WORKING / "v9_setup_env.json"
    env = os.environ.copy()
    env.update({"PYTHON": sys.executable, "TAAF_KAGGLE_BUNDLE_DIR": str(bundle),
                "TAAF_KAGGLE_WORKING_DIR": str(WORKING), "TAAF_KAGGLE_SETUP_ENV": str(env_file)})
    for command in json.loads(spec.read_text()):
        log(f"setup: {command}")
        subprocess.run(command, shell=True, check=True, env=env)
        if env_file.is_file():  # each command may export variables for the next
            for key, value in json.loads(env_file.read_text()).items():
                os.environ[str(key)] = str(value)
                env[str(key)] = str(value)
    log("setup commands complete")


# --- the competition arcade ------------------------------------------------------------------


def wait_for_gateway(base_url: str = GATEWAY, timeout_s: float = 600.0) -> None:
    import urllib.error
    import urllib.request
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/games", timeout=10) as response:
                if response.status < 500:
                    log(f"gateway ready ({response.status})")
                    return
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                log(f"gateway ready ({exc.code})")
                return
            last = repr(exc)
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"gateway did not become ready in {timeout_s:.0f}s: {last}")


def competition_arcade():
    """The submission arcade: one scorecard for the whole session, baselines hidden."""
    import arc_agi
    base_url = os.environ.get("ARC_BASE_URL", GATEWAY)
    wait_for_gateway(base_url)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION,
                            arc_base_url=base_url, environments_dir="")
    game_ids = [env.game_id for env in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("competition arcade exposed no games")
    log(f"competition arcade: {len(game_ids)} games")
    return arcade, game_ids


def find_env_dir(root: Optional[Path] = None) -> Optional[Path]:
    """The competition's `environment_files`, wherever this image happens to mount it.

    Two fixed paths were tried before, and neither is guaranteed: the public image mounts a
    competition at /kaggle/input/<slug>/ and the private BYOD image this notebook pins mounts it
    at /kaggle/input/competitions/<slug>/, while `DEFAULT_ENV_DIR` is a repo-relative path that
    resolves, on Kaggle, to a directory inside the unpacked source that does not exist. A wrong
    guess is not a small error here: it is discovered after a full wait in the queue, which is
    the same way the source-bundle search learned to glob instead of guess.
    """
    root = Path("/kaggle/input") if root is None else Path(root)
    for fixed in (root / "competitions" / COMPETITION / "environment_files",
                  root / COMPETITION / "environment_files"):
        if fixed.is_dir():
            return fixed
    if root.is_dir():
        for path in sorted(root.rglob("environment_files")):
            if path.is_dir():
                return path
    try:
        from v9.engine import DEFAULT_ENV_DIR
        if Path(DEFAULT_ENV_DIR).is_dir():
            return Path(DEFAULT_ENV_DIR)
    except Exception:  # noqa: BLE001
        pass
    return None


def offline_arcade():
    """Interactive Save & Run: the 25 public games from the competition's environment files."""
    from v9.engine import open_arcade
    env_dir = find_env_dir()
    if env_dir is None:
        seen = sorted(str(p) for p in Path("/kaggle/input").glob("*")) if Path("/kaggle/input").is_dir() else []
        raise RuntimeError(f"no environment_files under /kaggle/input; saw {seen[:20]}")
    arcade = open_arcade(env_dir)
    game_ids = [env.game_id for env in arcade.available_environments]
    log(f"offline arcade: {len(game_ids)} games from {env_dir}")
    return arcade, game_ids


# --- play ---------------------------------------------------------------------------------------


def play_game(arcade, game_id: str, scorecard_id: Optional[str], deadline: float, out_dir: Path,
              base_url: str, model: str) -> dict[str, Any]:
    """One game, start to finish, in its own thread. Never raises: a crashed game must not take
    the session down, because the other 109 are still worth playing."""
    from v9.agent import V9Agent
    from v9.client import ModelClient
    from v9.engine import make_game
    from v9.results import game_run_dict, finished_line
    from v9.results import utc_now
    from v9.session import GameSession
    from v9.vendor.tycho.harness import run_env

    started = utc_now()
    tag = f"{game_id}_p0"
    try:
        handle = make_game(arcade, game_id, scorecard_id=scorecard_id)
        client = ModelClient(base_url=base_url, model=model,
                             api_key=os.environ.get("V9_API_KEY", "EMPTY"), thinking=False,
                             timeout=float(os.environ.get("V9_REQUEST_TIMEOUT", "900")))
        config = {"cap_minutes": PER_GAME_S / 60.0, "no_repl": True, "thinking": False,
                  "game_store": str(out_dir / "v9_game_memory.json")}
        session = GameSession(model=client, config=config, out_dir=out_dir, tag=tag)
        agent = V9Agent(session)
        game_deadline = min(deadline, time.monotonic() + PER_GAME_S)
        record = run_env(agent, handle.env, game_id, handle.baselines, n_levels=handle.n_levels,
                         first_frame=handle.first_frame, keep_trace=False, deadline=game_deadline)
        tokens = int(getattr(session, "generated_tokens", 0) or 0)
        session.close()
        run = game_run_dict(record, started_at=started, history=[], generated_tokens=tokens)
        log(finished_line(run))
        return run
    except Exception as exc:  # noqa: BLE001 - one game must never end the session
        log(f"game {game_id} failed: {type(exc).__name__}: {exc}")
        return {"game_id": game_id, "number_of_levels": 0, "base_actions_per_level": None, "state": "crashed",
                "history": [], "actions_per_level": [], "levels_completed": 0, "final_score": 0.0,
                "solver_note": f"error: {type(exc).__name__}: {exc}", "started_at": started.replace(tzinfo=None).isoformat(),
                "final_generated_tokens": 0, "final_uncached_input_tokens": 0, "final_wallclock_seconds": 0.0,
                "v9": {"start_level": 1, "stop_reason": "exception"}}


async def main(*, model: Optional[str] = None, base_url: Optional[str] = None,
               concurrency: int = CONCURRENCY, notebook_started: Optional[float] = None) -> dict[str, Any]:
    """Set up, play every game concurrently under the notebook's soft deadline, write results."""
    from v9.results import utc_now, write_benchmark

    t0 = notebook_started if notebook_started is not None else time.monotonic()
    true_submission = set_environment()
    log(f"mounts: {describe_mounts()}")
    install_arc_runtime()
    # Before the model server, not after it. Starting the server costs ten minutes of the GPU
    # session, and a session that has no games to play should not spend them: the first attempt
    # brought vLLM up, reported the model was serving, and only then died on `import arc_agi`
    # because the competition was not mounted where it was looked for.
    if not true_submission:
        try:
            import arc_agi  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"arc_agi is not importable ({exc}) and no wheelhouse was found. A Save & Run has "
                f"no internet, so there is nothing to play. mounts: {describe_mounts()}") from exc
        if find_env_dir() is None:
            raise RuntimeError(
                "no environment_files anywhere under /kaggle/input: the competition's game data is "
                "not mounted in this session, so the offline 25-game run cannot start. This is "
                f"normal for an interactive session on a code competition. mounts: {describe_mounts()}")
    bundle = find_bundle()
    try:
        run_setup_commands(bundle)
    except Exception as exc:  # noqa: BLE001
        # A Save & Run All exists to prove the notebook is well formed and to write the artifact
        # Kaggle requires before a version can be submitted; the scoring rerun happens later, on
        # competition hardware, with the gateway live. A free session gets a Tesla P100, whose
        # sm_60 is below the floor of the pinned torch, so the model server cannot start and this
        # raised "nvidia-smi not found" -- which failed the version and blocked submission
        # entirely. On a real rerun a missing model server is still fatal: there is nothing to
        # play with.
        if true_submission:
            raise
        log(f"serving setup failed on this machine ({exc}); this is a Save & Run, so writing the "
            f"placeholder and stopping rather than failing the version")
        write_placeholder_submission()
        return {"setup_failed": repr(exc)[:200], "true_submission": False, "games": 0}

    model = model or os.environ.get("V9_MODEL") or os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
    base_url = base_url or os.environ.get("V9_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    if not model:
        raise RuntimeError("set V9_MODEL (or LOCAL_ANALYZER_MODEL_ID) to the served model id")
    log(f"model={model} base_url={base_url}")

    arcade, game_ids = competition_arcade() if true_submission else offline_arcade()
    scorecard_id = arcade.create_scorecard()  # competition mode allows exactly one
    out_dir = WORKING / "v9"
    out_dir.mkdir(parents=True, exist_ok=True)

    deadline = t0 + NOTEBOOK_BUDGET_S - SOFT_RESERVE_S
    log(f"{len(game_ids)} games, {concurrency} at a time, soft deadline in "
        f"{(deadline - time.monotonic()) / 60:.0f} min")

    start = utc_now()
    runs: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="v9-game")

    async def one(game_id: str) -> None:
        async with semaphore:
            if time.monotonic() >= deadline:
                return
            run = await loop.run_in_executor(pool, play_game, arcade, game_id, scorecard_id, deadline,
                                             out_dir, base_url, model)
            runs.append(run)

    tasks = [asyncio.create_task(one(g)) for g in game_ids]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        # The notebook is being stopped: let the in-flight games see their own deadline, close the
        # scorecard, then re-raise so the runtime records a cancellation rather than a success.
        log("cancelled: draining games")
        deadline = time.monotonic()
        await asyncio.wait(tasks, timeout=120)
        raise
    finally:
        pool.shutdown(wait=False)
        try:
            arcade.close_scorecard(scorecard_id)
        except Exception as exc:  # noqa: BLE001
            log(f"close_scorecard failed: {exc}")

    end = utc_now()
    write_benchmark(out_dir, label="v9-kaggle", runs=runs, start_time=start, end_time=end, n_passes=1,
                    extra={"model": model, "base_url": base_url, "true_submission": true_submission})
    levels = sum(int(r.get("levels_completed") or 0) for r in runs)
    log(f"finished {len(runs)} games, {levels} levels total -> {out_dir / 'benchmark.json'}")
    if not true_submission:
        write_placeholder_submission()
    return {"games": len(runs), "levels": levels, "out_dir": str(out_dir)}


def write_placeholder_submission() -> None:
    """Save & Run needs an output artifact; a real rerun is scored by the gateway and needs none."""
    try:
        import pandas as pd
        pd.DataFrame([{"row_id": "1_0", "game_id": "1", "end_of_game": True, "score": 1}]).to_parquet(
            WORKING / "submission.parquet", index=False)
        log("wrote the Save & Run placeholder submission.parquet")
    except Exception as exc:  # noqa: BLE001
        log(f"placeholder submission not written: {exc}")
