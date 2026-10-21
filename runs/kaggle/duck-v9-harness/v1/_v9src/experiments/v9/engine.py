"""Engine adapter: open the official arc_agi Arcade offline, make a game, jump to a level.

This is the only module that talks to `arc_agi`/`arcengine` directly for game setup. The play
loop itself is the vendored Tycho `run_env` (vendor/tycho/harness.py). Nothing here depends on
the Duck harness.

Facts this code rests on (taaf/game_api.py:215-222, experiments/level_start/level_start.py):
- `ONLY_RESET_LEVELS=true` must be set AFTER `arcade.make` so the make-time RESET still
  full-resets; afterwards a RESET is a level reset.
- A per-level test jumps the LOCAL engine with `env._game.set_level(n-1)`, sets the engine's
  `_score` to n-1 so the frame reports `levels_completed = n-1`, and resets once through the
  wrapper so the clean level renders. This is impossible on a remote (competition) arcade.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import arc_agi

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_DIR = REPO_ROOT / "data" / "environment_files"

# The 25 public games, notebook order (configs/duck-kaggle-frozen.json).
PUBLIC_GAME_IDS = (
    "tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb", "bp35-0a0ad940", "wa30-ee6fef47",
    "lp85-305b61c3", "r11l-495a7899", "tu93-0768757b", "sp80-589a99af", "m0r0-492f87ba",
    "vc33-5430563c", "ar25-0c556536", "ka59-38d34dbb", "sc25-635fd71a", "sk48-d8078629",
    "dc22-fdcac232", "cd82-fb555c5d", "ft09-0d8bbf25", "g50t-5849a774", "ls20-9607627b",
    "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44", "su15-1944f8ab", "tr87-cd924810",
)


def resolve_game_id(name: str, env_dir: Path = DEFAULT_ENV_DIR) -> str:
    """'re86' or 're86-8af5384d' -> 're86-8af5384d', using the env files on disk when present."""
    name = name.strip().lower()
    for gid in PUBLIC_GAME_IDS:
        if gid == name or gid.split("-")[0] == name:
            return gid
    prefix = name.split("-")[0]
    root = Path(env_dir) / prefix
    if root.is_dir():
        versions = sorted(p.name for p in root.iterdir() if p.is_dir())
        if versions:
            return f"{prefix}-{versions[-1]}"
    raise SystemExit(f"unknown game {name!r}")


def game_metadata(game_id: str, env_dir: Path = DEFAULT_ENV_DIR) -> dict[str, Any]:
    prefix, version = game_id.split("-", 1)
    path = Path(env_dir) / prefix / version / "metadata.json"
    if not path.is_file():
        raise SystemExit(f"missing environment files for {game_id}: {path}")
    return json.loads(path.read_text())


def open_arcade(env_dir: Path = DEFAULT_ENV_DIR) -> arc_agi.Arcade:
    """Offline Arcade over the bundled environment files: no network, no API key."""
    import logging
    logger = logging.getLogger("v9.arcade")
    logger.setLevel(logging.WARNING)
    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    return arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(env_dir),
        logger=logger,
    )


def jump_to_level(env: Any, level: int) -> Any:
    """Move a LOCAL environment to 1-indexed `level` and return the fresh frame.

    Re-implemented from experiments/level_start/level_start.py:621-641 (our own code) so v9 does
    not import an arm module by path.
    """
    game = getattr(env, "_game", None)
    if game is None:
        raise RuntimeError("level jump: the environment has no local game object (remote arcade?)")
    n_levels = len(getattr(game, "_levels"))
    if not 1 <= level <= n_levels:
        raise RuntimeError(f"level jump: level {level} out of range 1..{n_levels}")
    os.environ["ONLY_RESET_LEVELS"] = "true"
    game.set_level(level - 1)
    game._score = level - 1
    frame = env.reset()
    if frame is None:
        raise RuntimeError("level jump: env.reset() returned None after set_level")
    got = getattr(frame, "levels_completed", None)
    if got != level - 1:
        raise RuntimeError(f"level jump: frame reports levels_completed={got}, expected {level - 1}")
    return frame


@dataclass
class GameHandle:
    game_id: str
    env: Any
    first_frame: Any
    n_levels: int
    baselines: Optional[list[int]]
    start_level: int
    scorecard_id: Optional[str] = None


def make_game(arcade: arc_agi.Arcade, game_id: str, *, start_level: int = 1,
              scorecard_id: Optional[str] = None) -> GameHandle:
    """Make the environment, apply the level-reset policy, optionally jump to a level."""
    if scorecard_id is None:
        scorecard_id = arcade.create_scorecard()
    env = arcade.make(game_id, scorecard_id=scorecard_id)
    if env is None:
        raise RuntimeError(f"arcade.make({game_id!r}) returned None")
    # After make: a RESET is now a level reset, never a full-game reset.
    os.environ["ONLY_RESET_LEVELS"] = "true"
    info = env.environment_info
    initial = env.observation_space
    if initial is None:
        raise RuntimeError(f"env {game_id!r} has no observation after make()")
    n_levels = int(initial.win_levels)
    baselines = list(info.baseline_actions) if getattr(info, "baseline_actions", None) else None
    frame = initial
    if start_level > 1:
        frame = jump_to_level(env, start_level)
    else:
        # A fresh reset gives the canonical first frame (make already reset once; this is free).
        frame = env.reset() or initial
    return GameHandle(game_id=game_id, env=env, first_frame=frame, n_levels=n_levels,
                      baselines=baselines, start_level=int(start_level), scorecard_id=scorecard_id)


def settled_grid(frame: Any) -> list[list[int]]:
    """The playable grid after an action: the LAST layer of the frame list."""
    fl = frame.frame
    grid = fl[-1] if fl else []
    return [[int(v) for v in row] for row in grid]


def all_layers(frame: Any) -> list[list[list[int]]]:
    """Every layer (animation frames then the settled frame). The raw frame is always kept."""
    return [[[int(v) for v in row] for row in grid] for grid in (frame.frame or [])]
