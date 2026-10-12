"""Compressed play trace: the boards and action batches the history window dropped.

Sits right after the system prompt as one user message and is rebuilt only when
the window's first retained message changes, so between trims the prompt stays
append-only. Design and template: docs/harness/compressed-trace-design.md.

Indexing: `history[0]` is the starting board (step 0) and every executed action,
the automatic RESET included, appends one entry, so step n is `history[n]`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from inference.agent.runtime_state import Frame, HistoryEntry

# Private message keys; _strip_control_keys removes both before a request is sent.
TRACE_MESSAGE_KEY = "_arc3_trace"
STEP_KEY = "_arc3_step"
MAX_RUNS_PER_LINE = 12
_NO_OP_STOP = "no_op_action"
_AUTOMATIC_RESET = "automatic_reset"


def compressed_trace_batches() -> int:
    """ARC3_COMPRESSED_TRACE_BATCHES: batches kept in the trace; 0 (default) is off."""
    raw = os.environ.get("ARC3_COMPRESSED_TRACE_BATCHES", "").strip()
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        return 0


@dataclass(frozen=True)
class Batch:
    """One `action([...])` call's executed steps, inclusive."""

    start: int
    end: int
    stop_reason: str = ""
    skipped: int = 0


def run_length(actions: list[str]) -> list[str]:
    """Merge consecutive equal actions: ['LEFT', 'LEFT', 'UP'] -> ['LEFT×2', 'UP']."""
    runs: list[list[Any]] = []
    for action in actions:
        if runs and runs[-1][0] == action:
            runs[-1][1] += 1
        else:
            runs.append([action, 1])
    return [f"{action}×{count}" if count > 1 else action for action, count in runs]


def _result(entry: HistoryEntry) -> dict[str, Any]:
    return entry.result if isinstance(entry.result, dict) else {}


def _is_automatic_reset(entry: HistoryEntry) -> bool:
    return bool(_result(entry).get("automatic")) and str(entry.action).upper() == "RESET"


def _ends_level_or_life(entry: HistoryEntry) -> bool:
    result = _result(entry)
    return bool(result.get("level_completed") or result.get("game_over") or result.get("run_complete"))


def segment_batches(entries: list[HistoryEntry], recorded: list[Batch], last_step: int) -> list[Batch]:
    """Steps 1..last_step as batches, oldest first.

    Recorded batches are used as they are. Steps no recorded batch covers - the
    automatic RESET, which the solver runs between turns, or play from before
    recording started - become batches of their own, split where a level ends or
    a life is lost, as the solver splits a real batch.
    """
    last_step = min(last_step, len(entries) - 1)
    by_start = {b.start: b for b in recorded if 1 <= b.start <= b.end <= last_step}
    batches: list[Batch] = []
    n = 1
    while n <= last_step:
        batch = by_start.get(n)
        if batch is not None:
            batches.append(batch)
            n = batch.end + 1
            continue
        if _is_automatic_reset(entries[n]):
            batches.append(Batch(n, n, _AUTOMATIC_RESET))
            n += 1
            continue
        end = n
        while (
            end + 1 <= last_step
            and end + 1 not in by_start
            and not _is_automatic_reset(entries[end + 1])
            and not _ends_level_or_life(entries[end])
        ):
            end += 1
        batches.append(Batch(n, end))
        n = end + 1
    return batches


def _actions_text(names: list[str]) -> str:
    runs = run_length(names)
    if len(runs) <= MAX_RUNS_PER_LINE:
        return ", ".join(runs)
    shown = runs[:MAX_RUNS_PER_LINE]
    hidden = len(names) - sum(_run_count(run) for run in shown)
    return ", ".join(shown) + f", … (+{hidden} actions)"


def _run_count(run: str) -> int:
    head, sep, count = run.rpartition("×")
    return int(count) if sep and count.isdigit() else 1


def render_trace(
    entries: list[HistoryEntry],
    batches: list[Batch],
    solved_boards: dict[int, Any],
    image_part: Callable[[Frame], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Message content for the trace: text and image parts, board → actions → board.

    The last batch ends where the retained conversation starts, so its result is
    the first board of that conversation and is not repeated here.
    """
    parts: list[dict[str, Any]] = []

    def text(line: str) -> None:
        if parts and parts[-1].get("type") == "text":
            parts[-1]["text"] += "\n" + line
        else:
            parts.append({"type": "text", "text": line})

    def board(label: str, frame: Frame) -> bool:
        image = image_part(frame)
        if image is None:
            return False
        text(label)
        parts.append(image)
        return True

    shows_solved = any(
        _result(entries[b.end]).get("level_completed") and solved_boards.get(b.end) is not None
        for b in batches[:-1]
    )
    text(
        "Earlier play, compressed: the boards and the actions between them, oldest first. "
        "Your reasoning and tool output are not shown. Step n is `history[n]`; "
        "`history[0]` is the starting board."
        + (" A solved-level board is the only board here that is not in `history`." if shows_solved else "")
    )
    text("")
    first = batches[0].start - 1
    before = entries[first].frame
    start_label = "Starting board, step 0" if first == 0 else f"Board after step {first}"
    if not board(f"Level {before.level}. {start_label}:", before):
        text(f"Level {before.level}.")
    level_starts: dict[int, Frame] = {}
    for index, batch in enumerate(batches):
        level = entries[batch.start - 1].frame.level
        last = entries[batch.end]
        result = _result(last)
        steps = f"step {batch.start}" if batch.start == batch.end else f"steps {batch.start}–{batch.end}"
        if batch.stop_reason == _AUTOMATIC_RESET:
            text(f"→ {steps}: RESET (automatic), level {level} restarts")
        else:
            names = [str(entries[n].action) for n in range(batch.start, batch.end + 1)]
            line = f"→ {steps}: {_actions_text(names)}"
            if result.get("run_complete") or result.get("state") == "WIN":
                line += " · GAME COMPLETE"
            elif result.get("level_completed"):
                line += f" · LEVEL {level} CLEARED"
            elif result.get("game_over"):
                line += " · GAME OVER"
            if batch.stop_reason == _NO_OP_STOP:
                line += (
                    f" (stopped: no change, +{batch.skipped} skipped)" if batch.skipped
                    else " (stopped: no change)"
                )
            text(line)
        if index == len(batches) - 1:
            text(f"The board after step {batch.end} is the first board in the conversation below.")
            break
        after = last.frame
        if result.get("level_completed"):
            grid = solved_boards.get(batch.end)
            if grid is None or not board(
                f"Board as level {level} was solved (shown during step {batch.end}, not in history):",
                Frame(grid=grid, step=batch.end, level=level),
            ):
                text(f"Board as level {level} was solved: unavailable.")
            label = f"Level {after.level}. Board after step {batch.end}, the level {after.level} start:"
            if not board(label, after):
                text(f"Level {after.level} starts after step {batch.end}.")
            level_starts[after.level] = after
        elif result.get("game_over"):
            board(f"Board after step {batch.end}, the game-over board:", after)
        elif batch.stop_reason == _AUTOMATIC_RESET and level in level_starts and level_starts[level].grid == after.grid:
            text(f"Board after step {batch.end}: same as after step {level_starts[level].step}.")
        else:
            board(f"Board after step {batch.end}:", after)
    return parts


def build_trace_message(
    entries: list[HistoryEntry],
    recorded: list[Batch],
    solved_boards: dict[int, Any],
    last_step: int,
    max_batches: int,
    image_part: Callable[[Frame], dict[str, Any] | None],
) -> dict[str, Any] | None:
    """The trace message for play up to `last_step`, or None when there is none.

    Returns None as well when `history` does not satisfy the indexing the trace
    quotes to the model (`history[n].frame.step == n`), rather than quote steps
    that point at the wrong board.
    """
    if max_batches <= 0 or last_step <= 0 or len(entries) <= last_step:
        return None
    if any(entries[n].frame is None or entries[n].frame.step != n for n in range(last_step + 1)):
        return None
    batches = segment_batches(entries, recorded, last_step)[-max_batches:]
    if not batches:
        return None
    content = render_trace(entries, batches, solved_boards, image_part)
    return {"role": "user", "content": content, TRACE_MESSAGE_KEY: last_step}
