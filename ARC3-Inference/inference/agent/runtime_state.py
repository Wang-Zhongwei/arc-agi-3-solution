"""Structured runtime state shared with created Python tools."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference.utils.grid_utils import format_grid_ascii


RUNTIME_STATE_FILENAME = "tool_runtime_state.json"


@dataclass(frozen=True)
class Frame:
    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int

    @property
    def shape(self) -> tuple[int, int]:
        rows = len(self.grid)
        cols = max((len(row) for row in self.grid), default=0)
        return rows, cols

    @property
    def ascii(self) -> str:
        return format_grid_ascii(self.grid)

    def __str__(self) -> str:
        rows, cols = self.shape
        return (
            f"Level: {self.level}\n"
            f"Step: {self.step}\n"
            f"Grid shape: {rows} x {cols}\n"
            f"Grid contents:\n{self.ascii}"
        )


@dataclass(frozen=True)
class HistoryEntry:
    action: str
    frame: Frame
    result: dict[str, Any] = field(default_factory=dict)


def normalize_grid(raw: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    rows: list[tuple[int, ...]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)):
            continue
        cells: list[int] = []
        for cell in row:
            try:
                cells.append(int(cell))
            except (TypeError, ValueError):
                cells.append(0)
        rows.append(tuple(cells))
    return tuple(rows)


def frame_from_payload(payload: Any) -> Frame | None:
    if not isinstance(payload, dict):
        return None
    try:
        step = max(0, int(payload.get("step", 0) or 0))
    except (TypeError, ValueError):
        step = 0
    try:
        level = max(1, int(payload.get("level", 1) or 1))
    except (TypeError, ValueError):
        level = 1
    return Frame(
        grid=normalize_grid(payload.get("grid")),
        step=step,
        level=level,
    )


def frame_to_payload(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "grid": [list(row) for row in frame.grid],
        "step": frame.step,
        "level": frame.level,
    }


def history_entry_from_payload(payload: Any) -> HistoryEntry | None:
    if not isinstance(payload, dict):
        return None
    frame = frame_from_payload(payload.get("frame"))
    if frame is None:
        return None
    return HistoryEntry(
        action=str(payload.get("action", "")).strip(), frame=frame,
        result=dict(payload["result"]) if isinstance(payload.get("result"), dict) else {},
    )


def history_entry_to_payload(entry: HistoryEntry) -> dict[str, Any]:
    return {
        "action": entry.action,
        "frame": frame_to_payload(entry.frame),
        "result": dict(getattr(entry, "result", {}) or {}),
    }


# The solver and the tool agent run in one process, so the state the solver
# writes is served from memory. The file used to be the channel: rewritten as
# indented JSON after every action and re-parsed for every tool call and every
# action inside it. Both scale with the whole history (18.6 MB and ~0.5 s at
# 300 actions), and with every game a thread of one process that time came
# straight out of the GPU's schedule late in a run. The file is still written,
# for anything reading it from outside, but at most every
# RUNTIME_STATE_FILE_INTERVAL_SECONDS and on flush.
RUNTIME_STATE_FILE_INTERVAL_SECONDS = 30.0
_LIVE_STATE: dict[str, tuple[Frame | None, list[HistoryEntry]]] = {}
_LAST_FILE_WRITE: dict[str, float] = {}
_LIVE_LOCK = threading.Lock()


def _live_key(path: Path) -> str:
    return str(Path(path).resolve())


def load_runtime_state(path: Path) -> tuple[Frame | None, list[HistoryEntry]]:
    with _LIVE_LOCK:
        live = _LIVE_STATE.get(_live_key(path))
    if live is not None:
        # a copy of the list, not of the entries: entries are frozen and the
        # caller must not see later appends through its own reference
        return live[0], list(live[1])
    if not path.exists():
        return None, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_frame = frame_from_payload(payload.get("current_frame"))
    history_entries = [
        entry
        for raw_entry in payload.get("history", [])
        for entry in [history_entry_from_payload(raw_entry)]
        if entry is not None
    ]
    return current_frame, history_entries


def write_runtime_state(
    path: Path,
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
    flush: bool = False,
) -> None:
    """Publish the state to this process at once; write the file when due.

    `history` is copied, so the caller may keep appending to its own list.
    The file is written on the first call for a path, on `flush`, and
    otherwise at most every RUNTIME_STATE_FILE_INTERVAL_SECONDS.
    """
    key = _live_key(path)
    now = time.monotonic()
    with _LIVE_LOCK:
        _LIVE_STATE[key] = (current_frame, list(history))
        last = _LAST_FILE_WRITE.get(key)
        due = flush or last is None or now - last >= RUNTIME_STATE_FILE_INTERVAL_SECONDS
        if due:
            _LAST_FILE_WRITE[key] = now
    if not due:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_frame": frame_to_payload(current_frame),
        "history": [history_entry_to_payload(entry) for entry in history],
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)


def discard_runtime_state(path: Path) -> None:
    """Forget a finished game's state: the live copy and the file."""
    key = _live_key(path)
    with _LIVE_LOCK:
        _LIVE_STATE.pop(key, None)
        _LAST_FILE_WRITE.pop(key, None)
    Path(path).unlink(missing_ok=True)
