"""Compact records of transitions whose conversation exchanges were evicted."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.agent.runtime_state import HistoryEntry
from inference.agent.vision_context import frame_to_png_data_url


EXCHANGE_KEY = "_arc3_transition_exchange"
REMINDER_KIND = "transition_reminders"
REASONING_LIMIT = 2048
REASONING_HALF = 1024
OMISSION_MARKER = "[Middle of reasoning omitted]"


@dataclass(frozen=True)
class TransitionReminder:
    kind: str
    history_index: int
    source_exchange: int
    source_tool_call: str
    parts: list[dict[str, Any]]

    def source_retained(self, messages: list[dict[str, Any]]) -> bool:
        source = [m for m in messages if m.get(EXCHANGE_KEY) == self.source_exchange]
        return (
            any(m.get("role") == "assistant" for m in source)
            and any(m.get("role") == "tool" and m.get("tool_call_id") == self.source_tool_call
                    for m in source)
        )


def make_reminder(
    kind: str, entries: list[HistoryEntry], index: int, *, sequence_start: int,
    reasoning: str, source_exchange: int, source_tool_call: str,
) -> TransitionReminder:
    entry = entries[index]
    before = entries[index - 1].frame
    after = entry.frame
    outcome = (
        "Game completed" if entry.result.get("run_complete") else
        f"Last level-up {before.level} → {after.level}" if kind == "level_up" else
        f"Last GAME OVER (level {before.level})"
    )
    lines = [f"{outcome} — action {entry.result.get('action_num', after.step)} / history[{index}]"]
    if reasoning:
        label = (
            "Reasoning before the fatal action (may contain mistaken assumptions):"
            if kind == "game_over" else "Reasoning before the actions:"
        )
        lines.extend([label, reasoning])
    start = max(sequence_start, index - 7)
    if start > sequence_start:
        lines.append("[Earlier executed actions omitted]")
    lines.append("Executed actions: " + ", ".join(e.action for e in entries[start:index + 1]))
    lines.append("Triggering action: " + entry.action)
    if kind == "game_over" and index + 1 < len(entries):
        reset = entries[index + 1]
        if reset.result.get("automatic") and reset.action == "RESET":
            lines.append(f"The environment then automatically RESET at history[{index + 1}].")
    parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    boards = [(index - 1, before, "Before GAME OVER" if kind == "game_over" else "Before the level-up")]
    if kind == "game_over":
        boards.append((index, after, "Fatal board"))
    for board_index, board, label in boards:
        parts.extend([
            {"type": "text", "text": f"{label} (history[{board_index}].frame, level {board.level}):"},
            {"type": "image_url", "image_url": {"url": frame_to_png_data_url(board)}},
        ])
    return TransitionReminder(kind, index, source_exchange, source_tool_call, parts)
