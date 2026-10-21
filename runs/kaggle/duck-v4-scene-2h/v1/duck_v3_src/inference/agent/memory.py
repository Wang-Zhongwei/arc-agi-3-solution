"""Rulebook memory: a compact, per-level description of a game carried across levels and across runs.

When a level completes, one short model call turns the remembered facts, the carried world
model and the probe table into a rulebook (objects, action effects, goal, solution outline).
It is shown once at the start of the next level of this run and, on later runs of the same
game, once at the start of every level that has a rulebook. Stored as JSON per game under
``memory_dir`` (default ``memory/`` next to the harness).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

RULEBOOK_SYSTEM_PROMPT = (
    "You write the rulebook for one level of an ARC-AGI-3 grid puzzle game that an agent has just completed. "
    "Use only the evidence given (measured facts, the agent's world model, the probe table). Be concrete: name objects by colour and size, "
    "give each action's effect with numbers, state the goal and a 3-6 step solution outline, and list anything that looked like a HUD or timer. "
    "At most 150 words, plain lines starting with 'Objects:', 'Actions:', 'Goal:', 'Solution:', 'Notes:'. No preamble."
)


def game_id_from_state_path(state_path: Path) -> str:
    name = state_path.name
    stem = name.split("_tool_runtime_state")[0]
    if "_p" in stem:
        stem = stem.rsplit("_p", 1)[0]
    return stem or "unknown-game"


class GameMemory:
    def __init__(
        self,
        *,
        enabled: bool,
        memory_dir: Path,
        game_id: str,
        chat_completion: Callable[..., Any] | None,
        max_output_tokens: int = 600,
    ) -> None:
        self.enabled = enabled
        self.memory_dir = memory_dir
        self.game_id = game_id
        self.path = memory_dir / f"{game_id}.json"
        self._chat = chat_completion
        self._max_output_tokens = max_output_tokens
        self.data: dict[str, Any] = {"game_id": game_id, "levels": {}, "runs": 0}
        self.session_rulebooks: dict[int, str] = {}
        self._shown: set[int] = set()
        if enabled:
            self._load()

    @classmethod
    def from_env(cls, *, state_path: Path, chat_completion: Callable[..., Any] | None) -> "GameMemory":
        raw = os.environ.get("HARNESS_MEMORY_ENABLED", "true").strip().lower()
        enabled = raw not in {"0", "false", "no", "off"}
        memory_dir = Path(os.environ.get("HARNESS_MEMORY_DIR", "memory").strip() or "memory")
        return cls(enabled=enabled, memory_dir=memory_dir, game_id=game_id_from_state_path(state_path), chat_completion=chat_completion)

    def _load(self) -> None:
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
                    self.data.setdefault("levels", {})
        except (OSError, ValueError):
            pass

    def _save(self) -> None:
        if not self.enabled:
            return
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def stored_rulebook(self, level: int) -> str:
        entry = self.data.get("levels", {}).get(str(level))
        return str(entry.get("rulebook", "") or "") if isinstance(entry, dict) else ""

    def prompt_lines(self, level: int) -> list[str]:
        """Rulebooks worth showing now: the previous level of this run, and a stored one for this level. Once each."""
        if not self.enabled or level in self._shown:
            return []
        lines: list[str] = []
        previous = self.session_rulebooks.get(level - 1)
        if previous:
            lines.append(f"Rulebook for level {level - 1}, which you just completed (mechanics usually carry over; look for what is new):")
            lines.extend(f"  {line}" for line in previous.splitlines() if line.strip())
        stored = self.stored_rulebook(level)
        if stored and stored != previous:
            lines.append(f"Rulebook for level {level} from an earlier run of this game (verify before relying on it):")
            lines.extend(f"  {line}" for line in stored.splitlines() if line.strip())
        if lines:
            self._shown.add(level)
        return lines

    def write_rulebook(
        self,
        *,
        level: int,
        facts: dict[str, str],
        knowledge_lines: list[str],
        helper_names: list[str],
        actions_used: int,
    ) -> tuple[str, dict[str, Any]]:
        """Produce and store the rulebook for a completed level. Falls back to a deterministic digest."""
        fallback = self._fallback_rulebook(facts, knowledge_lines, actions_used)
        text = fallback
        meta: dict[str, Any] = {"source": "fallback"}
        if self.enabled and self._chat is not None:
            user = "\n".join(
                [
                    f"Level {level} of game {self.game_id} was completed in {actions_used} actions.",
                    "Measured facts and remembered rules:",
                    *[f"- {k}: {v}" for k, v in facts.items()],
                    "",
                    "The agent's carried world model:",
                    *(knowledge_lines or ["- (none)"]),
                    "",
                    f"Helper functions the agent wrote: {', '.join(helper_names) or 'none'}.",
                    "",
                    "Write the rulebook now.",
                ]
            )
            try:
                result = self._chat(
                    [{"role": "system", "content": RULEBOOK_SYSTEM_PROMPT}, {"role": "user", "content": user}],
                    tools=None,
                    max_tokens=self._max_output_tokens,
                    thinking=False,
                )
                message = result.message if hasattr(result, "message") else {}
                content = message.get("content", "") if isinstance(message, dict) else ""
                if isinstance(content, list):
                    content = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
                content = str(content or "").strip()
                if content:
                    text = content[:1500]
                    meta = {"source": "model", "usage": getattr(result, "usage", None)}
            except Exception as exc:  # never break the turn
                meta = {"source": "fallback", "error": str(exc)[:200]}
        self.session_rulebooks[level] = text
        levels = self.data.setdefault("levels", {})
        levels[str(level)] = {"rulebook": text, "actions_used": actions_used, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.data["runs"] = int(self.data.get("runs", 0)) + 1
        self._save()
        return text, meta

    @staticmethod
    def _fallback_rulebook(facts: dict[str, str], knowledge_lines: list[str], actions_used: int) -> str:
        parts = [f"Completed in {actions_used} actions."]
        if facts:
            parts.append("Facts: " + "; ".join(f"{k}: {v}" for k, v in list(facts.items())[:12]))
        for line in knowledge_lines:
            if line.startswith("- "):
                parts.append(line[2:])
        return "\n".join(parts)[:1500]
