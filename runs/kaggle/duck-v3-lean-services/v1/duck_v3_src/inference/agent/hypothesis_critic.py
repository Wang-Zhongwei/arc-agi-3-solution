"""Hypothesis-correction critic for the tool agent.

The duck's dominant failure mode is a wrong goal hypothesis that never gets
corrected: it keeps spending actions on one level while its carried world model
stays the same. This module adds a second, tool-less model call (the *critic*)
that fires when too many actions (or too many requests) have been spent on a
single level. The critic audits the carried world model against the level's
action log and returns either KEEP, REVISE, or REPLACE with concrete alternative
hypotheses and one discriminating test. The analyzer then folds the verdict into
its carried world model and shows the audit in the next few user prompts.

All thresholds are environment-configurable (see ``CriticConfig``).
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class CriticConfig:
    enabled: bool = True
    # Fire once this many real environment actions have been spent on the current level.
    action_threshold: int = 40
    # Also fire once this many analyzer requests have been spent on the level (covers
    # analysis paralysis, where the model keeps looking without acting).
    request_threshold: int = 15
    # After a critique, wait this many further actions (or requests) before re-firing.
    action_cooldown: int = 25
    request_cooldown: int = 10
    # How many analyzer requests the audit block stays in the user prompt.
    note_turns: int = 3
    # Generation settings for the critic call.
    max_output_tokens: int = 6144
    thinking: bool = True
    # Number of recent per-level actions shown to the critic.
    action_log_window: int = 40
    # Attach the same batched image the analyzer sees (when multimodal is on).
    attach_image: bool = True

    @classmethod
    def from_env(cls) -> "CriticConfig":
        return cls(
            enabled=_env_bool("HYPOTHESIS_CRITIC_ENABLED", True),
            action_threshold=_env_int("HYPOTHESIS_CRITIC_ACTION_THRESHOLD", 40, minimum=1),
            request_threshold=_env_int("HYPOTHESIS_CRITIC_REQUEST_THRESHOLD", 15, minimum=1),
            action_cooldown=_env_int("HYPOTHESIS_CRITIC_ACTION_COOLDOWN", 25, minimum=1),
            request_cooldown=_env_int("HYPOTHESIS_CRITIC_REQUEST_COOLDOWN", 10, minimum=1),
            note_turns=_env_int("HYPOTHESIS_CRITIC_NOTE_TURNS", 3, minimum=1),
            max_output_tokens=_env_int("HYPOTHESIS_CRITIC_MAX_OUTPUT", 6144, minimum=256),
            thinking=_env_bool("HYPOTHESIS_CRITIC_THINKING", True),
            action_log_window=_env_int("HYPOTHESIS_CRITIC_ACTION_LOG_WINDOW", 40, minimum=5),
            attach_image=_env_bool("HYPOTHESIS_CRITIC_ATTACH_IMAGE", True),
        )


@dataclass
class LevelProgress:
    """Per-level counters the critic trigger is judged on."""

    level: int = 1
    level_start_action_num: int = 0
    requests_on_level: int = 0
    game_overs_on_level: int = 0
    # Compact log of executed actions on this level: {"action", "board_changed", "action_num"}
    action_log: list[dict[str, Any]] = field(default_factory=list)
    critiques_on_level: int = 0
    last_critique_action_num: int | None = None
    last_critique_request: int | None = None

    def actions_on_level(self, action_num: int) -> int:
        return max(0, int(action_num) - int(self.level_start_action_num))

    def reset_for_level(self, level: int, action_num: int) -> None:
        self.level = level
        self.level_start_action_num = action_num
        self.requests_on_level = 0
        self.game_overs_on_level = 0
        self.action_log = []
        self.critiques_on_level = 0
        self.last_critique_action_num = None
        self.last_critique_request = None

    def record_action(self, *, action: str, board_changed: bool, action_num: Any, game_over: bool) -> None:
        try:
            parsed_num = int(action_num) if action_num is not None else None
        except (TypeError, ValueError):
            parsed_num = None
        self.action_log.append(
            {"action": str(action or "").strip(), "board_changed": bool(board_changed), "action_num": parsed_num}
        )
        if len(self.action_log) > 400:
            del self.action_log[: len(self.action_log) - 400]
        if game_over:
            self.game_overs_on_level += 1


CRITIC_LABELS = [
    "Verdict",
    "Evidence against current hypothesis",
    "Alternative hypotheses",
    "Discriminating test",
    "Revised world model",
    "Revised goal model",
    "Revised plan",
    "Stop doing",
]

VALID_VERDICTS = ("KEEP", "REVISE", "REPLACE")


def should_run_critic(config: CriticConfig, progress: LevelProgress, *, action_num: int) -> str | None:
    """Return a trigger reason when the critic should fire on this request, else None."""
    if not config.enabled:
        return None
    actions = progress.actions_on_level(action_num)
    requests = progress.requests_on_level
    if progress.critiques_on_level == 0:
        if actions >= config.action_threshold:
            return f"actions_on_level={actions} >= {config.action_threshold}"
        if requests >= config.request_threshold:
            return f"requests_on_level={requests} >= {config.request_threshold}"
        return None
    since_actions = actions - int(progress.last_critique_action_num or 0)
    since_requests = requests - int(progress.last_critique_request or 0)
    if since_actions >= config.action_cooldown:
        return f"actions_since_last_critique={since_actions} >= {config.action_cooldown}"
    if since_requests >= config.request_cooldown:
        return f"requests_since_last_critique={since_requests} >= {config.request_cooldown}"
    return None


def summarize_action_log(progress: LevelProgress, *, window: int) -> list[str]:
    log = progress.action_log
    lines: list[str] = []
    if not log:
        lines.append("No actions have been executed on this level yet (the agent has only been looking).")
        return lines
    total = len(log)
    unchanged = sum(1 for item in log if not item.get("board_changed"))
    counts = Counter(item.get("action", "") for item in log)
    top = ", ".join(f"{name}x{count}" for name, count in counts.most_common(6))
    lines.append(f"Actions executed on this level: {total}; without any board change: {unchanged}.")
    lines.append(f"Most used actions: {top}.")
    # Longest streak of no-change actions.
    streak = best = 0
    for item in log:
        streak = streak + 1 if not item.get("board_changed") else 0
        best = max(best, streak)
    if best >= 3:
        lines.append(f"Longest run of consecutive no-change actions: {best}.")
    if progress.game_overs_on_level:
        lines.append(f"GAME_OVER was reached {progress.game_overs_on_level} time(s) on this level (each one reset the level).")
    recent = log[-window:]
    rendered = " ".join(
        f"{item.get('action') or '?'}{'' if item.get('board_changed') else '(no-change)'}" for item in recent
    )
    lines.append(f"Most recent {len(recent)} actions (oldest first): {rendered}")
    return lines


def build_critic_messages(
    *,
    system_prompt: str,
    level: int,
    actions_on_level: int,
    requests_on_level: int,
    trigger_reason: str,
    world_model_lines: list[str],
    action_summary_lines: list[str],
    current_ascii: str,
    valid_actions: list[str],
    image_part: dict[str, Any] | None,
    previous_critique: dict[str, str] | None,
) -> list[dict[str, Any]]:
    user_lines = [
        f"Audit request: the agent has spent {actions_on_level} actions and {requests_on_level} model requests on level {level} "
        f"without completing it (trigger: {trigger_reason}).",
        "",
        "Carried world model the agent is currently acting on:",
        *(world_model_lines or ["- (empty: the agent has not written down a world model)"]),
        "",
        "Action evidence on this level:",
        *[f"- {line}" for line in action_summary_lines],
        "",
        f"Valid actions right now: {', '.join(valid_actions) if valid_actions else 'unknown'}.",
    ]
    if previous_critique:
        user_lines.extend(
            [
                "",
                "Your previous audit on this level (the agent still has not cleared the level since):",
                *[f"- {label}: {value}" for label, value in previous_critique.items() if value],
            ]
        )
    user_lines.extend(
        [
            "",
            "Current board (letter-coded ARC colors, one row per line):",
            current_ascii,
            "",
            "Respond with exactly these labeled sections, each label on its own line, no other headings:",
            "Verdict: one of KEEP, REVISE, REPLACE",
            "Evidence against current hypothesis: ...",
            "Alternative hypotheses: 2-3 ranked, one per line, most likely first",
            "Discriminating test: one concrete probe (actions or MOUSE row/col targets) that separates the top hypotheses",
            "Revised world model: ...",
            "Revised goal model: ...",
            "Revised plan: ...",
            "Stop doing: ...",
        ]
    )
    text = "\n".join(user_lines)
    content: Any = text
    if image_part is not None:
        content = [{"type": "text", "text": f"{text}\n\nRecent frames image (oldest left, newest right):"}, image_part]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_verdict(raw: str) -> str:
    text = str(raw or "").strip().upper()
    for verdict in VALID_VERDICTS:
        if text.startswith(verdict):
            return verdict
    for verdict in VALID_VERDICTS:
        if verdict in text:
            return verdict
    return "REVISE"


def critique_prompt_lines(critique: dict[str, str], *, level: int, actions_on_level: int) -> list[str]:
    """Lines injected into the analyzer's next user prompts after a critique."""
    verdict = critique.get("Verdict", "REVISE")
    header = (
        f"Hypothesis audit from the critic agent (after {actions_on_level} actions on level {level}), verdict {verdict}:"
    )
    lines = [header]
    for label in CRITIC_LABELS[1:]:
        value = critique.get(label, "")
        if value:
            lines.append(f"- {label}: {value}")
    if verdict == "KEEP":
        lines.append(
            "- The critic found the current hypothesis consistent with the evidence; execute its discriminating test or the plan instead of re-investigating."
        )
    else:
        lines.append(
            "- Default: adopt the revised world model and run the discriminating test first. Only keep the old hypothesis if you can cite concrete frame evidence the critic missed, and say so explicitly."
        )
    return lines


class HypothesisCritic:
    """Runs the audit call and turns the answer into a world-model update."""

    def __init__(
        self,
        *,
        config: CriticConfig,
        system_prompt: str,
        chat_completion: Callable[..., Any],
        extract_blocks: Callable[[str, list[str]], dict[str, str]],
    ) -> None:
        self.config = config
        self._system_prompt = system_prompt
        self._chat_completion = chat_completion
        self._extract_blocks = extract_blocks

    def run(
        self,
        *,
        progress: LevelProgress,
        action_num: int,
        trigger_reason: str,
        world_model_lines: list[str],
        current_ascii: str,
        valid_actions: list[str],
        image_part: dict[str, Any] | None,
        previous_critique: dict[str, str] | None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Return (critique, meta). ``meta`` carries raw text for the transcript."""
        messages = build_critic_messages(
            system_prompt=self._system_prompt,
            level=progress.level,
            actions_on_level=progress.actions_on_level(action_num),
            requests_on_level=progress.requests_on_level,
            trigger_reason=trigger_reason,
            world_model_lines=world_model_lines,
            action_summary_lines=summarize_action_log(progress, window=self.config.action_log_window),
            current_ascii=current_ascii,
            valid_actions=valid_actions,
            image_part=image_part if self.config.attach_image else None,
            previous_critique=previous_critique,
        )
        result = self._chat_completion(
            messages,
            tools=None,
            max_tokens=self.config.max_output_tokens,
            thinking=self.config.thinking,
        )
        message = result.message if hasattr(result, "message") else {}
        content = _message_text(message.get("content", ""))
        reasoning = _message_text(message.get("reasoning") or message.get("reasoning_content") or "")
        critique = self._extract_blocks(content, CRITIC_LABELS) if content else {}
        if not critique.get("Verdict") and reasoning:
            # Truncated or thinking-only reply: salvage labeled blocks from the reasoning.
            critique = self._extract_blocks(reasoning, CRITIC_LABELS) or critique
        critique = {label: value for label, value in critique.items() if value}
        if critique:
            critique["Verdict"] = parse_verdict(critique.get("Verdict", ""))
        meta = {
            "finish_reason": getattr(result, "finish_reason", ""),
            "usage": getattr(result, "usage", None),
            "content": content,
            "reasoning": reasoning,
            "messages": messages,
        }
        return critique, meta


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(part for part in parts if part).strip()
    return ""
