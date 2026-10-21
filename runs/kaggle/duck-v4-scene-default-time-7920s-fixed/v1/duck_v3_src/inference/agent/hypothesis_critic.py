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
    enabled: bool = False
    # "conditional" (default): fire only when the carried world model has stopped evolving while actions are
    # still being spent, or when the level shows a loop. "schedule": the older action/request thresholds.
    mode: str = "conditional"
    # Conditional triggers: no change to the world/goal/action model for this many requests, during which at
    # least this many actions were spent on the level.
    stagnation_requests: int = 8
    stagnation_actions: int = 12
    # A k-gram of actions (k <= 4) repeated at least this many times at the tail of the level log is a loop.
    loop_repeats: int = 3
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
    # Generation settings for the critic call. "audit" mode deliberates at length (thinking on, high reasoning
    # effort, full output budget) and tests every carried claim (world model, action model, mechanics, goal)
    # one by one against the recorded evidence instead of writing a replacement model.
    max_output_tokens: int = 6144
    # Flash-Next's chat template accepts only xhigh / medium / low (a 400 otherwise); "xhigh" is its default.
    reasoning_effort: str = "xhigh"
    thinking: bool = True
    # Number of recent per-level actions shown to the critic.
    action_log_window: int = 40
    # Attach the same batched image the analyzer sees (when multimodal is on).
    attach_image: bool = True

    @classmethod
    def from_env(cls) -> "CriticConfig":
        return cls(
            enabled=_env_bool("HYPOTHESIS_CRITIC_ENABLED", False),
            mode=(os.environ.get("HYPOTHESIS_CRITIC_MODE", "conditional").strip().lower() or "conditional"),
            stagnation_requests=_env_int("HYPOTHESIS_CRITIC_STAGNATION_REQUESTS", 8, minimum=2),
            stagnation_actions=_env_int("HYPOTHESIS_CRITIC_STAGNATION_ACTIONS", 12, minimum=1),
            action_threshold=_env_int("HYPOTHESIS_CRITIC_ACTION_THRESHOLD", 40, minimum=1),
            request_threshold=_env_int("HYPOTHESIS_CRITIC_REQUEST_THRESHOLD", 15, minimum=1),
            action_cooldown=_env_int("HYPOTHESIS_CRITIC_ACTION_COOLDOWN", 25, minimum=1),
            request_cooldown=_env_int("HYPOTHESIS_CRITIC_REQUEST_COOLDOWN", 10, minimum=1),
            note_turns=_env_int("HYPOTHESIS_CRITIC_NOTE_TURNS", 3, minimum=1),
            max_output_tokens=_env_int("HYPOTHESIS_CRITIC_MAX_OUTPUT", 6144, minimum=256),
            thinking=_env_bool("HYPOTHESIS_CRITIC_THINKING", True),
            reasoning_effort=(os.environ.get("HYPOTHESIS_CRITIC_REASONING_EFFORT", "xhigh").strip().lower() or "xhigh"),
            # In audit mode a loop must be longer before it is worth a long deliberation: 3 repeats is ordinary probing.
            loop_repeats=_env_int("HYPOTHESIS_CRITIC_LOOP_REPEATS", 6 if os.environ.get("HYPOTHESIS_CRITIC_MODE", "").strip().lower() == "audit" else 3, minimum=2),
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
    # (request index, fingerprint of world/goal/action model, actions on level) per request, for stagnation.
    model_fingerprints: list[tuple[int, str, int]] = field(default_factory=list)

    def record_model_fingerprint(self, fingerprint: str, actions_on_level: int) -> None:
        self.model_fingerprints.append((self.requests_on_level, fingerprint, actions_on_level))
        if len(self.model_fingerprints) > 200:
            del self.model_fingerprints[: len(self.model_fingerprints) - 200]

    def stagnation(self) -> tuple[int, int]:
        """(requests since the model last changed, actions spent in that span)."""
        if not self.model_fingerprints:
            return 0, 0
        latest = self.model_fingerprints[-1][1]
        span_start = len(self.model_fingerprints) - 1
        while span_start > 0 and self.model_fingerprints[span_start - 1][1] == latest:
            span_start -= 1
        first = self.model_fingerprints[span_start]
        last = self.model_fingerprints[-1]
        return len(self.model_fingerprints) - 1 - span_start, max(0, last[2] - first[2])

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
        self.model_fingerprints = []

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
# In conditional mode the critic answers with only these, in at most ~120 words.
TERSE_CRITIC_LABELS = ["Verdict", "Evidence against current hypothesis", "Discriminating test", "Stop doing"]

# Audit mode: every carried claim is tested, then a verdict and corrected mechanics.
AUDIT_CRITIC_LABELS = ["Claims", "Verdict", "Corrected mechanics", "Discriminating test", "Stop doing"]

VALID_VERDICTS = ("KEEP", "REVISE", "REPLACE")


def repeated_tail_pattern(actions: list[str], *, min_repeats: int, max_period: int = 4) -> tuple[list[str], int] | None:
    """The shortest action k-gram repeated at least ``min_repeats`` times at the end of the log, if any."""
    names = [a for a in actions if a]
    for period in range(1, max_period + 1):
        if len(names) < period * min_repeats:
            continue
        gram = names[-period:]
        repeats = 0
        i = len(names)
        while i - period >= 0 and names[i - period : i] == gram:
            repeats += 1
            i -= period
        if repeats >= min_repeats:
            return gram, repeats
    return None


def should_run_critic(
    config: CriticConfig,
    progress: LevelProgress,
    *,
    action_num: int,
    stuck: bool = False,
) -> str | None:
    """Return a trigger reason when the critic should fire on this request, else None."""
    if not config.enabled:
        return None
    actions = progress.actions_on_level(action_num)
    requests = progress.requests_on_level
    if config.mode in ("conditional", "audit"):
        # Cooldown after a critique, in actions and requests.
        if progress.critiques_on_level:
            if actions - int(progress.last_critique_action_num or 0) < config.action_cooldown and requests - int(progress.last_critique_request or 0) < config.request_cooldown:
                return None
        if stuck:
            return "stuck loop: the last batches produced the same or no board change"
        loop = repeated_tail_pattern([item.get("action", "") for item in progress.action_log], min_repeats=config.loop_repeats)
        if loop is not None:
            gram, repeats = loop
            return f"repeated actions: {' '.join(gram)} x{repeats} at the end of the level log"
        stale_requests, stale_actions = progress.stagnation()
        if stale_requests >= config.stagnation_requests and stale_actions >= config.stagnation_actions:
            return f"rules not evolving: world/goal/action model unchanged for {stale_requests} requests while {stale_actions} actions were spent"
        return None
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
    terse: bool = False,
    audit: bool = False,
    evidence_lines: list[str] | None = None,
) -> list[dict[str, Any]]:
    if audit:
        return _build_audit_messages(system_prompt=system_prompt, level=level, actions_on_level=actions_on_level, requests_on_level=requests_on_level,
                                     trigger_reason=trigger_reason, world_model_lines=world_model_lines, action_summary_lines=action_summary_lines,
                                     evidence_lines=evidence_lines or [], current_ascii=current_ascii, valid_actions=valid_actions,
                                     image_part=image_part, previous_critique=previous_critique)
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
    if terse:
        user_lines.extend(
            [
                "",
                "Current board (letter-coded ARC colors, one row per line):",
                current_ascii,
                "",
                "Answer in at most 120 words with exactly these labeled lines and nothing else:",
                "Verdict: KEEP (rules are evolving and the loop is explainable) or REVISE",
                "Evidence against current hypothesis: the one fact from the action log that the current rules do not explain",
                "Discriminating test: one concrete probe (actions or MOUSE row/col) that would break the loop or test the stale rule",
                "Stop doing: the repeated thing to stop",
            ]
        )
    else:
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


CLAIM_LABELS = ("world model", "goal model", "action model", "mechanics")


def _numbered_claims(world_model_lines: list[str]) -> list[str]:
    """Split the carried model into individually testable claims: one per sentence / numbered rule.

    Only the four model fields are claims (findings, open questions and the plan are not statements about the game);
    the ASCII map inside the world model is a picture, and an 'Evidence: ...' sentence stays attached to the claim it supports.
    """
    import re as _re
    claims: list[str] = []
    for line in world_model_lines:
        text = line.strip().lstrip("- ")
        label, _, body = text.partition(":")
        if label.strip().lower() not in CLAIM_LABELS or not body.strip():
            continue
        body = _re.split(r"\s*Map \(1 char", body, maxsplit=1)[0]
        raw_pieces = [p.strip() for p in _re.split(r"(?:(?<=[.;])\s+|\s(?=\d+\.\s))", body) if p.strip()]
        pieces: list[str] = []
        for piece in raw_pieces:
            if pieces and _re.fullmatch(r"\d+\.", pieces[-1]):
                pieces[-1] = f"{pieces[-1]} {piece}"     # keep '2.' with its rule
            else:
                pieces.append(piece)
        for piece in pieces:
            if _re.fullmatch(r"\d+\.", piece) or "chars omitted]" in piece:
                continue
            if piece.lower().startswith("evidence:") and claims:
                claims[-1] = f"{claims[-1]} {piece}"
                continue
            claims.append(f"[{label.strip()}] {piece}")
    return claims[:24]


def _build_audit_messages(*, system_prompt, level, actions_on_level, requests_on_level, trigger_reason, world_model_lines,
                          action_summary_lines, evidence_lines, current_ascii, valid_actions, image_part, previous_critique):
    claims = _numbered_claims(world_model_lines)
    user_lines = [
        f"Audit request: the agent has spent {actions_on_level} actions and {requests_on_level} model requests on level {level} "
        f"without completing it (trigger: {trigger_reason}). Take your time: test EVERY claim below against the evidence before judging.",
        "",
        "Claims the agent is acting on (its carried world model, split into testable statements):",
        *([f"{i + 1}. {c}" for i, c in enumerate(claims)] or ["(the agent has written no world model; audit its actions instead)"]),
        "",
        "Measured evidence (authoritative; each line is a harness measurement of a real transition):",
        *([f"- {line}" for line in evidence_lines] or ["- (none recorded yet)"]),
        "",
        "Action log on this level:",
        *[f"- {line}" for line in action_summary_lines],
        "",
        f"Valid actions right now: {', '.join(valid_actions) if valid_actions else 'unknown'}.",
        "",
        "Current board (letter-coded ARC colors, one row per line):",
        current_ascii,
    ]
    if previous_critique:
        user_lines += ["", "Your previous audit on this level:", *[f"- {k}: {v}" for k, v in previous_critique.items() if v]]
    user_lines += [
        "",
        "Respond with exactly these labeled sections, each label on its own line:",
        "Claims: one line per claim number: 'k: SUPPORTED by <the transition or measurement>' or 'k: CONTRADICTED by <the transition or measurement>' or 'k: UNTESTED; cheapest test: <actions>'. Every claim must appear.",
        "Verdict: KEEP if no claim is contradicted, else REVISE",
        "Corrected mechanics: numbered rules that survive the audit, each with its supporting transition; rewrite contradicted rules from the evidence, do not invent new ones",
        "Discriminating test: the one probe (actions or MOUSE row/col) that settles the most important untested or contradicted claim",
        "Stop doing: the repeated thing in the action log that the evidence says cannot work",
    ]
    text = "\n".join(user_lines)
    content: Any = text
    if image_part is not None:
        content = [{"type": "text", "text": f"{text}\n\nRecent frames image (oldest left, newest right):"}, image_part]
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def parse_verdict(raw: str) -> str:
    text = str(raw or "").strip().upper()
    for verdict in VALID_VERDICTS:
        if text.startswith(verdict):
            return verdict
    for verdict in VALID_VERDICTS:
        if verdict in text:
            return verdict
    return "REVISE"


def _normalize(value: str, max_chars: int = 400) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


def critique_prompt_lines(critique: dict[str, str], *, level: int, actions_on_level: int) -> list[str]:
    """Lines injected into the analyzer's next user prompts after a critique."""
    verdict = critique.get("Verdict", "REVISE")
    header = (
        f"Hypothesis audit from the critic agent (after {actions_on_level} actions on level {level}), verdict {verdict}:"
    )
    lines = [header]
    for label in (AUDIT_CRITIC_LABELS if "Claims" in critique else CRITIC_LABELS)[1:]:
        value = critique.get(label, "")
        if value:
            lines.append(f"- {label}: {_normalize(value, max_chars=700 if label == 'Corrected mechanics' else 400)}")
    if "Claims" in critique:
        lines.append(f"- Claims audited: {_normalize(critique['Claims'], max_chars=900)}")
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
        audit_system_prompt: str | None = None,
    ) -> None:
        self.config = config
        self._system_prompt = system_prompt
        self._audit_system_prompt = audit_system_prompt or system_prompt
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
        evidence_lines: list[str] | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Return (critique, meta). ``meta`` carries raw text for the transcript."""
        audit = self.config.mode == "audit"
        messages = build_critic_messages(
            system_prompt=self._audit_system_prompt if audit else self._system_prompt,
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
            terse=self.config.mode == "conditional",
            audit=audit,
            evidence_lines=evidence_lines,
        )
        kwargs: dict[str, Any] = {}
        if audit:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        result = self._chat_completion(
            messages,
            tools=None,
            max_tokens=min(self.config.max_output_tokens, 1200) if self.config.mode == "conditional" else self.config.max_output_tokens,
            thinking=True if audit else self.config.thinking,
            **kwargs,
        )
        message = result.message if hasattr(result, "message") else {}
        content = _message_text(message.get("content", ""))
        reasoning = _message_text(message.get("reasoning") or message.get("reasoning_content") or "")
        labels = AUDIT_CRITIC_LABELS if audit else CRITIC_LABELS
        critique = self._extract_blocks(content, labels) if content else {}
        if not critique.get("Verdict") and reasoning:
            # Truncated or thinking-only reply: salvage labeled blocks from the reasoning.
            critique = self._extract_blocks(reasoning, labels) or critique
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
