"""Explore / exploit phase controller for the tool agent.

The model tends to keep exploring and doubting itself indefinitely. This controller
turns the level into explicit phases with deterministic budgets:

* **explore** (learn, do not solve): capped at ``explore_budget`` actions on the level
  (deterministic probes included). The model reports ``Confidence: action model NN%,
  goal model NN%`` on every turn; once both reach ``confidence_threshold`` -- or the
  budget runs out -- the current world/goal/action model is frozen and the phase
  switches to exploit.
* **exploit** (solve with the committed rules): edits to the frozen hypotheses are
  ignored; the model must act every request. The attempt ends when
  ``exploit_max_contradictions`` predictions have been contradicted, a GAME_OVER
  happens, or the budget (``exploit_budget_factor`` x the level's human benchmark,
  or ``exploit_budget_default``) runs out. Then a short re-exploration with
  ``re_explore_budget`` actions starts, focused on the rules that failed.

Every executed action batch is logged with the model's stated hypothesis, prediction
and the deterministic verdict, and the most recent ones are shown in the prompt so the
history carries the *why*, not just the *what*.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
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
class PhaseConfig:
    enabled: bool = False
    explore_budget: int = 30
    re_explore_budget: int = 8
    confidence_threshold: int = 80
    # Primary gate: the model's own p in [0, 1] that it understands the game well enough to attempt it.
    p_threshold: float = 0.7
    exploit_budget_factor: float = 1.5
    exploit_budget_default: int = 40
    exploit_max_contradictions: int = 1
    # Consecutive executed batches with an identical (or empty) measured diff that count as "stuck".
    stuck_repeats: int = 3
    recent_reasons: int = 8

    @classmethod
    def from_env(cls) -> "PhaseConfig":
        return cls(
            enabled=_env_bool("PHASES_ENABLED", False),
            explore_budget=_env_int("PHASES_EXPLORE_BUDGET", 30, minimum=1),
            re_explore_budget=_env_int("PHASES_RE_EXPLORE_BUDGET", 8, minimum=1),
            confidence_threshold=_env_int("PHASES_CONFIDENCE_THRESHOLD", 80, minimum=1),
            p_threshold=max(0.0, min(1.0, _env_float("PHASES_P_THRESHOLD", 0.7))),
            exploit_budget_factor=max(0.1, _env_float("PHASES_EXPLOIT_BUDGET_FACTOR", 1.5)),
            exploit_budget_default=_env_int("PHASES_EXPLOIT_BUDGET_DEFAULT", 40, minimum=1),
            exploit_max_contradictions=_env_int("PHASES_EXPLOIT_MAX_CONTRADICTIONS", 1, minimum=1),
            stuck_repeats=_env_int("PHASES_STUCK_REPEATS", 3, minimum=2),
            recent_reasons=_env_int("PHASES_RECENT_REASONS", 8, minimum=0),
        )


FROZEN_KEYS = ("world_model", "goal_model", "action_model")

_CONF_PATTERNS = {
    "action_model": re.compile(r"action(?:[\s_]+model|[\s_]+rules?|s)?\s*[:=]?\s*(\d{1,3})\s*%?", re.IGNORECASE),
    "goal_model": re.compile(r"goal(?:[\s_]+model)?\s*[:=]?\s*(\d{1,3})\s*%?", re.IGNORECASE),
}
_CONF_LINE_RE = re.compile(r"confidence\s*:\s*(.+)", re.IGNORECASE)
_P_RE = re.compile(r"\bp\s*[:=]\s*(0?\.\d+|1(?:\.0+)?|0|\d{1,3}\s*%)", re.IGNORECASE)


def parse_p(value: Any) -> float | None:
    """Normalize a self-rated understanding probability: 0.8, '0.8', '80%', 'p=0.8' -> 0.8."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1.0:
            number = number / 100.0
        return max(0.0, min(1.0, number))
    text = str(value).strip()
    if not text:
        return None
    match = _P_RE.search(text) if not re.fullmatch(r"[0-9.]+\s*%?", text) else None
    raw = match.group(1) if match else text
    raw = raw.strip()
    try:
        if raw.endswith("%"):
            return max(0.0, min(1.0, float(raw[:-1].strip()) / 100.0))
        number = float(raw)
    except ValueError:
        return None
    if number > 1.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def parse_confidence(*texts: str) -> dict[str, int]:
    """Extract ``Confidence: action model NN%, goal model NN%`` percentages (last occurrence wins)."""
    found: dict[str, int] = {}
    for text in texts:
        if not text:
            continue
        for line in text.splitlines():
            match = _CONF_LINE_RE.search(line)
            if not match:
                continue
            body = match.group(1)
            for key, pattern in _CONF_PATTERNS.items():
                hit = pattern.search(body)
                if hit:
                    found[key] = max(0, min(100, int(hit.group(1))))
    return found


@dataclass
class ActionReason:
    actions: list[str]
    hypothesis: str
    prediction: str
    verdict: str
    action_num: int | None
    phase: str
    board_changed: bool = False
    game_over: bool = False

    def line(self) -> str:
        names = ", ".join(self.actions[:8]) + (" ..." if len(self.actions) > 8 else "")
        where = f" (ending at action {self.action_num})" if self.action_num is not None else ""
        parts = [f"[{self.phase}] {names}{where}"]
        parts.append(f"why: {self.hypothesis}" if self.hypothesis else "why: (no hypothesis stated)")
        if self.prediction:
            parts.append(f"predicted: {self.prediction}")
        if self.verdict:
            parts.append(f"verdict: {self.verdict}")
        elif not self.board_changed:
            parts.append("board unchanged")
        if self.game_over:
            parts.append("GAME_OVER")
        return " | ".join(parts)


@dataclass
class PhaseState:
    level: int
    phase: str = "explore"
    cycle: int = 0
    phase_start_action: int = 0
    budget: int = 30
    confidence: dict[str, int] = field(default_factory=dict)
    p: float | None = None
    frozen: dict[str, str] = field(default_factory=dict)
    # Edits to frozen fields made during an exploitation attempt: adopted (with their reason) when it ends.
    proposed_revisions: list[dict[str, Any]] = field(default_factory=list)
    contradictions_in_phase: int = 0
    game_overs_in_phase: int = 0
    stuck: bool = False
    recent_signatures: list[str] = field(default_factory=list)
    transition_reason: str = ""
    failure_notes: list[str] = field(default_factory=list)
    reasons: list[ActionReason] = field(default_factory=list)
    benchmark_actions: int | None = None

    def actions_in_phase(self, action_num: int) -> int:
        return max(0, int(action_num) - int(self.phase_start_action))


def augment_field(old: str, new: str, reason: str, *, attempt: int | None = None, keep_chars: int = 220) -> str:
    """Replace a memory field without losing the old version: new text, the reason, and a trimmed 'previously'."""
    new = " ".join(str(new or "").split())
    old = " ".join(str(old or "").split())
    if not old or old == new:
        return new
    tag = f"revised{' after attempt ' + str(attempt) if attempt else ''}"
    reason_text = f" because {reason}" if reason else ""
    previous = old if len(old) <= keep_chars else old[:keep_chars].rstrip() + "..."
    return f"{new} [{tag}{reason_text}; previously: {previous}]"


class PhaseController:
    def __init__(self, config: PhaseConfig) -> None:
        self.config = config
        self.state: PhaseState | None = None

    def reset(self) -> None:
        self.state = None

    # -- lifecycle ----------------------------------------------------------

    def level_start(self, level: int, action_num: int) -> PhaseState:
        self.state = PhaseState(level=level, phase="explore", phase_start_action=action_num, budget=self.config.explore_budget)
        return self.state

    @property
    def phase(self) -> str:
        return self.state.phase if self.state is not None and self.config.enabled else "explore"

    def frozen_keys(self) -> tuple[str, ...]:
        if not self.config.enabled or self.state is None or self.state.phase != "exploit":
            return ()
        return FROZEN_KEYS

    def exploit_budget(self, benchmark: int | None) -> int:
        if benchmark is None or benchmark <= 0:
            return self.config.exploit_budget_default
        return max(self.config.exploit_budget_default, int(round(self.config.exploit_budget_factor * benchmark)))

    # -- observations -------------------------------------------------------

    def record_confidence(self, *texts: str) -> dict[str, int]:
        parsed = parse_confidence(*texts)
        if self.state is not None and parsed:
            self.state.confidence.update(parsed)
        if self.state is not None:
            for text in texts:
                if text and "p=" in text.lower() or (text and "p:" in text.lower()):
                    p = parse_p(text)
                    if p is not None:
                        self.state.p = p
        return parsed

    def record_p(self, value: Any) -> float | None:
        p = parse_p(value)
        if self.state is not None and p is not None:
            self.state.p = p
        return p

    def propose_revision(self, *, field_name: str, value: str, reason: str, action_num: int | None) -> None:
        if self.state is None:
            return
        self.state.proposed_revisions.append(
            {"field": field_name, "value": value, "reason": reason, "action_num": action_num, "attempt": self.state.cycle}
        )
        if len(self.state.proposed_revisions) > 30:
            del self.state.proposed_revisions[: len(self.state.proposed_revisions) - 30]

    def record_step(
        self,
        *,
        actions: list[str],
        hypothesis: str,
        prediction: str,
        verdict: str,
        action_num: int | None,
        board_changed: bool,
        game_over: bool,
        signature: str = "",
    ) -> None:
        if self.state is None:
            return
        state = self.state
        state.recent_signatures.append(signature or ("nochange" if not board_changed else "unknown"))
        if len(state.recent_signatures) > 20:
            del state.recent_signatures[: len(state.recent_signatures) - 20]
        window = state.recent_signatures[-self.config.stuck_repeats :]
        if len(window) >= self.config.stuck_repeats and len(set(window)) == 1 and window[0] != "unknown":
            state.stuck = True
            if state.phase == "exploit":
                kind = "no board change" if window[0] == "nochange" else "the identical board change"
                state.failure_notes.append(
                    f"the last {len(window)} action batches produced {kind} each time ({', '.join(actions[:4])} ...): a stuck loop"
                )
        state.reasons.append(
            ActionReason(
                actions=list(actions),
                hypothesis=hypothesis,
                prediction=prediction,
                verdict=verdict,
                action_num=action_num,
                phase=state.phase,
                board_changed=board_changed,
                game_over=game_over,
            )
        )
        if len(state.reasons) > 200:
            del state.reasons[: len(state.reasons) - 200]
        if verdict == "CONTRADICTED":
            state.contradictions_in_phase += 1
            if state.phase == "exploit":
                state.failure_notes.append(f"prediction '{prediction}' for {', '.join(actions[:4])} was contradicted")
        if game_over:
            state.game_overs_in_phase += 1
            if state.phase == "exploit":
                state.failure_notes.append(f"{', '.join(actions[:4])} led to GAME_OVER")

    # -- transitions --------------------------------------------------------

    def evaluate(self, *, action_num: int, benchmark: int | None, knowledge: dict[str, str]) -> str | None:
        """Apply at most one transition for this request; returns a description or None."""
        if not self.config.enabled or self.state is None:
            return None
        state = self.state
        state.benchmark_actions = benchmark
        used = state.actions_in_phase(action_num)
        threshold = self.config.confidence_threshold
        if state.phase == "explore":
            action_conf = state.confidence.get("action_model", 0)
            goal_conf = state.confidence.get("goal_model", 0)
            confident = action_conf >= threshold and goal_conf >= threshold
            p_ready = state.p is not None and state.p >= self.config.p_threshold
            if p_ready or confident or used >= state.budget:
                if p_ready:
                    reason = f"self-rated understanding p={state.p:.2f} >= {self.config.p_threshold:.2f}"
                elif confident:
                    reason = f"confidence reached (action model {action_conf}%, goal model {goal_conf}% >= {threshold}%)"
                else:
                    reason = f"exploration budget exhausted ({used}/{state.budget} actions)"
                state.frozen = {key: str(knowledge.get(key, "") or "") for key in FROZEN_KEYS}
                state.proposed_revisions = []
                state.stuck = False
                state.recent_signatures = []
                state.cycle += 1
                state.phase = "exploit"
                state.phase_start_action = action_num
                state.budget = self.exploit_budget(benchmark)
                state.contradictions_in_phase = 0
                state.game_overs_in_phase = 0
                state.failure_notes = []
                state.transition_reason = reason
                return f"explore -> exploit (attempt {state.cycle}): {reason}; budget {state.budget} actions"
            return None
        # exploit
        failure = None
        if state.contradictions_in_phase >= self.config.exploit_max_contradictions:
            failure = f"{state.contradictions_in_phase} prediction(s) were contradicted"
        elif state.game_overs_in_phase >= 1:
            failure = "the attempt hit GAME_OVER"
        elif state.stuck:
            failure = f"a stuck loop ({self.config.stuck_repeats} consecutive batches with the same or no board change)"
        elif used >= state.budget:
            failure = f"exploitation budget exhausted ({used}/{state.budget} actions) without completing the level"
        if failure is None:
            return None
        adopted = self.adopt_revisions(knowledge)
        state.phase = "explore"
        state.phase_start_action = action_num
        state.budget = self.config.re_explore_budget
        state.confidence = {}
        state.p = None
        state.contradictions_in_phase = 0
        state.game_overs_in_phase = 0
        state.stuck = False
        state.recent_signatures = []
        state.transition_reason = failure
        suffix = f"; adopted {adopted} proposed revision(s)" if adopted else ""
        return f"exploit -> re-explore (after attempt {state.cycle}): {failure}{suffix}; budget {state.budget} actions"

    def adopt_revisions(self, knowledge: dict[str, str]) -> int:
        """Merge the revisions proposed during the attempt into the knowledge, keeping the old text."""
        if self.state is None or not self.state.proposed_revisions:
            return 0
        latest: dict[str, dict[str, Any]] = {}
        for item in self.state.proposed_revisions:
            latest[str(item["field"])] = item
        for field_name, item in latest.items():
            knowledge[field_name] = augment_field(
                str(knowledge.get(field_name, "") or ""), str(item["value"]), str(item.get("reason", "") or ""), attempt=self.state.cycle
            )
        count = len(self.state.proposed_revisions)
        self.state.proposed_revisions = []
        return count

    # -- prompt -------------------------------------------------------------

    def prompt_lines(self, *, action_num: int) -> list[str]:
        if not self.config.enabled or self.state is None:
            return []
        state = self.state
        used = state.actions_in_phase(action_num)
        threshold = self.config.confidence_threshold
        lines: list[str] = []
        if state.phase == "explore":
            kind = "RE-EXPLORATION" if state.cycle > 0 else "EXPLORATION"
            lines.append(
                f"Phase: {kind} (learn the rules; do not try to solve yet). Actions used in this phase: {used} of {state.budget} on level {state.level}."
            )
            if state.cycle > 0:
                lines.append(f"- Exploitation attempt {state.cycle} failed: {state.transition_reason}.")
                for note in state.failure_notes[-3:]:
                    lines.append(f"  - {note}")
                lines.append("- Fix only the rules that failed; keep everything that held. Then report confidence again.")
            lines.append(
                "- Run probes that discriminate between hypotheses; every action carries the Hypothesis/Test/Prediction/Next note."
            )
            reported = f" Last reported p={state.p:.2f}." if state.p is not None else " No p reported yet."
            lines.append(
                f"- Rate yourself in every note with `p=<0..1>`: your probability that you understand the game well enough to attempt the level now. "
                f"When p >= {self.config.p_threshold:.2f} you switch to EXPLOITATION.{reported}"
            )
            lines.append(
                "- Be decisive: every exploration action is one fewer for solving. A slightly wrong model is cheap, because a failed attempt "
                f"brings you back here for a short {self.config.re_explore_budget}-action fix; a long exploration is not. The budget of {state.budget} actions is only a ceiling."
            )
        else:
            benchmark = f" (a human needs about {state.benchmark_actions} actions on this level)" if state.benchmark_actions else ""
            lines.append(
                f"Phase: EXPLOITATION attempt {state.cycle} (solve the level with the committed rules). Actions used in this attempt: {used} of {state.budget}{benchmark}."
            )
            lines.append(f"- Committed because: {state.transition_reason}.")
            lines.append("- Committed hypotheses (frozen: edits to the World/Goal/Action model are ignored until this attempt ends):")
            for key, label in (("world_model", "World model"), ("goal_model", "Goal model"), ("action_model", "Action model")):
                value = state.frozen.get(key, "")
                lines.append(f"  - {label}: {value or '(empty)'}")
            lines.append(
                "- Execute the plan: batch the actions, act on every request, no probing, no re-derivation of the rules. Keep writing the note with a concrete Prediction so failures are detected."
            )
            lines.append(
                "- If you learn something that changes a committed field, pass it in `note(...)` with a `why`: it is recorded as a proposed revision "
                "(shown below) and merged into the field, together with the old text and your reason, when this attempt ends."
            )
            if state.proposed_revisions:
                lines.append("- Revisions proposed during this attempt (not yet adopted):")
                for item in state.proposed_revisions[-4:]:
                    lines.append(f"  - {item['field']}: {str(item['value'])[:200]} (why: {str(item.get('reason', ''))[:120] or 'not given'})")
            lines.append(
                f"- This attempt ends and a short re-exploration ({self.config.re_explore_budget} actions) starts if {self.config.exploit_max_contradictions} prediction(s) are contradicted, "
                f"a GAME_OVER happens, {self.config.stuck_repeats} consecutive batches produce the same or no board change, or the budget runs out."
            )
            if state.contradictions_in_phase:
                lines.append(f"- Contradicted predictions so far in this attempt: {state.contradictions_in_phase}.")
        return lines

    def reason_lines(self) -> list[str]:
        if not self.config.enabled or self.state is None or not self.state.reasons or self.config.recent_reasons <= 0:
            return []
        recent = self.state.reasons[-self.config.recent_reasons :]
        return ["Recent actions with the reasons you gave (oldest first):", *[f"- {item.line()}" for item in recent]]
