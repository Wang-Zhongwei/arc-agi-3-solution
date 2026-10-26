"""GRPO mode for the harness: records what training needs while an episode is played, and says when it ends.

On when ARC3_RL_RECORD=1 (configs/grpo/*.json). The model sees none of it; the episode is the evaluation harness
with an untrimmed context (the config switches trimming and turn yield off) and an end.

- Every gameplay request also asks vLLM for the token ids it saw and sampled and their processed logprobs
  (`return_token_ids`, `logprobs`), and its `max_tokens` is lowered to what the episode's context budget leaves, so
  no sequence passes the budget (the thinking budget stays on: it is set from the default `max_tokens`).
- One JSON line per request in <job>/rl/<stem>.jsonl ("request"): the prompt's length, the sampled ids and their
  logprobs, the finish reason, the longest common prefix of this prompt with the previous prompt + reply (a reply
  rebuilt differently from what was sampled shows up as a prefix shorter than the previous sequence), and the levels
  completed and actions taken when it was sent.
- At the end one line ("episode"): the last prompt's ids, TAAF's per-level action counts, the human counts, the
  levels cleared, the budget and why the episode ended; the last request's images go to <job>/rl/<stem>/NNNN.png,
  in the order the chat template renders them.
- The episode ends (the solver's should_stop) once ARC3_RL_STOP_AFTER_LEVELS levels are cleared, or when the budget
  cannot hold another request.

Budget per episode: min(ARC3_RL_BUDGET_CAP, ARC3_RL_BUDGET_PER_HUMAN_ACTION x human actions of the levels played),
in context tokens (decisions.md).
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


def enabled() -> bool:
    return os.environ.get("ARC3_RL_RECORD", "").strip().lower() in ("1", "true", "on", "yes")


STOP_AFTER_LEVELS = _env_int("ARC3_RL_STOP_AFTER_LEVELS", 2)
BUDGET_PER_HUMAN_ACTION = _env_int("ARC3_RL_BUDGET_PER_HUMAN_ACTION", 1750)
BUDGET_CAP = _env_int("ARC3_RL_BUDGET_CAP", 130_000)
REQUEST_MARGIN = _env_int("ARC3_RL_REQUEST_MARGIN", 256)  # what the anchored estimate may be short by (p99 +1.3%)
MIN_REQUEST_TOKENS = _env_int("ARC3_RL_MIN_REQUEST_TOKENS", 1024)  # less left than this ends the episode


def episode_budget(base_actions_per_level: list[int] | None, levels: int = STOP_AFTER_LEVELS) -> int:
    """Context tokens one episode may reach: min(cap, per human action x human actions of the first `levels`)."""
    if not base_actions_per_level:
        return BUDGET_CAP
    return min(BUDGET_CAP, BUDGET_PER_HUMAN_ACTION * sum(int(a) for a in base_actions_per_level[:levels]))


def common_prefix(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two id lists."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # a[:lo] == b[:lo] holds, a[:hi] == b[:hi] does not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def message_images(messages: list[dict[str, Any]]) -> list[bytes]:
    """Every image of a request, in the order the chat template renders them (messages, then content parts)."""
    out = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = (part.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                raise ValueError(f"image is not inline data: {url[:40]}")
            out.append(base64.b64decode(url.split(",", 1)[1]))
    return out


class EpisodeRecorder:
    """One game's GRPO record. The solver creates it, sets the budget and the game callbacks, and calls finish();
    the analyzer calls request_cap() / before_request() / after_response() around each gameplay request."""

    def __init__(self, path: Path, budget: int, levels_completed: Callable[[], int], action_count: Callable[[], int],
                 stop_after_levels: int = STOP_AFTER_LEVELS):
        self.path = Path(path)
        self.image_dir = self.path.with_suffix("")
        self.budget = int(budget)
        self.levels_completed = levels_completed
        self.action_count = action_count
        self.stop_after_levels = stop_after_levels
        self.requests = 0
        self.context_tokens = 0  # last request's real prompt + reply
        self.exhausted = False
        self.stop_reason: str | None = None
        self._previous: list[int] = []  # last prompt + reply ids
        self._last_prompt: list[int] = []
        self._last_messages: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._finished = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def should_stop(self) -> bool:
        if self.levels_completed() >= self.stop_after_levels:
            self.stop_reason = self.stop_reason or f"cleared {self.stop_after_levels} levels"
            return True
        if self.exhausted or self.budget - self.context_tokens - REQUEST_MARGIN < MIN_REQUEST_TOKENS:
            self.exhausted = True
            self.stop_reason = self.stop_reason or "budget"
            return True
        return False

    def request_cap(self, estimate: int) -> int | None:
        """max_tokens for the next request from the anchored estimate of its prompt, or None when the budget
        cannot hold it (the episode then ends instead of sending)."""
        left = self.budget - max(int(estimate), self.context_tokens) - REQUEST_MARGIN
        if left < MIN_REQUEST_TOKENS:
            self.exhausted = True
            self.stop_reason = self.stop_reason or "budget"
            return None
        return left

    @staticmethod
    def augment(payload: dict[str, Any], max_tokens: int | None) -> None:
        """Ask for ids and logprobs, and lower max_tokens to the budget's cap."""
        payload["logprobs"] = True
        payload["top_logprobs"] = 0
        payload["return_token_ids"] = True
        if max_tokens is not None:
            current = payload.get("max_tokens")
            payload["max_tokens"] = int(max_tokens) if current is None else min(int(current), int(max_tokens))

    @staticmethod
    def extract(response: dict[str, Any]) -> dict[str, Any]:
        """prompt_token_ids, the reply's token_ids and their logprobs (None where vLLM gave none)."""
        choice = (response.get("choices") or [{}])[0]
        content = ((choice.get("logprobs") or {}).get("content")) or []
        return dict(prompt_token_ids=response.get("prompt_token_ids"), token_ids=choice.get("token_ids"),
                    logprobs=[c.get("logprob") for c in content])

    def after_response(self, rl: dict[str, Any] | None, *, messages: list[dict[str, Any]], finish_reason: str,
                       max_tokens: int | None, usage: dict[str, Any] | None) -> None:
        prompt = list((rl or {}).get("prompt_token_ids") or [])
        ids = list((rl or {}).get("token_ids") or [])
        logprobs = list((rl or {}).get("logprobs") or [])
        with self._lock:
            row = dict(kind="request", i=self.requests, t=round(time.time(), 2), level=self.levels_completed() + 1,
                       actions=self.action_count(), prompt_len=len(prompt), lcp=common_prefix(prompt, self._previous),
                       previous_len=len(self._previous), ids=ids, logprobs=logprobs, finish=finish_reason,
                       max_tokens=max_tokens, usage=usage or {})
            if not prompt or not ids or len(logprobs) != len(ids):
                row["error"] = (f"missing ids or logprobs: prompt {len(prompt)}, ids {len(ids)}, "
                                f"logprobs {len(logprobs)}")
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.requests += 1
            self._previous = prompt + ids
            self._last_prompt = prompt
            self._last_messages = messages
            self.context_tokens = len(prompt) + len(ids) if prompt else self.context_tokens

    def finish(self, run: Any, note: str | None = None) -> None:
        """The episode line and the last request's images. `run` is TAAF's GameRun."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            images = []
            error = None
            try:
                images = message_images(self._last_messages)
            except Exception as exc:  # noqa: BLE001 - an episode without images is dropped later, not fatal here
                error = f"images: {type(exc).__name__}: {exc}"
            self.image_dir.mkdir(parents=True, exist_ok=True)
            for i, data in enumerate(images):
                (self.image_dir / f"{i:04d}.png").write_bytes(data)
            levels = int(getattr(run, "levels_completed", 0) or 0)
            if self.stop_reason is None:
                self.stop_reason = f"cleared {levels} levels" if levels >= self.stop_after_levels else (
                    note or str(getattr(run, "state", "") or "ended"))
            row = dict(kind="episode", game_id=getattr(run, "game_id", None), requests=self.requests,
                       budget=self.budget, context_tokens=self.context_tokens, stop_reason=self.stop_reason,
                       levels_completed=levels, number_of_levels=getattr(run, "number_of_levels", None),
                       actions_per_level=list(getattr(run, "actions_per_level", []) or []),
                       base_actions_per_level=list(getattr(run, "base_actions_per_level", None) or []),
                       state=str(getattr(run, "state", "")), images=len(images), final_prompt_ids=self._last_prompt,
                       solver_note=getattr(run, "solver_note", None))
            if error:
                row["error"] = error
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
