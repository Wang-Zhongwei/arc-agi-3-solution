"""Model client: one structured-JSON request per role, engine-agnostic.

Talks to any OpenAI-compatible `/v1/chat/completions` (SGLang on the Spark and Kaggle; vLLM as
the fallback). Constrained decoding is applied only to the final answer through
`response_format={"type":"json_schema",...}` (verified live on the frozen vLLM 2026-09-10: with
`enable_thinking` the model reasons first, then emits schema-valid JSON). Reasoning text is
never persisted into memory; it is written to the per-game trace only.

Sampling follows the Qwen3.8 model card: non-thinking temperature 0.7 / top_p 0.8 / top_k 20 /
presence_penalty 1.5; thinking 1.0 / 0.95 / 20.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .vendor.tycho.serving import _post_json, _with_retry, is_context_limit_error, normalize_usage

NON_THINKING_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def served_models(base_url: str, api_key: str = "EMPTY", timeout: float = 10.0) -> list[str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return [m.get("id") for m in payload.get("data", [])]


@dataclass
class CallResult:
    obj: Optional[dict[str, Any]]
    text: str
    reasoning: str
    usage: dict[str, int]
    latency_s: float
    finish: str
    error: Optional[str] = None
    attempts: int = 1
    raw: dict[str, Any] = field(default_factory=dict)


class ModelClient:
    """Structured calls with one repair retry. `validate(obj) -> None | str` returns an error
    message the model gets to fix once; the second failure is reported, not retried."""

    def __init__(self, *, base_url: str, model: str, api_key: str = "EMPTY", thinking: bool = False,
                 timeout: float = 1800.0, sampling: Optional[dict[str, Any]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        # OpenRouter is a gateway, not an engine: it owns its own switch for reasoning and its own
        # routing, and the engine-level knobs below do not reach the provider intact.
        self.openrouter = "openrouter.ai" in self.base_url
        self.model = model
        self.api_key = api_key
        self.thinking = bool(thinking)
        self.timeout = int(timeout)
        self.sampling = dict(sampling) if sampling else None
        self.total_usage = {"in": 0, "out": 0}
        self.calls = 0
        # Real money when the endpoint is a paid gateway. OpenRouter reports the charge for each
        # call in `usage.cost`, so this is what was actually billed, not an estimate from a price
        # list that may be stale.
        self.total_cost = 0.0
        self.cost_limit: Optional[float] = None
        # Measured output speed, and the wall clock a single answer is allowed to take. A token
        # budget is really a time budget in disguise, and the exchange rate differs by two orders
        # of magnitude between endpoints: through OpenRouter 2800 tokens is a few seconds, while on
        # the GB10 shared with another job it is 2.2 tokens a second, so the same ceiling bought a
        # 1251-second call that was then truncated and repeated. Budgets tuned for one endpoint are
        # ruinous on the other, so the harness stops tuning tokens and starts tuning seconds.
        self.tok_per_s: Optional[float] = None
        self.seconds_per_call: Optional[float] = None
        # The floor has to fit the longest answer a role actually needs, not the shortest. At 384
        # the time budget trimmed `decide` until it truncated on BOTH attempts and lost the turn
        # anyway -- 365 seconds spent to produce nothing, which is the exact failure the budget
        # exists to prevent. A slow call that finishes beats a fast one that does not.
        self.min_tokens = 768

    # -- transport -----------------------------------------------------------------------------

    def budget_for(self, max_tokens: int) -> int:
        """Trim a role's token ceiling to what fits the time this endpoint is allowed per answer.

        Typical answers here run 330 to 420 tokens, so on a slow server this changes nothing for
        them; what it stops is the occasional ramble that spends twenty minutes and is cut off at
        the end anyway. On a fast endpoint the computed cap is far above any role's ceiling and
        this does nothing at all.
        """
        if not self.seconds_per_call:
            return int(max_tokens)
        ceiling = int(max_tokens)
        floor = min(ceiling, max(self.min_tokens, ceiling // 2))
        if not self.tok_per_s:
            # The first call of a run, before any speed is known. The operator asked for a time
            # budget, so the trim applies from the start: untrimmed, `orient` took its full 2400
            # and ran into the ceiling at 437 s, and only the half-size repair recovered the turn.
            # One call at the floor measures the speed that sizes every call after it.
            return floor
        fits = int(self.seconds_per_call * self.tok_per_s)
        # `max_tokens` is a ceiling, not a target: a role that answers in 330 tokens costs the
        # same whether the ceiling is 768 or 2800, so trimming it buys almost nothing and risks
        # everything. Measured live on ls20 on 2026-09-11, at 2.1 tok/s with a 240-second budget:
        # `fits` was 504, every role in ROLE_PLAN was trimmed to the flat 768 floor, and the full
        # `decide` form -- twelve required memory-patch arrays plus a batch and six step
        # predictions -- came back finish=length at 768 on BOTH attempts. 352 s, then 694 s, for
        # nothing, on the turn after a failure. That is the exact failure this budget exists to
        # prevent, caused by the budget. So the floor is half the role's own ceiling now: a
        # runaway is still cut off, and a role that needs room to finish keeps it.
        # The role's own ceiling is the outer bound in BOTH directions. The floor exists to stop
        # the time budget trimming a role into guaranteed truncation, not to hand a small role
        # room it was never meant to have: on the GB10 (2.1 tok/s, 240 s a call) `fits` is 504, so
        # the old `max(min_tokens, ...)` returned 768 for every role -- it raised `goal` from 400
        # and flattened ORIENT's 2400 and RECOVER's 2800 to the same number, which made every
        # per-role budget in modes.ROLE_PLAN inert on the only endpoint that needed them.
        return max(floor, min(ceiling, fits))

    def note_speed(self, out_tokens: int, latency_s: float) -> None:
        """Fold one observation into the running output speed."""
        if out_tokens <= 0 or latency_s <= 0:
            return
        rate = out_tokens / latency_s
        self.tok_per_s = rate if self.tok_per_s is None else (0.7 * self.tok_per_s + 0.3 * rate)

    def over_budget(self) -> bool:
        """Has this client spent what it was allowed to? Checked before a call is sent, so the
        limit is a ceiling on what is spent rather than on what is noticed afterwards."""
        return self.cost_limit is not None and self.total_cost >= self.cost_limit

    def _body(self, messages: list[dict[str, Any]], schema: dict[str, Any], *, max_tokens: int,
              thinking: Optional[bool], schema_name: str) -> dict[str, Any]:
        think = self.thinking if thinking is None else bool(thinking)
        sampling = self.sampling or (THINKING_SAMPLING if think else NON_THINKING_SAMPLING)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": think},
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}},
        }
        body.update(sampling)
        if self.openrouter:
            # Measured against qwen/qwen3.8-27b on 2026-09-11, and the difference is not cosmetic:
            # `chat_template_kwargs` alone left 68 reasoning tokens on the bill, `reasoning.exclude`
            # hid the reasoning and still generated and charged for 79 of it, and only
            # `reasoning.enabled=false` produced none (13 output tokens, $0.00005, 3.7 s).
            body["reasoning"] = {"enabled": think}
            # Only route to providers that honour response_format and the sampling knobs; a
            # provider that quietly drops the schema would hand the validator free text.
            body["provider"] = {"require_parameters": True}
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        return _post_json(f"{self.base_url}/chat/completions", body, headers, self.timeout, _with_retry)

    # -- the one public call -------------------------------------------------------------------

    def call_json(self, *, role: str, system: str, user: str, schema: dict[str, Any], max_tokens: int,
                  thinking: Optional[bool] = None, validate: Optional[Callable[[dict[str, Any]], Optional[str]]] = None,
                  on_attempt: Optional[Callable[[dict[str, Any]], None]] = None) -> CallResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        started = time.monotonic()
        last: Optional[CallResult] = None
        cut_off = False
        for attempt in (1, 2):
            # Per attempt, not since the call began: on a repair round the second answer's tokens
            # were being divided by both attempts' wall clock, which halved the measured speed,
            # which trimmed the next budget, which made truncation (and another repair) likelier.
            attempt_started = time.monotonic()
            ceiling = self.budget_for(max_tokens)
            if cut_off:
                # The first answer ran into the ceiling, so the second will too -- the model fills
                # the room it is given, and the repair note asking for fewer and shorter entries
                # does not change that. Measured on ls20: 631 s truncated, then 1266 s truncated
                # again, half an hour on one role. Halve the room so the harness finds out fast
                # and falls back to its own plan with most of the turn still left.
                ceiling = max(256, ceiling // 2)
            body = self._body(messages, schema, max_tokens=ceiling,
                               thinking=thinking, schema_name=role)
            try:
                response = self._post(body)
            except Exception as exc:  # noqa: BLE001 - the caller decides what a failed request means
                kind = "context_limit" if is_context_limit_error(exc) else "request_error"
                return CallResult(obj=None, text="", reasoning="", usage={"in": 0, "out": 0},
                                  latency_s=time.monotonic() - started, finish="", error=f"{kind}: {exc}"[:500],
                                  attempts=attempt)
            self.calls += 1
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            usage = normalize_usage(response.get("usage"))
            self.total_usage["in"] += usage["in"]
            self.total_usage["out"] += usage["out"]
            self.note_speed(int(usage.get("out") or 0), time.monotonic() - attempt_started)
            try:
                self.total_cost += float((response.get("usage") or {}).get("cost") or 0.0)
            except (TypeError, ValueError):
                pass
            finish = str(choice.get("finish_reason") or "")
            obj: Optional[dict[str, Any]] = None
            error: Optional[str] = None
            try:
                obj = json.loads(text) if text.strip() else None
                if obj is None:
                    # An empty body is a failure like any other, and it has to say so: without a
                    # reason the caller logs `error: None` and the repair round asks the model to
                    # fix "None".
                    error = ("your answer was cut off before any JSON was produced"
                             if finish == "length" else "the reply was empty")
                elif not isinstance(obj, dict):
                    error = "response was not a JSON object"
                    obj = None
            except json.JSONDecodeError as exc:
                error = f"invalid JSON: {exc}"
                if finish == "length":
                    error = "your answer was cut off at the token limit; use fewer and shorter entries"
            if obj is not None and validate is not None:
                problem = validate(obj)
                if problem:
                    error = problem
                    obj = None
            last = CallResult(obj=obj, text=text, reasoning=reasoning, usage=usage,
                              latency_s=time.monotonic() - started, finish=finish, error=error,
                              attempts=attempt, raw={"finish": finish, "reasoning_chars": len(reasoning)})
            if on_attempt is not None:
                try:
                    on_attempt({"role": role, "attempt": attempt, "usage": usage, "finish": finish,
                                "reasoning": reasoning, "text": text, "error": error,
                                "latency_s": last.latency_s})
                except Exception:  # noqa: BLE001
                    pass
            if obj is not None:
                return last
            cut_off = finish == "length"
            # One repair round: hand back the validator's complaint with the bad answer.
            messages = messages[:2] + [
                {"role": "assistant", "content": text[:4000]},
                {"role": "user", "content": f"Your answer was rejected: {error}. Reply again with JSON that satisfies the schema and the rule above; change only what the rejection names."},
            ]
        assert last is not None
        return last
