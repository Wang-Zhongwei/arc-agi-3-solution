"""Prefix-cache reuse across the previous reply (#3).

A request reuses the cache past the previous prompt only if every token of the
previous reply re-renders exactly as it was generated. The hybrid model's
recurrent state is checkpointed only at the end of prefill and the end of
decode, so one differing character anywhere in the reply sends the server back
to the prompt-end checkpoint, and the whole reply is prefilled again.

This module holds two things:
- the outer trim the chat template applies, which is the only change history
  may make to reply text without breaking reuse;
- a probe (ARC3_PREFIX_REUSE_DIAG) that compares each previous reply with the
  history message that re-renders it, next to the cache hit the server
  reported, so a run's log separates harness-side mismatches from cache
  pressure.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any

# The served config's --page-size: the server caches, and reports, whole pages.
PAGE_TOKENS = 64
# Characters of context printed on each side of a divergence.
_NEAR_CHARS = 24
# Requests between two running-totals lines.
_TOTALS_EVERY = 100


def template_trimmed_text(value: Any) -> str:
    """Message text with only the outer trim the chat template applies.

    The Qwen3.x template renders `reasoning_content|trim` and `content|trim`,
    so this changes nothing the server renders. Collapsing blank lines, as the
    transcript does, makes the re-rendered reply differ from what was generated.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        # the template's render_content concatenates text parts as they are
        parts = [
            str(item.get("text", ""))
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "".join(parts).strip()
    return ""


def reply_reasoning(message: dict[str, Any]) -> Any:
    """A reply's reasoning field: `reasoning`, else `reasoning_content`."""
    reasoning = message.get("reasoning")
    if reasoning in (None, ""):
        reasoning = message.get("reasoning_content", "")
    return reasoning


def _tool_call_arguments(function: dict[str, Any]) -> Any:
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        if not arguments:
            return {}
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"<unparsed>": arguments}
    return arguments


def render_assistant_turn(reasoning: Any, content: Any, tool_calls: Any) -> str:
    """An assistant turn as the Qwen3.x template renders it with preserve_thinking.

    The text between `<|im_start|>assistant\\n` and `<|im_end|>`. The server
    parses tool-call parameters by removing one newline on each side, and the
    template adds them back, so a parameter value re-renders as generated
    only when it is stored exactly as the server returned it.
    """
    content_text = template_trimmed_text(content)
    reasoning_text = reasoning.strip() if isinstance(reasoning, str) else ""
    parts = ["<think>\n", reasoning_text, "\n</think>\n\n", content_text]
    calls = tool_calls if isinstance(tool_calls, list) else []
    for index, call in enumerate(calls):
        function = call if isinstance(call, dict) else {}
        if isinstance(function.get("function"), dict):
            function = function["function"]
        if index > 0:
            parts.append("\n<tool_call>\n<function=")
        elif content_text:
            parts.append("\n\n<tool_call>\n<function=")
        else:
            parts.append("<tool_call>\n<function=")
        parts.append(f"{function.get('name', '')}>\n")
        arguments = _tool_call_arguments(function)
        if isinstance(arguments, dict):
            for name, value in arguments.items():
                rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                parts.append(f"<parameter={name}>\n{rendered}\n</parameter>\n")
        parts.append("</function>\n</tool_call>")
    return "".join(parts)


def render_reply(message: dict[str, Any]) -> str:
    """What the model generated for a reply, rebuilt from the parsed fields."""
    return render_assistant_turn(
        reply_reasoning(message), message.get("content"), message.get("tool_calls")
    )


def render_history_message(message: dict[str, Any]) -> str:
    """What the template renders for a stored message: it reads only `reasoning_content`."""
    return render_assistant_turn(
        message.get("reasoning_content"), message.get("content"), message.get("tool_calls")
    )


def _first_difference(left: str, right: str) -> int:
    """Index of the first differing character, or -1 when the strings are equal."""
    if left == right:
        return -1
    low, high = 0, min(len(left), len(right))
    # binary search on slice equality: a few memcmp calls instead of a char loop
    while low < high:
        middle = (low + high + 1) // 2
        if left[:middle] == right[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def cached_tokens(usage: dict[str, Any] | None) -> int | None:
    """`usage.prompt_tokens_details.cached_tokens` (SGLang --enable-cache-report)."""
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return _int_or_none(details.get("cached_tokens"))
    return _int_or_none(usage.get("cached_tokens"))


def cache_outcome(prompt_tokens: int | None, output_tokens: int | None, cached: int | None) -> str:
    """Where this request's cache hit ended, relative to the previous request.

    reply_end: the previous reply was reused; prompt_end: the hit stopped at
    the previous prompt's last page, so the whole reply was prefilled again;
    partial: in between; below_prompt: less than the previous prompt.
    """
    if prompt_tokens is None or output_tokens is None or cached is None:
        return "unknown"
    reply_end = (prompt_tokens + output_tokens) // PAGE_TOKENS * PAGE_TOKENS
    prompt_end = prompt_tokens // PAGE_TOKENS * PAGE_TOKENS
    if cached >= reply_end:
        return "reply_end"
    if cached == prompt_end:
        return "prompt_end"
    if cached > prompt_end:
        return "partial"
    return "below_prompt"


@dataclass
class _Reply:
    stored: dict[str, Any] | None  # the message history holds for it, or None
    replaced: bool  # history holds a stand-in (a stub), not the reply
    text: str
    request_messages: list[dict[str, Any]]
    prompt_tokens: int | None
    output_tokens: int | None
    finished_at: float


_TOTALS_LOCK = threading.Lock()
_TOTALS: dict[str, int] = {}


def _count(increments: dict[str, int]) -> dict[str, int] | None:
    """Add to the process-wide totals; return a copy every _TOTALS_EVERY requests."""
    with _TOTALS_LOCK:
        for key, amount in increments.items():
            _TOTALS[key] = _TOTALS.get(key, 0) + amount
        _TOTALS["requests"] = _TOTALS.get("requests", 0) + 1
        if _TOTALS["requests"] % _TOTALS_EVERY == 0:
            return dict(_TOTALS)
    return None


class PrefixReuseProbe:
    """Per-stream record of the previous reply, checked against the next request.

    Call `check` with the messages a request is about to send, `report` with
    that check and the response's usage, and `note_reply` once the reply's
    message is in history (stored=None when nothing was kept).

    history: match (the stored message re-renders as the reply), mismatch
    (it does not; diverge_at and the text on each side say where), replaced
    (a stub stands in for it), absent (not in this request: rolled back or
    trimmed), none (no previous reply on this stream).
    prefix: same when every message of the previous request is still in
    place, so only the reply can explain a hit that stops at its prompt.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._reply: _Reply | None = None
        self._request_messages: list[dict[str, Any]] = []

    def check(self, messages: list[dict[str, Any]], now: float) -> dict[str, Any]:
        """Compare the previous reply with its history message, before the request.

        The reply stays until `report` runs, so a request retried after a
        failure (a context overflow, a timeout) is still checked against it.
        """
        reply = self._reply
        self._request_messages = list(messages)
        fields: dict[str, Any] = {"history": "none"}
        if reply is None:
            return fields
        fields.update(
            prev_prompt=reply.prompt_tokens,
            prev_output=reply.output_tokens,
            gap_s=f"{max(0.0, now - reply.finished_at):.2f}",
        )
        sent_before = reply.request_messages
        fields["prefix"] = (
            "same"
            if len(messages) >= len(sent_before)
            and all(a is b or a == b for a, b in zip(sent_before, messages))
            else "changed"
        )
        stored = reply.stored
        if stored is None or not any(message is stored for message in messages):
            fields["history"] = "absent"
            return fields
        if reply.replaced:
            fields["history"] = "replaced"
            return fields
        history_text = render_history_message(stored)
        at = _first_difference(reply.text, history_text)
        if at < 0:
            fields["history"] = "match"
            return fields
        fields.update(
            history="mismatch",
            diverge_at=f"{at}/{len(reply.text)}",
            reply_sha=_sha(reply.text),
            history_sha=_sha(history_text),
            reply_near=repr(reply.text[max(0, at - _NEAR_CHARS): at + _NEAR_CHARS]),
            history_near=repr(history_text[max(0, at - _NEAR_CHARS): at + _NEAR_CHARS]),
        )
        return fields

    def report(self, fields: dict[str, Any], usage: dict[str, Any] | None, **context: Any) -> list[str]:
        """Log lines for one request: the check, the cache hit, and periodic totals.

        lost: previous-reply tokens prefilled again, counted only when the
        previous request's messages are all still in place (prefix=same).
        """
        # a request completed after the reply: the next one follows a new reply
        self._reply = None
        cached = cached_tokens(usage)
        history = fields.get("history")
        outcome = "first" if history == "none" else cache_outcome(
            fields.get("prev_prompt"), fields.get("prev_output"), cached
        )
        lost = 0
        if outcome in ("prompt_end", "partial", "below_prompt") and fields.get("prefix") == "same":
            prompt_end = fields["prev_prompt"] // PAGE_TOKENS * PAGE_TOKENS
            reply_end = (fields["prev_prompt"] + fields["prev_output"]) // PAGE_TOKENS * PAGE_TOKENS
            lost = max(0, reply_end - max(cached or 0, prompt_end))
        # space-free fields first, so a line splits on whitespace up to the
        # quoted text around a divergence
        ordered = {
            **context,
            "history": history,
            "prefix": fields.get("prefix", "-"),
            "cache": outcome,
            "cached": cached,
            "prompt": _int_or_none((usage or {}).get("prompt_tokens")),
            **{k: fields[k] for k in ("prev_prompt", "prev_output", "gap_s") if k in fields},
            "lost": lost,
            **{k: v for k, v in fields.items() if k not in ("history", "prefix", "prev_prompt", "prev_output", "gap_s")},
        }
        lines = ["PREFIXDIAG " + self.name + " " + " ".join(f"{k}={v}" for k, v in ordered.items())]
        totals = _count({
            f"history_{history}": 1,
            f"cache_{outcome}": 1,
            f"{history}_{outcome}": 1,
            "lost_tokens": lost,
            f"lost_tokens_{history}": lost,
        })
        if totals is not None:
            lines.append(
                "PREFIXDIAG totals " + " ".join(f"{k}={v}" for k, v in sorted(totals.items()))
            )
        return lines

    def note_reply(
        self,
        *,
        reply: dict[str, Any],
        stored: dict[str, Any] | None,
        usage: dict[str, Any] | None,
        now: float,
        replaced: bool = False,
    ) -> None:
        """Remember the reply the request just produced and the message that holds it."""
        usage = usage if isinstance(usage, dict) else {}
        self._reply = _Reply(
            stored=stored,
            replaced=replaced,
            text=render_reply(reply),
            request_messages=self._request_messages,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            finished_at=now,
        )
