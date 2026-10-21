# Modified from Tycho tycho/serving/public_backends.py and tycho/serving/llm_client.py @f68912a
# (Apache-2.0, copyright 2026 Jens Lehmann, Andrei Aioanei, Sahar Vahdati).
# Changes: only the transport helpers are carried (_post_json, normalize_usage, the retry loop and
# the context-limit classifier); the Anthropic/Responses backends, plugin extensions, recording and
# telemetry hooks are dropped. See PROVENANCE.md.
"""HTTP transport helpers for an OpenAI-compatible chat endpoint.

`_post_json` is the one place a request leaves the process; the retry policy is Tycho's
(exponential back-off with jitter under a wall-clock budget, retrying only on the status codes
and connection errors that a serving stack recovers from).
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Callable

Retry = Callable[[Callable[[], dict]], dict]


def normalize_usage(usage: dict | None, *, input_tokens_include_cache: bool = False) -> dict:
    """Normalize provider usage into fresh input, output, cache-read, and cache-write tokens."""
    usage = usage or {}
    cache_read = int(
        usage.get("cacheReadInputTokens")
        or usage.get("cache_read_input_tokens")
        or ((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        or 0
    )
    cache_write = int(
        usage.get("cacheWriteInputTokens") or usage.get("cache_creation_input_tokens") or 0
    )
    if "inputTokens" in usage:
        total = int(usage.get("inputTokens") or 0)
        fresh = max(0, total - cache_read - cache_write) if input_tokens_include_cache else total
    elif "input_tokens" in usage:
        fresh = int(usage.get("input_tokens") or 0)
    else:
        fresh = max(0, int(usage.get("prompt_tokens") or 0) - cache_read)
    return {
        "in": fresh,
        "out": int(
            usage.get("outputTokens")
            or usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        ),
        "cache_read": cache_read,
        "cache_write": cache_write,
    }


def _post_json(url: str, body: dict, headers: dict, timeout: int, retry) -> dict:
    data = json.dumps(body).encode()

    def send() -> dict:
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            exc._tycho_body = detail  # type: ignore[attr-defined]
            if detail:
                exc.msg = f"{exc.reason}; provider response: {detail[:2000]}"
            raise

    return retry(send)


_CONTEXT_MARKERS = (
    "context length", "context window", "maximum context", "max context", "input length",
    "too many tokens", "token limit", "prompt is too long", "prompt too long",
    "exceeds maximum", "exceeded maximum", "max_model_len", "request too large",
)


def is_context_limit_error(error: Exception) -> bool:
    body = getattr(error, "_tycho_body", "") or ""
    text = f"{type(error).__name__}: {error} {body}".lower()
    return any(marker in text for marker in _CONTEXT_MARKERS)


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 409, 429) or 500 <= error.code < 600
    if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    return False


def _with_retry(fn, *, base: float = 4.0, cap: float = 30.0, is_retryable=None):
    retryable = is_retryable or _is_retryable
    budget = float(os.environ.get("LLM_RETRY_BUDGET_S", "1200"))
    started = time.monotonic()
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - provider-neutral retry boundary
            elapsed = time.monotonic() - started
            if not retryable(error) or elapsed >= budget:
                raise
            delay = min(cap, base * (2 ** min(attempt, 5))) * random.uniform(0.75, 1.25)
            if elapsed + delay > budget:
                raise
            time.sleep(delay)
            attempt += 1
