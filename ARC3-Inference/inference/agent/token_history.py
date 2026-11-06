"""ARC3_TOKEN_EXACT_HISTORY: history sends the model's sampled token ids, not re-rendered text.

A stored reply keeps its ids beside the parsed fields. The fields stay what every decision, trim and estimate
reads; at send time a reply whose fields are unchanged goes out as a marker with its ids, which the patched server
splices into the prompt (inference/framework/lordhansolo/token_exact/). A reply the harness has since edited goes out
as text: its ids no longer describe it.
"""
import array
import hashlib
import json
from typing import Any, Sequence

MARKER_OPEN = "ARC3TOKENS:"
MARKER_CLOSE = ""
REPLY_TOKENS_KEY = "_reply_tokens"
REPLY_SEAL_KEY = "_reply_tokens_seal"
REPLY_KEYS = (REPLY_TOKENS_KEY, REPLY_SEAL_KEY)
_SEALED_FIELDS = ("content", "reasoning", "reasoning_content", "tool_calls")


def _seal(message: dict[str, Any]) -> str:
    fields = {key: message.get(key) for key in _SEALED_FIELDS}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


def seal_reply_tokens(message: dict[str, Any]) -> None:
    """Fix the fields the stored ids describe. Call once the message is complete."""
    if REPLY_TOKENS_KEY in message:
        message[REPLY_SEAL_KEY] = _seal(message)


def strip_reply_tokens(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in message.items() if key not in REPLY_KEYS}
        if REPLY_TOKENS_KEY in message else message
        for message in messages
    ]


def render_reply_tokens(
    messages: list[dict[str, Any]], reasoning_keys: Sequence[str]
) -> tuple[list[dict[str, Any]], str, int]:
    """Wire messages, the last reply's ids as sent, and how many replies fell back to text."""
    wire: list[dict[str, Any]] = []
    last_sent = ""
    stale = 0
    for message in messages:
        tokens = message.get(REPLY_TOKENS_KEY)
        if tokens is None:
            wire.append(message)
        elif not tokens or message.get(REPLY_SEAL_KEY) != _seal(message):
            stale += 1
            wire.append({key: value for key, value in message.items() if key not in REPLY_KEYS})
        else:
            marker = f"{MARKER_OPEN}{tokens}{MARKER_CLOSE}"
            wire.append({"role": "assistant", "content": "", **{key: marker for key in reasoning_keys}})
            last_sent = tokens
    return wire, last_sent, stale


def prompt_holds_reply(prompt: Sequence[int], tokens: str) -> bool:
    """Whether the ids sent for a reply are in the prompt the server built, consecutively and in order."""
    needle = array.array("I", (int(value) for value in tokens.split(","))).tobytes()
    return array.array("I", prompt).tobytes().find(needle) >= 0
