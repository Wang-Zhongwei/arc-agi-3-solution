"""ARC3_NORMALIZE_HISTORY chooses the reply text history stores.

On (the default) history gets the normalised reasoning and content the turn
loop works with; off it gets the raw texts, blank lines kept, so the next
request renders the reply as generated.
"""

import pytest

from inference.agent import tool_agent
from inference.agent.tool_agent import (
    _contains_tool_call_markup,
    _extract_reasoning_text,
    _history_reply_texts,
    _normalize_message_content,
)

REASONING = "  Let me look at the board.\n\nThe red block moved left.\n   \nSo LEFT moves it.\n"
CONTENT = "Trying LEFT next.\n\nThen UP.  "
MARKUP = '<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>'


def _texts(message):
    reasoning = _extract_reasoning_text(message)
    content = _normalize_message_content(message.get("content", ""))
    markup = _contains_tool_call_markup(reasoning, content)
    if markup:
        reasoning = tool_agent._strip_tool_call_markup(reasoning)
        content = tool_agent._strip_tool_call_markup(content)
    return reasoning, content, markup


@pytest.mark.parametrize("value", ["", "1", "true"])
def test_default_and_on_store_the_normalised_text(monkeypatch, value):
    monkeypatch.setenv("ARC3_NORMALIZE_HISTORY", value)
    message = {"reasoning": REASONING, "content": CONTENT}
    reasoning, content, markup = _texts(message)

    assert _history_reply_texts(message, reasoning, content, markup) == (reasoning, content)
    assert "\n\n" not in reasoning


def test_off_stores_the_reply_as_generated(monkeypatch):
    monkeypatch.setenv("ARC3_NORMALIZE_HISTORY", "0")
    message = {"reasoning": REASONING, "content": CONTENT}
    reasoning, content, markup = _texts(message)

    assert _history_reply_texts(message, reasoning, content, markup) == (
        REASONING.strip(),
        CONTENT.strip(),
    )


def test_off_reads_reasoning_content_when_reasoning_is_empty(monkeypatch):
    monkeypatch.setenv("ARC3_NORMALIZE_HISTORY", "0")
    message = {"reasoning": "", "reasoning_content": REASONING, "content": None}
    reasoning, content, markup = _texts(message)

    assert _history_reply_texts(message, reasoning, content, markup) == (REASONING.strip(), "")


def test_off_still_removes_tool_call_markup(monkeypatch):
    monkeypatch.setenv("ARC3_NORMALIZE_HISTORY", "0")
    message = {"reasoning": f"Plan first.\n\n{MARKUP}\n\nDone.", "content": ""}
    reasoning, content, markup = _texts(message)
    history_reasoning, _ = _history_reply_texts(message, reasoning, content, markup)

    assert markup
    assert "<tool_call>" not in history_reasoning
    assert history_reasoning.startswith("Plan first.\n\n")
