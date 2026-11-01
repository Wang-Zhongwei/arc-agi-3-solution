"""ARC3_CONFIRMED_RULES: carry the model's own "confirmed" findings across context trims.

Every assistant reply that enters the conversation is tagged with a sequence
number and the level it was written on, and the sentences of its THINKING that
say "confirmed" are set aside (tool-call code and reply text are not read).
When a context trim drops a reply, its sentences move into the rules, grouped
by level. The rules are sent as one user message pinned after the system prompt
(and after the compressed trace, when that is on), ahead of the retained
history. Until a trim has dropped a reply with a usable sentence there are no
rules and no message. The message is rebuilt only when a trim adds rules, so
between trims every request carries the identical text and the prefix cache
holds.

Sentence filter (judgment calls, checked against main-tuned-drafter v02
thinking: 251 of 384 "confirmed" sentences kept):
- the word "confirmed" itself, not confirm/confirms/confirming;
- no "?" and no hedge or conditional word (if, whether, would, maybe, let me, ...):
  those are plans or open questions, e.g. "If the plus turns orange, the
  mechanic is confirmed.";
- at least MIN_WORDS words, which drops "Confirmed." and "Confirmed again.";
- leading list/markdown markers stripped; one sentence is cut at
  MAX_SENTENCE_CHARS (v02's longest kept sentence is 114 tokens, ~450 chars);
- exact duplicates (case and spacing ignored) are kept once per game.
The section is capped at MAX_SECTION_CHARS by dropping the oldest sentences;
v02's largest whole-game ledger was 781 tokens (~3,100 chars), so the cap only
binds on games far longer than any seen so far.
"""

from __future__ import annotations

import re
from typing import Any

RULES_MESSAGE_KEY = "_arc3_rules"
SEQ_KEY = "_arc3_rules_seq"

MIN_WORDS = 5
MAX_SENTENCE_CHARS = 500
MAX_SECTION_CHARS = 8000

_CONFIRMED = re.compile(r"\bconfirmed\b", re.IGNORECASE)
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_HEDGED = re.compile(
    r"\b(if|whether|unless|once|until|would|could|might|maybe|perhaps|probably|"
    r"likely|hopefully|should|not yet|not( been)? confirmed|unconfirmed|"
    r"isn't confirmed|wasn't confirmed|hasn't been confirmed|to be confirmed|"
    r"let me|let's|i'll|i will)\b",
    re.IGNORECASE,
)
_LEAD = re.compile(r"^[\s\-*#>•]+")

HEADER = "## Confirmed rules from trimmed turns"
INTRO = (
    "Earlier turns of this game were trimmed from this conversation. These are "
    "the statements you marked as confirmed in your reasoning during those turns, "
    "grouped by the level you were on when you wrote them. They are your own past "
    "conclusions, not verified facts: use them as a starting point and re-check "
    "any that new evidence contradicts."
)


def confirmed_sentences(reasoning: str) -> list[str]:
    """The usable "confirmed" sentences of one reply's thinking, in order."""
    found: list[str] = []
    for raw in _SPLIT.split(reasoning or ""):
        sentence = _LEAD.sub("", raw).strip()
        if not _CONFIRMED.search(sentence) or "?" in sentence or _HEDGED.search(sentence):
            continue
        if len(sentence.split()) < MIN_WORDS:
            continue
        if len(sentence) > MAX_SENTENCE_CHARS:
            sentence = sentence[: MAX_SENTENCE_CHARS - 1].rstrip() + "…"
        found.append(sentence)
    return found


def _dedupe_key(sentence: str) -> str:
    return " ".join(sentence.lower().split())


class ConfirmedRules:
    """Per-game ledger: pending sentences per tagged reply, and the rules."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._next_seq = 0
        # seq -> (level, sentences) for replies still in the conversation
        self._pending: dict[int, tuple[int, list[str]]] = {}
        # (level, sentence) in the order the trims released them
        self._rules: list[tuple[int, str]] = []
        self._seen: set[str] = set()
        self._version = 0
        self._cache: tuple[int, dict[str, Any] | None] = (0, None)
        self.logged_version = 0

    @property
    def version(self) -> int:
        return self._version

    def tag(self, message: dict[str, Any], level: int, reasoning: str) -> None:
        """Tag an assistant reply as it enters the conversation."""
        if str(message.get("role", "")).strip() != "assistant" or SEQ_KEY in message:
            return
        seq = self._next_seq
        self._next_seq += 1
        message[SEQ_KEY] = seq
        sentences = confirmed_sentences(reasoning)
        if sentences:
            self._pending[seq] = (max(1, int(level)), sentences)

    def harvest(self, retained: list[dict[str, Any]]) -> bool:
        """Move the sentences of every tagged reply no longer in `retained`
        into the rules. True when the rules changed."""
        if not self._pending:
            return False
        present = {m.get(SEQ_KEY) for m in retained if SEQ_KEY in m}
        dropped = sorted(seq for seq in self._pending if seq not in present)
        added = False
        for seq in dropped:
            level, sentences = self._pending.pop(seq)
            for sentence in sentences:
                key = _dedupe_key(sentence)
                if key in self._seen:
                    continue
                self._seen.add(key)
                self._rules.append((level, sentence))
                added = True
        if added:
            self._enforce_cap()
            self._version += 1
        return added

    def _enforce_cap(self) -> None:
        total = sum(len(s) + 3 for _, s in self._rules)
        while self._rules and total > MAX_SECTION_CHARS:
            _, sentence = self._rules.pop(0)
            total -= len(sentence) + 3

    def text(self) -> str:
        if not self._rules:
            return ""
        by_level: dict[int, list[str]] = {}
        for level, sentence in self._rules:
            by_level.setdefault(level, []).append(sentence)
        blocks = [HEADER, INTRO]
        for level in sorted(by_level):
            blocks.append(f"Level {level}:\n" + "\n".join(f"- {s}" for s in by_level[level]))
        return "\n\n".join(blocks)

    def message(self) -> dict[str, Any] | None:
        """The pinned user message, or None while there are no rules."""
        version, cached = self._cache
        if version != self._version or (cached is None and self._rules):
            text = self.text()
            cached = {"role": "user", "content": text, RULES_MESSAGE_KEY: self._version} if text else None
            self._cache = (self._version, cached)
        return cached
