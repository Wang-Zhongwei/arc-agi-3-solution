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
ARC3_REFUTED_RULES adds a "Ruled out" list beside each level's confirmed
list, from thinking sentences with a strong refutation word only (refuted,
disproved, falsified, debunked, ruled out; "my model was wrong" and
"contradicts" are left out as too weak or too temporary). Same filter as above
on text with curly apostrophes made straight, plus no "may"/"will" and no
negated or planned refutation ("NOT ruled out", "can't rule out", "to refute").
A sentence with a refutation word goes to the ruled-out list only. v01+v02:
about 20 such sentences in 50 games.

ARC3_RULES_CONTENT_FILTER drops low-information sentences from both lists
(v01+v02: 43 of 496 dropped). A sentence ending in ":" first takes in up to
COLON_JOIN_PIECES following lines when the part before the colon says nothing
on its own ("The push rule is CONFIRMED:" - its list is the content). Leading fillers
(Hmm, So, OK, Good, ...) and trailing check marks are stripped. A sentence is
dropped when it never says what was confirmed ("That's confirmed by the win.",
"Confirmed by this death.") or has fewer than MIN_CONTENT_WORDS words left once
the marker words, fillers, function words and generic nouns (model,
hypothesis, rule, mechanic, goal, target, map, ...) are removed: "So the target
is confirmed." goes, "White = wall, confirmed." stays.

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
MIN_CONTENT_WORDS = 2
COLON_JOIN_PIECES = 4
COLON_JOIN_CHARS = 400

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
_REFUTED = re.compile(
    r"\b(refut\w*|disprov\w*|falsif\w*|debunk\w*|ruled out|rules out|rule out)\b", re.IGNORECASE
)
_REFUTED_EXTRA = re.compile(
    r"\b(may|will|won't|wouldn't)\b|\bfalsifiab\w*|"
    r"\b(to|can't|cannot|can not|couldn't|not( yet)?|haven't|hasn't|never)\s+(be\s+)?"
    r"(refute|disprove|falsify|debunk|rule out|ruled out|refuted|disproved|falsified)\b",
    re.IGNORECASE,
)

_FILLER_WORDS = frozenset("""
confirmed confirm confirms refuted refute refutes refutation disproved disproves disprove
falsified debunked ruled rules rule out
so hmm hmmm ok okay good enough great yes wait well now anyway also then thus hence again
definitively finally indeed really actually fully clearly partially directly empirically
strongly definitely obviously basically
the a an is are was were be been being this that it its it's that's there's my our we i
and or but of to in on at by for with as all both here there which what
level level's case model hypothesis theory idea guess mechanic mechanics goal target test
result results behavior behaviour pattern picture map
""".split())
_TOKEN_EDGES = ".,:;!()[]\"'“”‘’✓✗*"
_LEAD_FILLER = re.compile(
    r"^(?:(?:hmm+|ok(?:ay)?|so|good|great|yes|wait|right|anyway|and|but|now|alright|well)\b[\s,:;!.—-]*)+",
    re.IGNORECASE,
)
_TRAIL_MARKS = re.compile(r"[\s✓✗!]+(?=[.)]*$)")
_SUBJECTLESS = re.compile(
    r"^(?:(?:that|this|it|which)(?:'s| is| was)?\s+(?:also\s+)?(?:confirmed|refuted|ruled out|disproved)\b"
    r"|confirmed by\b[^:]*$|confirmed[.!]?$)",
    re.IGNORECASE,
)


def _tidy(sentence: str) -> str:
    sentence = _LEAD_FILLER.sub("", sentence).strip()
    sentence = _TRAIL_MARKS.sub("", sentence).strip()
    return sentence[:1].upper() + sentence[1:]


def _informative(sentence: str) -> bool:
    if _SUBJECTLESS.search(sentence.replace("\u2019", "'")):
        return False
    words = [w.strip(_TOKEN_EDGES) for w in sentence.split()]
    content = [
        w for w in words
        if w and w.lower() not in _FILLER_WORDS and not set(w) <= set("✓✗=→-") and not w.isdigit()
    ]
    return len(content) >= MIN_CONTENT_WORDS


def _segments(reasoning: str, join_colon: bool) -> list[str]:
    pieces = [p for p in _SPLIT.split(reasoning or "") if p and p.strip()]
    if not join_colon:
        return pieces
    out: list[str] = []
    index = 0
    while index < len(pieces):
        piece = pieces[index]
        index += 1
        if piece.rstrip().endswith(":") and not _informative(_tidy(_LEAD.sub("", piece).strip())):
            taken = 0
            while index < len(pieces) and taken < COLON_JOIN_PIECES and len(piece) < COLON_JOIN_CHARS:
                piece = piece.rstrip() + " " + _LEAD.sub("", pieces[index]).strip()
                index += 1
                taken += 1
        out.append(piece)
    return out


CONFIRMED = "confirmed"
REFUTED = "refuted"

HEADER = "## Confirmed rules from trimmed turns"
INTRO = (
    "Earlier turns of this game were trimmed from this conversation. These are "
    "the statements you marked as confirmed in your reasoning during those turns, "
    "grouped by the level you were on when you wrote them. They are your own past "
    "conclusions, not verified facts: use them as a starting point and re-check "
    "any that new evidence contradicts."
)


HEADER_BOTH = "## Confirmed and ruled-out rules from trimmed turns"
INTRO_BOTH = (
    "Earlier turns of this game were trimmed from this conversation. These are "
    "the statements your reasoning in those turns marked as confirmed, and the "
    "hypotheses it marked as refuted or ruled out, grouped by the level you were "
    "on when you wrote them. They are your own past conclusions, not verified "
    "facts: use them as a starting point and re-check any that new evidence "
    "contradicts."
)


def _cut(sentence: str) -> str:
    if len(sentence) > MAX_SENTENCE_CHARS:
        return sentence[: MAX_SENTENCE_CHARS - 1].rstrip() + "…"
    return sentence


def refuted_sentences(reasoning: str, *, content_filter: bool = False) -> list[str]:
    """The usable strong-refutation sentences of one reply's thinking, in order."""
    found: list[str] = []
    for raw in _segments(reasoning, content_filter):
        sentence = _LEAD.sub("", raw).strip()
        plain = sentence.replace("\u2019", "'")
        if not _REFUTED.search(plain) or "?" in plain:
            continue
        if _HEDGED.search(plain) or _REFUTED_EXTRA.search(plain):
            continue
        if len(sentence.split()) < MIN_WORDS:
            continue
        if content_filter:
            sentence = _tidy(sentence)
            if not _informative(sentence):
                continue
        found.append(_cut(sentence))
    return found


def classified_sentences(
    reasoning: str, *, confirmed: bool, refuted: bool, content_filter: bool = False
) -> list[tuple[str, str]]:
    """(kind, sentence) in order; a sentence with a refutation word is never
    also listed as confirmed when the ruled-out list is on."""
    found: list[tuple[str, str]] = []
    ruled_out = refuted_sentences(reasoning, content_filter=content_filter) if refuted else []
    if confirmed:
        for sentence in confirmed_sentences(reasoning, content_filter=content_filter):
            if refuted and _REFUTED.search(sentence.replace("\u2019", "'")):
                continue
            found.append((CONFIRMED, sentence))
    found.extend((REFUTED, s) for s in ruled_out)
    return found


def confirmed_sentences(reasoning: str, *, content_filter: bool = False) -> list[str]:
    """The usable "confirmed" sentences of one reply's thinking, in order."""
    found: list[str] = []
    for raw in _segments(reasoning, content_filter):
        sentence = _LEAD.sub("", raw).strip()
        if not _CONFIRMED.search(sentence) or "?" in sentence or _HEDGED.search(sentence):
            continue
        if len(sentence.split()) < MIN_WORDS:
            continue
        if content_filter:
            sentence = _tidy(sentence)
            if not _informative(sentence):
                continue
        found.append(_cut(sentence))
    return found


def _dedupe_key(sentence: str) -> str:
    return " ".join(sentence.lower().split())


class ConfirmedRules:
    """Per-game ledger: pending sentences per tagged reply, and the rules."""

    def __init__(self, *, confirmed: bool = True, refuted: bool = False, content_filter: bool = False) -> None:
        self.confirmed = confirmed
        self.refuted = refuted
        self.content_filter = content_filter
        self.reset()

    def reset(self) -> None:
        self._next_seq = 0
        # seq -> (level, [(kind, sentence)]) for replies still in the conversation
        self._pending: dict[int, tuple[int, list[tuple[str, str]]]] = {}
        # (level, kind, sentence) in the order the trims released them
        self._rules: list[tuple[int, str, str]] = []
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
        sentences = classified_sentences(
            reasoning, confirmed=self.confirmed, refuted=self.refuted, content_filter=self.content_filter
        )
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
            for kind, sentence in sentences:
                key = _dedupe_key(sentence)
                if key in self._seen:
                    continue
                self._seen.add(key)
                self._rules.append((level, kind, sentence))
                added = True
        if added:
            self._enforce_cap()
            self._version += 1
        return added

    def _enforce_cap(self) -> None:
        total = sum(len(s) + 3 for _, _, s in self._rules)
        while self._rules and total > MAX_SECTION_CHARS:
            _, _, sentence = self._rules.pop(0)
            total -= len(sentence) + 3

    def text(self) -> str:
        if not self._rules:
            return ""
        by_level: dict[int, dict[str, list[str]]] = {}
        for level, kind, sentence in self._rules:
            by_level.setdefault(level, {}).setdefault(kind, []).append(sentence)
        if not self.refuted:
            # the confirmed-only layout, unchanged from ARC3_CONFIRMED_RULES alone
            blocks = [HEADER, INTRO]
            for level in sorted(by_level):
                lines = by_level[level].get(CONFIRMED, [])
                blocks.append(f"Level {level}:\n" + "\n".join(f"- {s}" for s in lines))
            return "\n\n".join(blocks)
        blocks = [HEADER_BOTH, INTRO_BOTH]
        for level in sorted(by_level):
            parts = [f"Level {level}:"]
            for kind, title in ((CONFIRMED, "Confirmed:"), (REFUTED, "Ruled out:")):
                lines = by_level[level].get(kind)
                if lines:
                    parts.append(title + "\n" + "\n".join(f"- {s}" for s in lines))
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)

    def message(self) -> dict[str, Any] | None:
        """The pinned user message, or None while there are no rules."""
        version, cached = self._cache
        if version != self._version or (cached is None and self._rules):
            text = self.text()
            cached = {"role": "user", "content": text, RULES_MESSAGE_KEY: self._version} if text else None
            self._cache = (self._version, cached)
        return cached
