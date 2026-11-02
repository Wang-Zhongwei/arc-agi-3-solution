"""ARC3_CONFIRMED_RULES: carry the model's own "confirmed" findings across context trims.

Every assistant reply that enters the conversation is tagged with a sequence
number and the level it was written on, and the sentences of its THINKING that
say "confirmed" are set aside (tool-call code and reply text are not read).
When a context trim drops a reply, its sentences move into the rules, grouped
by level. The rules are sent as one user message pinned after the system prompt
(and after the compressed trace, when that is on), ahead of the retained
history. Until a trim has released a usable sentence there are no rules and no
message. The message is rebuilt only when a trim adds rules, so between trims
every request carries the identical text and the prefix cache holds.

Confirmed sentences show only for cleared levels. A trim that drops a reply
written on the level still being played holds its confirmed sentences back; the
first trim after that level is cleared adds them. Lines written on a level the
model never clears were wrong about twice as often as lines from cleared ones.
Ruled-out sentences do not wait.

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
Two kinds of sentence state the surviving rule, not the refuted one, and would
read backwards under "Ruled out": one that also says confirm*/support*
("the trail model is strongly supported (... a refutation of greedy)"), which
goes to the confirmed list when it says "confirmed" and is dropped otherwise;
and one that ends in a bare verdict with no subject ("So the moves are
absolute ✓✓ REFUTED definitively."), which is dropped.

Both lists then drop low-information sentences (always on). A sentence ending in ":" first takes in up to
COLON_JOIN_PIECES following lines when the part before the colon says nothing
on its own ("The push rule is CONFIRMED:" - its list is the content). Leading fillers
(Hmm, So, OK, Good, ...) and trailing check marks are stripped. A sentence is
dropped when it never says what was confirmed ("That's confirmed by the win.",
"Confirmed by this death.") or has fewer than MIN_CONTENT_WORDS words left once
the marker words, fillers, function words and generic nouns (model,
hypothesis, rule, mechanic, goal, target, map, ...) are removed: "So the target
is confirmed." goes, "White = wall, confirmed." stays.

Confirmed sentences then go through confirmed_line_filters (always on; not used
for ruled-out sentences). It drops a sentence when the one after it hedges, and
sentences that a reader on a later level cannot use. Labelled on main-tuned-drafter
v01 and v02, see docs/harness/confirmed-rules-audit.md.

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
NEXT_SENTENCE_CHARS = 600

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
    "the statements you marked as confirmed in your reasoning during those turns "
    "on levels you have since cleared, grouped by the level. They are your own past "
    "conclusions, not verified facts: use them as a starting point and re-check "
    "any that new evidence contradicts."
)


HEADER_BOTH = "## Confirmed and ruled-out rules from trimmed turns"
INTRO_BOTH = (
    "Earlier turns of this game were trimmed from this conversation. These are "
    "the statements your reasoning in those turns marked as confirmed on levels "
    "you have since cleared, and the hypotheses it marked as refuted or ruled out "
    "on any level, grouped by the level you were on when you wrote them. They are "
    "your own past conclusions, not verified "
    "facts: use them as a starting point and re-check any that new evidence "
    "contradicts."
)


_SUPPORTS = re.compile(r"\b(confirm\w*|support\w*)\b", re.IGNORECASE)
_VERDICT_TAIL = re.compile(
    r"(\S+)\s+(?:refuted|ruled out|disproved|falsified|debunked)\b(?:\s+(?:definitively|empirically|again))?\W*$",
    re.IGNORECASE,
)
_VERDICT_SUBJECT_LINKS = {"is", "was", "are", "were", "be", "been", "being", "now", "also", "definitively"}


def _bare_verdict(sentence: str) -> bool:
    """`<claim> ✓✓ REFUTED`: the verdict word closes the sentence with no
    is/was before it, so it is unclear what it refutes."""
    match = _VERDICT_TAIL.search(sentence)
    if not match:
        return False
    return match.group(1).strip(_TOKEN_EDGES).lower() not in _VERDICT_SUBJECT_LINKS


def _cut(sentence: str) -> str:
    if len(sentence) > MAX_SENTENCE_CHARS:
        return sentence[: MAX_SENTENCE_CHARS - 1].rstrip() + "…"
    return sentence


def refuted_sentences(reasoning: str, *, content_filter: bool = True) -> list[str]:
    """The usable strong-refutation sentences of one reply's thinking, in order."""
    found: list[str] = []
    for raw in _segments(reasoning, content_filter):
        sentence = _LEAD.sub("", raw).strip()
        plain = sentence.replace("\u2019", "'")
        if not _REFUTED.search(plain) or "?" in plain:
            continue
        if _HEDGED.search(plain) or _REFUTED_EXTRA.search(plain):
            continue
        if _SUPPORTS.search(plain) or _bare_verdict(plain):
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
    reasoning: str, *, confirmed: bool, refuted: bool, content_filter: bool = True
) -> list[tuple[str, str]]:
    """(kind, sentence) in order. The lists never share a sentence: one that
    says "confirmed" is a confirmed rule even when it also names what it refuted."""
    found: list[tuple[str, str]] = []
    ruled_out = refuted_sentences(reasoning, content_filter=content_filter) if refuted else []
    if confirmed:
        found.extend((CONFIRMED, s) for s in confirmed_sentences(reasoning, content_filter=content_filter))
    found.extend((REFUTED, s) for s in ruled_out)
    return found


# Filters on one confirmed sentence and the sentence after it in the thinking.
# hedged_next: the next sentence doubts or re-checks this one.
_HEDGE_NEXT = re.compile(r"^\W*(hmm+|wait|but|however|unless|actually|although|though|maybe|perhaps|or\b|not sure|hold on|let me (re-?check|double|verify|re-?read)|is that|did\b|does\b|was\b|could|might|need to (check|verify))|\?\s*$|\b(not sure|unclear|double[- ]check|re-?check|verify|uncertain|suspicious|contradict|doesn't (match|fit)|wrong)\b", re.I)
# generic_artifact: an earlier plan, model, map... is called confirmed without restating it.
_ARTIFACT = r"(plan|model|map|board|state|layout|schedule|legend|mapping|shapes?|experiment|search|analysis|hypothesis|theory|table|picture|reading|interpretation|origin|geometry|order|sequence|solution|answer|prediction|setup|configuration|structure|assignment|rule set|facts?)"
_GENERIC_ARTIFACT = re.compile(
    r"^\W*(so|ok|okay|good|great|yes|and|also)?[,\s]*(the|my|our|this|that|these|those)?\s*([\w-]+\s){0,2}?" + _ARTIFACT + r"s?\b[^.:;]{0,30}?\b(is|are|was|were|has been|have been)?\s*(now\s+|also\s+|again\s+|fully\s+)?(confirmed|verified)\b(?!\s*:)[^:]{0,40}$"
    r"|^\W*(i|we)\s+(have\s+|'ve\s+)?(now\s+)?(confirmed|verified)\s+(the|my|our)\s+([\w-]+\s){0,2}?" + _ARTIFACT + r"s?\b[^:]{0,40}$"
    r"|^\W*(the|my|our)\s+([\w-]+\s){0,2}?" + _ARTIFACT + r"s?\s+confirmed\s+(the|that|it)\b[^:]{0,40}$",
    re.I,
)
# A location: <r,c> (see _main_clause), "rows 3", "col 2".
_PLACE = re.compile(r"<[-+]?\d+,[-+]?\d+>|\b(rows?|cols?|columns?)\s+\d+", re.I)
# The sentence says something holds in general or names an input: not a read-out.
_GENERAL = re.compile(r"→|\b(when|if|until|per|each|every|always)\b", re.I)
_INPUT = re.compile(r"\b(click|press|mov|push)\w*", re.I)
# fragment: a heading whose list was cut off, or a short sentence whose subject is a cell.
_CELL_OPEN = re.compile(r"^\W*(then\s+)?\(\s*-?\d+\s*,\s*\d+\s*\)", re.I)
FRAGMENT_MAX_WORDS = 12
# snapshot: where something is at one moment, in the main clause.
_MOMENT = re.compile(r"\bnow\b(?!\s+(confirmed|verified))|\bcurrently\b|\bremain(s|ing)?\b|\bfresh\b|\binitial\b", re.I)
_LOCATED = re.compile(r"\b(is|are|was|were|occup(y|ies|ied))\s+(at\s+)?(rows?\s*\d|cols?\s*\d|<)", re.I)
_AT_STEP = re.compile(r"\bat steps?\s+\d|^\W*step\s+\d+\s*:", re.I)
# dead_coords: a place in the main clause and nothing said to happen there.
_EVENT = re.compile(r"\b(?!confirm|verifi)\w{3,}ed\b", re.I)  # a past-tense event other than the claim word
_NAMED_RULE = re.compile(r"\b(mechanics?|rules?)\b", re.I)  # the sentence calls itself a rule
_CELL_VERB = re.compile(r">\s+[a-z]{3,}s\b")  # a verb right after the cell: "<2,2> turns"
# dead_layout: one level's layout was checked, and it comes with numbers.
_LAYOUT_WORD = r"(map|geometry|layout|regions?)"
_LAYOUT = re.compile(r"\b" + _LAYOUT_WORD + r"\s+(is\s+)?confirmed|\bconfirmed\s+(the\s+)?([\w-]+\s+){0,2}?" + _LAYOUT_WORD + r"\b", re.I)
# dead_literal: a private symbol set to a literal, a set-builder formula, or a quoted grid.
_LITERAL = re.compile(r"(?<![\w-])[A-Z][-+]?\s*=\s*[{\d+-]|\{[^}]*[<>=]|[\"'][.A-Za-z]{2,}\s?/")


def _mark_cells(sentence: str) -> str:
    """Spell a (r,c) cell as <r,c> so it survives the aside stripping in _main_clause."""
    return re.sub(r"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)", r"<\1,\2>", sentence)


def _main_clause(sentence: str) -> str:
    """The sentence without asides, nested ones included; a bare cell stays."""
    text, previous = _mark_cells(sentence), None
    while previous != text:
        previous, text = text, re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", text)
    return text


def confirmed_line_filters(sentence: str, next_sentence: str | None) -> list[str]:
    """The names of the filters that drop this confirmed sentence; empty keeps it."""
    text = sentence.strip()
    main = _main_clause(text)
    general = bool(_GENERAL.search(text) or _INPUT.search(text))
    found = []
    if _HEDGE_NEXT.search(next_sentence or ""):
        found.append("hedged_next")
    if _GENERIC_ARTIFACT.search(text):
        found.append("generic_artifact")
    if text.endswith(":") or (_CELL_OPEN.search(text) and len(text.split()) <= FRAGMENT_MAX_WORDS):
        found.append("fragment")
    if not general and (
        (_MOMENT.search(main) and _PLACE.search(_mark_cells(text))) or _LOCATED.search(main) or _AT_STEP.search(main)
    ):
        found.append("snapshot")
    if _PLACE.search(main) and not (_CELL_VERB.search(main) or general or _EVENT.search(text) or _NAMED_RULE.search(text)):
        found.append("dead_coords")
    if _LAYOUT.search(text) and re.search(r"\d", text):
        found.append("dead_layout")
    if _LITERAL.search(text):
        found.append("dead_literal")
    return found


def confirmed_sentences(reasoning: str, *, content_filter: bool = True) -> list[str]:
    """The usable "confirmed" sentences of one reply's thinking, in order."""
    found: list[str] = []
    pieces = _segments(reasoning, content_filter)
    for index, raw in enumerate(pieces):
        sentence = _LEAD.sub("", raw).strip()
        if not _CONFIRMED.search(sentence) or "?" in sentence or _HEDGED.search(sentence):
            continue
        if len(sentence.split()) < MIN_WORDS:
            continue
        if content_filter:
            sentence = _tidy(sentence)
            if not _informative(sentence):
                continue
            following = pieces[index + 1].strip()[:NEXT_SENTENCE_CHARS] if index + 1 < len(pieces) else None
            if confirmed_line_filters(_cut(sentence), following):
                continue
        found.append(_cut(sentence))
    return found


def _dedupe_key(sentence: str) -> str:
    return " ".join(sentence.lower().split())


class ConfirmedRules:
    """Per-game ledger: pending sentences per tagged reply, and the rules."""

    def __init__(self, *, confirmed: bool = True, refuted: bool = False, content_filter: bool = True) -> None:
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
        # (level, sentence): confirmed sentences released while their level is still being played
        self._held: list[tuple[int, str]] = []
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

    def harvest(self, retained: list[dict[str, Any]], current_level: int) -> bool:
        """Move the sentences of every tagged reply no longer in `retained` into
        the rules. Confirmed sentences wait until `current_level` is past theirs.
        True when the rules changed."""
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
                if kind == CONFIRMED:
                    self._held.append((level, sentence))
                else:
                    self._rules.append((level, kind, sentence))
                    added = True
        cleared = [held for held in self._held if held[0] < current_level]
        if cleared:
            self._held = [held for held in self._held if held[0] >= current_level]
            self._rules.extend((level, CONFIRMED, sentence) for level, sentence in cleared)
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
