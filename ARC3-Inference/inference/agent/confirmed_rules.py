"""ARC3_MEM_RULES: the rules the model confirmed, quoted from its own reasoning.

Pure python, no I/O. The state lab (2026-10-13) found the model losing rules
it had confirmed on earlier levels once their turns were trimmed (r11l A's
"the match is by color SET", m0r0 A's "the selected dot moves while the
blocks are frozen"); quoting them back moved m0r0 L6 from 3/20 to 9/20.

A sentence is kept when it:

- carries a confirmation mark (a check mark, "confirmed", "verified"), or
  follows a marked sentence that was not kept and opens with a conclusion
  word ("So the selected dot is a movable object ..." after "CONFIRMED: The
  yellow dot MOVED UP from (19,11) to (15,11) ...": the mark sits on the
  evidence, the rule follows it);
- states a rule (a generic verb or quantifier: moves, only, must, always,
  never, kills, opens, teleports, absorbs, matches, same, goal, win, ...);
- holds at most one coordinate pair, is 40-260 chars, is no question, no
  plan ("Let me ...", "I'll ...", "plan") and no hedge ("might", "should",
  "presumably", "what if", ...), holds no python, at most 8 digits and one
  "=", and does not open with "Wait", "Then" or an imperative ("Click ...").

A leading interjection ("Hmm, ", "OK, ") is cut, never the sentence: two of
the eleven rules the lab quoted open with "Hmm," ("Hmm, and the knot's path
doesn't matter (teleport) ✓."). What is kept is otherwise the model's words
verbatim. Per level the ``PER_LEVEL`` best by ``rule_score`` are shown (see
``RuleLedger``); a sentence whose word overlap with a kept one is >= ``SAME``
replaces it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PER_LEVEL = 3
# sentences remembered per level, of which the best PER_LEVEL are shown
CANDIDATES = 24
SAME = 0.6
MIN_CHARS = 40
MAX_CHARS = 260
MAX_DIGITS = 8

_MARK_RE = re.compile(r"[✓✔✅]|\bconfirm(?:ed|s)?\b|\bverified\b", re.I)
_RULE_RE = re.compile(
    r"\b(?:mov(?:e|es|ed|ing)|only|must|always|never|each|every|kill(?:s|ed)?|destroy(?:s|ed)?|"
    r"remov(?:e|es|ed)|open(?:s|ed)?|clos(?:e|es|ed)|teleport(?:s|ed|ing)?|select(?:s|ed)?|"
    r"absorb(?:s|ed)?|merg(?:e|es|ed)|match(?:es|ed)?|same|cycl(?:e|es|ed)|goal|win(?:s|ning)?|"
    r"rule|swap(?:s|ped)?|rotat(?:e|es|ed)|push(?:es|ed)?|toggl(?:e|es|ed)|flip(?:s|ped)?|"
    r"consum(?:e|es|ed)|collect(?:s|ed)?|complet(?:e|es|ed)|clear(?:s|ed)|regardless|"
    r"doesn't matter|does not matter|frozen|blocked|game over)\b",
    re.I,
)
_COORD_RE = re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)")
# a sentence end, or a line break (bullets and list items are their own sentence)
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_BULLET_RE = re.compile(r"^(?:[-*•]+|\d+[.)])\s+")
_INTERJECTION_RE = re.compile(r"^(?:(?:hmm+|hm|ok|okay|ah|oh|well|yes|right)\b[,.:!]*\s+)+", re.I)
_PLAN_RE = re.compile(
    r"^(?:let me|let's|i'll|i will|i need|i should|i want|i must|we need|we should|now i|next,? i|"
    r"wait\b|but wait\b|actually,? wait\b|maybe\b|perhaps\b|todo\b|then\b|"
    r"(?:also |now |first,? )?(?:confirm|check|verify|test|try|click|press|move|execute|run)\b)",
    re.I,
)
# a hedge or a plan anywhere: not a confirmed rule
_HEDGE_RE = re.compile(
    r"\b(?:might|maybe|presumably|probably|perhaps|likely|should|would|could|assum\w*|guess\w*|"
    r"seems?|hypothes\w*|unless|not sure|i think|plan|planned|planning|what if|need to|only need|"
    r"hopefully|e\.g|risk)\b",
    re.I,
)
# the model's own tools and code: bookkeeping about its solver, not the game
_TOOL_RE = re.compile(
    r"\b(?:my|i|planner|solver|search|bfs|template|code|function|helper|script|detector|detection|"
    r"variable|model's|snippet|print(?:ed)?)\b",
    re.I,
)
_STRONG_RE = re.compile(
    r"\b(?:must|always|never|only|rule|goal|win|wins|each|every|any|same|regardless|doesn't matter|"
    r"does not matter|teleports?|absorbs?|match(?:es)?|mechanic|whenever)\b",
    re.I,
)
_CONCLUSION_RE = re.compile(r"^(?:so|thus|therefore|hence|this means|that means|i\.e\.)\b", re.I)
# python and markup: kept sentences are prose
_CODE_RE = re.compile(r"==|!=|\bdef \b|\bprint\(|\[\w+\]\[|</?\w+>|```|\bimport\b|\blambda\b|\b[a-z]+_[a-z_0-9]+\b")
_WORD_RE = re.compile(r"[a-z0-9]+")

# ARC3_RULES_FILTER_V2 (arm/mem-repair-rules): what else is not a rule. The
# run-wide audit of stall-fix-10x's 512 quoted rules (hand-labelled): 36% real
# mechanics, the rest board trivia, plans, hypotheses stated as fact, notes on
# the model's own code and echoes of the memory or prompt.
# (a) a coordinate pair or a row/col range: the board's state, not a rule
_V2_COORD_RE = re.compile(
    r"[(\[]\s*-?\d+(?:\s*[-–]\s*\d+)?\s*,\s*-?\d+(?:\s*[-–]\s*\d+)?\s*[)\]]|"
    r"\(\s*row\s*=\s*\d+\s*,\s*col\s*=\s*\d+\s*\)|"
    r"\b(?:rows?|cols?|columns?)\s*\d+\s*(?:-|–|\.\.|to)\s*\d+",
    re.I,
)
# (b) a fragment: cut off mid-thought, or a piece of a bracketed aside
_V2_FRAGMENT_RE = re.compile(r"(?::|…|\.\.\.)\s*$|^[)\],;→—–-]")
# (c) an echo: a quote of the memory, the prompt or an earlier level's note
_V2_QUOTE_RE = re.compile(r"\"[^\"]+\"|“[^”]+”")
_V2_LEVEL_RE = re.compile(r"\blevels?[- ]?\d|\bL\d+\b", re.I)
_V2_ECHO_RE = re.compile(r"\b(?:note|noted|notes|recap|harness|memory|says|said)\b|^(?:level|L)\s*\d+\s*[:\-—]|^[\"“]", re.I)
# (d) a plan, or about the model's own code
_V2_PLAN_RE = re.compile(
    r"\b(?:let me|let's|i'll|i will|i need|we need|first,|actually|next)\b|"
    r"\b(?:my|the)\s+(?:code|solver|search|bfs|helper|function|simulator|model)\b",
    re.I,
)
# (e) a hedge or a condition
_V2_HEDGE_RE = re.compile(
    r"\b(?:may|might|maybe|perhaps|probably|likely|seems?|appears?|could|possibly|guess|assumed?)\b|\?|"
    r"^(?:if|unless|assuming|suppose)\b",
    re.I,
)
# (f) an absolute in a conclusion that carries no mark of its own
_V2_ABSOLUTE_RE = re.compile(r"\b(?:never|cannot|can't|can not|impossible|only|must|always)\b", re.I)
_STOP = frozenset(
    "a an the is are was were be been being to of and or in on at it its this that these those with for "
    "by as from into than then so but if when while there here i we you".split()
)


def _words(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall(text.lower()) if w not in _STOP)


def overlap(a: str, b: str) -> float:
    """Word overlap (Jaccard over content words) of two sentences."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for piece in _SPLIT_RE.split(str(text or "")):
        piece = _BULLET_RE.sub("", piece.strip()).strip()
        if piece:
            out.append(piece)
    return out


def _rule_ok(sentence: str) -> bool:
    if not (MIN_CHARS <= len(sentence) <= MAX_CHARS):
        return False
    if sentence.rstrip().endswith("?") or _PLAN_RE.match(sentence) or _CODE_RE.search(sentence):
        return False
    if len(_COORD_RE.findall(sentence)) > 1 or _HEDGE_RE.search(sentence):
        return False
    # arithmetic and bookkeeping: many digits or a chain of equalities
    if sum(ch.isdigit() for ch in sentence) > MAX_DIGITS or sentence.count("=") > 1:
        return False
    return bool(_RULE_RE.search(sentence))


def _balanced(sentence: str) -> bool:
    return (
        sentence.count("(") == sentence.count(")")
        and sentence.count("[") == sentence.count("]")
        and sentence.count('"') % 2 == 0
        and sentence.count("“") == sentence.count("”")
    )


def _rule_ok_v2(sentence: str, *, marked: bool) -> bool:
    """ARC3_RULES_FILTER_V2: a sentence ``_rule_ok`` keeps that is still no
    rule: (a) it holds a coordinate pair or a row/col range; (b) it is a
    fragment (ends in ":" or "...", opens with ")", unbalanced brackets or
    quotes); (c) it echoes a note (quoted text naming a level, "note",
    "recap", "harness", "says", a leading "Level 6:" or quote); (d) it is a
    plan or about the model's code ("let me", "I'll", "actually", "next",
    "the solver", ...); (e) it is hedged ("may", "seems", "appears", "could",
    a "?", a leading "If"); (f) it is an unmarked conclusion stating an
    absolute ("So ... never / only / must / always ...")."""
    if _V2_COORD_RE.search(sentence) or _V2_FRAGMENT_RE.search(sentence) or not _balanced(sentence):
        return False
    if _V2_ECHO_RE.search(sentence) or (_V2_QUOTE_RE.search(sentence) and _V2_LEVEL_RE.search(sentence)):
        return False
    if _V2_PLAN_RE.search(sentence) or _V2_HEDGE_RE.search(sentence):
        return False
    return marked or not _V2_ABSOLUTE_RE.search(sentence)


def rule_score(sentence: str) -> float:
    """How much a kept sentence reads as a general rule: distinct rule words
    (strong ones count twice), less digits, coordinates, symbols and words
    about the model's own code. Ranks a level's sentences; ties go to the
    newest."""
    strong = {m.group(0).lower() for m in _STRONG_RE.finditer(sentence)}
    weak = {m.group(0).lower() for m in _RULE_RE.finditer(sentence)} - strong
    score = 2.0 * len(strong) + len(weak)
    score -= sum(ch.isdigit() for ch in sentence) / 4
    score -= 2 * len(_COORD_RE.findall(sentence))
    score -= sentence.count("=") + sentence.count("\u2192") + sentence.count("[")
    score -= 2 * len({m.group(0).lower() for m in _TOOL_RE.finditer(sentence)})
    return score


def rule_sentences(text: str, *, v2: bool = False) -> list[str]:
    """The confirmed rules in one reasoning or reply, in the order written.
    ``v2`` (ARC3_RULES_FILTER_V2) also drops what ``_rule_ok_v2`` rejects."""
    found: list[str] = []
    carry = False
    for raw in _sentences(text):
        sentence = _INTERJECTION_RE.sub("", raw).strip()
        marked = bool(_MARK_RE.search(sentence))
        inherits = carry and not marked and bool(_CONCLUSION_RE.match(sentence))
        carry = False
        if not (marked or inherits):
            continue
        if _rule_ok(sentence) and (not v2 or _rule_ok_v2(sentence, marked=marked)):
            found.append(sentence)
        elif marked:
            # the mark is on the evidence; a conclusion right after it is the rule
            carry = True
    return found


class TaggedRule(str):
    """ARC3_RULES_ACTION_TAG: a rule (the sentence itself) with the game
    action number it was written at (``quoted`` adds ``(action 63)``)."""

    def __new__(cls, text: str, action: int | None) -> "TaggedRule":
        rule = super().__new__(cls, text)
        rule.action = action
        return rule

    def __reduce__(self):
        return (TaggedRule, (str(self), self.action))


@dataclass
class RuleLedger:
    """Per level, the confirmed rules: up to ``CANDIDATES`` sentences, of
    which ``rules_for`` shows the ``per_level`` best by ``rule_score``
    (ties to the newest), in the order they were written. A sentence whose
    word overlap with a kept one is >= ``SAME`` replaces it (newer wins).

    The spec asked for the newest three. On run A's logs the newest three
    were mostly late solver bookkeeping ("my template matches any color ✓
    ..."): r11l L4 lost "So each knot must go to the ring with the SAME color
    set ✓✓." (call 95) and m0r0 L3 "So the selected (yellow) dot is a movable
    object ..." (call 64), the rules the lab quoted; ranked, both are kept."""

    per_level: int = PER_LEVEL
    levels: dict[int, list[tuple[int, str]]] = field(default_factory=dict)
    written: int = 0
    # ARC3_RULES_FILTER_V2: rule_sentences(v2=True)
    filter_v2: bool = False
    # ARC3_RULES_ACTION_TAG: rules_for gives TaggedRules (written index -> action)
    tag_actions: bool = False
    actions: dict[int, int] = field(default_factory=dict)

    def absorb(self, level: int, text: str, action: int | None = None) -> int:
        """Record the confirmed rules in ``text`` against ``level``, written
        after game action ``action``; how many."""
        added = 0
        for sentence in rule_sentences(text, v2=self.filter_v2):
            self.written += 1
            if action is not None:
                self.actions[self.written] = int(action)
            kept = self.levels.setdefault(int(level), [])
            kept[:] = [(n, old) for n, old in kept if old != sentence and overlap(old, sentence) < SAME]
            kept.append((self.written, sentence))
            if len(kept) > CANDIDATES:
                worst = min(kept, key=lambda item: (rule_score(item[1]), item[0]))
                kept.remove(worst)
            added += 1
        return added

    def rules_for(self, level: int) -> list[str]:
        """``level``'s best ``per_level`` rules, in the order written (with
        ``tag_actions``, as TaggedRules: oldest action first)."""
        kept = self.levels.get(int(level), [])
        best = sorted(kept, key=lambda item: (-rule_score(item[1]), -item[0]))[: self.per_level]
        if self.tag_actions:
            return [TaggedRule(sentence, self.actions.get(n)) for n, sentence in sorted(best)]
        return [sentence for _, sentence in sorted(best)]


def quoted(rules: list[str]) -> str:
    """``"rule one" | "rule two"``; a TaggedRule with an action number is
    ``"rule one" (action 63)``."""
    return " | ".join(
        f'"{rule}"' + (f" (action {rule.action})" if getattr(rule, "action", None) is not None else "")
        for rule in rules
    )
