"""What the agent is told it already knows, carried turn to turn.

The analyzer used to parse ``World model:``-style lines out of its own prose and
then never send them back (``ToolAgent._summarized_knowledge_lines`` had no
caller), so every turn re-derived the game from scratch. This module keeps that
knowledge in three small ledgers and renders them into the next prompt:

- **World model** -- what the board is and what the actions do.
- **Hypotheses** -- split into confirmed and refuted, because "UP moves the
  player" and "UP does NOT move the player" are both worth carrying, and a
  refuted one is what stops the model retrying a dead idea.
- **Level notes** -- one compressed line per level, distilled from the model's
  own ``<think>`` trace, so a level transition does not erase everything learned.

The model writes them as Qwen3-style XML tags anywhere in its reply::

    <world_model>4-colour board; blue block is the player</world_model>
    <hypothesis status="confirmed">UP moves the blue block up one cell</hypothesis>
    <hypothesis status="refuted">MOUSE on the wall opens it</hypothesis>
    <level_note>level 3 needs the key before the door</level_note>

Tags are extracted and stripped; everything outside them is left alone, so the
model keeps writing freely and only the parts it wants remembered are structured.
Parsing is deliberately tolerant: unknown ``status`` values fall back to
``open``, unclosed tags are ignored, and malformed XML never raises.

Used by ``inference/agent/tool_agent.py``: ``absorb()`` after each assistant
message, ``note_think()`` with the reasoning text, ``on_level_change()`` when a
step summary reports a new level, and ``render()`` when building the next turn's
user prompt.

memory-v2 (``card_mode``, ``solved_mode``, ``delta_mode``; all off by default,
so ``MemoryLedger()`` behaves exactly as above) adds a second renderer,
``render_card()``, that puts harness-observed facts (solved levels, action
effects, hazards; see ``memory_facts.py``) ahead of the model's own notes, and
``take_delta()``, the lines added since the model last saw the card. Items are
versioned: ``mark_shown(version)`` after a head card drops the deltas it
covers, and ``snapshot_display()``/``restore_display()`` undo a turn whose
opener was never kept.

ARC3_MEM_EVERY_CALL (``layout_mode``) and ARC3_MEM_LEGIBLE (``legible``), both
off by default, are rendered by ``memory_layout.py``: ``layout_commit()``
freezes cleared levels into ``prev_text`` (the ``<previous_levels>`` message),
``notes_for_level()`` gives the model's notes for ``<this_level>``, and
``legible`` words every card line in plain English.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from inference.agent.memory_facts import (
    SOLVED_PATH_CHARS,
    split_recap,
    cut_words,
    extract_debrief,
    extract_recap,
    render_solved,
    solved_head,
    solved_line,
)
from inference.agent import memory_layout as layout

# One <tag ...>text</tag>. DOTALL so an entry may span lines; non-greedy so two
# tags on one line stay separate.
_TAG_RE = re.compile(
    r"<(world_model|hypothesis|level_note)(\s[^>]*)?>(.*?)</\1>",
    flags=re.DOTALL | re.IGNORECASE,
)
# The scratchpad is matched separately: its body is free text kept verbatim,
# including newlines, where the other tags are collapsed to one line each.
_SCRATCHPAD_RE = re.compile(
    r"<scratchpad>(.*?)</scratchpad>", flags=re.DOTALL | re.IGNORECASE
)
_STATUS_RE = re.compile(r'status\s*=\s*["\']?(\w+)', flags=re.IGNORECASE)

# Keep the block small enough that it never crowds out the board itself.
MAX_WORLD_MODEL = 6
MAX_PER_STATUS = 8
MAX_LEVEL_NOTES = 6
MAX_ENTRY_CHARS = 200
# The scratchpad is free-form, so it gets a character budget instead of a line
# count: enough for a small map or table, not enough to crowd out the board.
MAX_SCRATCHPAD_CHARS = 1200

# memory-v2 card mode (ARC3_MEM_CARD): longer model entries (88% of world-model
# tags were cut at 200), deduplicated by word overlap, newest wins.
CARD_ENTRY_CHARS = {"world_model": 400, "hypothesis": 250, "level_note": 200}
JACCARD_SAME = 0.6
JACCARD_CORRECTION = 0.35
_CORRECTION_RE = re.compile(r"^\s*(?:correction\s*:|fix\s*:|actually\s*,)\s*", re.IGNORECASE)
_STOP_WORDS = frozenset(
    "a an the is are was were be been being to of and or in on at it its this that these "
    "those with for by as from into than then so but if when while there here i we you".split()
)
# debrief mode: reasonings checked for a recap after a level change
DEBRIEF_TRIES = 2
# Card sections: soft caps are applied only when the card is over its hard cap.
SOFT_CAPS = {"solved": 1300, "actions": 1000, "hazard": 200, "beliefs": 1400}
MAX_HAZARDS = 3
# Delta lines about one level (its deaths, the first effects seen on it):
# once the level is over they are no longer news.
_LEVEL_EVENT_PREFIXES = ("hazard:", "fx:")
_SECTION_LABELS = {"world_model": "world_model", "confirmed": "confirmed", "refuted": "refuted", "level_note": "level_note"}

CONFIRMED = "confirmed"
REFUTED = "refuted"
OPEN = "open"
_STATUS_ALIASES = {
    CONFIRMED: CONFIRMED, "true": CONFIRMED, "verified": CONFIRMED, "success": CONFIRMED,
    REFUTED: REFUTED, "false": REFUTED, "disproved": REFUTED, "failed": REFUTED, "failure": REFUTED,
}


def _tidy(text: str) -> str:
    """Collapse an entry to one capped line.

    Args:
        text (str): raw tag body.

    Returns:
        str: whitespace collapsed, truncated to ``MAX_ENTRY_CHARS`` with an
        ellipsis. ``""`` when the body is blank.
    """
    line = " ".join(str(text or "").split())
    if len(line) <= MAX_ENTRY_CHARS:
        return line
    return line[:MAX_ENTRY_CHARS].rstrip() + "..."


def _push(entries: list[str], line: str, cap: int) -> None:
    """Append ``line`` unless already present, newest last, capped at ``cap``.

    Re-stating an entry moves it to the end rather than duplicating it, so the
    model repeating itself refreshes the entry instead of filling the ledger.

    Args:
        entries (list[str]): ledger list, mutated in place.
        line (str): tidied entry; blank lines are ignored.
        cap (int): maximum entries kept; the oldest are dropped first.
    """
    if not line:
        return
    lowered = line.lower()
    for existing in list(entries):
        if existing.lower() == lowered:
            entries.remove(existing)
    entries.append(line)
    del entries[:-cap]


def summarize_think(text: str, *, max_chars: int = MAX_ENTRY_CHARS) -> str:
    """Compress a ``<think>`` trace into one line, without a model call.

    Deliberately cheap: reasoning traces here average ~1000 tokens and run to
    39k, so a second model call per turn would cost more than the turn itself.
    This keeps the sentences that state a finding ("the door opened", "so the
    key must come first") and drops narration ("let me look at the board").

    Args:
        text (str): the model's reasoning text.
        max_chars (int): cap on the returned line.

    Returns:
        str: a one-line summary, or ``""`` if nothing looked like a finding.
    """
    flat = " ".join(str(text or "").split())
    if not flat:
        return ""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]
    keep = [
        s for s in sentences
        # Findings state a conclusion or a rule; narration does not.
        if re.search(r"\b(so|therefore|means|because|confirms?|shows?|must|cannot|"
                     r"can't|doesn't|does not|opened|blocked|moves?|needs?)\b", s, re.I)
    ]
    # Prefer the last findings: later sentences reflect the settled view.
    chosen = " ".join(keep[-2:]) if keep else ""
    return _tidy(chosen)[:max_chars] if chosen else ""


def strip_memory_tags(text: str) -> str:
    """Remove the memory tags from a reply, leaving the prose.

    Args:
        text (str): assistant content.

    Returns:
        str: the text with every recognised tag block removed and blank runs
        collapsed. Unclosed or unknown tags are left untouched.
    """
    stripped = _TAG_RE.sub("", str(text or ""))
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def _norm_tokens(text: str) -> frozenset[str]:
    """Lowercased words without punctuation or stop words, for overlap tests."""
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return frozenset(w for w in words if w not in _STOP_WORDS)


def jaccard(a: str, b: str) -> float:
    """Word overlap of two entries (0 when either has no content words)."""
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


_NOTE_LEVEL_RE = re.compile(r"\b(?:level|L)\s*(\d+)\b", re.IGNORECASE)
_LEVEL_LABEL_RE = re.compile(r"^(?:L(\d+): |level (\d+): )")


def _unlabel(entry: str) -> str:
    """An entry without its ``L3: `` / ``level 3: `` label."""
    return _LEVEL_LABEL_RE.sub("", str(entry or ""), count=1)


def _entry_level(entry: str) -> int | None:
    match = _LEVEL_LABEL_RE.match(str(entry or ""))
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def strip_tags_in_code(code: str) -> str:
    """Remove recognised memory tags from python code, keeping its layout."""
    return _TAG_RE.sub("", str(code or ""))


@dataclass
class _Item:
    """A harness fact on the card: replaced in place under a stable id."""

    version: int
    section: str
    text: str
    order: float = 0.0
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Event:
    """One delta line waiting for the next turn message."""

    version: int
    id: str
    text: str
    bypass: bool = False


@dataclass
class MemoryLedger:
    """Per-game memory: world model, confirmed/refuted hypotheses, level notes.

    Attributes:
        world_model (list[str]): claims about the board and the actions.
        confirmed (list[str]): hypotheses the model reported as holding.
        refuted (list[str]): hypotheses it reported as failing -- kept so a dead
            idea is not retried.
        level_notes (list[str]): one ``level N: ...`` line per level seen.
        scratchpad (str): free-form notepad, replaced wholesale whenever the
            model writes one and returned verbatim otherwise.
        level (int): the level the ledger is currently recording against.
    """

    world_model: list[str] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)
    refuted: list[str] = field(default_factory=list)
    level_notes: list[str] = field(default_factory=list)
    scratchpad: str = ""
    level: int = 1
    # Findings from this level's think traces, folded into a note on level change.
    _pending_think: list[str] = field(default_factory=list, repr=False)
    # memory-v2. card_mode: ARC3_MEM_CARD (beliefs rules, no think fold into
    # level notes); solved_mode: ARC3_MEM_SOLVED (solved lines; the think fold
    # stays, but its notes never ride on a solved line); delta_mode: deltas
    # are queued for the turn message (placement delta or both).
    card_mode: bool = False
    solved_mode: bool = False
    delta_mode: bool = False
    # ARC3_MEM_DEBRIEF: a solved line carries the model's recap from its first
    # reasoning after the clear instead of "how" (a sentence of the clearing
    # turn's plan); model notes carry their level; the card puts the model's
    # text before the harness rows and cuts it last.
    debrief_mode: bool = False
    # ARC3_MEM_RECAP_ASK: the level-start message asks for "Level N recap:"
    # and the recap is read by extract_recap (labelled, then precise, then the
    # opening of the reasoning)
    recap_ask: bool = False
    # ARC3_MEM_RECAP_CHARS (with recap_ask): the recap's one cap for capture
    # and display, read by extract_recap(tidy=True); 0 keeps RECAP_CHARS
    recap_chars: int = 0
    # ARC3_MEM_RECAP_ASK_V2: a recap written as `Goal: ... Rules: ...` fills
    # the entry's Goal and Rules lines; `Corrects level K: ...` marks entry K
    recap_v2: bool = False
    _recap_parts: dict = field(default_factory=dict, repr=False)
    _corrections: dict = field(default_factory=dict, repr=False)
    card_chars: int = 3900
    version: int = 0
    shown_version: int = 0
    # raw reasoning of the last assistant message (the one that issued the
    # latest action); a cleared level's "how" sentence is distilled from it
    last_think: str = field(default="", repr=False)
    delta_chars_used: int = 0
    seen_chars_used: int = 0
    _items: dict = field(default_factory=dict, repr=False)
    _queue: list = field(default_factory=list, repr=False)
    _journal: list = field(default_factory=list, repr=False)
    _carry_streak: int = field(default=0, repr=False)
    _entry_versions: dict = field(default_factory=dict, repr=False)
    # queued level lines at or below this version belong to a finished level
    _level_floor: int = field(default=0, repr=False)
    # level notes made by the think fold, not written by the model
    _folded_notes: set = field(default_factory=set, repr=False)
    # debrief_mode: the level whose recap the next reasonings may hold, how
    # many reasonings are still checked, and the recaps found by level
    _debrief_for: int | None = field(default=None, repr=False)
    _debrief_tries: int = field(default=0, repr=False)
    _debriefs: dict = field(default_factory=dict, repr=False)
    # ARC3_MEM_EVERY_CALL: notes carry their level, open hypotheses are kept
    # (per level) instead of folded away, a recap must name its level, and
    # cleared levels are frozen into prev_entries / prev_text at commits
    layout_mode: bool = False
    # ARC3_MEM_LEGIBLE: card lines in plain words (memory_layout)
    legible: bool = False
    head_chars: int = layout.HEAD_CHARS
    # memory v2: how <previous_levels> entries are drawn (ARC3_MEM_* flags)
    head_options: layout.HeadOptions = layout.DEFAULT_HEAD
    open_notes: list[str] = field(default_factory=list)
    prev_entries: dict = field(default_factory=dict, repr=False)
    prev_text: str = ""
    # this level's GAME OVER lines for <this_level>
    layout_deaths: list = field(default_factory=list, repr=False)
    # ARC3_MEM_RULES: confirmed_rules.RuleLedger, the model's confirmed rules
    # per level (None: the flag is off and nothing is recorded or shown)
    rules: object = field(default=None, repr=False)
    # ARC3_CLEARED_BOARDS: level -> the clear line its entry carries
    cleared_facts: dict = field(default_factory=dict, repr=False)

    def absorb(self, text: str) -> int:
        """Record every memory tag in an assistant message.

        Args:
            text (str): the assistant's content (and/or reasoning).

        Returns:
            int: how many tags were recorded. 0 means the model wrote no tags,
            which is not an error -- the prose is still its own answer.
        """
        recorded = 0
        # Whole-notepad replacement: the last one written in this reply wins,
        # and a reply with no scratchpad leaves the existing page untouched.
        pages = _SCRATCHPAD_RE.findall(str(text or ""))
        if pages:
            page = pages[-1].strip()
            self.scratchpad = page[:MAX_SCRATCHPAD_CHARS].rstrip()
            recorded += 1
        for tag, attributes, body in _TAG_RE.findall(str(text or "")):
            if self.card_mode or self.layout_mode:
                recorded += self._absorb_card(tag.lower(), attributes, body)
                continue
            line = _tidy(body)
            if not line:
                continue
            tag = tag.lower()
            if tag == "world_model":
                _push(self.world_model, line, MAX_WORLD_MODEL)
            elif tag == "level_note":
                _push(self.level_notes, f"level {self.level}: {line}", MAX_LEVEL_NOTES)
            else:
                status = OPEN
                match = _STATUS_RE.search(attributes or "")
                if match is not None:
                    status = _STATUS_ALIASES.get(match.group(1).lower(), OPEN)
                if status == CONFIRMED:
                    _push(self.confirmed, line, MAX_PER_STATUS)
                elif status == REFUTED:
                    _push(self.refuted, line, MAX_PER_STATUS)
                else:
                    # An open hypothesis is a question, not knowledge: it is
                    # carried only until the level ends.
                    _push(self._pending_think, f"open: {line}", MAX_PER_STATUS)
            recorded += 1
        return recorded

    def note_think(self, reasoning: str) -> None:
        """Fold a turn's reasoning into this level's pending findings.

        Args:
            reasoning (str): the model's ``<think>`` text.

        Returns:
            None. Nothing is kept when the trace states no finding.
        """
        self.last_think = str(reasoning or "")
        if self.debrief_mode and self._debrief_for is not None:
            self._capture_debrief(self.last_think)
        if self.card_mode:
            # the card keeps the raw text for the solved line's "how" and no
            # longer turns cut-off fragments of it into level notes
            return
        line = summarize_think(reasoning)
        if line:
            _push(self._pending_think, line, MAX_PER_STATUS)

    def on_level_change(self, level: int) -> None:
        """Close the current level's note and start recording against ``level``.

        The pending findings are what the model worked out on the level it just
        left, so they are written as that level's note before switching.

        Args:
            level (int): the new level number.

        Returns:
            None. Does nothing when ``level`` is the current one.
        """
        if level == self.level:
            return
        if self.debrief_mode and int(level) > self.level:
            # the next two reasonings are the first on the new level; a recap
            # there is about the level just cleared
            self._debrief_for = int(level) - 1
            self._debrief_tries = DEBRIEF_TRIES
        if self._pending_think and not self.card_mode:
            note = f"level {self.level}: {_tidy(' '.join(self._pending_think))}"
            _push(self.level_notes, note, MAX_LEVEL_NOTES)
            if self.solved_mode:
                self._folded_notes.add(note)
        self._pending_think.clear()
        self.level = max(1, int(level))

    def render(self) -> str:
        """Render the ledgers as the memory block for the next prompt.

        Returns:
            str: a ``<memory>`` block in the same XML the model writes, so the
            format it reads is the format it is asked to produce. ``""`` when
            every ledger is empty, so turn one carries no dead weight.
        """
        lines: list[str] = []
        for entry in self.world_model:
            lines.append(f"  <world_model>{entry}</world_model>")
        for entry in self.confirmed:
            lines.append(f'  <hypothesis status="confirmed">{entry}</hypothesis>')
        for entry in self.refuted:
            lines.append(f'  <hypothesis status="refuted">{entry}</hypothesis>')
        for entry in self.level_notes:
            lines.append(f"  <level_note>{entry}</level_note>")
        if self.scratchpad:
            # Verbatim, including newlines: reflowing someone's sketch of a
            # board into one line destroys the thing they drew.
            lines.append(f"  <scratchpad>\n{self.scratchpad}\n  </scratchpad>")
        if not lines:
            return ""
        return "<memory>\n" + "\n".join(lines) + "\n</memory>"

    # ------------------------------------------------------------ memory-v2

    def _absorb_card(self, tag: str, attributes: str, body: str) -> int:
        """Card-mode recording of one tag: longer caps, overlap dedup.

        In debrief mode world-model and hypothesis entries are labelled with
        the level they were written on (``L3: ...``), as level notes are."""
        line = " ".join(str(body or "").replace("</memory>", " ").split())
        if not line:
            return 0
        level = self._note_level(line) if self.layout_mode else self.level
        label = f"L{level}: " if (self.debrief_mode or self.layout_mode) else ""
        if tag == "world_model":
            self._push_belief("world_model", cut_words(line, CARD_ENTRY_CHARS["world_model"]), label=label)
        elif tag == "level_note":
            note = cut_words(line, CARD_ENTRY_CHARS["level_note"])
            self._push_belief("level_note", f"level {level}: {note}")
        else:
            status = OPEN
            match = _STATUS_RE.search(attributes or "")
            if match is not None:
                status = _STATUS_ALIASES.get(match.group(1).lower(), OPEN)
            line = cut_words(line, CARD_ENTRY_CHARS["hypothesis"])
            if status in (CONFIRMED, REFUTED):
                self._push_belief(status, line, label=label)
            elif self.layout_mode:
                self._push_open(label + line)
            else:
                _push(self._pending_think, f"open: {line}", MAX_PER_STATUS)
        return 1

    def _note_level(self, text: str) -> int:
        """The level a note is about (layout mode): the first level it names
        ("Level2 solved by ...", "L1 goal: ...") when that level is not
        later than the current one and not yet frozen into
        ``<previous_levels>``; the current level otherwise. Notes written in
        the first turn after a clear are often about the level just cleared:
        in user-combo-10x v1's tn36 they were labelled with the new level."""
        match = _NOTE_LEVEL_RE.search(str(text or ""))
        if match:
            named = int(match.group(1))
            if 1 <= named < self.level and named not in self.prev_entries:
                return named
        return self.level

    def _push_open(self, line: str) -> None:
        """An open hypothesis (layout mode): kept until it is confirmed or
        refuted, newest last, overlap-deduplicated like the other notes."""
        for entry in list(self.open_notes):
            if entry.lower() == line.lower() or jaccard(_unlabel(entry), _unlabel(line)) >= JACCARD_SAME:
                self.open_notes.remove(entry)
                self._entry_versions.pop(("open", entry), None)
        self.open_notes.append(line)
        while len(self.open_notes) > MAX_PER_STATUS:
            self._entry_versions.pop(("open", self.open_notes.pop(0)), None)
        self._entry_versions[("open", line)] = self._bump()

    def _belief_lists(self) -> dict[str, list[str]]:
        return {
            "world_model": self.world_model,
            "confirmed": self.confirmed,
            "refuted": self.refuted,
            "level_note": self.level_notes,
        }

    def _drop_entry(self, section: str, entry: str) -> None:
        entries = self._belief_lists()[section]
        if entry in entries:
            entries.remove(entry)
        self._entry_versions.pop((section, entry), None)
        event_id = f"m:{section}:{entry}"
        self._queue = [e for e in self._queue if e.id != event_id]

    def _push_belief(self, section: str, line: str, *, label: str = "") -> bool:
        """Record one model entry; newest wins on overlap.

        - Same list, word overlap >= 0.6: the older entry is replaced and the
          new one goes last.
        - A confirmed/refuted pair at >= 0.6: the older one is removed.
        - ``Correction:``, ``Fix:`` or ``Actually,`` drops the prefix and
          replaces the best match at >= 0.35 across the world model and the
          hypotheses.
        - An exact restatement moves the entry last and adds no delta line.
        - ``label`` (debrief mode, ``L<N>: ``) goes in front after a
          correction prefix is handled.

        Returns:
            bool: True when a new entry was recorded.
        """
        lists = self._belief_lists()
        caps = {"world_model": MAX_WORLD_MODEL, "confirmed": MAX_PER_STATUS,
                "refuted": MAX_PER_STATUS, "level_note": MAX_LEVEL_NOTES}
        target = lists[section]
        if section != "level_note":
            match = _CORRECTION_RE.match(line)
            if match:
                line = line[match.end():].strip()
                if not line:
                    return False
                line = line[0].upper() + line[1:]
                best: tuple[float, str, str] | None = None
                for name in ("world_model", "confirmed", "refuted"):
                    for entry in lists[name]:
                        score = jaccard(line, entry)
                        if best is None or score > best[0]:
                            best = (score, name, entry)
                if best is not None and best[0] >= JACCARD_CORRECTION:
                    self._drop_entry(best[1], best[2])
        if label:
            line = label + line
        tokens = _norm_tokens(line)
        for entry in list(target):
            if entry.lower() == line.lower() or (tokens and _norm_tokens(entry) == tokens):
                target.remove(entry)
                target.append(entry)
                return False
        for entry in list(target):
            if jaccard(line, entry) >= JACCARD_SAME:
                self._drop_entry(section, entry)
        if section in (CONFIRMED, REFUTED):
            other = REFUTED if section == CONFIRMED else CONFIRMED
            for entry in list(lists[other]):
                if jaccard(line, entry) >= JACCARD_SAME:
                    self._drop_entry(other, entry)
            if self.layout_mode:
                # a tested hypothesis is no longer open
                for entry in list(self.open_notes):
                    if jaccard(_unlabel(line), _unlabel(entry)) >= JACCARD_SAME:
                        self.open_notes.remove(entry)
                        self._entry_versions.pop(("open", entry), None)
        target.append(line)
        while len(target) > caps[section]:
            self._drop_entry(section, target[0])
        version = self._bump()
        self._entry_versions[(section, line)] = version
        self._queue_event(f"m:{section}:{line}", f"{_SECTION_LABELS[section]}: {line}", version)
        return True

    # ---- harness items

    def _bump(self) -> int:
        self.version += 1
        return self.version

    def _queue_event(self, event_id: str, text: str, version: int, bypass: bool = False) -> None:
        if not self.delta_mode:
            return
        event = _Event(version, event_id, text.replace("</memory>", ""), bypass)
        self._queue = [e for e in self._queue if e.id != event_id]
        self._queue.append(event)
        self._journal.append(event)

    def set_item(
        self,
        item_id: str,
        section: str,
        text: str,
        *,
        order: float = 0.0,
        data: dict | None = None,
        event: str | None = None,
        bypass: bool = False,
        bump: bool = True,
    ) -> int:
        """Put a harness fact on the card under a stable id (replaced, never duplicated).

        Args:
            bump: False updates the text in place without a new version (the
                action counts change every turn; only class changes are news).
            event: the delta line announcing it, if any.
            bypass: the delta line ignores the per-turn hold (solved levels,
                first death, clears).

        Returns:
            int: the item's version.
        """
        text = str(text or "").replace("</memory>", "")
        item = self._items.get(item_id)
        version = self._bump() if (item is None or bump) else item.version
        self._items[item_id] = _Item(version, section, text, float(order), dict(data or {}))
        if event:
            self._queue_event(item_id, event, version, bypass)
        return version

    def add_event(self, event_id: str, text: str, *, bypass: bool = False) -> int:
        """A delta-only line (its fact is already in another card item)."""
        version = self._bump()
        self._queue_event(event_id, text, version, bypass)
        return version

    def remove_section(self, section: str) -> None:
        for item_id in [k for k, v in self._items.items() if v.section == section]:
            del self._items[item_id]
        self._queue = [e for e in self._queue if self._event_alive(e)]

    def clear_level_facts(self) -> None:
        """A new level started: its hazards leave the card, and the queued
        hazard and first-effect lines of the level just left are dropped (a
        clear before the hold ran out would show them a level late). Solved
        lines and model notes stay queued."""
        self._level_floor = self.version
        self.remove_section("hazard")

    def items(self, section: str) -> list[_Item]:
        return sorted((v for v in self._items.values() if v.section == section), key=lambda v: v.order)

    def level_note_for(self, level: int) -> str:
        prefix = f"level {int(level)}: "
        for entry in reversed(self.level_notes):
            # a solved line carries the model's own note, never the think fold
            if entry.startswith(prefix) and entry not in self._folded_notes:
                return entry[len(prefix):]
        return ""

    def add_solved(self, level: int, path: list[str], spent: int, deaths: int, *, winning: int | None = None,
                   keys: list[str] | None = None, attempts: int | None = None) -> str:
        """Write the never-evicted ``<solved>`` record of a cleared level.

        ``winning`` (ARC3_MEM_PATH): the winning attempt's loop-free length,
        given when ``path`` is a shorter route on record. In debrief mode the
        line carries the level's recap (if one was found yet) and no "how"
        or note; a recap found later is added by ``_capture_debrief``.
        ``keys`` (ARC3_MEM_ROUTE_FORMAT=readable, passed only then): each
        path action's effect key, kept in the record's data. ``attempts``
        (ARC3_MEM_HEAD_WORDS, passed only then): the level's attempt count."""
        if self.debrief_mode:
            line, kept = solved_line(level, path, spent, deaths, winning=winning,
                                     debrief=self._debriefs.get(int(level), ""))
            data = {"level": int(level), "path": list(path), "spent": int(spent), "deaths": int(deaths), **kept}
            if winning:
                data["winning"] = int(winning)
            if keys is not None:
                data["keys"] = [str(k) for k in keys]
            if attempts is not None:
                data["attempts"] = int(attempts)
            if self.legible:
                line = self._solved_text(data, kept)
            self.set_item(f"solved:{level}", "solved", line, order=level, data=data,
                          event=line if self.legible else f"solved {line}", bypass=True)
            return line
        note = self.level_note_for(level)
        line, kept = solved_line(level, path, spent, deaths, think=self.last_think, note=note, winning=winning)
        data = {"level": int(level), "path": list(path), "spent": int(spent), "deaths": int(deaths), **kept}
        if winning:
            data["winning"] = int(winning)
        if keys is not None:
            data["keys"] = [str(k) for k in keys]
        if attempts is not None:
            data["attempts"] = int(attempts)
        if kept.get("note"):
            # the note now rides on the solved line; its own delta would repeat it
            prefix = f"m:level_note:level {int(level)}: "
            self._queue = [e for e in self._queue if not e.id.startswith(prefix)]
        if self.legible:
            line = self._solved_text(data, kept)
        self.set_item(f"solved:{level}", "solved", line, order=level, data=data,
                      event=line if self.legible else f"solved {line}", bypass=True)
        return line

    def _solved_text(self, d: dict, parts: dict, *, path_chars: int = SOLVED_PATH_CHARS) -> str:
        """A solved line from its data: the compact form, or (``legible``)
        ``Level 2: cleared in 16 actions (23 spent, no deaths). Route: ...``
        with ``How:``, ``Your note:`` and ``Your recap:`` parts."""
        if not self.legible:
            head = solved_head(d["level"], d["path"], d["spent"], d["deaths"],
                               path_chars=path_chars, winning=d.get("winning"))
            return render_solved(head, parts)
        text = layout.solved_text(d["level"], d["path"], d["spent"], d["deaths"],
                                  route_chars=layout.ROUTE_CHARS if path_chars > 0 else 0,
                                  winning=d.get("winning"))
        for name, label in (("how", "How"), ("note", "Your note"), ("debrief", "Your recap")):
            if parts.get(name):
                text += f" {label}: {parts[name]}"
        return text

    def _capture_debrief(self, reasoning: str) -> None:
        """Look for the recap of ``_debrief_for`` in one reasoning (debrief mode).

        The first recap found is kept; after ``DEBRIEF_TRIES`` reasonings
        without one the level gets none. A solved line already written is
        rewritten in place with the recap: the card that shows it is drawn at
        the next trim, so no rendered prompt changes."""
        level = int(self._debrief_for or 0)
        self._debrief_tries -= 1
        # precise: only a recap about this level (the PR #148 review: 14 of 83
        # captures named another level, 4 were not recaps)
        if self.recap_ask and self.recap_chars > 0:
            text = extract_recap(reasoning, level, max_chars=self.recap_chars, tidy=True)
        elif self.recap_ask:
            text = extract_recap(reasoning, level)
        else:
            text = extract_debrief(reasoning, level + 1, precise=True)
        if text or self._debrief_tries <= 0:
            self._debrief_for = None
        if not text or level < 1:
            return
        self._debriefs[level] = text
        if self.recap_v2:
            parts = split_recap(text, level)
            if parts is not None:
                goal, rules, corrections = parts
                self._recap_parts[level] = (goal, rules)
                for earlier, claim in corrections:
                    self._corrections.setdefault(int(earlier), []).append((level, claim))
        item = self._items.get(f"solved:{level}")
        if item is not None and item.data:
            d = item.data
            line, kept = solved_line(d["level"], d["path"], d["spent"], d["deaths"],
                                     winning=d.get("winning"), debrief=text)
            data = {**d, **kept}
            if self.legible:
                line = self._solved_text(data, kept)
            self.set_item(f"solved:{level}", "solved", line, order=item.order, data=data)

    # ---- the card

    def _solved_lines(self, stage: int) -> list[str]:
        """Stage 0: every line whole; 1: lines older than the newest 3 lose
        how and note; 2: those are compressed to counts; 3: all are."""
        solved = self.items("solved")
        older, newest = solved[:-3], solved[-3:]
        if stage >= 3:
            older, newest = solved, []

        def head_only(item: _Item) -> str:
            d = item.data
            if not d:
                return item.text
            return self._solved_text(d, {})

        lines: list[str] = []
        if older:
            if stage <= 0:
                lines.extend(item.text for item in older)
            elif stage == 1:
                lines.extend(head_only(item) for item in older)
            else:
                data = [item.data for item in older if item.data]
                if len(data) == len(older) and self.legible:
                    lines.extend(self._solved_text(d, {}, path_chars=0) for d in data)
                elif len(data) == len(older):
                    first, last = data[0]["level"], data[-1]["level"]
                    span = f"L{first}" if first == last else f"L{first}-L{last}"
                    lines.append(
                        f"{span}: {'/'.join(str(len(d['path'])) for d in data)} acts, "
                        f"spent {'/'.join(str(d['spent']) for d in data)}, "
                        f"deaths {'/'.join(str(d['deaths']) for d in data)}"
                    )
                else:
                    lines.extend(item.text for item in older)
        lines.extend(item.text for item in newest)
        return [f"  <solved>{line}</solved>" for line in lines]

    def _belief_entries(self, solved_levels: set[int]) -> list[tuple[str, str, int]]:
        """(kind, rendered line, version) for the model's own entries, card order."""
        out: list[tuple[str, str, int]] = []
        version = self._entry_versions.get
        for entry in self.world_model:
            out.append(("world_model", f"  <world_model>{entry}</world_model>", version(("world_model", entry), 0)))
        for entry in self.confirmed:
            out.append(("confirmed", f'  <hypothesis status="confirmed">{entry}</hypothesis>',
                        version(("confirmed", entry), 0)))
        for entry in self.refuted:
            out.append(("refuted", f'  <hypothesis status="refuted">{entry}</hypothesis>',
                        version(("refuted", entry), 0)))
        for entry in self.level_notes:
            match = re.match(r"level (\d+): ", entry)
            if match and int(match.group(1)) in solved_levels and entry not in self._folded_notes:
                continue  # carried on that level's <solved> line
            out.append(("level_note", f"  <level_note>{entry}</level_note>", version(("level_note", entry), 0)))
        if self.scratchpad:
            out.append(("scratchpad", f"  <scratchpad>\n{self.scratchpad}\n  </scratchpad>", -1))
        return out

    def render_card(self) -> str:
        """The memory-v2 card: harness facts first, then the model's notes.

        Order (also the order that survives truncation): ``<solved>`` by level,
        ``<actions level="N">``, ``<hazard>`` (newest 3, current level), then world model,
        confirmed, refuted (oldest first), level notes. Over ``card_chars``:
        the oldest non-refuted model entries go first, then older solved lines
        lose their how and note, then their paths; action rows are cut last. Refuted entries and hazards go only if nothing else is
        left to cut.

        Returns:
            str: a ``<memory>`` block, ``""`` when there is nothing to show.
        """
        if self.debrief_mode:
            return self._render_card_debrief()
        solved_levels = {
            int(item.data["level"]) for item in self.items("solved")
            if item.data and item.data.get("note")
        }
        beliefs = self._belief_entries(solved_levels)
        dropped: set[int] = set()
        actions = self._items.get("actions")
        action_rows = [row for row in (actions.text.split("\n") if actions and actions.text else []) if row]
        hazards = [f"  <hazard>{item.text}</hazard>" for item in self.items("hazard")][-MAX_HAZARDS:]
        stage = 0

        def section_len(lines: list[str]) -> int:
            return sum(len(line) + 1 for line in lines)

        def belief_lines() -> list[str]:
            return [line for i, (_, line, _) in enumerate(beliefs) if i not in dropped]

        # the level the counts are for: a head card can outlive its level
        level = actions.data.get("level") if actions else None
        opening = f'  <actions level="{level}">' if level else "  <actions>"

        def action_block() -> list[str]:
            if not action_rows:
                return []
            return [opening, *[f"  {row}" for row in action_rows], "  </actions>"]

        def build() -> str:
            lines = [*self._solved_lines(stage), *action_block(), *hazards, *belief_lines()]
            if not lines:
                return ""
            return "<memory>\n" + "\n".join(lines) + "\n</memory>"

        def next_droppable(include_refuted: bool = False) -> int | None:
            order = sorted(
                (version, i) for i, (kind, _, version) in enumerate(beliefs)
                if i not in dropped and (include_refuted or kind != "refuted")
            )
            return order[0][1] if order else None

        text = build()
        cap = max(200, int(self.card_chars))
        while len(text) > cap:
            candidate = next_droppable()
            if candidate is not None and section_len(belief_lines()) > SOFT_CAPS["beliefs"]:
                dropped.add(candidate)
            elif stage < 2 and len(self.items("solved")) > 3 and section_len(self._solved_lines(stage)) > SOFT_CAPS["solved"]:
                stage += 1
            elif len(action_rows) > 1 and section_len(action_block()) > SOFT_CAPS["actions"]:
                action_rows.pop()
            elif candidate is not None:
                dropped.add(candidate)
            elif stage < 3 and self.items("solved"):
                stage += 1
            elif action_rows:
                action_rows.pop()
            elif next_droppable(include_refuted=True) is not None:
                dropped.add(next_droppable(include_refuted=True))
            elif len(hazards) > 1:
                hazards.pop(0)
            else:
                break
            text = build()
        return text

    def _render_card_debrief(self) -> str:
        """The debrief-mode card: the model's text first, cut last.

        Order: ``<solved>`` by level (counts, path, the model's ``debrief:``),
        then the model's notes (world model, confirmed, refuted, level notes,
        scratchpad), then ``<hazard>``, then ``<actions level="N">``. Over
        ``card_chars`` the harness rows go first: action rows (all of them),
        then the paths of solved lines older than the newest 3, then every
        path, then all but the newest hazard. Only then the oldest model
        notes (refuted ones last), the last hazard, the older debriefs cut to
        120 chars, and finally the oldest debriefs.
        """
        solved = self.items("solved")
        beliefs = self._belief_entries(set())
        dropped: set[int] = set()
        actions = self._items.get("actions")
        action_rows = [row for row in (actions.text.split("\n") if actions and actions.text else []) if row]
        hazards = [f"  <hazard>{item.text}</hazard>" for item in self.items("hazard")][-MAX_HAZARDS:]
        level = actions.data.get("level") if actions else None
        opening = f'  <actions level="{level}">' if level else "  <actions>"
        path_stage = 0  # 0: every path; 1: the newest 3 keep theirs; 2: none
        short: set[int] = set()
        gone: set[int] = set()

        def solved_lines() -> list[str]:
            out: list[str] = []
            for i, item in enumerate(solved):
                d = item.data
                if not d:
                    out.append(f"  <solved>{item.text}</solved>")
                    continue
                keep_path = path_stage == 0 or (path_stage == 1 and i >= len(solved) - 3)
                debrief = "" if i in gone else d.get("debrief", "")
                if debrief and i in short:
                    debrief = cut_words(debrief, 120)
                text = self._solved_text(d, {"debrief": debrief} if debrief else {},
                                         path_chars=SOLVED_PATH_CHARS if keep_path else 0)
                out.append(f"  <solved>{text}</solved>")
            return out

        def action_block() -> list[str]:
            if not action_rows:
                return []
            return [opening, *[f"  {row}" for row in action_rows], "  </actions>"]

        def build() -> str:
            lines = [*solved_lines(),
                     *[line for i, (_, line, _) in enumerate(beliefs) if i not in dropped],
                     *hazards, *action_block()]
            if not lines:
                return ""
            return "<memory>\n" + "\n".join(lines) + "\n</memory>"

        def next_belief(include_refuted: bool) -> int | None:
            order = sorted(
                (version, i) for i, (kind, _, version) in enumerate(beliefs)
                if i not in dropped and (include_refuted or kind != "refuted")
            )
            return order[0][1] if order else None

        with_debrief = [i for i, item in enumerate(solved) if item.data and item.data.get("debrief")]
        text = build()
        cap = max(200, int(self.card_chars))
        while len(text) > cap:
            if action_rows:
                action_rows.pop()
            elif path_stage < 2 and any(item.data and item.data.get("path") for item in solved):
                path_stage += 1
            elif len(hazards) > 1:
                hazards.pop(0)
            elif next_belief(False) is not None:
                dropped.add(next_belief(False))
            elif next_belief(True) is not None:
                dropped.add(next_belief(True))
            elif hazards:
                hazards.pop(0)
            elif [i for i in with_debrief[:-1] if i not in short]:
                short.add(next(i for i in with_debrief[:-1] if i not in short))
            elif [i for i in with_debrief if i not in gone]:
                gone.add(next(i for i in with_debrief if i not in gone))
            else:
                break
            text = build()
        return text

    # ---- deltas

    def take_delta(self, *, hold: bool = False, cap: int = 480, game_cap: int = 12000) -> str:
        """The ``<memory_update>`` block for a new turn message, or ``""``.

        Lines added since they were last shown, oldest first, up to ``cap``
        chars; the rest is carried to the next message, and on the third
        message in a row that carries, the cap doubles to catch up. While
        ``hold`` is set only bypass lines (solved levels, a first death, a
        clear) are taken. Nothing more is sent once ``game_cap`` chars of
        deltas went out this game.
        """
        if not self._queue or self.delta_chars_used >= game_cap:
            return ""
        eligible = [e for e in self._queue if e.bypass or not hold]
        if not eligible:
            return ""
        limit = cap * 2 if self._carry_streak >= 2 else cap
        # the wrapper costs 34 chars; the game cap is never exceeded
        room = game_cap - self.delta_chars_used - 34
        taken: list[_Event] = []
        size = 0
        for event in eligible:
            need = len(event.text) + 3
            if size + need > room or (taken and size + need > limit):
                break
            taken.append(event)
            size += need
        if not taken:
            return ""
        self._carry_streak = self._carry_streak + 1 if len(taken) < len(eligible) else 0
        taken_ids = {id(e) for e in taken}
        self._queue = [e for e in self._queue if id(e) not in taken_ids]
        lines: list[str] = []
        previous_fx = False
        for event in taken:
            is_fx = event.id.startswith("fx:")
            if is_fx and previous_fx and lines:
                lines[-1] += f"; {event.text}"
            else:
                lines.append(f"+ {event.text}")
            previous_fx = is_fx
        block = "<memory_update>\n" + "\n".join(lines) + "\n</memory_update>"
        self.delta_chars_used += len(block)
        return block

    def pending(self) -> list[_Event]:
        return list(self._queue)

    def mark_shown(self, version: int | None = None) -> None:
        """A head card at ``version`` was shown: drop the deltas it covers."""
        version = self.version if version is None else int(version)
        self.shown_version = version
        self._queue = [e for e in self._queue if e.version > version]

    def snapshot_display(self) -> tuple:
        """What a dropped opener could have consumed; see ``restore_display``."""
        self._journal = []
        return (self.shown_version, list(self._queue), self._carry_streak,
                self.delta_chars_used, self.seen_chars_used)

    def restore_display(self, snapshot: tuple) -> None:
        """Undo the display state of an abandoned turn.

        Lines taken for the dropped opener (or covered by a dropped head card)
        go back into the queue; lines recorded during the turn stay queued.
        """
        shown, queue, carry, delta_chars, seen_chars = snapshot
        merged = {e.id: e for e in queue if self._event_alive(e)}
        for event in self._journal:
            if self._event_alive(event):
                merged[event.id] = event
        self._queue = sorted(merged.values(), key=lambda e: e.version)
        self._journal = []
        self.shown_version = shown
        self._carry_streak = carry
        self.delta_chars_used = delta_chars
        self.seen_chars_used = seen_chars

    # ---- ARC3_MEM_EVERY_CALL

    def notes_for_level(self, level: int) -> list[tuple[str, str]]:
        """The model's own notes written on ``level``, newest first, as
        (label, text): world model, confirmed, refuted, open hypotheses and
        level notes, each without its level label."""
        kinds = (
            ("world_model", "World model", self.world_model),
            ("confirmed", "Confirmed", self.confirmed),
            ("refuted", "Refuted", self.refuted),
            ("open", "Open", self.open_notes),
            ("level_note", "Level note", self.level_notes),
        )
        found: list[tuple[int, str, str]] = []
        for section, label, entries in kinds:
            for entry in entries:
                if _entry_level(entry) == int(level) and entry not in self._folded_notes:
                    found.append((self._entry_versions.get((section, entry), 0), label, _unlabel(entry)))
        found.sort(key=lambda item: -item[0])
        return [(label, text) for _, label, text in found]

    def layout_pending(self) -> list[int]:
        """Cleared levels (a solved item exists) not yet in ``<previous_levels>``."""
        return sorted(
            int(item.data["level"]) for item in self.items("solved")
            if item.data and int(item.data["level"]) not in self.prev_entries
        )

    def layout_has_recap(self, level: int) -> bool:
        """Whether ``level`` has the model's recap (ARC3_LEVEL_COMPACT_NEEDS_RECAP)."""
        if self._debriefs.get(int(level)):
            return True
        return bool((getattr(self, "debriefs", None) or {}).get(int(level)))

    def layout_waiting_for_recap(self, level: int) -> bool:
        """Debrief mode: the next reasonings may still hold ``level``'s recap."""
        return bool(self.debrief_mode and self._debrief_for == int(level))

    def _layout_entry(self, level: int) -> layout.LevelEntry | None:
        item = self._items.get(f"solved:{level}")
        if item is None or not item.data:
            return None
        d = item.data
        goal = rules = ""
        # arm/level-debrief's MemoryLedger.debriefs: level -> (fields, free text)
        pinned = (getattr(self, "debriefs", None) or {}).get(int(level))
        if pinned:
            fields = pinned[0] if isinstance(pinned, tuple) else {}
            goal = str(fields.get("GOAL", "") or "")
            rules = str(fields.get("MECHANICS", "") or "")
        recap = self._debriefs.get(int(level), "")
        if int(level) in self._recap_parts:
            # ARC3_MEM_RECAP_ASK_V2: the recap is shown as its Goal and Rules
            goal, rules = self._recap_parts[int(level)]
            recap = ""
        return layout.LevelEntry(
            cleared=str(self.cleared_facts.get(int(level), "")),
            level=int(level), path=list(d.get("path") or []), spent=int(d.get("spent", 0)),
            deaths=int(d.get("deaths", 0)), winning=d.get("winning"),
            recap=recap, goal=goal, rules=rules,
            notes=self.notes_for_level(level)[: layout.NOTES_PER_LEVEL],
            keys=list(d.get("keys") or []), attempts=int(d.get("attempts") or 0),
            corrected=list(self._corrections.get(int(level), [])),
            confirmed=self.rules.rules_for(level) if self.rules is not None else [],
        )

    def layout_commit(self, levels: list[int]) -> bool:
        """Freeze ``levels`` into ``<previous_levels>`` and redraw it.

        A committed entry keeps the notes it had; a recap found after its
        commit is added at the next redraw. Returns True when the rendered
        text changed (the caller then pays one cache re-prefill)."""
        for level in levels:
            entry = self._layout_entry(level)
            if entry is not None:
                self.prev_entries[int(level)] = entry
        return self.layout_redraw()

    def layout_stage(self, levels: list[int]) -> list[str]:
        """Freeze ``levels`` as ``layout_commit`` does, without redrawing
        ``prev_text``; return each entry as a ``<cleared_level>`` block, the
        text ``<previous_levels>`` will hold for it (ARC3_MEM_APPEND_COMMIT)."""
        blocks: list[str] = []
        for level in levels:
            entry = self._layout_entry(level)
            if entry is None:
                continue
            self.prev_entries[int(level)] = entry
            text = layout.render_entry(entry, legible=self.legible, options=self.head_options)
            blocks.append(f"<{layout.CLEARED_TAG}>\n{text}\n</{layout.CLEARED_TAG}>")
        return blocks

    def layout_redraw(self) -> bool:
        for level, entry in self.prev_entries.items():
            if level in self._recap_parts:
                # ARC3_MEM_RECAP_ASK_V2: a recap found after the commit, as Goal and Rules
                if not (entry.recap or entry.goal or entry.rules):
                    entry.goal, entry.rules = self._recap_parts[level]
            elif not entry.recap and self._debriefs.get(level):
                entry.recap = self._debriefs[level]
            if self._corrections.get(level):
                entry.corrected = list(self._corrections[level])
            if not entry.cleared and self.cleared_facts.get(level):
                # ARC3_CLEARED_BOARDS: a clear line recorded after the commit
                entry.cleared = str(self.cleared_facts[level])
        text = layout.render_previous_levels(
            list(self.prev_entries.values()), max_chars=self.head_chars, legible=self.legible,
            options=self.head_options,
        )
        changed = text != self.prev_text
        self.prev_text = text
        return changed

    def _event_alive(self, event: _Event) -> bool:
        if event.id.startswith(_LEVEL_EVENT_PREFIXES) and event.version <= self._level_floor:
            return False
        if event.id.startswith("m:"):
            _, section, entry = event.id.split(":", 2)
            return (section, entry) in self._entry_versions
        if event.id in ("actions",) or event.id.startswith(("solved:", "hazard:")):
            return event.id in self._items
        return True
