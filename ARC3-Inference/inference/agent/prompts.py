"""Prompt templates for the analyzer agent."""

import logging

from inference.utils.grid_utils import ARC_COLOR_LEGEND

log = logging.getLogger(__name__)


def apply_rows(text: str, rows, *, where: str) -> str:
    """Franzen's `(old, new)` replacements, guarded.

    A bare `str.replace` does nothing when `old` has drifted and replaces
    twice when it occurs twice; here a row whose `old` does not occur exactly
    once in `text` is logged and skipped, and the text is returned without it.
    Tests assert `count(old) == 1` on the real rendered text for every row.
    """
    for old, new in rows:
        found = text.count(old)
        if found != 1:
            log.warning("prompt row skipped (%d matches): %s", found, where)
            continue
        text = text.replace(old, new)
    return text

LEVEL_TRANSFER_SYSTEM_GUIDANCE = (
    "- Levels usually build on mechanics learned in earlier levels, especially the most "
    "recent one. Carry forward supported knowledge as a starting hypothesis, while "
    "re-checking anything contradicted by new evidence. New levels often introduce "
    "additional mechanics, sometimes through unfamiliar board elements. These additions "
    "are often important for solving the level. The goal may remain the same but require "
    "new mechanics to reach it, or the goal itself may change.\n"
)

# Ported from the Wang-fork memory ledger (inference/agent/memory_ledger.py),
# shown only right after history is trimmed (ARC3_LEDGER_ON_TRIM).
LEDGER_ON_TRIM_ADDENDUM = (
    "\n\nMemory that survives history trimming:\n"
    "- When this conversation grows too long, its oldest turns are dropped. The oldest message that "
    "remains then starts with a `<memory>` block holding what you recorded, so findings from dropped "
    "turns are not lost.\n"
    "- To record something, wrap one line in a tag anywhere in your reply or reasoning. Text outside "
    "the tags is unaffected.\n"
    "    `<world_model>the blue block is the piece you steer</world_model>` -- what the board is and "
    "what the actions do.\n"
    "    `<hypothesis status=\"confirmed\">UP moves the blue block up one cell</hypothesis>` -- "
    "tested, and it held.\n"
    "    `<hypothesis status=\"refuted\">MOUSE on a wall opens it</hypothesis>` -- tested, and it "
    "failed.\n"
    "    `<level_note>level 2 needed the key before the door</level_note>` -- what mattered on this "
    "level.\n"
    "- One claim per tag. A refuted hypothesis is as valuable as a confirmed one: it is what stops you "
    "repeating a dead experiment once the evidence for it has been dropped.\n"
)

# memory-v2 (ARC3_MEM_*): static system-prompt lines describing the harness
# memory. Built once per run from the enabled components, so the cached system
# prefix never changes during a game. Facts, not advice.
_MEMORY_TAG_EXAMPLES = (
    "    `<world_model>the blue block is the piece you steer</world_model>` -- what the board is and "
    "what the actions do.\n"
    "    `<hypothesis status=\"confirmed\">UP moves the blue block up one cell</hypothesis>` -- "
    "tested, and it held.\n"
    "    `<hypothesis status=\"refuted\">MOUSE on a wall opens it</hypothesis>` -- tested, and it "
    "failed.\n"
    "    `<level_note>level 2 needed the key before the door</level_note>` -- what mattered on this "
    "level.\n"
)
MEMORY_HEAD_CARD_LINE = (
    "- When this conversation grows too long, its oldest turns are dropped. The oldest message that "
    "remains then starts with a `<memory>` block.\n"
)
MEMORY_DELTA_LINE = (
    "- A turn message may carry a `<memory_update>` block: the memory lines added since you last "
    "saw them.\n"
)
MEMORY_SOLVED_LINE = (
    "- `<solved>` lines are written by the harness when a level is cleared: the winning attempt's "
    "path with its loops cut out (U D L R S are UP DOWN LEFT RIGHT SPACE, M(r,c) a click at row r, "
    "column c, a number repeats the action), the actions spent on the level and its deaths.\n"
)
# ARC3_MEM_PATH: the solved path is the shortest route on record
MEMORY_SOLVED_ROUTE_LINE = (
    "- `<solved>` lines are written by the harness when a level is cleared: the shortest route on "
    "record to the winning board, then the winning action (U D L R S are UP DOWN LEFT RIGHT SPACE, "
    "M(r,c) a click at row r, column c, a number repeats the action); \"(K in the winning attempt)\" "
    "follows when that attempt, with its loops cut out, was longer; then the actions spent on the "
    "level and its deaths.\n"
)
MEMORY_EFFECTS_LINE = (
    "- `<actions level=\"N\">` lines are counted by the harness from the frames: what each action, or a "
    "click on a cell of each colour, did on level N, then after `| game` in the whole game. moved = "
    "cells moved and the colour counts stayed the same; changed = colour counts changed; edge only = "
    "only cells near the edge changed; "
    "no change; anim only = cells changed and changed back. `<hazard>` lines give the action and the "
    "board from which an attempt ended in GAME OVER.\n"
)
# Placement delta has no head card, so the facts arrive only as the
# `<memory_update>` lines' plain text: these describe those forms (and the
# per-action count table, which exists only on the card, is not mentioned).
MEMORY_SOLVED_DELTA_LINE = (
    "- `solved L<N>:` lines are written by the harness when a level is cleared: the winning "
    "attempt's path with its loops cut out (U D L R S are UP DOWN LEFT RIGHT SPACE, M(r,c) a click at "
    "row r, column c, a number repeats the action), the actions spent on the level and its deaths.\n"
)
# ARC3_MEM_PATH without a head card (placement delta): the same facts as
# MEMORY_SOLVED_ROUTE_LINE in the `solved L<N>:` delta form
MEMORY_SOLVED_ROUTE_DELTA_LINE = (
    "- `solved L<N>:` lines are written by the harness when a level is cleared: the shortest route "
    "on record to the winning board, then the winning action (U D L R S are UP DOWN LEFT RIGHT "
    "SPACE, M(r,c) a click at row r, column c, a number repeats the action); \"(K in the winning "
    "attempt)\" follows when that attempt, with its loops cut out, was longer; then the actions "
    "spent on the level and its deaths.\n"
)
MEMORY_EFFECTS_DELTA_LINE = (
    "- `<action>: first <effect>` lines are written by the harness from the frames the first time in "
    "this game an action, or a click on a cell of one colour (`MOUSE@<colour letter>`), has that "
    "effect: moved = cells moved and the colour counts stayed the same; changed = colour counts "
    "changed; edge only = only cells near the edge changed; no change; anim only = cells changed and "
    "changed back. `L<N>: attempt <K> ended in GAME OVER` lines give the action and the board from "
    "which an attempt ended.\n"
)
MEMORY_SEEN_LINE = (
    "- A `Seen before` line in a turn message means the board interior matches one from earlier on "
    "this level, and lists what was tried from it and what happened.\n"
)
# ARC3_MEM_PATH: {b} is the cropped border, {c} the turn lines' per-game
# char cap and {hold} the token hold, filled once per run
MEMORY_PATH_LINE = (
    "- Route facts come from links the harness records on each level: a board-changing action links "
    "one board to the next (boards compared with the outer {b} cells cropped; UNDO, RESET, GAME OVER, "
    "clears and no-change actions add no link; a click is its exact cell). The route is the fewest "
    "links from this level's start board to the current board. Each action(...) call prints route N "
    "and what changed; a turn message after actions on the current level carries a Shortest route on "
    "record line{hold}, up to {c} characters of these lines per game. In python, level_path is that "
    "route as a list action(...) accepts and level_path_len its length.\n"
)


def memory_path_line(border: int, game_chars: int = 16000, after_tokens: int = 0) -> str:
    """The route line with the run's knobs: static for the run, so it is
    true of every turn (ARC3_MEM_PATH_GAME_CHARS, ARC3_MEM_PATH_AFTER_TOKENS)."""
    after = max(0, int(after_tokens))
    hold = f" once the level has used {after} generated tokens" if after else ""
    return (
        MEMORY_PATH_LINE.replace("{b}", str(int(border)))
        .replace("{hold}", hold)
        .replace("{c}", str(max(0, int(game_chars))))
    )


MEMORY_NOTES_LINE = (
    "- The other memory lines are your own notes. To record one, wrap one line in a tag anywhere in "
    "your reply, your reasoning or a comment in your python code; text outside the tags is "
    "unaffected.\n"
)
# ARC3_MEM_DEBRIEF: the head-card line in the ledger's words ("holding what
# you recorded"), the debrief fact, and the ledger's closing line
MEMORY_HEAD_CARD_RECORDED_LINE = (
    "- When this conversation grows too long, its oldest turns are dropped. The oldest message that "
    "remains then starts with a `<memory>` block holding what you recorded and what the harness "
    "recorded, so findings from dropped turns are not lost.\n"
)
MEMORY_DEBRIEF_LINE = (
    "- After a level is cleared, the harness copies your recap of it from your first reasoning on "
    "the next level (\"Level N solved: ...\") onto that level's `<solved>` line as `debrief:`.\n"
)
MEMORY_NOTES_CLOSING_LINE = (
    "- One claim per tag. A refuted hypothesis is as valuable as a confirmed one: it is what stops you "
    "repeating a dead experiment once the evidence for it has been dropped.\n"
)
MEMORY_NOTES_SCOPE_LINE = (
    "- Paths, action effects and deaths are already recorded by the harness, so a note holds what "
    "the frames do not show: the goal, the rules, and why.\n"
)


# ARC3_MEM_LEGIBLE: the same facts, written out in words, so no codes to explain
MEMORY_SOLVED_LEGIBLE_LINE = (
    "- `<solved>` lines are written by the harness when a level is cleared: the actions it took (the "
    "winning attempt with its loops cut out) and spent, its deaths, and the winning route.\n"
)
MEMORY_SOLVED_LEGIBLE_DELTA_LINE = (
    "- `Level N: cleared in ...` lines are written by the harness when a level is cleared: the "
    "actions it took (the winning attempt with its loops cut out) and spent, its deaths, and the "
    "winning route.\n"
)
MEMORY_EFFECTS_LEGIBLE_LINE = (
    "- `<actions level=\"N\">` lines are counted by the harness from the frames: what each action, "
    "and clicks on cells of each colour, did on level N, and in the whole game where that differs. "
    "\"Moved cells\" means the colour counts stayed the same; \"changed cell colours\" means they "
    "changed. `<hazard>` lines say which action, from which board, ended an attempt in GAME OVER.\n"
)
MEMORY_EFFECTS_LEGIBLE_DELTA_LINE = (
    "- `... for the first time in this game` lines are written by the harness from the frames the "
    "first time an action, or a click on a cell of one colour, has that effect. \"Moved cells\" means "
    "the colour counts stayed the same; \"changed cell colours\" means they changed. `Level N: attempt "
    "K ended in GAME OVER` lines say which action, from which board, ended an attempt.\n"
)
MEMORY_SEEN_LEGIBLE_LINE = (
    "- A `The current board was seen before` line in a turn message means the board inside its edge "
    "matches one from earlier on this level, and says what each action tried from it did.\n"
)

# ARC3_MEM_EVERY_CALL: the two blocks. Facts about where the memory is, not advice.
MEMORY_LAYOUT_PREVIOUS_LINE = (
    "- Once you clear a level, a `<previous_levels>` message right after these instructions holds one "
    "entry per cleared level: the actions it took and spent, its deaths, the winning route, and what "
    "you recorded about it. It changes only when a level is cleared.\n"
)
MEMORY_LAYOUT_RECAP_LINE = (
    "- An entry's recap is copied from your first reasoning on the next level when it names the "
    "level just cleared (\"Level N solved: the goal was ...\").\n"
)
MEMORY_LAYOUT_THIS_LEVEL_LINE = (
    "- Each turn message ends with a `<this_level>` block about the current level only: what each "
    "action and each click has done so far, the actions not tried yet, how attempts ended in GAME "
    "OVER, whether the current board was seen before, and the notes you wrote on this level.\n"
)
MEMORY_LAYOUT_CODES_LINE = (
    "- In routes, U D L R S are UP DOWN LEFT RIGHT SPACE, M(r,c) a click at row r, column c, and a "
    "number repeats the action.\n"
)
MEMORY_LAYOUT_COMPACT_LINE = (
    "- When a level is cleared, its turns are dropped from this conversation except the last {k}; "
    "its `<previous_levels>` entry replaces them.\n"
)
# ARC3_MEM_APPEND_COMMIT: a clear no longer redraws <previous_levels>. The old
# sentence was already false in pf-keep-10x (trim redraws add late recaps).
# The last sentence answers the reading A and PF made of "turns are dropped"
# (40 reasonings: "history is trimmed", one fatal live probe instead of a free
# lookup): the python lists are rebuilt from the solver's full history.
MEMORY_LAYOUT_APPEND_COMMIT_ROWS = (
    (
        " It changes only when a level is cleared.\n",
        " A newly cleared level's entry is shown once, in a `<cleared_level>` block near the end of a "
        "turn message, until older turns are next dropped from this conversation; then it joins "
        "`<previous_levels>`. The python `history` and `transitions` lists are never trimmed: they hold "
        "every action of this game.\n",
    ),
)
# ARC3_MEM_RECAP_FROZEN_FACT: a later level may contradict an earlier recap
# (tu93 L2 vs L3 in run B), and nothing said the recaps are frozen
MEMORY_LAYOUT_RECAP_FROZEN_ROWS = (
    (
        " and what you recorded about it.",
        " and what you recorded about it. Each recap is copied right after its level is cleared and is not "
        "updated afterwards.",
    ),
)
# ARC3_MEM_RECAP_ASK / ARC3_LEVEL_COMPACT_NEEDS_RECAP
MEMORY_LAYOUT_RECAP_ASK_LINE = (
    "- When you clear a level, the next turn message asks for its recap: write `Level N recap:` in your "
    "reasoning before you study the new board, with the level's goal and the rules you confirmed on it. "
    "The harness copies it into that level's entry.\n"
)
MEMORY_LAYOUT_COMPACT_RECAP_LINE = (
    "- When a level is cleared and its recap is saved, its turns are dropped from this conversation "
    "except the last {k}; its `<previous_levels>` entry replaces them. A level without a recap keeps "
    "its turns.\n"
)


def level_recap_ask(level: int) -> str:
    """ARC3_MEM_RECAP_ASK: the line after the level-start text."""
    n = int(level)
    return (
        f"Before you study the new board, write `Level {n} recap:` in your reasoning: one or two "
        f"sentences with the goal of level {n} and the rules you confirmed on it (what the actions "
        f"do, what completed the level). The harness copies this recap into `<previous_levels>`; "
        f"it is what you keep of level {n} once its turns leave this conversation."
    )


def recap_ask_v2_rows(level: int) -> tuple[tuple[str, str], ...]:
    """ARC3_MEM_RECAP_ASK_V2: `(old, new)` rows on `level_recap_ask(level)`.
    The recap is asked goal first, as `Goal:` and `Rules:` parts, with an
    optional last `Corrects level K:` when this level showed an earlier
    recap's rule was wrong. The label `Level {n} recap:` is kept exactly
    (memory_facts.extract_recap keys on it)."""
    n = int(level)
    return (
        (
            f"write `Level {n} recap:` in your reasoning: one or two sentences with the goal of level {n} "
            f"and the rules you confirmed on it (what the actions do, what completed the level).",
            f"write `Level {n} recap: Goal: ... Rules: ...` in your reasoning: after `Goal:`, what completed "
            f"level {n}; after `Rules:`, what the actions and objects did on level {n}, as you confirmed them. "
            f"If level {n} showed that a rule in an earlier level's recap was wrong, end with "
            f"`Corrects level K: ...`, K being that level.",
        ),
    )


# ARC3_MEM_STUCK_FACT: one harness count per level, once the level has used
# more generated tokens than this game's cleared levels usually took. Facts in
# a unit any reader has; no advice and no "budget" words.
STUCK_COUNT_LINE = (
    "Harness count: you have written {tokens} tokens of reasoning and code on level {level} without "
    "clearing it{cleared}."
)


def thousands(tokens: float) -> str:
    """``77.4k``."""
    return f"{float(tokens) / 1000:.1f}k"


def stuck_count_line(level: int, tokens: float, cleared: list[tuple[int, float]]) -> str:
    """The count line: this level's generated tokens, and each cleared
    level's (with their median from two levels on)."""
    tail = ""
    if len(cleared) == 1:
        tail = f"; level {cleared[0][0]} took {thousands(cleared[0][1])} tokens"
    elif cleared:
        levels = [lv for lv, _ in cleared]
        if levels == list(range(levels[0], levels[-1] + 1)):
            name = f"levels {levels[0]}-{levels[-1]}"
        else:
            name = "levels " + ", ".join(map(str, levels[:-1])) + f" and {levels[-1]}"
        counts = [thousands(t) for _, t in cleared]
        values = sorted(t for _, t in cleared)
        mid = len(values) // 2
        median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
        tail = (f"; {name} took " + ", ".join(counts[:-1]) + f" and {counts[-1]} tokens "
                f"(median {thousands(median)})")
    return STUCK_COUNT_LINE.format(tokens=thousands(tokens), level=int(level), cleared=tail)


MEMORY_LAYOUT_NOTES_LINE = (
    "- To record a note, wrap one line in a tag anywhere in your reply, your reasoning or a comment "
    "in your python code; text outside the tags is unaffected. Your notes appear in `<this_level>` "
    "while you play the level and move into `<previous_levels>` when it is cleared.\n"
)
_MEMORY_LAYOUT_TAG_EXAMPLES = _MEMORY_TAG_EXAMPLES + (
    "    `<hypothesis>the bar on the right counts the moves left</hypothesis>` -- not tested yet.\n"
)


def memory_layout_addendum(
    *, debrief: bool, legible: bool, compact_keep: int = 0,
    recap_ask: bool = False, compact_needs_recap: bool = False,
    append_commit: bool = False, recap_frozen: bool = False,
) -> str:
    """The system-prompt text for ARC3_MEM_EVERY_CALL (static per run):
    where the two memory blocks are, what is in them, and the tags.
    ``compact_keep`` > 0 (ARC3_LEVEL_COMPACT) states that a cleared level's
    turns are dropped except the last ``compact_keep``. ``append_commit``
    (ARC3_MEM_APPEND_COMMIT) says where a new entry waits until a trim."""
    previous = MEMORY_LAYOUT_PREVIOUS_LINE
    if recap_frozen:
        previous = apply_rows(previous, MEMORY_LAYOUT_RECAP_FROZEN_ROWS, where="ARC3_MEM_RECAP_FROZEN_FACT")
    if append_commit:
        previous = apply_rows(previous, MEMORY_LAYOUT_APPEND_COMMIT_ROWS, where="ARC3_MEM_APPEND_COMMIT")
    text = "\n\nMemory kept by the harness:\n" + previous
    if debrief:
        text += MEMORY_LAYOUT_RECAP_ASK_LINE if recap_ask else MEMORY_LAYOUT_RECAP_LINE
    text += MEMORY_LAYOUT_THIS_LEVEL_LINE
    if not legible:
        text += MEMORY_LAYOUT_CODES_LINE
    if compact_keep > 0:
        line = MEMORY_LAYOUT_COMPACT_RECAP_LINE if compact_needs_recap else MEMORY_LAYOUT_COMPACT_LINE
        text += line.replace("{k}", str(int(compact_keep)))
    text += MEMORY_LAYOUT_NOTES_LINE + _MEMORY_LAYOUT_TAG_EXAMPLES + MEMORY_NOTES_CLOSING_LINE
    return text


def _legible_facts(text: str) -> str:
    """ARC3_MEM_LEGIBLE: the fact lines in their plain-words form."""
    for old, new in (
        (MEMORY_SOLVED_LINE, MEMORY_SOLVED_LEGIBLE_LINE),
        (MEMORY_SOLVED_DELTA_LINE, MEMORY_SOLVED_LEGIBLE_DELTA_LINE),
        (MEMORY_EFFECTS_LINE, MEMORY_EFFECTS_LEGIBLE_LINE),
        (MEMORY_EFFECTS_DELTA_LINE, MEMORY_EFFECTS_LEGIBLE_DELTA_LINE),
        (MEMORY_SEEN_LINE, MEMORY_SEEN_LEGIBLE_LINE),
    ):
        text = text.replace(old, new)
    return text


def memory_addendum(
    *,
    ledger: bool,
    card: bool,
    solved: bool,
    effects: bool,
    seen: bool,
    head_card: bool,
    deltas: bool,
    path: bool = False,
    border: int = 4,
    path_chars: int = 16000,
    path_after: int = 0,
    debrief: bool = False,
    legible: bool = False,
) -> str:
    """The system-prompt text for the enabled memory components.

    With only the ledger on this is LEDGER_ON_TRIM_ADDENDUM, unchanged. With
    the card on (ARC3_MEM_CARD) the card text replaces it; otherwise the
    harness-fact lines are appended to it, or stand alone when there is no
    ledger or no head card. Without a head card (placement delta) the solved
    and effects lines describe the `<memory_update>` forms instead of the
    card's `<solved>`, `<actions>` and `<hazard>` tags, which never appear.

    ``debrief`` (ARC3_MEM_DEBRIEF, with the card and solved lines): the head
    card is described in the ledger's words ("holding what you recorded"),
    the `debrief:` part of a solved line is stated, and the notes close with
    the ledger's line instead of the scope line (that paths and effects are
    already recorded).
    """
    facts = ""
    if head_card and not ledger and not card:
        facts += MEMORY_HEAD_CARD_LINE
    if deltas and (solved or effects or card):
        facts += MEMORY_DELTA_LINE
    if solved:
        if head_card:
            facts += MEMORY_SOLVED_ROUTE_LINE if path else MEMORY_SOLVED_LINE
        else:
            facts += MEMORY_SOLVED_ROUTE_DELTA_LINE if path else MEMORY_SOLVED_DELTA_LINE
    if effects:
        facts += MEMORY_EFFECTS_LINE if head_card else MEMORY_EFFECTS_DELTA_LINE
    if seen:
        facts += MEMORY_SEEN_LINE
    if path:
        facts += memory_path_line(border, path_chars, path_after)
    if legible:
        facts = _legible_facts(facts)
    if card and debrief and solved:
        text = "\n\nMemory kept by the harness:\n"
        if head_card:
            text += MEMORY_HEAD_CARD_RECORDED_LINE
        text += facts.replace(MEMORY_HEAD_CARD_LINE, "")
        text += MEMORY_DEBRIEF_LINE
        text += MEMORY_NOTES_LINE + _MEMORY_TAG_EXAMPLES + MEMORY_NOTES_CLOSING_LINE
        return text
    if card:
        text = "\n\nMemory kept by the harness:\n"
        if head_card:
            text += MEMORY_HEAD_CARD_LINE
        text += facts.replace(MEMORY_HEAD_CARD_LINE, "")
        text += MEMORY_NOTES_LINE + _MEMORY_TAG_EXAMPLES + "- One claim per tag.\n"
        if solved or effects:
            text += MEMORY_NOTES_SCOPE_LINE
        return text
    if ledger and head_card:
        # without a head card (placement delta) the ledger is never shown,
        # so its text, which promises one, is left out
        return LEDGER_ON_TRIM_ADDENDUM + facts
    if facts:
        return "\n\nMemory kept by the harness:\n" + facts
    return ""


LEVEL_START_USER_PROMPT = (
    "You have completed the previous level. `current_frame` now contains the starting "
    "board of the next level; any accompanying current-grid image shows this new board. "
    "Build a new plan for this layout rather than continuing the previous level's action "
    "sequence.\n\n"
    "Start from the mechanics you established on the previous level; do not rediscover "
    "them without reason. Inspect the new board for unfamiliar elements, changed "
    "arrangements, or interactions your previous understanding does not explain. New "
    "elements often introduce mechanics needed to solve this level, so prioritize small, "
    "informative tests when their behavior is unclear.\n\n"
    "Reassess the goal: does the previous objective still apply, now requiring the new "
    "mechanics, or does the evidence suggest a different objective? Combine retained "
    "knowledge with new findings to plan for this board."
)


def apply_level_transfer_system_guidance(prompt: str) -> str:
    """Opt-in system wording; the caller gates ARC3_LEVEL_TRANSFER_GUIDANCE.

    Preserve the original constants for the default arm. The first user prompt
    after completion gives the new-board instructions; the system retains flag
    definitions and stop instructions for failure and run completion.
    """
    return (
        prompt.replace(
            "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n",
            LEVEL_TRANSFER_SYSTEM_GUIDANCE,
        )
        .replace(
            "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n",
            "",
        )
        .replace(
            "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n",
            "",
        )
        .replace(
            "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n",
            "- `WIN` means the whole game is solved.\n",
        )
        .replace(
            "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n",
            "- If an action result reports `game_over`, `run_complete`, or `done`, stop acting immediately and re-ground on the next turn.\n",
        )
    )


# ARC3_BUDGET_TRUTH: exact (old, new) substitutions on the finished system
# prompt. The edge bar is a per-level action budget, but the default text calls
# it a timer and tells the model to ignore it, while every action result also
# showed the run clock; in fz r1, r2 and control the model called the bar a
# real-time timer in about 11 of 25 games per run (the regex counts 12, with at
# least 2 false positives). Facts only, no new advice, and the result is
# shorter than the text it replaces. Other model-visible "timer" and
# "step/time budget" wording, including the RESET line's "remaining-action/time
# bar" (RESET_INFO_ADDENDUM below), is left as it is (see CONFIGURATION.md).
BUDGET_TRUTH_BAR_REPLACEMENTS = (
    (
        "is a timer or remaining-steps bar. It often shrinks or changes each step. If you "
        "identify such a bar, do not get distracted by it or treat it as core gameplay state "
        "unless there is concrete evidence that it interacts with the puzzle mechanics.\n",
        "is a step bar. It usually counts down a per-level action budget: it shrinks as "
        "actions are taken, not with wall-clock time.\n",
    ),
    (
        "classify it as HUD/timer state, not as an object to click through segment by "
        "segment. DON'T DO THIS!\n",
        "classify it as HUD state, not as an object to click through segment by segment.\n",
    ),
    (
        "affected gameplay state or only HUD elements such as countdown bars.\n",
        "affected gameplay state or only HUD elements.\n",
    ),
)

# Applied only when persistent functions are on: there, retained functions DO
# survive between calls, so "ephemeral" is false. The tool schema's two
# matching phrases live in tool_agent.py.
BUDGET_TRUTH_PERSISTENT_REPLACEMENTS = (
    (
        "- The only tool is `python`; call it with one ephemeral `code` string.\n",
        "- The only tool is `python`; call it with one `code` string.\n",
    ),
)


def apply_budget_truth(prompt: str, *, persistent_functions: bool) -> str:
    """Opt-in system wording; the caller gates ARC3_BUDGET_TRUTH."""
    replacements = BUDGET_TRUTH_BAR_REPLACEMENTS + (
        BUDGET_TRUTH_PERSISTENT_REPLACEMENTS if persistent_functions else ()
    )
    for old, new in replacements:
        prompt = prompt.replace(old, new)
    return prompt


# ARC3_PROMPT_HYGIENE: exact (old, new) substitutions on the finished system
# prompt that fix wording defects without changing what it says. The entity
# bullet was missing its newline, so it ran into the next bullet
# ("...1x1 tokens.- Some games ..."), and had two typos; the segmented-bar
# line and the ARC3_DIFF_IMAGE line had no "- " marker (the latter is matched
# only when it starts a line, as it does unless ARC3_AUTO_FRAME_DIFF puts a
# sentence before it). The flag also moves the memory-ledger section out
# from between the Python guidance bullets (tool_agent._build_system_prompt).
PROMPT_HYGIENE_REPLACEMENTS = (
    (
        "- Game entities are usually be rendered as connected multi-tile shapes such as 2×2, "
        "2×3, 3×3, or longer patterned structures. Sometime they might also be 1x1 tokens.",
        "- Game entities are usually rendered as connected multi-tile shapes such as 2×2, "
        "2×3, 3×3, or longer patterned structures. Sometimes they might also be 1x1 tokens.\n",
    ),
    (
        "\nA common failure mode is to mistake a segmented edge bar for clickable puzzle pieces.",
        "\n- A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces.",
    ),
    (
        "\n Alongside the current grid image you receive a diff image:",
        "\n- Alongside the current grid image you receive a diff image:",
    ),
)

# Applied only when persistent functions are on, where "ephemeral" is false:
# eligible functions a snippet defines are retained, and the line two bullets
# up plus the tool-session rules already say how and for how long. Identical
# to the ARC3_BUDGET_TRUTH substitution, so the two flags compose in any order.
PROMPT_HYGIENE_PERSISTENT_REPLACEMENTS = (
    (
        "- The only tool is `python`; call it with one ephemeral `code` string.\n",
        "- The only tool is `python`; call it with one `code` string.\n",
    ),
)


def apply_prompt_hygiene(prompt: str, *, persistent_functions: bool) -> str:
    """Opt-in system wording; the caller gates ARC3_PROMPT_HYGIENE."""
    replacements = PROMPT_HYGIENE_REPLACEMENTS + (
        PROMPT_HYGIENE_PERSISTENT_REPLACEMENTS if persistent_functions else ()
    )
    for old, new in replacements:
        prompt = prompt.replace(old, new)
    return prompt


# The game-over opener's two pointers to where a diagnosis goes. "Your world
# model" names the memory sections; with ARC3_MEMORY_SECTIONS=off they do not
# exist, so ARC3_PROMPT_HYGIENE points at what does: the ledger's tags when
# ARC3_LEDGER_ON_TRIM is on (a cause is a fact about the board and actions, a
# falsified belief is a refuted hypothesis), otherwise the reasoning itself.
GAME_OVER_RECORD_DEFAULT = (
    "Record the diagnosed cause in your world model.",
    "record the correction in your world model instead of retesting it.",
)
GAME_OVER_RECORD_LEDGER = (
    "Record the diagnosed cause in a `<world_model>` tag.",
    "record it in a `<hypothesis status=\"refuted\">` tag instead of retesting it.",
)
GAME_OVER_RECORD_REASONING = (
    "State the diagnosed cause in your reasoning.",
    "state the correction in your reasoning instead of retesting it.",
)

# ARC3_IMAGE_GEOMETRY. Every ARC-AGI-3 frame is 64x64 cells (the overview and
# the MOUSE range in this prompt say so); vision_context.frame_to_png_data_url
# draws one pixel per cell and resizes by MULTIMODAL_UPSCALE (nearest
# neighbour by default), so each cell is a scale x scale block of the image.
BOARD_CELLS = 64


def image_geometry_line(scale: int, cells: int = BOARD_CELLS) -> str:
    """One bullet: the board image's size and where each cell sits in it."""
    px = cells * scale
    if scale == 1:
        where = "is the pixel at x c, y r"
    else:
        where = f"spans x {scale}c..{scale}c+{scale - 1}, y {scale}r..{scale}r+{scale - 1}"
    return (
        f"- The board image is {px}x{px} px: {cells}x{cells} cells at {scale} px per cell; "
        f"cell (row r, col c) {where}.\n"
    )


TOOL_CALL_FORMAT_GUIDANCE = (
    "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
    "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
    "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
)

GAME_OVERVIEW_ADDENDUM = (
    "\n\nGame overview:\n"
    "- You are solving a multi-level grid puzzle game. \n"
    "- You are called repeatedly over the course of a run. Treat each turn as one observe-plan-act cycle: re-understand the current state from the newest frame, update your working world model in Python, choose the next best action or short sequence against the goal as currently understood, execute it, and expect to re-evaluate on the next turn from the updated state.\n"
    "- Your job is to solve the entire game by clearing every level, not just the current screen.\n"
    "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n"
    "- Optimize for as few in-game actions as possible while still being reliable.\n"
    "- In this environment, boards are presented as 64 x 64 color grids rendered with ARC color symbols.\n"
    f"- Color legend: {ARC_COLOR_LEGEND}.\n"
)

VISUAL_GAME_ADDENDUM = (
    "\n\nVisual-game guidance:\n"
    "- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.\n"
    "- Game entities are usually be rendered as connected multi-tile shapes such as 2×2, 2×3, 3×3, or longer patterned structures. Sometime they might also be 1x1 tokens."
    "- Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board configuration.\n"
    "- Background colors are often white or gray/black-ish large regions, but not always. Verify background hypotheses by area, stability, and object boundaries rather than assuming them.\n"
    "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often shrinks or changes each step. If you identify such a bar, do not get distracted by it or treat it as core gameplay state unless there is concrete evidence that it interacts with the puzzle mechanics.\n"
    "A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. If a repeated strip of small blocks sits flush against the top, bottom, left, or right border and actions only change that strip while the interior board stays the same, classify it as HUD/timer state, not as an object to click through segment by segment. DON'T DO THIS!\n"
    "- Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a specific absolute row or column.\n"
    "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n"
    "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n"
    "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n"
    "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"
)

ACTION_INFO_ADDENDUM = (
    "\n\nAction meanings (use only actions currently available):\n"
    "- `UP`, `DOWN`, `LEFT`, and `RIGHT` are directional controls; what they affect "
    "depends on the game.\n"
    "- When available, `SPACE` performs a game-specific action, such as interacting, "
    "selecting, rotating, attaching/detaching, or executing. Test its effect rather "
    "than assuming what it does.\n"
    "- When available, `MOUSE` clicks a board location. Pass integer `row` and `col` "
    "fields from 0 to 63. Coordinates are zero-based from the top-left: `row` "
    "increases downward and `col` increases rightward.\n"
)

UNDO_INFO_ADDENDUM = (
    "- When available, `UNDO` reverses a previous action, usually the last turn. "
    "Check what it restores. It cannot recover a failed attempt after game over.\n"
)

# Appended to the UNDO line only when RESET is exposed too, so the two ways
# back state their costs side by side.
UNDO_COST_ADDENDUM = (
    " UNDO itself counts as one action and does not refund the action it reverses.\n"
)

RESET_INFO_ADDENDUM = (
    "- When available, `RESET` usually restores the current level to its starting "
    "state, including the remaining-action/time bar, while keeping completed levels. "
    "Use it to recover from an unrecoverable position or start a different approach. "
    "RESET itself counts as one action, and actions already spent still count toward "
    "your score. RESET must be the first game action in a Python snippet; inspection "
    "or computation may precede it. RESET is refused, and spends nothing, when no "
    "action has been executed since the level started or was last reset.\n"
)

GAMEPLAY_CHANGED_ADDENDUM = (
    # One region, named once and referred back to twice. "outside that margin"
    # and "inside that border" both read as possibly including the border
    # cells themselves; "ignores those edge cells" states the rule before any
    # geometry, and "further in" cannot be read as the border.
    "- `board_changed` is true when ANY cell differs, including cells within {border} of the "
    "edge where a timer or remaining-steps bar usually sits; `gameplay_changed` ignores those "
    "edge cells and is true only when a cell further in differs. An action that only shrinks "
    "a timer bar in those edge cells sets `board_changed` but not `gameplay_changed`.\n"
)

STRUCTURED_RUNTIME_STATE_ADDENDUM = (
    "\n\nRuntime variables inside every `python` tool call:\n"
    "- `current_frame` is a lightweight frame view for the latest environment state.\n"
    "- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- `current_frame.ascii` is a single newline-delimited string containing the latest board rendered with the letter-coded ARC color symbols.\n"
    "- `current_frame.segmentation` parses the board into objects. It returns `{'nodes': [...], 'adjacency_list': [...]}`.\n"
    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that ignores position AND rotation -- equal hashes indicate matching color and shape up to rotation; they do not establish object identity), `rotation` (degrees clockwise from the object's canonical orientation: 0/90/180/270, reduced to 0/90 for 2-fold symmetric shapes, `None` for fully symmetric ones), `rotational_symmetry` (1, 2, or 4), `pose_hash` (the rotation-SENSITIVE variant, if you need to match exact orientation), `shape_hash` (color-independent pose shape signature), `pixels` (cell count), `boundary` (clockwise outer-perimeter corner points as `[row, col]` cell coordinates), and `children` (ids of objects fully enclosed by this one).\n"
    "- `segmentation['adjacency_list']` is a list of `[i, j]` node-id pairs whose objects share an edge.\n"
    
    "- `current_frame.step` is the current environment step count.\n"
    "- `current_frame.level` is the current level number.\n"
    "- `current_frame.shape` is a `(rows, cols)` tuple.\n"
    "- The raw numeric grid is intentionally not exposed. Use `current_frame.segmentation` as your primary view of the board -- objects, colors, shapes, containment, adjacency, and cross-frame object hashes. Use `current_frame.ascii` only to read a small, specific region; do not scan the whole board with it.\n"

    "- `history` is a chronological list of action/frame snapshots.\n"
    "- `history` is a Python list of objects, not a dict.\n"
    "- Each history entry exposes `.action`, `.frame`, and `.result`; entries are not subscriptable like `entry['action']`.\n"
    "- Each `history[i].frame` is the frame after `history[i].action`; each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- Important history semantics: when `history` is non-empty, `history[-1].frame` is the same latest/post-action board as `current_frame`. It is not the previous board. To inspect the state before the latest action, use `previous_frame` or `history[-2].frame` when available.\n"
    "- `previous_frame` is the frame before the most recent real environment action, or `None` if no previous frame is available.\n"
    "- `last_action` is the most recent real environment action name/display, or `None` before any real action.\n"
    "- `last_action_frame` is the post-action frame for `last_action`; it matches `current_frame` after a real action.\n"
    "- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), and `.result`. Each `.result` describes that individual action and is retained with its history entry.\n"
    "- `last_transition` is `transitions[-1]` or `None`. A refused action creates no transition and does not overwrite existing transition results. For before/after diffs, compare `last_transition.before_frame` to `last_transition.after_frame`; do not compare `current_frame` to `history[-1].frame`.\n"
    "- `last_action_call_result` contains the result of the latest model-issued `action(...)` call, including batch totals, executed/skipped actions, and stop reasons. `action(...)` returns this result. It remains available across later Python inspection calls and analyzer turns, and is `{}` before any action call result exists.\n"
    "- On the next analyzer turn after automatic RESET, `current_frame` shows the restarted level. The fatal action and RESET are separate transitions. Each transition result has `automatic=True` for a harness-initiated action and `automatic=False` for a model-issued action. Automatic RESET does not overwrite `last_action_call_result`: it still describes the call that caused death.\n"
    "- `valid_actions` is the current list of valid action names.\n"
    "- Call `action(actions)` to execute one or more real environment actions from Python.\n"
    "- Pass `action(actions)` a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n"
    "- One action usually returns one frame, but a single action can result in a short multi-frame animation.\n"
    "- After `action(actions)` returns, `current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, and `last_action_call_result` are refreshed.\n"
)

ANIMATION_ADDENDUM_TIMELINE = (
    "- `last_animation_timeline` is a diff timeline of those frames: it "
    "shows which cells changed at each step. It is a dict with `action`, `frames` and "
    "`steps`, one entry per step that changed anything. Each entry has `step`, "
    "`changed`, `bbox`, and "
    "either `changes` (the cells, as `old>new @ (row,col) ...`) or `transitions` (a "
    "count per colour change, when there were too many cells to list).\n"
)

ANIMATION_ADDENDUM = (
    "- One action can return a short animation. `current_frame` is its final frame.\n"
    "- When the final executed action of the latest model-issued call animated, `last_action_call_result['animation']` "
    "reports `frames`, `transient_pixels` and `transient_bbox`. Transient cells "
    "changed and then changed BACK during the animation, so they appear in no frame "
    "you can otherwise reach - not in `current_frame`, not in `previous_frame`, not "
    "in `history`.\n"
    "- `last_animation_frames` is the frames of the last executed action that "
    "animated, as views with "
    "`.ascii`, `.shape` and `.segmentation`, the same as `current_frame`. Index 0 is "
    "the first frame of the animation and the last entry is the board it settled on. "
    "After a death these are the frames of the FATAL action, not of the automatic "
    "reset that followed it. Crop "
    "their `.ascii` yourself; printing a whole 64x64 frame will not fit the tool "
    "budget. Empty when the action did not animate, and reading them costs no "
    "in-game action.\n"
)

# ARC3_ANIM_TRUTH only. Constant text, so the prompt prefix stays cacheable.
ANIMATION_TRUTH_ADDENDUM = (
    "- An action whose animation changed cells and changed them back, while the board "
    "area ended unchanged, is reported as `ANIMATED (...)`, not `NO-OP`: it did "
    "something, and what it did is visible only in its animation frames.\n"
)

# ARC3_ANIM_DIGEST only. Constant text, so the treatment's prompt prefix is
# as cacheable as the control's.
ANIMATION_DIGEST_ADDENDUM = (
    "- When an action's animation is new or hid something, the next turn message "
    "carries an \"Animation digest\" after the animation line. S0 is the board before "
    "that action and S<n> the board it settled on (`current_frame` when it was the last "
    "action, except after a game over: then S<n> is the death frame and `current_frame` "
    "is the restarted level); timeline step k equals `last_animation_frames[k-1]`. A "
    "trail lists where a blob of one colour and size was seen, top-left (row,col), in "
    "time order - a heuristic, not proof that one object moved.\n"
)

# ARC3_ANIM_RING only (the digest needs it). Constant text, as above.
ANIMATION_RING_ADDENDUM = (
    "- Recent animated actions stay readable, not only the last: `animations` lists "
    "them (`action_num`, `action`, `step_count`, `transient_pixels`, `level`; "
    "`step_count` is an int) and `animation_for(action_num)` returns that action's "
    "`{'action_num', 'action', 'frames', 'timeline'}`. S0 is the board before an "
    "action and `frames[k-1]` is step k (state S<k>), the same indexing as "
    "`last_animation_frames`, which hold the newest action that animated. A timeline "
    "has `action_num`, `action`, `frames` (a count) and `steps`. An action result's "
    "`animation` has `frames` (the count of states after S0), `transient_pixels` and, "
    "when that is above 0, `transient_bbox`; a batch's `animations` lists only the "
    "actions whose animation hid something. Re-reading any of them costs no game "
    "action, so do not spend UNDO or repeat a click just to see an animation again.\n"
)

MULTIMODAL_CONTEXT_ADDENDUM = (
    "\n\nMultimodal context:\n"
    "- User turns include an attached image of the current ARC grid.\n"
    "- The image and `current_frame.ascii` are two representations of the same current frame.\n"
    "- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n"
)

PYTHON_ADDENDUM_HEAD = (
    "\n\nPython tool guidance:\n"
    "- Use `current_frame.segmentation` as your primary view of the board -- objects, colors, containment, adjacency, and cross-frame object hashes.\n"
    "- Use `current_frame.ascii` only to read a small, specific region of the board when `segmentation` is not enough; never use it to scan or summarize the whole board.\n"
    "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n"
    "- The only importable standard-library modules are: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.\n"
    "- The only tool is `python`; call it with one ephemeral `code` string.\n"
    "- Always inspect `current_frame`, `history`, and `valid_actions` from Python instead of reasoning from the raw board by eye.\n"
    "- For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`. `history[-1].frame` is the current frame, so comparing it to `current_frame` only compares the board to itself.\n"
)

# split so the world-model lines can be swapped in at their original
# position - immediately before the IMPORTANT search guidance - rather than
# appended after the whole addendum
PYTHON_ADDENDUM_TAIL = (
    "- IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n"
    "- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n"
    "- Inspect current and history frames from Python instead of describing frames freehand.\n"
    "- Never print or echo full board frames. Return only compact derived summaries such as object lists, diffs, coordinates, counts, or tiny local crops.\n"
    "- Keep tool-output context size minimal and decision-oriented so you can quickly compare before/after state. It's fine to write a lot of python code, just make the output short and interpretable\n"
    "- A strong default loop is: summarize the board, infer the desired environment change, write a small scorer or search over candidate sequences, execute the best probe or plan with `action(...)`, then inspect again until you understand exactly what changed.\n"
    "- For object tracking, match objects by color, overlap, bounding box proximity, area change, and edge contact rather than by exact coordinates alone.\n"
    "- For frame diffs, summarize changed cells, color transitions, appearing/disappearing components, movement candidates, and small local row slices around the changed region.\n"
    "- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n"
    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\n"
    "- Call `action(...)` inside Python rather than returning action text in the chat.\n"
    "- `action(...)` accepts an ordered list of one or more actions. Once your code has selected a reliable sequence, it is often useful to batch it.\n"
    "- You can also call `action(...)` multiple times in one Python snippet, including inside loops. Each call updates the preloaded variables before execution continues.\n"
    "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n"
    "- Flag semantics: `game_over` = this attempt FAILED (death or a limit ran out) — the level auto-resets to its initial state and completed levels are kept; it never means the run is won. `level_completed` = advanced one level. `run_complete`/`done` = the whole game is won. Deaths come either from the action itself (e.g., a hazard cell) or from a step/time budget depleting — watch for a bar at the grid border that shrinks each action, and count remaining budget into your plans.\n"
)

FRAME_DIFF_HINT_ADDENDUM = (
    "- `frame_diff(before=None, after=None)` compares two frames. Omitted arguments default to `previous_frame` and `current_frame`, respectively.\n"
    "- It returns a dictionary containing `changed_cell_count` (an integer) and lists named `moved`, `rotated`, `appeared`, `disappeared`, `changed_color`, and `resized`. Unchanged objects are omitted.\n"
    "- Object-level changes are inferred by matching same-color, 4-connected components between frames: cells connect through shared edges, not diagonally. A game object may therefore be split into several components, or touching same-color objects may form one component.\n"
    "- `moved` describes components matched at different positions. Matching does not establish persistent object identity: with multiple similar components, the reported correspondence may be ambiguous or incorrect. Movement combined with rotation includes `rotated_by` in degrees clockwise; `rotated` describes rotation in place.\n"
    "- `changed_color` requires identical occupied cells with a different color. Entries contain `at`, `pixels`, `before_color`, `after_color`, `before_hash`, and `after_hash`.\n"
    "- `resized` describes same-color components with overlapping footprints whose size changed. This can include backgrounds and budget bars. `at_before` and `at_after` can differ.\n"
    "- `appeared` and `disappeared` entries include `bbox` in `[r0, c0, r1, c1]` order. Object hashes use the same rotation-invariant shape-and-color representation as segmentation; they are not unique object identities, and multiple objects can share a hash.\n"
    "- Examples: `frame_diff()` compares the latest action’s before/after frames; `frame_diff(history[-3].frame, history[-2].frame)` compares two explicitly selected historical frames, when available.\n"
)

STEP_VERIFICATION_ADDENDUM = (
    "- During a multi-action plan, verify important effects by comparing the actual game "
    "state with your predictions—for example, changed cells, object properties, positions, "
    "or counters. Make these comparisons in code so the batch can stop when a mismatch "
    "undermines the remaining plan. Test uncertain interactions before committing to long "
    "batches.\n"
    "- If your search finds no solution under your current model of the game, remember "
    "that the game is solvable. Reconsider your mechanics, goal, search implementation, "
    "or search limits, including interactions with new elements. Take a targeted action "
    "to test an uncertain rule or overlooked interaction, then update your model from "
    "the result.\n"
)

WORLD_MODEL_ADDENDUM = (
    "- Maintain a compact working world model: what entities or regions exist, what actions "
    "seem to do, what the goal likely is, what remains uncertain, and what plan best fits the "
    "evidence so far.\n"
    "- Optimize for the shortest reliable sequence that advances the current goal as described "
    "by your world model. If confidence is low, program a discriminating probe and revise the "
    "world model from the result.\n"
)

WORLD_MODEL_FREE_ADDENDUM = (
    "- Optimize for the shortest reliable sequence that advances the current goal. If "
    "confidence is low, program a discriminating probe and revise your approach from the "
    "result.\n"
)

DEATH_GUARD_ADDENDUM = (
    "- An action that ended a previous attempt from this EXACT board state is refused before "
    "execution: it raises `KnownDeathActionError`, nothing runs, and no attempt is lost. "
    "`stop_reason` is `known_death`. The guard checks every position of a batch, not just the "
    "first, so a fatal move is caught whichever route reached it. To repeat such an action "
    "deliberately, issue it again as your very next action.\n"
)

NOOP_GUARD_ADDENDUM = (
    "- An action already proven to change nothing in this EXACT board state is refused before "
    "execution: it raises `KnownNoOpActionError`, which is NOT catchable and ends the snippet - "
    "nothing runs, "
    "and no action budget is spent. `stop_reason` is `known_noop`. If you have a specific reason "
    "to believe the action will behave differently now, issue that same action again as your very "
    "next action and it will execute.\n"
)

PREFER_TOOL_CALLS_LINE = (
    # The closing clause used to warn that a long chain "wastes the whole turn".
    # That overstated it - a truncated reply keeps its reasoning in history, and
    # what is lost is only that nothing executed - and it aimed at the tail,
    # since truncation touched about 4% of replies. The speed claim is the
    # stronger one and it is true: decode runs at tens of tokens per second per
    # stream, while a tool call is a short generation plus a mostly-cached
    # prefill, so 2,000 tokens of further reasoning costs far more wall-clock
    # than running the code that would settle the question. The opening clause
    # used to say reading and computing "cost nothing", which is false of the
    # game's wall-clock budget that every tool call spends; it now says only
    # what is true, that they spend no actions.
    "- Reading and computing spend no actions; only `action(...)` spends the level budget. When you "
    "are unsure, prefer another tool call over more reasoning: the code answers what the "
    "reasoning would only guess at, and running it is faster than thinking your way to the "
    "same answer.\n"
)

EPHEMERAL_FUNCTIONS_LINE = (
    "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need.\n"
)

PERSISTENT_FUNCTIONS_LINE = (
    "- Python variables reset between tool calls. Eligible functions you define are "
    "retained automatically {lifetime}. Call your retained functions directly without "
    "repeating their definitions. Redefine a function when you need to change it.\n"
    "- For a function to be retained, its dependencies must also be available next call. "
    "Pass snippet-specific data as arguments. You may use Python builtins, provided "
    "globals such as `current_frame`, and other retained functions. Provided globals "
    "are refreshed for each call. {import_guidance} Use plain "
    "top-level definitions without decorators or annotations, and literal defaults. "
    "Tool results report which functions were retained and explain any rejection.\n"
)


COMPACT_TOOL_SESSION_ADDENDUM = (
    "\n\nTool session rules:\n"
    "- You have exactly one tool: `python`.\n"
    f"- {TOOL_CALL_FORMAT_GUIDANCE}\n"
    "{persistent_functions}"
    "- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear probe or plan.\n"
    "- Do not ration tool calls when the state is unclear. Spend extra tool calls to confirm what changed between frames and whether the last action affected gameplay state or only HUD elements such as countdown bars.\n"
    "{prefer_tool_calls}"
    "- After `action(...)` returns, the structured runtime state is refreshed before the next Python statement and before the next tool call. Inspection-only Python calls do not clear `last_action_call_result`.\n"
    "- Each `python` tool call has a hard time limit of 30 seconds.\n"
    "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool result will tell you that.\n"
    "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
)


SUMMARY_REQUEST_PROMPT = (
    "Summarize everything above: what you have established about this game, what you have "
    "ruled out, what you were in the middle of, and anything else a later turn would need. "
    "Keep uncertain ideas tentative, and preserve unresolved alternatives. "
    "Do not act and do not call any tool - write the summary only.\n"
    # No "the turns after this one stay visible": the summary is appended at
    # the end of history, so nothing follows it when it is written. That
    # sentence belonged to an earlier design where the summary was inserted
    # ahead of a retained tail, and here it points at messages that do not
    # exist yet.
    "Write it to stand completely on its own. Everything above this point will eventually be "
    "dropped and replaced by this summary, and earlier summaries go with it, so anything you "
    "leave out is gone.\n"
    "Length is not a problem here; leaving something out is."
)


# ---------------------------------------------------------------------------
# ARC3_LEGIBILITY (inference/agent/legibility.py): harness text the model
# misread, restated as facts. The system-prompt part of each fix is a list of
# exact (old, new) rows applied once per run to the finished prompt, after
# ARC3_BUDGET_TRUTH, so the prompt stays static for the run. Each row is a
# tuple of alternatives; the first whose old text occurs exactly once is
# applied. A row with no such alternative is skipped with a warning instead of
# replacing several places or none: Franzen's bare str.replace fails silently.
# tests/test_legibility.py asserts every row applies under the pf-keep-10x env.

# ARC3_TURN_BASED_FACT: appended to the edge-bar bullet.
TURN_BASED_FACT = (
    " The game is turn-based: the board, bars included, changes only when an action "
    "executes; time spent reasoning or in tool calls changes nothing. One action can remove "
    "no bar cell, one, or several, depending on the game and the action."
)

# ARC3_STALE_TRUTH: the end of the stale-state refusal (solver._stale_detail).
# The default said the error "can be caught if a snippet has good reason to
# keep acting through no-ops", but the latch is cleared only by an executed
# action that changes the board, and a refused call executes nothing: all 51
# action() calls made after a caught refusal in runs A, B and PF were refused.
# The sandbox rebuilds the latch for every python call, and the latch is
# armed only by an action that executed and changed nothing.
def stale_truth_line(stale_after: str) -> str:
    return (
        "(Catching StaleStateActionError does not lift this halt: every later action() call "
        "in this python call is refused the same way, separate action() calls included, and "
        "a refused action is not executed and costs nothing. Your next python call can act "
        f"again. {stale_after} itself was executed and counted as one action.)"
    )

# ARC3_RESET_BAR_FACT: the RESET line's new first sentence.
RESET_BAR_FIRST_SENTENCE = (
    "When available, `RESET` restores the current level to the board it started with, step "
    "bar included, and keeps completed levels."
)

LEGIBILITY_SYSTEM_ROWS: dict[str, tuple[tuple[tuple[str, str], ...], ...]] = {
    # Nodes had no extent field, so `boundary[0]`/`[2]` was read as a box:
    # 25 IndexErrors on `boundary[2]` and 25 misreads in 27 of 75 game-runs.
    # A filled rectangle has 4 corner points, a one-cell-wide line 2, a cell 1.
    "ARC3_SEG_BBOX": (
        ((
            "`boundary` (clockwise outer-perimeter corner points as `[row, col]` cell "
            "coordinates)",
            "`bbox` (`[r0, c0, r1, c1]`: the first and last row and column the object covers, "
            "inclusive, the same order as `frame_diff`'s `bbox`), `boundary` (the `[row, col]` "
            "cells where the outline turns, clockwise: 4 for a filled rectangle, 2 for a "
            "one-cell-wide line, 1 for a single cell; it is not a bounding box)",
        ),),
        ((
            "include `bbox` in `[r0, c0, r1, c1]` order.",
            "include `bbox` in `[r0, c0, r1, c1]` order, inclusive.",
        ),),
    ),
    # 64 bare letters per row: crops sliced out of `.ascii` were miscounted in
    # about 52 requests in 35 game-runs; tn36-B lost L7 to one. `.ascii` itself
    # is unchanged, since a header line would shift every split('\n')[r].
    "ARC3_FRAME_RUNS": (
        ((
            "- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, and "
            "`.segmentation`.\n",
            "- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, "
            "`.segmentation` and `.runs(r0, r1, c0, c1)`.\n",
        ),),
        ((
            "containing the latest board rendered with the letter-coded ARC color symbols.\n",
            "containing the latest board rendered with the letter-coded ARC color symbols. "
            "`frame.runs(r0, r1, c0, c1)` returns the same cells, rows r0 to r1 inclusive, as "
            "one line per row in which each run of one letter is written with its first and "
            "last column, for example `28: c38-40 Y41 B42-43 Y44 c45`.\n",
        ),),
        ((
            "each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, and "
            "`.segmentation`.\n",
            "each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, `.segmentation` "
            "and `.runs(r0, r1, c0, c1)`.\n",
        ),),
    ),
    # animation_for(n) of an action that is not kept returned a truthy dict
    # without `action` (its `timeline` held only `error`), and step `changed`
    # is an int: 30 of 612 animation-helper snippets failed on KeyError
    # 'action'/'frames'/'steps' or len(int).
    "ARC3_ANIM_SCHEMA": (
        ((
            "`changed`, `bbox`, and ",
            "`changed` (an int: how many cells changed at that step), `bbox`, and ",
        ),),
        ((
            "so do not spend UNDO or repeat a click just to see an animation again.\n",
            "so do not spend UNDO or repeat a click just to see an animation again. For an "
            "action whose animation is not kept, `animation_for(n)` tests false and has the "
            "same keys, with `action` None, no frames or steps, and an `error` naming the kept "
            "action numbers.\n",
        ),),
    ),
    # "usually restores ... including the remaining-action/time bar" and
    # "counts as one action ... still count toward your score" mixed the
    # level's action count with the on-board bar: ~50 doubt messages in ~30
    # game-runs, 9 wrong budget beliefs. All 188 RESETs in runs A, B and PF
    # reproduce the level's first board exactly, bar included.
    "ARC3_RESET_BAR_FACT": (
        ((
            "When available, `RESET` usually restores the current level to its starting "
            "state, including the remaining-action/time bar, while keeping completed levels.",
            RESET_BAR_FIRST_SENTENCE,
        ),),
        ((
            "RESET itself counts as one action, and actions already spent still count toward "
            "your score.",
            "RESET adds one action to the level's action count, which the score uses, as every "
            "action already spent does; it does not shorten the restored bar.",
        ),),
        ((
            "the level auto-resets to its initial state and completed levels are kept",
            "the level auto-resets to its initial state, step bar included, and completed "
            "levels are kept",
        ),),
    ),
    # "Timer" and "step/time" leftovers, the "usually" hedge and bars that lose
    # 0, 1 or 2 cells per action keep a real-time reading alive: 80-122
    # real-time messages in 28-43 of 75 game-runs, 9 of them citing the
    # wording; hurry decisions. ARC3_BUDGET_TRUTH's rewrite of the bar bullet
    # barely moved the rate, so the fact added here says what triggers a
    # change. The bar row extends budget truth's bullet, or Franzen's when
    # budget truth is off.
    "ARC3_TURN_BASED_FACT": (
        (
            (
                "it shrinks as actions are taken, not with wall-clock time.\n",
                "it shrinks as actions are taken, not with wall-clock time." + TURN_BASED_FACT + "\n",
            ),
            (
                "unless there is concrete evidence that it interacts with the puzzle mechanics.\n",
                "unless there is concrete evidence that it interacts with the puzzle mechanics."
                + TURN_BASED_FACT + "\n",
            ),
        ),
        (
            ("including the remaining-action/time bar,", "including the step bar,"),
            # ARC3_RESET_BAR_FACT rewrote that sentence first: leave it as it is
            (RESET_BAR_FIRST_SENTENCE, RESET_BAR_FIRST_SENTENCE),
        ),
        (("(death or a limit ran out)", "(death, or the level's step budget ran out)"),),
        (("from a step/time budget depleting", "from a step budget depleting"),),
        ((
            "a bar at the grid border that shrinks each action",
            "a bar at the grid border that shrinks as actions are taken",
        ),),
        ((
            "edge where a timer or remaining-steps bar usually sits",
            "edge where a step bar usually sits",
        ),),
        ((
            "An action that only shrinks a timer bar in those edge cells",
            "An action that only shrinks a step bar in those edge cells",
        ),),
        ((
            "only a timer, progress bar, or remaining-step bar moved",
            "only a progress or step bar moved",
        ),),
    ),
    # Digest numbers - top-left, HxW size, d(), a subsampled path - were read
    # as object moves and sizes: about 28 reasonings in 35 matched game-runs,
    # 6 explicit HxW misreads, one wrong physical hypothesis (sp80-A).
    "ARC3_DIGEST_COORD_FACT": (
        ((
            "was seen, top-left (row,col), in time order",
            "was seen, top-left (row,col) and size (rows x cols), in time order",
        ),),
    ),
    # frame_diff matches same-colour components, so a component that merged
    # into a neighbour is listed as disappeared although none of its cells
    # changed: bp35-A took a 276-cell "disappearance" for a drained room
    # (~9.7k tokens); about 14 requests in 11 game-runs name the artifact.
    "ARC3_DIFF_SAME_CELLS": (
        ((
            "they are not unique object identities, and multiple objects can share a hash.\n",
            "they are not unique object identities, and multiple objects can share a hash. "
            "`same_colour_cells` equal to `pixels` means none of those cells changed colour: "
            "only the component around them merged or split. `same_hash_candidates` above 1 "
            "means another component with the same hash could be the one that moved.\n",
        ),),
    ),
    # "Current state: step N" is the next action's number, while the digest,
    # animation_for(n), current_frame.step and the diff image count actions
    # taken: 3 misreads in ls20 (A and B) rebuilt a trajectory from history.
    # Without the animation ring there is no animation_for (ACTION_INDEX_NO_RING_ROW).
    "ARC3_ACTION_INDEX_FACT": (
        ((
            "- `current_frame.step` is the current environment step count.\n",
            "- `current_frame.step` is the current environment step count. Actions are "
            "numbered from 1 across the whole game: action n is `transitions[n-1]` and "
            "`animation_for(n)`; `current_frame.step` and the diff image's step numbers count "
            "the actions taken, and `Current state: step N` is the number the next action "
            "will get.\n",
        ),),
    ),
    # Retained functions are listed by signature only and `inspect` is not
    # importable, so after a trim their code was unreadable: 18 ImportErrors
    # from inspect attempts in 16 game-runs, then rewrites from scratch.
    "ARC3_RETAINED_SOURCE": (
        ((
            "Redefine a function when you need to change it.\n",
            "Redefine a function when you need to change it.\n"
            "- `retained_source` maps the name of each retained function to its source text.\n",
        ),),
    ),
    # Held out of the ARC3_LEGIBILITY bundle: 7 case-pair slips (b blue read
    # as B black, ...) in 75 game-runs, one consequential (s5i5-B L5).
    "ARC3_LEGEND_CASE_PAIRS": (
        ((
            f"- Color legend: {ARC_COLOR_LEGEND}.\n",
            f"- Color legend: {ARC_COLOR_LEGEND}. Letter case changes the colour: B black / "
            "b blue, W white / w light gray, G dark gray / g gray, R red / r dark red, P pink "
            "/ p purple.\n",
        ),),
    ),
}
ACTION_INDEX_NO_RING_ROW = ((
    "- `current_frame.step` is the current environment step count.\n",
    "- `current_frame.step` is the current environment step count. Actions are numbered from "
    "1 across the whole game: action n is `transitions[n-1]`; `current_frame.step` and the "
    "diff image's step numbers count the actions taken, and `Current state: step N` is the "
    "number the next action will get.\n",
),)
# ARC3_DIGEST_COORD_FACT: the digest's closing line (utils/animation.py builds
# it; tool_agent swaps the sentence in after the digest is fitted to its
# budget, so the same trail lines are kept and the game's digest budget is
# charged the same).
DIGEST_HEURISTIC_ROW = (
    "Trails are heuristic (top-left of each blob not in the settled board). ",
    "Trails are heuristic (top-left of each blob not in the settled board). Positions are "
    "the (row, col) of a blob's top-left cell and sizes are rows x cols, both taken from "
    "stored states that can skip steps, so a tween's d() or a path's extreme point can "
    "differ from the settled move. ",
)
# The same substitutions in the turn messages and tool results that carry the
# wording (tool_agent: the terminal stop detail, the game-over opener, the
# board-edge line; solver: the no-op stop detail). Each text holds at most one.
TURN_BASED_TEXT_ROWS = (
    ("a depleted step/time budget (usually", "a depleted step budget (usually"),
    ("or by a step/time budget running out", "or by a step budget running out"),
    ("(usually a timer or remaining-steps bar)", "(usually a step bar)"),
    (
        "cells, where a timer or remaining-steps bar usually sits, could have moved",
        "cells, where a step bar usually sits, could have moved",
    ),
)


# ARC3_RESET_BAR_FACT in the game-over opener, as on the flag-semantics line.
RESET_BAR_TEXT_ROWS = (
    (
        "the current level restarted from its initial state and previously completed",
        "the current level restarted from its initial state, step bar included, and "
        "previously completed",
    ),
)
TURN_TEXT_ROWS = {
    "ARC3_RESET_BAR_FACT": RESET_BAR_TEXT_ROWS,
    "ARC3_TURN_BASED_FACT": TURN_BASED_TEXT_ROWS,
}


def apply_turn_text(text: str, flags) -> str:
    """The turn-text rows of every flag in ``flags`` on one turn message or tool
    result. Each text holds at most one occurrence of a row's old text."""
    for name, rows in TURN_TEXT_ROWS.items():
        if name in flags:
            for old, new in rows:
                if text.count(old) == 1:
                    text = text.replace(old, new)
    return text


# The python tool's description says the frames "expose only" five fields,
# which the runs helper would make false.
FRAME_RUNS_TOOL_ROW = (
    ", and `.shape` (a `(rows, cols)` tuple); ",
    ", `.shape` (a `(rows, cols)` tuple) and `.runs(r0, r1, c0, c1)`; ",
)


def apply_once(text: str, row: tuple[tuple[str, str], ...], *, where: str = "") -> str:
    """Apply the first (old, new) alternative of ``row`` whose old text occurs once."""
    for old, new in row:
        if text.count(old) == 1:
            return text.replace(old, new)
    log.warning("legibility row skipped (%s): %r", where, row[0][0][:80])
    return text


def apply_legibility(prompt: str, flags) -> str:
    """The system-prompt rows of every legibility flag in ``flags``, in a fixed order."""
    for name, rows in LEGIBILITY_SYSTEM_ROWS.items():
        if name in flags:
            for row in rows:
                prompt = apply_once(prompt, row, where=name)
    return prompt
