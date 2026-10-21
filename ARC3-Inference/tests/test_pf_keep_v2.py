"""Memory v2 (arm/mem-v2) and legibility (arm/legibility) with every flag on together.

Both branches change model-visible text with exact (old, new) rows, and a row
whose old text has drifted, or occurs twice, does nothing or too much without
an error (Franzen's bare str.replace; apply_rows and apply_once skip it with a
warning). Each branch tests its own rows with the other branch's flags off.
Here every row is checked with ALL flags on, in two envs on top of the
notebook's setup_env:

- ``arm``: pf-keep-v2-all-10x from kaggle/arms.json (pf-keep-10x plus
  ARC3_MEM_APPEND_COMMIT, pf-keep-memv2-10x's flags, ARC3_MEM_STUCK_FACT and
  ARC3_LEGIBILITY), the combined run;
- ``every``: that plus every flag either branch has (the six flag-only
  legibility fixes, ARC3_MEM_RECAP_FROZEN_FACT, ARC3_MEM_TRIM_STRIP and
  ARC3_MEM_APPEND_NO_HANDOVER).

System prompt: each row of each stage (the memory section's rows, then level
transfer, hygiene, budget truth and legibility, in _build_system_prompt's
order) is logged as it is applied; its old text must occur exactly once at
that point, each stage must apply all of its rows, pf-keep-10x's own rows must
apply as they do without the new flags, every new text must appear exactly
once in the finished prompt (or be extended by a later row, which then
appears once), and the prompt must be pf-keep-10x's with the two branches'
rows applied.

Turn messages: the three openers of tests/flags_off_snapshot.py (an animated
no-op, a game over, an ordinary step) on a level past the stuck threshold,
plus a level-start opener: every memory-v2 row (the stuck count line, the v2
recap ask) and every legibility turn row (RESET bar, turn-based wording) must
find its old text once and leave its new text once.

tests/test_pf_keep_v2_loop.py runs the arm through the real game loop.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import namedtuple
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import flags_off_snapshot as fos  # noqa: E402
from inference.agent import legibility as lg  # noqa: E402
from inference.agent import prompts  # noqa: E402
from inference.agent import tool_agent as ta  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
from notebook_env import notebook_env  # noqa: E402

ARMS = json.loads((HERE.parents[1] / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
ARM, BASE = "pf-keep-v2-all-10x", "pf-keep-10x"
EVERY_FLAG = {
    **{name: "1" for name in lg.FLAGS},
    "ARC3_MEM_RECAP_FROZEN_FACT": "1",
    "ARC3_MEM_TRIM_STRIP": "1",
    "ARC3_MEM_APPEND_NO_HANDOVER": "1",
}
ENVS = {
    "base": ARMS[BASE]["env"],
    "arm": ARMS[ARM]["env"],
    "every": {**ARMS[ARM]["env"], **EVERY_FLAG},
}
ALL_ON = ("arm", "every")

# _build_system_prompt's row stages, in the order it applies them
MEMORY_ROWS = "apply_rows"  # prompts.memory_layout_addendum's rows (the memory section)
PF_STAGES = ("apply_level_transfer_system_guidance", "apply_prompt_hygiene", "apply_budget_truth")
LEGIBILITY = "apply_legibility"
STAGES = (MEMORY_ROWS, *PF_STAGES, LEGIBILITY)

Row = namedtuple("Row", "stage old new found")


def _set_env(monkeypatch, name: str) -> None:
    for key in list(os.environ):
        if key.startswith(("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_", "EXPOSE_")):
            monkeypatch.delenv(key)
    for key, value in {**notebook_env(), **ENVS[name]}.items():
        monkeypatch.setenv(key, str(value))
    for attr, key in (("_DIFF_IMAGE", "ARC3_DIFF_IMAGE"), ("_AUTO_FRAME_DIFF", "ARC3_AUTO_FRAME_DIFF")):
        monkeypatch.setattr(ta, attr, ta._get_env_bool(key, False))  # read at import time


def _logged_system(monkeypatch, name: str) -> tuple[str, list[Row]]:
    """The system prompt under ENVS[name], and every replace() each row stage made."""
    _set_env(monkeypatch, name)
    log: list[Row] = []
    stage = [""]

    class Logged(str):
        def replace(self, old, new, count=-1):
            log.append(Row(stage[0], old, new, str.count(self, old)))
            return Logged(str.replace(self, old, new, count))

    def wrap(module, attr):
        real = getattr(module, attr)

        def logged(text, *args, **kwargs):
            outer = stage[0]
            stage[0] = outer or attr
            try:
                return str(real(Logged(text), *args, **kwargs))
            finally:
                stage[0] = outer

        monkeypatch.setattr(module, attr, logged)

    for attr in (*PF_STAGES, LEGIBILITY, "apply_once"):
        wrap(ta, attr)
    wrap(prompts, MEMORY_ROWS)
    return ta._build_system_prompt(tool_output_tokens=3072), log


@pytest.fixture(scope="module")
def systems():
    mp = pytest.MonkeyPatch()
    try:
        out = {}
        for name in ENVS:
            out[name] = _logged_system(mp, name)
            mp.undo()
        return out
    finally:
        mp.undo()


def _legibility_rows(name: str) -> int:
    on = {flag for flag in lg.FLAGS if ENVS[name].get(flag) == "1"}
    if ENVS[name].get(lg.UMBRELLA) == "1":
        on |= set(lg.BUNDLE)
    return sum(len(rows) for flag, rows in prompts.LEGIBILITY_SYSTEM_ROWS.items() if flag in on)


# ---------------------------------------------------------------- the system prompt


@pytest.mark.parametrize("name", ALL_ON)
def test_every_system_row_finds_its_old_text_once(systems, name):
    _, log = systems[name]
    assert {row.stage for row in log} == set(STAGES)
    applied: set[tuple[str, str]] = set()
    for row in log:
        if row.found == 0 and (row.old, row.new) in applied:
            # budget truth's "ephemeral" row is hygiene's, applied just before
            assert row.stage == "apply_budget_truth" and "ephemeral" in row.old
            continue
        assert row.found == 1, f"{row.stage}: {row.found} matches for {row.old[:70]!r}"
        applied.add((row.old, row.new))


@pytest.mark.parametrize("name", ALL_ON)
def test_each_stage_applies_all_of_its_rows(systems, name):
    _, log = systems[name]
    by_stage = {stage: [row for row in log if row.stage == stage] for stage in STAGES}
    memory = [*prompts.MEMORY_LAYOUT_APPEND_COMMIT_ROWS]
    if ENVS[name].get("ARC3_MEM_RECAP_FROZEN_FACT") == "1":
        memory = [*prompts.MEMORY_LAYOUT_RECAP_FROZEN_ROWS, *memory]
    assert [(row.old, row.new) for row in by_stage[MEMORY_ROWS]] == memory
    # one alternative of every row of every legibility flag that is on
    assert len(by_stage[LEGIBILITY]) == _legibility_rows(name)
    assert len(by_stage["apply_prompt_hygiene"]) == (
        len(prompts.PROMPT_HYGIENE_REPLACEMENTS) + len(prompts.PROMPT_HYGIENE_PERSISTENT_REPLACEMENTS))
    assert len(by_stage["apply_budget_truth"]) == (
        len(prompts.BUDGET_TRUTH_BAR_REPLACEMENTS) + len(prompts.BUDGET_TRUTH_PERSISTENT_REPLACEMENTS))


@pytest.mark.parametrize("name", ALL_ON)
def test_pf_keep_rows_apply_as_without_the_new_flags(systems, name):
    """Level transfer, hygiene and budget truth run before the legibility rows
    and after the memory section is built: the new flags must not change what
    they find."""
    _, base = systems["base"]
    _, log = systems[name]
    assert [row for row in log if row.stage in PF_STAGES] == [row for row in base if row.stage in PF_STAGES]
    assert {row.stage for row in base} == set(PF_STAGES)


@pytest.mark.parametrize("name", ALL_ON)
def test_every_new_text_appears_once_and_no_old_text_is_left(systems, name):
    prompt, log = systems[name]
    applied = [row for row in log if row.found == 1]
    for i, row in enumerate(applied):
        later = [r for r in applied[i + 1:] if r.old in row.new and r.old != r.new]
        if not later and row.new:
            assert prompt.count(row.new) == 1, f"{row.stage}: {row.new[:70]!r}"
        if row.old not in row.new:
            assert row.old not in prompt, f"{row.stage}: {row.old[:70]!r} is still there"
    # the texts each branch adds, by name
    (old, new), = prompts.MEMORY_LAYOUT_APPEND_COMMIT_ROWS
    for text in (new, prompts.RESET_BAR_FIRST_SENTENCE, prompts.TURN_BASED_FACT,
                 "is a step bar. It usually counts down a per-level action budget: it shrinks as actions "
                 "are taken, not with wall-clock time." + prompts.TURN_BASED_FACT + "\n",
                 "- `retained_source` maps the name of each retained function to its source text.\n"):
        assert prompt.count(text) == 1, text[:70]
    assert old not in prompt
    for stale in ("step/time", "a timer or remaining", "timer bar", "remaining-action/time bar"):
        assert stale not in prompt, stale


def _with_rows(prompt: str, flags) -> str:
    for flag, rows in prompts.LEGIBILITY_SYSTEM_ROWS.items():
        if flag in flags:
            for row in rows:
                old, new = next(pair for pair in row if prompt.count(pair[0]) == 1)
                prompt = prompt.replace(old, new)
    return prompt


@pytest.mark.parametrize("name", ALL_ON)
def test_system_prompt_is_pf_keep_with_both_branches_rows(systems, name):
    base, _ = systems["base"]
    got, _ = systems[name]
    expected = base
    if ENVS[name].get("ARC3_MEM_RECAP_FROZEN_FACT") == "1":
        expected = expected.replace(*prompts.MEMORY_LAYOUT_RECAP_FROZEN_ROWS[0])
    expected = expected.replace(*prompts.MEMORY_LAYOUT_APPEND_COMMIT_ROWS[0])
    flags = set(lg.BUNDLE) if name == "arm" else set(lg.FLAGS)
    assert got == _with_rows(expected, flags) != base


def test_the_system_prompt_is_the_same_on_every_build(monkeypatch, systems):
    for name in ALL_ON:
        _set_env(monkeypatch, name)
        assert ta._build_system_prompt(tool_output_tokens=3072) == systems[name][0]


# ---------------------------------------------------------------- turn messages


def _text(message) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


def _openers(monkeypatch, tmp_path, name: str) -> tuple[dict[str, str], list, list]:
    """Turn messages under ENVS[name] (text parts only), and the per-turn rows
    applied while building them: (where, rows, text) for apply_rows and
    (text in, flags, text out) for the legibility turn rows."""
    _set_env(monkeypatch, name)
    monkeypatch.setenv("ARC3_MEM_STUCK_MIN_TOKENS", "60000")
    rows_log: list = []
    turn_log: list = []
    real_rows, real_turn = ta.apply_rows, ta.apply_turn_text

    def logged_rows(text, rows, *, where):
        rows_log.append((where, tuple(rows), text))
        return real_rows(text, rows, where=where)

    def logged_turn(text, flags):
        out = real_turn(text, flags)
        turn_log.append((text, frozenset(flags), out))
        return out

    monkeypatch.setattr(ta, "apply_rows", logged_rows)
    monkeypatch.setattr(ta, "apply_turn_text", logged_turn)
    agent = ta.ToolAgent(model="snapshot-model")
    agent._ensure_session(tmp_path / "state.json")
    # level 1 is past the stuck threshold (60k, no level cleared yet): the
    # count line goes on each of the three openers
    agent._session_generated_tokens = agent._tokens_at_level_start + 77_400
    openers = {key: _text(message) for key, message in fos._openers(agent, Frame, HistoryEntry).items()}
    # a level start: level 1 was cleared by the previous action (the recap ask)
    agent._tokens_at_level_start = agent._session_generated_tokens
    board = fos._grid(0, {(3, 3): 9})
    frame = Frame(grid=board, step=20, level=2)
    summary = {
        "start_action_num": 20, "end_action_num": 20, "executed_count": 1, "executed_actions": ["UP"],
        "level": 2, "level_transition": True, "run_complete": False, "game_over": False,
        "game_over_trace": [], "fatal_action": "", "board_changed": True, "gameplay_changed": True,
        "stop_reason": None, "no_op_stops": [], "animation": None, "animation_action": "UP", "guard_stops": [],
    }
    history = [HistoryEntry(action="", frame=Frame(grid=board, step=0, level=1)),
               HistoryEntry(action="UP", frame=frame)]
    agent._last_animation_chain = []
    prompt = agent._build_user_prompt(20, valid_actions=["UP", "DOWN"], current_frame=frame,
                                      history_entries=history, previous_step_summary=summary)
    openers["level_start"] = _text(agent._build_user_message(prompt, frame))
    return openers, rows_log, turn_log


@pytest.fixture(params=ALL_ON)
def turns(request, monkeypatch, tmp_path):
    return request.param, *_openers(monkeypatch, tmp_path, request.param)


def test_memory_v2_turn_rows_find_their_old_text_once(turns):
    _, openers, rows_log, _ = turns
    where = [entry[0] for entry in rows_log]
    assert where == ["ARC3_MEM_STUCK_FACT"] * 3 + ["ARC3_MEM_RECAP_ASK_V2"]
    for _, rows, text in rows_log:
        for old, _new in rows:
            assert text.count(old) == 1, old[:70]


COUNT_LINE = prompts.stuck_count_line(1, 77_400, [])


@pytest.mark.parametrize("opener", ["animated_noop", "game_over", "plain"])
def test_stuck_count_line_once_right_after_the_valid_actions_line(turns, opener):
    _, openers, _, _ = turns
    text = openers[opener]
    assert text.count("Harness count:") == 1 and text.count(COUNT_LINE) == 1
    valid = re.findall(r"Valid actions right now: [^\n]*\.", text)
    assert len(valid) == 1
    assert f"{valid[0]}\n{COUNT_LINE}\n" in text
    assert text.index(COUNT_LINE) < text.index("Only tool:"), "before the opener's static tail"


def test_level_start_v2_recap_ask_once_and_no_count_line(turns):
    _, openers, _, _ = turns
    text = openers["level_start"]
    (old, new), = prompts.recap_ask_v2_rows(1)
    assert text.count(new) == 1 and old not in text
    assert text.count("Level 1 recap:") == 1 and "`Level 1 recap: Goal: ... Rules: ...`" in text
    assert "Harness count:" not in text
    assert text.index(new) < text.index("Current state: step 21, level 2.")


def test_legibility_turn_rows_find_their_old_text_at_most_once(turns):
    name, openers, _, turn_log = turns
    assert turn_log, "the game-over opener goes through the turn rows"
    for text_in, flags, text_out in turn_log:
        assert {lg.RESET_BAR_FACT, lg.TURN_BASED_FACT} <= flags
        for flag, rows in prompts.TURN_TEXT_ROWS.items():
            for old, new in rows:
                assert text_in.count(old) <= 1, old
                if old in text_in:
                    assert old not in text_out and text_out.count(new) == 1, new


def test_game_over_opener_carries_both_branches_texts_once(turns):
    _, openers, _, _ = turns
    text = openers["game_over"]
    (old, new), = prompts.RESET_BAR_TEXT_ROWS
    assert text.count(new) == 1 and old not in text
    assert text.count("or by a step budget running out") == 1
    assert text.count(COUNT_LINE) == 1
    for opener in openers.values():
        assert "step/time" not in opener and "a timer or remaining" not in opener


def test_terminal_game_over_detail_uses_the_turn_based_row(turns):
    detail = ta._terminal_action_stop_detail("game_over")
    assert detail.count("a depleted step budget (usually") == 1 and "step/time" not in detail
