"""Memory v2 (arm/mem-v2) unit checks: every new flag's pieces on their own.

The game-loop checks (flags off byte-identical to pf-keep-10x, each arm's
requests) are in test_memory_v2_loop.py.
"""
from __future__ import annotations

import logging
import re

from inference.agent import prompts

# ---------------------------------------------------------------- apply_rows


def test_apply_rows_replaces_a_row_that_occurs_once():
    assert prompts.apply_rows("a b c", (("b", "B"),), where="t") == "a B c"


def test_apply_rows_skips_a_missing_or_repeated_row(caplog):
    with caplog.at_level(logging.WARNING, logger=prompts.__name__):
        assert prompts.apply_rows("a b b", (("b", "B"), ("x", "X"), ("a", "A")), where="t") == "A b b"
    assert sum("prompt row skipped" in r.getMessage() for r in caplog.records) == 2


# ---------------------------------------------------------------- render_entry / layout_stage

from inference.agent import memory_layout as ML  # noqa: E402
from inference.agent.memory_ledger import MemoryLedger  # noqa: E402

ENTRIES = [
    ML.LevelEntry(level=1, path=["UP", "UP", "MOUSE(3,4)"], spent=5, deaths=0,
                  recap="The goal is to reach the exit; UP moves the block one cell."),
    ML.LevelEntry(level=2, path=["LEFT"] * 30 + ["RIGHT"], spent=40, deaths=2, winning=35,
                  notes=[("World model", "the red bar is a wall"), ("Refuted", "SPACE opens it")]),
    ML.LevelEntry(level=3, path=["SPACE"], spent=1, deaths=0),
]


def test_render_entry_is_the_entry_as_previous_levels_draws_it():
    block = ML.render_previous_levels(ENTRIES)
    body = block.split("\n", 1)[1].rsplit("\n", 1)[0]
    assert body == "\n".join(ML.render_entry(e) for e in ENTRIES)
    for e in ENTRIES:
        assert ML.render_entry(e) in block


def _solved_ledger() -> MemoryLedger:
    ledger = MemoryLedger(card_mode=True, solved_mode=True, debrief_mode=True, recap_ask=True)
    ledger.layout_mode = ledger.legible = True
    ledger.add_solved(1, ["UP", "LEFT"], 3, 0)
    ledger.add_solved(2, ["RIGHT"], 1, 1)
    return ledger


def test_layout_stage_freezes_the_entry_without_a_redraw():
    staged, committed = _solved_ledger(), _solved_ledger()
    blocks = staged.layout_stage([1, 2])
    assert staged.prev_text == "" and sorted(staged.prev_entries) == [1, 2]
    assert committed.layout_commit([1, 2])
    for level, block in zip((1, 2), blocks):
        text = ML.render_entry(committed.prev_entries[level])
        assert block == f"<cleared_level>\n{text}\n</cleared_level>"
    assert staged.layout_redraw() and staged.prev_text == committed.prev_text
    assert staged.layout_pending() == []


# ---------------------------------------------------------------- ARC3_MEM_APPEND_COMMIT

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from inference.agent import tool_agent as ta  # noqa: E402
from notebook_env import notebook_env  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text())["arms"]
PF_KEEP = ARMS["pf-keep-10x"]["env"]


def _system(monkeypatch, **extra) -> str:
    for key in list(os.environ):
        if key.startswith(("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_", "EXPOSE_")):
            monkeypatch.delenv(key)
    for key, value in {**notebook_env(), **PF_KEEP, **extra}.items():
        monkeypatch.setenv(key, value)
    return ta._build_system_prompt(tool_output_tokens=3072)


def test_append_commit_row_matches_once_in_the_real_prompt(monkeypatch):
    (old, new), = prompts.MEMORY_LAYOUT_APPEND_COMMIT_ROWS
    base = _system(monkeypatch)
    assert base.count(old) == 1
    assert prompts.memory_layout_addendum(debrief=True, legible=True, recap_ask=True).count(old) == 1
    on = _system(monkeypatch, ARC3_MEM_APPEND_COMMIT="1")
    assert on == base.replace(old, new) and on.count(new) == 1 and old not in on
    assert "The python `history` and `transitions` lists are never trimmed: they hold every action" in on
    # inert with level compaction (the commit stays) and without the layout
    assert _system(monkeypatch, ARC3_MEM_APPEND_COMMIT="1", ARC3_LEVEL_COMPACT="1") == \
        _system(monkeypatch, ARC3_LEVEL_COMPACT="1")
    assert _system(monkeypatch, ARC3_MEM_APPEND_COMMIT="1", ARC3_MEM_EVERY_CALL="0") == \
        _system(monkeypatch, ARC3_MEM_EVERY_CALL="0")


def test_recap_frozen_row_matches_once_and_stacks_with_append_commit(monkeypatch):
    (old, new), = prompts.MEMORY_LAYOUT_RECAP_FROZEN_ROWS
    (a_old, a_new), = prompts.MEMORY_LAYOUT_APPEND_COMMIT_ROWS
    base = _system(monkeypatch)
    assert base.count(old) == 1
    assert _system(monkeypatch, ARC3_MEM_RECAP_FROZEN_FACT="1") == base.replace(old, new)
    both = _system(monkeypatch, ARC3_MEM_RECAP_FROZEN_FACT="1", ARC3_MEM_APPEND_COMMIT="1")
    assert both == base.replace(old, new).replace(a_old, a_new)
    assert "The python `history` and `transitions` lists are never trimmed" in both


def test_append_commit_addendum_off_is_unchanged():
    kwargs = dict(debrief=True, legible=True, recap_ask=True)
    assert prompts.memory_layout_addendum(**kwargs) == prompts.memory_layout_addendum(**kwargs, append_commit=False)


BLOCK = "<cleared_level>\nLevel 1: cleared in 2 actions (3 spent, no deaths). Route: UP, LEFT.\n</cleared_level>"
THIS_LEVEL = {"type": "text", "text": "\n\n<this_level>\nLevel 2 so far: no actions yet.\n</this_level>"}
IMAGE = {"type": "image_url", "image_url": {"url": "data:x"}}


def test_tail_goes_right_before_this_level_in_a_fresh_turn_message():
    opener = {"role": "user", "content": [{"type": "text", "text": "Current state: step 4, level 2."}, IMAGE, THIS_LEVEL]}
    ta._mark_control_message(opener, "resume")
    out = ta._with_tail_before_this_level(opener, [BLOCK])
    assert out["content"] == [opener["content"][0], IMAGE, {"type": "text", "text": "\n\n" + BLOCK}, THIS_LEVEL]
    assert len(opener["content"]) == 3, "the original message is not mutated"
    assert out[ta._CONTROL_MESSAGE_KEY] == "resume"


def test_tail_is_appended_to_a_resumption_list_or_string():
    resumed = {"role": "user", "content": [{"type": "text", "text": "Your last python snippet ..."}, IMAGE]}
    out = ta._with_tail_before_this_level(resumed, [BLOCK, BLOCK])
    assert out["content"][2:] == [{"type": "text", "text": "\n\n" + BLOCK}] * 2
    as_string = {"role": "user", "content": "Current grid image: unchanged since the last grid image above"}
    assert ta._with_tail_before_this_level(as_string, [BLOCK])["content"] == as_string["content"] + "\n\n" + BLOCK
    with_tail = {"role": "user", "content": "state" + THIS_LEVEL["text"]}
    assert ta._with_tail_before_this_level(with_tail, [BLOCK])["content"] == "state\n\n" + BLOCK + THIS_LEVEL["text"]
    assert ta._with_tail_before_this_level(resumed, []) is resumed


def test_strip_cleared_level_removes_only_the_blocks():
    fresh = ta._with_tail_before_this_level(
        {"role": "user", "content": [{"type": "text", "text": "Current state: step 4, level 2."}, THIS_LEVEL]}, [BLOCK])
    string = {"role": "user", "content": "Your last python snippet\n\n" + BLOCK}
    quoted = [
        {"role": "assistant", "content": "", "reasoning_content": "the " + BLOCK + " says"},
        {"role": "tool", "tool_call_id": "c1", "content": "printed " + BLOCK},
    ]
    plain = {"role": "user", "content": [{"type": "text", "text": "no block"}]}
    out, removed = ta._strip_cleared_level([fresh, string, *quoted, plain])
    assert removed == 2
    assert out[0]["content"] == [fresh["content"][0], THIS_LEVEL]
    assert out[1]["content"] == "Your last python snippet"
    assert out[2] is quoted[0] and out[3] is quoted[1] and out[4] is plain
    assert BLOCK in "".join(p["text"] for p in fresh["content"]), "the input is not mutated"


# ---------------------------------------------------------------- ARC3_MEM_TRIM_STRIP


def _this_level(level: int, n: int) -> dict:
    return {"type": "text", "text": f"\n\n<this_level>\nLevel {level} so far: {n} actions, attempt 1, no deaths.\n</this_level>"}


def test_trim_strip_keeps_the_newest_block_of_each_level():
    opener = lambda level, n: {"role": "user", "content": [  # noqa: E731
        {"type": "text", "text": f"Current state: step {n}, level {level}."}, IMAGE, _this_level(level, n)]}
    quoted = {"role": "assistant", "content": "", "reasoning_content": _this_level(3, 1)["text"]}
    tool = {"role": "tool", "tool_call_id": "c", "content": _this_level(3, 2)["text"]}
    string_old = {"role": "user", "content": "state" + _this_level(4, 5)["text"]}
    history = [opener(3, 1), quoted, tool, opener(3, 2), string_old, opener(4, 6), opener(4, 7)]
    out, removed = ta._strip_stale_this_level(history)
    assert removed == 3
    kept = [m for m in out if m["role"] == "user"]
    assert kept[0]["content"] == history[0]["content"][:2]           # level 3, older: stripped
    assert kept[1] is history[3]                                     # newest of level 3: kept
    assert kept[2]["content"] == "state"                             # level 4, string form: stripped
    assert kept[3]["content"] == history[5]["content"][:2]           # level 4, older: stripped
    assert kept[4] is history[6]                                     # the current turn message: kept
    assert out[1] is quoted and out[2] is tool
    assert ta._strip_stale_this_level(out) == (out, 0)


# ---------------------------------------------------------------- ARC3_MEM_RECAP_CHARS

from inference.agent import memory_facts as MF  # noqa: E402

# vc33 level 2's first reasoning in pf-keep-10x (run B), up to its turn to the new board
VC33_L1 = (
    "Level 1 recap: The board had two horizontally-sliding bands (top and bottom) separated by a fixed black "
    "wall with a 2-wide yellow gap; clicking the blue handle block above the wall slid the top band left / "
    "bottom band right (each step 4 cols), and clicking the blue handle below the wall did the opposite. The "
    "goal was to slide the charcoal+yellow \"key\" piece (embedded in the bottom band's gray/white boundary, "
    "always 2 cols left of it) until its yellow blade aligned with the wall's yellow gap columns — that "
    "alignment completed the level (the 6th click at k=-3, i.e., piece yellow at cols 38-39, triggered the win, "
    "with an animation showing the piece moving up through the gap).\nWait — actually the win happened at "
    "the 6th click, when piece was at [38..43] (yellow at 42-43), not 38-39. Hmm. Let me check: at i=4 (step "
    "19) piece=[38..43] → charcoal 38-41, yellow 42-43. Then i=5 (step 20) → LEVEL COMPLETED.\n"
    "Now Level 2: the layout is mirrored. Let me inspect."
)
CORRECTED = "piece yellow at cols 38-39, triggered the win"
CORRECTION = "Wait — actually the win happened at the 6th click"


def test_tidy_never_shows_a_corrected_claim_without_its_correction():
    recap = MF.extract_recap(VC33_L1, 1, max_chars=800, tidy=True)
    assert recap.startswith("The board had two horizontally-sliding bands")
    assert "The goal was to slide" in recap and len(recap) <= 800
    # the correction and the sentence it corrects stand or fall together
    assert (CORRECTED in recap) == (CORRECTION in recap)
    assert CORRECTION in recap and "Hmm" not in recap and not recap.endswith("...")
    # B (420 capture, 300 shown) showed neither, nor the goal
    shown_b = MF.cut_words(MF.extract_recap(VC33_L1, 1), 300)
    assert CORRECTED not in shown_b and CORRECTION not in shown_b and "goal" not in shown_b


def test_tidy_keeps_corrections_and_drops_sentences_with_no_rule():
    text = ("Level 2 recap: Level 2 completed! Pushing the box onto the pad opens the door; UP moves the block 3 "
            "cells. Wait, actually the door opens only when both pads are covered. Did the bar shrink? Nice. "
            "Now level 3.")
    assert MF.extract_recap(text, 2, max_chars=800, tidy=True) == (
        "Pushing the box onto the pad opens the door; UP moves the block 3 cells. Wait, actually the door "
        "opens only when both pads are covered.")
    # tidy off: today's capture, the celebration and the question included
    assert MF.extract_recap(text, 2).startswith("Level 2 completed! Pushing the box")


def test_brackets_are_not_split():
    text = ("Level 4 recap: Solving = every block consistent with all covering displays (19 clicks incl. 3 probe "
            "clicks). Each click toggles a block red/blue. Now level 5.")
    assert MF.extract_recap(text, 4, max_chars=800, tidy=True) == (
        "Solving = every block consistent with all covering displays (19 clicks incl. 3 probe clicks). Each "
        "click toggles a block red/blue.")


MECHANICS = " ".join(f"Rule {i}: the lever {i} shifts band {i} by four columns to the left." for i in range(8))


def test_a_cut_is_marked_and_keeps_the_goal_sentence():
    text = f"Level 3 recap: {MECHANICS} The goal is to align the key with the gap. Now level 4."
    for cap in (300, 420):
        recap = MF.extract_recap(text, 3, max_chars=cap, tidy=True)
        assert len(recap) <= cap
        assert recap.endswith("The goal is to align the key with the gap.") and " ... " in recap
        assert recap.startswith("Rule 0:")
    # B's fit loses the goal and marks nothing
    assert "goal" not in MF.extract_recap(text, 3, max_chars=300)
    plain = f"Level 3 recap: {MECHANICS} Now level 4."
    recap = MF.extract_recap(plain, 3, max_chars=300, tidy=True)
    assert recap.endswith(" ...") and len(recap) <= 300
    assert MF.extract_recap("Level 3 recap: Hmm. Nice. Level 3 done! Now level 4.", 3, max_chars=800, tidy=True) == ""


def test_cut_words_marker():
    assert MF.cut_words("one two three four", 12) == "one two..."
    assert MF.cut_words("one two three four", 12, marker=" ...") == "one two ..."
    assert MF.cut_words("one two", 12, marker=" ...") == "one two"


def test_ledger_captures_the_tidy_recap_under_the_cap():
    ledger = MemoryLedger(card_mode=True, solved_mode=True, debrief_mode=True, recap_ask=True, recap_chars=800)
    ledger.level = 1
    ledger.on_level_change(2)
    ledger.note_think(VC33_L1)
    assert ledger._debriefs[1] == MF.extract_recap(VC33_L1, 1, max_chars=800, tidy=True)


def test_entry_lines_show_the_recap_up_to_the_cap():
    recap = MF.extract_recap(VC33_L1, 1, max_chars=800, tidy=True)
    entry = ML.LevelEntry(level=1, path=["UP"], spent=1, deaths=0, recap=recap)
    options = ML.HeadOptions(recap_chars=800, cut_marker=" ...")
    assert ML.render_entry(entry, options=options).endswith(f"Your recap: {recap}")
    assert ML.render_entry(entry).endswith(f"Your recap: {MF.cut_words(recap, 300)}")
    short = ML.HeadOptions(recap_chars=200, cut_marker=" ...")
    assert ML.render_entry(entry, options=short).endswith(" ...")


# ---------------------------------------------------------------- ARC3_MEM_COMPRESS_ROUTES_FIRST

LONG_RECAP = ("The lever moves the band four columns; the goal is to align every key with its gap. " * 10)[:780]


def _ten_entries() -> list:
    route = ["UP", "LEFT", "MOUSE(12,30)", "DOWN", "MOUSE(40,7)", "RIGHT"] * 8
    return [ML.LevelEntry(level=n, path=route, spent=60, deaths=0, recap=LONG_RECAP) for n in range(1, 11)]


def test_routes_first_drops_every_older_route_before_cutting_a_recap():
    options = ML.HeadOptions(recap_chars=800, cut_marker=" ...", routes_first=True)
    text = ML.render_previous_levels(_ten_entries(), max_chars=6000, options=options)
    assert len(text) <= 6000
    lines = text.split("\n")[1:-1]
    heads = [line for line in lines if line.startswith("Level ")]
    recaps = [line for line in lines if line.startswith("  Your recap: ")]
    cut = [line.endswith(" ...") for line in recaps]
    routed = ["Route:" in head for head in heads]
    if any(cut):
        assert not any(routed[:-1]), "a recap was cut while an older entry kept its route"
    # the newest entry is compressed last: it is whole while any older one is not
    assert routed[-1] and not cut[-1]
    # the 10 x 800-char recaps do not all fit: some recap text was cut, oldest first
    assert any(cut) and cut == sorted(cut, reverse=True)


def test_without_routes_first_the_oldest_entry_loses_its_recap_first():
    options = ML.HeadOptions(recap_chars=800, cut_marker=" ...")
    text = ML.render_previous_levels(_ten_entries(), max_chars=6000, options=options)
    lines = text.split("\n")[1:-1]
    heads = [line for line in lines if line.startswith("Level ")]
    # B's order: newer entries keep their routes while the oldest lose recap text
    assert "Route:" in heads[-2] and not any("Route:" in h for h in heads[:2])


# ---------------------------------------------------------------- ARC3_MEM_ROUTE_FORMAT=readable

PATH = ["UP", "UP", "UP", *["MOUSE(48,57)"] * 4, "MOUSE(48,12)", "MOUSE(5,5)", "MOUSE(5,5)", "MOUSE(5,5)", "LEFT"]
KEYS = ["UP", "UP", "UP", *["MOUSE@R"] * 4, "MOUSE@b", "MOUSE@R", "MOUSE@R", "MOUSE@b", "LEFT"]


def test_readable_route_names_each_click_by_colour():
    assert ML.route_words(PATH, 320, keys=KEYS) == (
        "UP x3, click (48,57) red x4, click (48,12) blue, click (5,5) red x2, click (5,5) blue, LEFT")
    # pf-keep-10x's form without keys
    assert ML.route_words(PATH, 320) == "UP x3, click (48, 57) x4, click (48, 12), click (5, 5) x3, LEFT"
    # an unknown colour leaves the click unnamed
    assert ML.route_words(["MOUSE(1,2)"], 320, keys=["MOUSE@?"]) == "click (1,2)"
    assert ML.route_words(["MOUSE(1,2)"], 320, keys=[]) == "click (1,2)"


def test_readable_route_is_cut_only_between_runs_and_keeps_its_end():
    path = [f"MOUSE({r},{r})" for r in range(10, 60)] + ["LEFT", "LEFT"]
    keys = ["MOUSE@N"] * 50 + ["LEFT", "LEFT"]
    text = ML.route_words(path, 320, keys=keys)
    assert len(text) <= 320
    head, rest = text.split(", ... ", 1)
    gap, tail = rest.split(" ..., ")
    assert tail == "click (59,59) light green, LEFT x2"
    assert all(re.fullmatch(r"click \(\d+,\d+\) light green", part) for part in head.split(", "))
    assert int(gap.split()[0]) + len(head.split(", ")) + 3 == 52
    # a single run longer than the cap is never cut inside
    assert ML.route_words(["MOUSE(1,2)"] * 3, 10, keys=["MOUSE@R"] * 3) == "3 actions ..."
    # pf-keep-10x's form is unchanged: head only
    assert ML.route_words(path, 240).endswith(" more actions")


def test_recorder_reports_each_path_action_key_only_when_asked():
    def clear(route_keys: bool) -> dict:
        recorder = MF.FactRecorder()
        recorder.route_keys = route_keys
        events: list = []
        for sig, key, pre, post, cls in [("MOUSE(3,4)", "MOUSE@R", "a", "b", MF.SHIFT), ("UP", "UP", "b", "c", MF.SHIFT),
                                         ("DOWN", "DOWN", "c", "b", MF.SHIFT), ("UP", "UP", "b", "c", MF.SHIFT),
                                         ("MOUSE(9,9)", "MOUSE@b", "c", "d", MF.CLEAR)]:
            recorder.step(1, sig, key, cls, pre, post, events=events)
        return next(e.data for e in events if e.kind == "clear")

    on, off = clear(True), clear(False)
    assert on["path"] == off["path"] == ["MOUSE(3,4)", "UP", "MOUSE(9,9)"]
    assert on["keys"] == ["MOUSE@R", "UP", "MOUSE@b"] and "keys" not in off
    assert {k: v for k, v in on.items() if k != "keys"} == off


# ---------------------------------------------------------------- ARC3_MEM_HEAD_WORDS


def test_head_words_say_what_each_count_is():
    assert ML.solved_words(2, ["UP"] * 12, 20, 0, 1, route_chars=0) == (
        "Level 2: cleared. 20 actions spent in 1 attempt, no deaths. Winning route without loops: 12 actions.")
    assert ML.solved_words(2, ["UP"] * 12, 20, 2, 3) == (
        "Level 2: cleared. 20 actions spent in 3 attempts, 2 deaths. Winning route without loops (12 actions): UP x12.")
    entry = ML.LevelEntry(level=2, path=PATH, spent=30, deaths=1, keys=KEYS, attempts=2)
    options = ML.HeadOptions(head_words=True, readable_routes=True, route_chars=320)
    assert ML.render_entry(entry, options=options) == (
        "Level 2: cleared. 30 actions spent in 2 attempts, 1 death. Winning route without loops (12 actions): "
        + ML.route_words(PATH, 320, keys=KEYS) + ".")
    # no attempt count recorded: pf-keep-10x's line
    plain = ML.LevelEntry(level=2, path=PATH, spent=30, deaths=1)
    assert ML.render_entry(plain, options=options) == ML.solved_text(2, PATH, 30, 1, route_chars=320, keys=[])


def _agent_records(monkeypatch, **flags) -> dict:
    for key in list(os.environ):
        if key.startswith(("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_", "EXPOSE_")):
            monkeypatch.delenv(key)
    for key, value in {**notebook_env(), **PF_KEEP, **flags}.items():
        monkeypatch.setenv(key, value)
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    agent._ledger = ta._new_memory_ledger()
    agent._mem_recorder = None
    agent._current_valid_actions = []
    data = {"key": "MOUSE@b", "sig": "MOUSE(9,9)", "path": PATH, "spent": 30, "deaths": 1, "attempt": 2,
            "game_first": False, "actions": 12, "keys": KEYS}
    ta.ToolAgent._mem_apply_events(agent, [MF.FactEvent("clear", 2, data)])
    return agent._ledger._items["solved:2"].data


def test_record_fields_are_added_only_under_their_flags(monkeypatch):
    off = _agent_records(monkeypatch)
    zero = _agent_records(monkeypatch, ARC3_MEM_HEAD_WORDS="0", ARC3_MEM_ROUTE_FORMAT="")
    assert off == zero and "attempts" not in off and "keys" not in off
    assert set(off) == {"level", "path", "spent", "deaths"}
    words = _agent_records(monkeypatch, ARC3_MEM_HEAD_WORDS="1")
    assert words == {**off, "attempts": 2}
    readable = _agent_records(monkeypatch, ARC3_MEM_ROUTE_FORMAT="readable")
    assert readable == {**off, "keys": KEYS}


# ---------------------------------------------------------------- ARC3_MEM_RECAP_ASK_V2

V2_RECAP = ("Level 3 recap: Goal: push both boxes onto the pads; that opened the exit. Rules: UP/DOWN move the "
            "green piece 3 cells; a pushed box slides until blocked. Corrects level 2: guards lunge along their "
            "facing direction, not along their row. Now level 4. Let me inspect the board.")


@pytest.mark.parametrize("level", [1, 2, 9, 12])
def test_v2_ask_rows_match_once_and_keep_the_label(level):
    ask = prompts.level_recap_ask(level)
    (old, new), = prompts.recap_ask_v2_rows(level)
    assert ask.count(old) == 1
    v2 = prompts.apply_rows(ask, prompts.recap_ask_v2_rows(level), where="t")
    assert f"`Level {level} recap: Goal: ... Rules: ...`" in v2 and f"Level {level} recap:" in v2
    assert "`Corrects level K: ...`" in v2 and v2.endswith(ask[ask.index(" The harness copies"):])


def test_split_recap_parts_and_corrections():
    recap = MF.extract_recap(V2_RECAP, 3)
    assert recap.startswith("Goal: push both boxes") and "Let me inspect" not in recap
    goal, rules, corrections = MF.split_recap(recap, 3)
    assert goal == "push both boxes onto the pads; that opened the exit."
    assert rules == "UP/DOWN move the green piece 3 cells; a pushed box slides until blocked."
    assert corrections == [(2, "guards lunge along their facing direction, not along their row.")]
    # markdown labels, rules first, text before the first label, a later level is no correction
    assert MF.split_recap("Mirrored arrows. **Rules:** LEFT moves right. **Goal:** cover the yellow. "
                          "Corrects level 5: x", 3) == (
        "cover the yellow. Corrects level 5: x", "Mirrored arrows. LEFT moves right.", [])
    assert MF.split_recap("the goal was to reach the exit; UP moves one cell.", 3) is None


def _v2_ledger(recap_v2: bool) -> MemoryLedger:
    ledger = MemoryLedger(card_mode=True, solved_mode=True, debrief_mode=True, recap_ask=True)
    ledger.layout_mode = ledger.legible = True
    ledger.recap_v2 = recap_v2
    ledger.level = 1
    for level, reasoning in ((1, "Level 1 recap: Goal: reach the exit. Rules: UP moves the block. Now level 2."),
                             (2, "Level 2 recap: the goal is to reach the exit; guards lunge along their row. "
                                 "Now level 3."),
                             (3, V2_RECAP)):
        ledger.add_solved(level, ["UP"], 1, 0)
        ledger.on_level_change(level + 1)
        ledger.note_think(reasoning)
    ledger.layout_commit([1, 2, 3])
    return ledger


def test_v2_recap_is_shown_as_goal_and_rules_and_marks_the_corrected_entry():
    on, off = _v2_ledger(True), _v2_ledger(False)
    assert "  Goal: reach the exit.\n  Rules: UP moves the block." in on.prev_text
    assert "  Your recap: the goal is to reach the exit; guards lunge along their row." in on.prev_text
    assert ("  Corrected by level 3: guards lunge along their facing direction, not along their row."
            in on.prev_text)
    assert on.prev_text.index("Corrected by level 3") < on.prev_text.index("Level 3: cleared")
    assert "  Goal:" not in off.prev_text and "Corrected by" not in off.prev_text
    assert "  Your recap: Goal: reach the exit. Rules: UP moves the block." in off.prev_text
    assert on._debriefs == off._debriefs, "the captured recap text is the same"


# ---------------------------------------------------------------- the <this_level> words


def _summary_line(**kw) -> str:
    base = dict(level=3, attempt=2, actions=61, deaths=0, rows=[], hazards=[], seen="", notes=[])
    return ML.render_this_level(**{**base, **kw}).split("\n")[1]


def test_attempt_counts_say_which_count_is_which():
    assert _summary_line(attempt_actions=22) == (
        "Level 3 so far: 61 actions over 2 attempts, 22 of them in this attempt (attempt 2), no deaths.")
    assert _summary_line() == "Level 3 so far: 61 actions, attempt 2, no deaths."
    assert _summary_line(attempt=1, actions=9, attempt_actions=9) == "Level 3 so far: 9 actions, attempt 1, no deaths."
    assert _summary_line(actions=0, attempt_actions=0) == "Level 3 so far: no actions yet."


def _played() -> tuple[MF.FactRecorder, list]:
    """Level 1 from game action 140: UP, RIGHT, LEFT (back), RESET, UP, DOWN (death)."""
    recorder = MF.FactRecorder()
    recorder._acts = 140
    events: list = []
    recorder._begin_level(1, None, events, start_h="s", announce=False)
    for act, sig, cls, pre, post in [(141, "UP", MF.SHIFT, "s", "a"), (142, "RIGHT", MF.CHANGE, "a", "b"),
                                     (143, "LEFT", MF.SHIFT, "b", "a")]:
        recorder.step(1, sig, sig, cls, pre, post, events=events, act=act)
    return recorder, events


def test_attempt_actions_come_from_the_recorder():
    recorder, events = _played()
    recorder.reset_step(1, "s", None, manual=True, events=events, act=144)
    recorder.step(1, "UP", "UP", MF.SHIFT, "s", "a", events=events, act=145)
    tracker = recorder.attempts
    assert (tracker.spent, tracker.attempt, len(tracker.steps)) == (5, 2, 1)


def test_seen_and_game_over_lines_give_the_game_action_first():
    recorder, events = _played()
    valid = ["UP", "DOWN", "LEFT", "RIGHT"]
    assert ML.seen_text(recorder, valid) == (
        "The current board was seen before on this level: after action 1 of this attempt. "
        "From it: RIGHT reached the board after action 2. Not tried from it: UP, DOWN, LEFT.")
    assert ML.seen_text(recorder, valid, game_actions=True) == (
        "The current board was seen before on this level: after action 141 (action 1 of this attempt). "
        "From it: RIGHT reached the board after action 142 (action 2 of this attempt). "
        "Not tried from it: UP, DOWN, LEFT.")
    recorder.step(1, "DOWN", "DOWN", MF.DEATH, "a", "x", events=events, act=144)
    death = next(e.data for e in events if e.kind == "death")
    assert (death["act"], death["first_act"], death["first_index"], death["length"]) == (144, 141, 1, 4)
    args = (1, "DOWN", death["first_index"], death["attempt"], death["length"])
    assert ML.hazard_text(*args) == (
        "Level 1: attempt 1 ended in GAME OVER after 4 actions. The last action was DOWN, "
        "from the board first reached after action 1 of that attempt.")
    assert ML.hazard_text(*args, act=death["act"], first_act=death["first_act"]) == (
        "Level 1: attempt 1 ended in GAME OVER at action 144 (action 4 of that attempt). The last action "
        "was DOWN, from the board first reached after action 141 (action 1 of that attempt).")
    # the next attempt: its start is named by the game's number too
    recorder.reset_step(1, "s", None, manual=False, events=events, act=145)
    recorder.step(1, "UP", "UP", MF.SHIFT, "s", "a", events=events, act=146)
    recorder.step(1, "RIGHT", "RIGHT", MF.SHIFT, "a", "s", events=events, act=147)
    assert ML.seen_text(recorder, valid, game_actions=True).startswith(
        "The current board was seen before on this level: at the start of attempt 1 (after action 140); "
        "at the start of this attempt (after action 145).")


def test_effect_words_name_what_each_class_measures():
    recorder = MF.FactRecorder()
    events: list = []
    recorder._begin_level(1, None, events, start_h="s", announce=False)
    for sig, cls in [("UP", MF.SHIFT), ("UP", MF.CHANGE), ("SPACE", MF.ANIM), ("LEFT", MF.HUD), ("DOWN", MF.NOOP)]:
        recorder.step(1, sig, sig, cls, "s", "s" if cls in MF.STILL else "t", events=events)
    valid = ["UP", "SPACE", "LEFT", "DOWN"]
    old = ML.effect_rows(recorder.effects, valid, game=False)
    new = ML.effect_rows(recorder.effects, valid, game=False, words=ML.effect_words_v2(4))
    assert old == ML.effect_rows(recorder.effects, valid, game=False, words=None)
    assert new[0] == ("UP (2 presses on this level): once moved cells (no colour count changed), "
                      "once changed colour counts (cells may also have moved).")
    assert old[0] == "UP (2 presses on this level): moved cells once, changed cell colours once."
    # click rows too: today's order with the flag off
    recorder.step(1, "MOUSE(5,5)", "MOUSE@R", MF.SHIFT, "s", "u", events=events)
    clicks = [r for r in ML.effect_rows(recorder.effects, [*valid, "MOUSE"], game=False) if r.startswith("Clicks")]
    assert clicks == ["Clicks on red cells (1 click on 1 cell): moved cells once. Last: (5, 5) moved cells."]
    clicks = [r for r in ML.effect_rows(recorder.effects, [*valid, "MOUSE"], game=False, words=ML.effect_words_v2(4))
              if r.startswith("Clicks")]
    assert clicks == ["Clicks on red cells (1 click on 1 cell): once moved cells (no colour count changed). "
                      "Last: (5, 5) moved cells (no colour count changed)."]
    assert "SPACE (1 press on this level): the board inside the edge never changed, on 1 board." in new
    words = ML.effect_words_v2(3)
    assert words[MF.ANIM] == "animated, then left the board inside the 3-cell border as it was"
    assert words[MF.HUD] == "changed only cells in the 3-cell border"
    assert words[MF.NOOP] == ML.EFFECT_WORDS[MF.NOOP] and words[MF.CLEAR] == ML.EFFECT_WORDS[MF.CLEAR]


# ---------------------------------------------------------------- ARC3_MEM_STUCK_FACT


@pytest.mark.parametrize("cleared,want", [
    ([], 60000), ([49700], 60000), ([14600, 49500, 27000, 49600], 76500),
    ([2400, 30500, 2600, 2500, 2800, 2100], 60000), ([40000, 50000], 90000),
])
def test_stuck_threshold(monkeypatch, cleared, want):
    for key in ("ARC3_MEM_STUCK_MIN_TOKENS", "ARC3_MEM_STUCK_MEDIAN_MULT", "ARC3_MEM_STUCK_MEDIAN_FROM"):
        monkeypatch.delenv(key, raising=False)
    assert ta.mem_stuck_threshold(cleared) == want
    monkeypatch.setenv("ARC3_MEM_STUCK_MEDIAN_FROM", "1")
    if len(cleared) == 1:
        assert ta.mem_stuck_threshold(cleared) == max(60000, 2 * cleared[0])


def test_stuck_count_line_goldens():
    # run B: bp35 L2 at ridx 49, ls20 L5 at ridx 126, m0r0 L1
    assert prompts.stuck_count_line(2, 68206, [(1, 49700)]) == (
        "Harness count: you have written 68.2k tokens of reasoning and code on level 2 without clearing it; "
        "level 1 took 49.7k tokens.")
    assert prompts.stuck_count_line(5, 77376, [(1, 14600), (2, 49500), (3, 27000), (4, 49600)]) == (
        "Harness count: you have written 77.4k tokens of reasoning and code on level 5 without clearing it; "
        "levels 1-4 took 14.6k, 49.5k, 27.0k and 49.6k tokens (median 38.2k).")
    assert prompts.stuck_count_line(1, 61000, []) == (
        "Harness count: you have written 61.0k tokens of reasoning and code on level 1 without clearing it.")
    assert "recap" not in prompts.STUCK_COUNT_LINE and "budget" not in prompts.STUCK_COUNT_LINE


def _stuck_agent(monkeypatch, tokens: int, cleared: list) -> "ta.ToolAgent":
    monkeypatch.setenv("ARC3_MEM_STUCK_MIN_TOKENS", "60000")
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    agent._reset_mem_layout_state()
    agent._session_generated_tokens, agent._tokens_at_level_start = 100000 + tokens, 100000
    agent._mem_level_tokens = list(cleared)
    return agent


TURN = ["The code executed 1 action in the previous sequence.", "You are still on the same level.",
        "Current state: step 37, level 2.", "Valid actions right now: RESET, LEFT, RIGHT, MOUSE, UNDO."]


def test_stuck_row_follows_the_valid_actions_line_once(monkeypatch):
    agent = _stuck_agent(monkeypatch, 68206, [(1, 49700)])
    valid = TURN[-1]
    assert "\n".join(TURN).count(valid) == 1
    out = agent._mem_stuck_rows(list(TURN), 2, valid, run_complete=False)
    assert out == ["\n".join(TURN) + "\n" + prompts.stuck_count_line(2, 68206, [(1, 49700)])]
    assert agent._mem_stuck_pending == 2
    # fired once the turn is kept: never again on this level
    agent._mem_stuck_fired.add(2)
    assert agent._mem_stuck_rows(list(TURN), 2, valid, run_complete=False) == TURN
    assert agent._mem_stuck_pending is None


def test_stuck_row_holds_below_threshold_after_run_complete_and_on_a_missing_anchor(monkeypatch):
    valid = TURN[-1]
    assert _stuck_agent(monkeypatch, 59000, []) ._mem_stuck_rows(list(TURN), 2, valid, run_complete=False) == TURN
    assert _stuck_agent(monkeypatch, 90000, [])._mem_stuck_rows(list(TURN), 2, valid, run_complete=True) == TURN
    agent = _stuck_agent(monkeypatch, 90000, [])
    assert agent._mem_stuck_rows(TURN[:-1], 2, valid, run_complete=False) == TURN[:-1]
    assert agent._mem_stuck_pending is None
    # the median of two cleared levels raises the bar
    assert _stuck_agent(monkeypatch, 90000, [(1, 50000), (2, 60000)])._mem_stuck_rows(
        list(TURN), 3, valid, run_complete=False) == TURN
