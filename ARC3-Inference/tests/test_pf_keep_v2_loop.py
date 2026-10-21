"""pf-keep-v2-all-10x (memory v2 and legibility together) through the real game loop.

tests/anim_loop_sim.py (the solver session, ToolAgent.analyze(), the python
sandbox, the priority gate, a fake SGLang that renders every request through
the Qwen3.8 chat template) with the arm's env from kaggle/arms.json and, for
comparison, pf-keep-10x's, both with the same replies: a model RESET, memory
tags in three replies, the asked recap of level 1, and the legibility probes
on level 2 (ANIM_SIM_LEGIBILITY=1). Three windows: 30k (history is trimmed
several times; the clear is staged on a trim request and folds at once), 45k
(the clear is staged well before the next trim, so its `<cleared_level>` block
is shown) and the arm's 98k (no trim). The stuck threshold is lowered to 200
tokens so the count line fires in the sim.

Checks: the same game is played; the system prompt and tool schema are static
and are pf-keep-10x's with both branches' rows; no request edits text already
sent except at a trim; each branch's texts reach the model where their event
happens, once.
"""
from __future__ import annotations

import re

import pytest

from inference.agent import legibility as lg
from inference.agent import prompts
from test_anim_loop import TIGHT, check_static_system, run_sim, texts
from test_legibility_loop import _printed, _tool_texts, _user_texts
from test_pf_keep_v2 import ARMS, _with_rows

ARM, BASE = "pf-keep-v2-all-10x", "pf-keep-10x"
SCRIPT = {"ANIM_SIM_RESET": "1", "ANIM_SIM_MEMORY_TAGS": "1", "ANIM_SIM_RECAP_ASKED": "1",
          "ANIM_SIM_LEGIBILITY": "1"}
WINDOWS = {"tight": TIGHT, "mid": {"LOCAL_ANALYZER_CONTEXT_WINDOW": "45000"}, "wide": {}}
STUCK = {"ARC3_MEM_STUCK_MIN_TOKENS": "200"}
COUNT_RE = re.compile(r"Harness count: you have written [\d.]+k tokens of reasoning and code on level (\d+)")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pf-keep-v2")
    out = {}
    for window, extra in WINDOWS.items():
        out[f"base-{window}"] = run_sim(tmp, f"base-{window}", {**ARMS[BASE]["env"], **SCRIPT, **extra})
        out[window] = run_sim(tmp, window, {**ARMS[ARM]["env"], **SCRIPT, **STUCK, **extra})
    return out


def _text(message) -> str:
    return "\n".join(texts(message))


def _head(request) -> str:
    messages = request["messages"]
    if len(messages) > 1 and messages[1]["role"] == "user":
        text = _text(messages[1])
        if text.startswith("<previous_levels>"):
            return text
    return ""


def _blocks(message, tag: str) -> list[str]:
    return re.findall(rf"<{tag}>\n(.*?)\n</{tag}>", _text(message), re.S)


def _breaks(run) -> list[int]:
    reqs = run["requests"]
    return [i for i in range(1, len(reqs)) if not reqs[i]["prompt"].startswith(reqs[i - 1]["prompt"])]


def _system(run) -> str:
    return "".join(texts(run["requests"][0]["messages"][0]))


@pytest.mark.parametrize("window", WINDOWS)
def test_the_same_game_is_played(runs, window):
    run, base = runs[window], runs[f"base-{window}"]
    for r in (run, base):
        assert r["replies_used"] >= r["script_len"] and r["levels_completed"] == 1
    assert run["engine_actions"] == base["engine_actions"]
    assert run["actions"] == base["actions"]


@pytest.mark.parametrize("window", WINDOWS)
def test_system_prompt_and_tools_are_static_and_both_branches_rows(runs, window):
    run, base = runs[window], runs[f"base-{window}"]
    assert check_static_system(run) == [] and check_static_system(base) == []
    expected = _system(base).replace(*prompts.MEMORY_LAYOUT_APPEND_COMMIT_ROWS[0])
    assert _system(run) == _with_rows(expected, lg.BUNDLE) != _system(base)
    assert run["requests"][0]["tools"] == base["requests"][0]["tools"]


@pytest.mark.parametrize("window", WINDOWS)
def test_sent_text_changes_only_at_trims(runs, window):
    run = runs[window]
    reqs = run["requests"]
    for i in _breaks(run):
        assert reqs[i]["evictions"] != reqs[i - 1]["evictions"], f"request {i}: a prefix break without a trim"
    if window == "wide":
        # no trim: one unbroken prefix for the whole game (pf-keep-10x breaks it at the commit)
        assert _breaks(run) == [] and all(not _head(r) for r in reqs)
        assert _breaks(runs["base-wide"])


def _first_entry(run) -> int:
    return next(i for i, r in enumerate(run["requests"])
                if _head(r) or any(_blocks(m, "cleared_level") for m in r["messages"]))


def test_cleared_level_is_shown_once_then_folded(runs):
    run = runs["mid"]
    reqs = run["requests"]
    staged = _first_entry(run)
    block, = _blocks(reqs[staged]["messages"][-1], "cleared_level")
    assert block.startswith("Level 1: cleared. ") and " actions spent in " in block  # ARC3_MEM_HEAD_WORDS
    folded = False
    for i, request in enumerate(reqs[staged:], staged):
        count = sum(len(_blocks(m, "cleared_level")) for m in request["messages"])
        if not _head(request):
            assert count == 1, f"request {i}: the entry is in context exactly once until the fold"
        else:
            folded = True
            head_entry = _head(request).split("\n", 1)[1].rsplit("\n", 1)[0]
            assert count == 0 and head_entry.startswith(block), i
    assert folded, "the 45k window trims after the clear"


def test_tight_window_folds_the_staged_clear_at_once(runs):
    reqs = runs["tight"]["requests"]
    assert not any(_blocks(m, "cleared_level") for r in reqs for m in r["messages"])
    heads = [h for h in map(_head, reqs) if h]
    assert heads and heads[-1].startswith("<previous_levels>\nLevel 1: cleared. ")


@pytest.mark.parametrize("window", WINDOWS)
def test_stuck_line_once_per_level_after_the_valid_actions_line(runs, window):
    reqs = runs[window]["requests"]
    carriers = [i for i, r in enumerate(reqs)
                if r["messages"][-1]["role"] == "user" and COUNT_RE.search(_text(r["messages"][-1]))]
    levels = sorted(int(m.group(1)) for i in carriers for m in COUNT_RE.finditer(_text(reqs[i]["messages"][-1])))
    assert levels == [1, 2], "one count line per level"
    for i in carriers:
        text = _text(reqs[i]["messages"][-1])
        valid = re.search(r"Valid actions right now: [^\n]*\.\n", text)
        assert valid and COUNT_RE.match(text, valid.end())


@pytest.mark.parametrize("window", WINDOWS)
def test_level_two_opener_asks_the_v2_recap_once(runs, window):
    (old, new), = prompts.recap_ask_v2_rows(1)
    openers = [t for t in _user_texts(runs[window]) if "You have completed the previous level" in t]
    assert openers
    for text in openers:
        assert text.count(new) == 1 and old not in text and text.count("Level 1 recap:") == 1


@pytest.mark.parametrize("window", WINDOWS)
def test_legibility_texts_reach_the_model(runs, window):
    run, base = runs[window], runs[f"base-{window}"]
    node = _printed(run, "NODE ")[0]
    assert node.startswith("NODE ['bbox', 'boundary',") and _printed(base, "NODE ")[0].endswith("None")
    assert _printed(run, "NOTKEPT ") == ["NOTKEPT False ['action', 'action_num', 'error', 'frames', 'timeline']"]
    assert _printed(run, "SRC ") == ["SRC def twice(x):"]
    stale = _printed(run, "STALE")
    assert len(stale) == 2 and all(line.endswith("itself was executed and counted as one action.)") for line in stale)
    every = [*_user_texts(run), *_tool_texts(run), _system(run)]
    assert not any("step/time" in t or "a timer or remaining" in t for t in every)
    assert any("restarted from its initial state, step bar included," in t for t in _user_texts(run))
    # the flag-only fixes stay out
    for text in ("of the game;", "same_colour_cells", "-cell border, row", "Letter case changes the colour"):
        assert not any(text in t for t in every), text


def test_memory_words_reach_the_model(runs):
    for window in ("tight", "mid"):
        heads = [h for h in map(_head, runs[window]["requests"]) if h]
        # ARC3_MEM_HEAD_WORDS and the readable route, with clicks named by colour
        assert heads[-1].startswith(
            "<previous_levels>\nLevel 1: cleared. 19 actions spent in 2 attempts, 1 death. "
            "Winning route without loops (11 actions): click (14,6) white, "), window
        # the asked recap, whole (it was not written in the v2 form, so it is "Your recap:")
        assert "\n  Your recap: the goal is to slide the block onto the exit; LEFT moves the block one cell " \
               "and reaching the exit completes the level.\n" in heads[-1], window
