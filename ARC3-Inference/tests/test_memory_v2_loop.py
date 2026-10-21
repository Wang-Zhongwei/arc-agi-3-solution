"""Memory v2 (arm/mem-v2) through the real game loop.

Each run is tests/anim_loop_sim.py (the real solver session, ToolAgent.analyze(),
the python sandbox, the priority gate; a fake SGLang that renders every request
through the Qwen3.8 chat template) under pf-keep-10x's env from
kaggle/arms.json, in a 30k window (history is trimmed several times) and in
the arm's own 98k window (no trim), with a model RESET, memory tags in three
replies and the asked recap of level 1 in the first reasoning on level 2.

Checks:
- flags off (unset, or every new flag set to 0) every request is byte-identical
  to arm/pf-mem @ 1882bcb3 (tests/fixtures/pf_keep_golden.json);
- every memory-v2 arm keeps the system prompt and the tool schema static;
- each flag's own placement and cache checks (below, per flag).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from memory_v2_golden import CONFIGS as GOLDEN_CONFIGS
from memory_v2_golden import fingerprint
from test_anim_loop import check_static_system, run_sim, texts

from inference.agent import prompts

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text())["arms"]
GOLDEN = json.loads((HERE / "fixtures" / "pf_keep_golden.json").read_text())

# every memory-v2 flag and knob, set to its off value
FLAGS_OFF = {
    "ARC3_MEM_APPEND_COMMIT": "0",
    "ARC3_MEM_APPEND_NO_HANDOVER": "0",
    "ARC3_MEM_TRIM_STRIP": "0",
    "ARC3_MEM_RECAP_CHARS": "0",
    "ARC3_MEM_COMPRESS_ROUTES_FIRST": "0",
    "ARC3_MEM_ROUTE_FORMAT": "",
    "ARC3_MEM_HEAD_WORDS": "0",
    "ARC3_MEM_RECAP_ASK_V2": "0",
    "ARC3_MEM_STUCK_FACT": "0",
    "ARC3_MEM_STUCK_MIN_TOKENS": "200",
    "ARC3_MEM_STUCK_MEDIAN_MULT": "1",
    "ARC3_MEM_STUCK_MEDIAN_FROM": "1",
    "ARC3_MEM_ATTEMPT_COUNTS": "0",
    "ARC3_MEM_SEEN_GAME_ACTION": "0",
    "ARC3_MEM_EFFECT_WORDS": "0",
    "ARC3_MEM_RECAP_FROZEN_FACT": "0",
}

TIGHT_ENV, WIDE_ENV = GOLDEN_CONFIGS["tight"], GOLDEN_CONFIGS["wide"]
BASE = "pf-keep-10x"
MEMV2_ARMS = ("pf-keep-append-10x", "pf-keep-recap-10x", "pf-keep-memv2-10x", "pf-keep-stuck-10x")
APPEND = {"ARC3_MEM_APPEND_COMMIT": "1"}
CONFIGS = {
    **{name: env for name, env in GOLDEN_CONFIGS.items()},
    **{f"{name}-flags-0": {**env, **FLAGS_OFF} for name, env in GOLDEN_CONFIGS.items()},
    "tight-append": {**TIGHT_ENV, **APPEND},
    # a 45k window: the clear is staged well before the next trim folds it
    "mid-append": {**TIGHT_ENV, **APPEND, "LOCAL_ANALYZER_CONTEXT_WINDOW": "45000"},
    "wide-append": {**WIDE_ENV, **APPEND},
    "wide-append-no-handover": {**WIDE_ENV, **APPEND, "ARC3_MEM_APPEND_NO_HANDOVER": "1"},
    "tight-strip": {**TIGHT_ENV, "ARC3_MEM_TRIM_STRIP": "1"},
    "tight-append-strip": {**TIGHT_ENV, **APPEND, "ARC3_MEM_TRIM_STRIP": "1"},
    "tight-stuck": {**TIGHT_ENV, "ARC3_MEM_STUCK_FACT": "1", "ARC3_MEM_STUCK_MIN_TOKENS": "200"},
    "wide-stuck": {**WIDE_ENV, "ARC3_MEM_STUCK_FACT": "1", "ARC3_MEM_STUCK_MIN_TOKENS": "200"},
    "wide-stuck-fail": {**WIDE_ENV, "ARC3_MEM_STUCK_FACT": "1", "ARC3_MEM_STUCK_MIN_TOKENS": "200",
                        "ANIM_SIM_FAIL_ON": "Harness count:", "ARC3_ANALYZER_RETRY_BACKOFF_SECONDS": "0"},
    # the arm's own threshold: never reached in the sim
    "wide-stuck-default": {**WIDE_ENV, "ARC3_MEM_STUCK_FACT": "1"},
    # each memory-v2 arm's own env (kaggle/arms.json), in the 30k window
    **{f"arm-{arm}": {**ARMS[arm]["env"], **{k: v for k, v in TIGHT_ENV.items() if k not in ARMS[BASE]["env"]}}
       for arm in MEMV2_ARMS},
    # the turn that first carries the block fails once and is retried
    "wide-append-fail": {**WIDE_ENV, **APPEND, "ANIM_SIM_FAIL_ON": "<cleared_level>",
                         "ARC3_ANALYZER_RETRY_BACKOFF_SECONDS": "0"},
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("memv2")
    return {name: run_sim(tmp, name, env) for name, env in CONFIGS.items()}


def _text(message) -> str:
    return "\n".join(texts(message))


# ---------------------------------------------------------------- flags off


@pytest.mark.parametrize("name", sorted(GOLDEN_CONFIGS))
def test_flags_off_match_the_base_branch(runs, name):
    want = GOLDEN["configs"][name]
    for run in (runs[name], runs[f"{name}-flags-0"]):
        got = fingerprint(run)
        assert got["system"] == want["system"] and got["tools"] == want["tools"]
        assert got["evictions"] == want["evictions"]
        mismatched = [i for i, (a, b) in enumerate(zip(got["requests"], want["requests"])) if a != b]
        assert not mismatched, f"first differing request: {mismatched[0]}"
        assert len(got["requests"]) == len(want["requests"])


def test_the_golden_exercises_a_clear_trims_and_the_head(runs):
    # a golden that never clears, trims or draws <previous_levels> would prove nothing
    tight = runs["tight"]
    assert tight["levels_completed"] == 1 and tight["requests"][-1]["evictions"] >= 2
    assert any(_text(r["messages"][1]).startswith("<previous_levels>") for r in tight["requests"])
    assert all(check_static_system(runs[name]) == [] for name in GOLDEN_CONFIGS)


# ---------------------------------------------------------------- helpers


def _head(request) -> str:
    messages = request["messages"]
    if len(messages) > 1 and messages[1]["role"] == "user":
        text = _text(messages[1])
        if text.startswith("<previous_levels>"):
            return text
    return ""


def _breaks(run) -> list[int]:
    reqs = run["requests"]
    return [i for i in range(1, len(reqs)) if not reqs[i]["prompt"].startswith(reqs[i - 1]["prompt"])]


def _blocks(message, tag: str) -> list[str]:
    return re.findall(rf"<{tag}>\n(.*?)\n</{tag}>", _text(message), re.S)


def _entry(head: str, level: int) -> str:
    """Level `level`'s lines in a `<previous_levels>` block."""
    body = head.split("\n", 1)[1].rsplit("\n", 1)[0]
    entries = re.split(r"\n(?=Level \d+: )", body)
    return next(e for e in entries if e.startswith(f"Level {level}: "))


# ---------------------------------------------------------------- ARC3_MEM_APPEND_COMMIT


def test_b_pays_a_break_at_the_commit(runs):
    # the cost the flag removes: without a trim, B still breaks the prefix once, at the commit
    run = runs["wide"]
    assert all(r["evictions"] == 0 for r in run["requests"])
    breaks = _breaks(run)
    assert breaks and all(_head(run["requests"][i]) != _head(run["requests"][i - 1]) for i in breaks)


@pytest.mark.parametrize("name", ["tight-append", "mid-append", "wide-append"])
def test_append_breaks_the_prefix_only_at_trims(runs, name):
    run = runs[name]
    assert run["levels_completed"] == 1 and run["replies_used"] >= run["script_len"]
    assert check_static_system(run) == []
    reqs = run["requests"]
    for i in _breaks(run):
        assert reqs[i]["evictions"] != reqs[i - 1]["evictions"], f"request {i}: break without a trim"
    if name == "wide-append":
        assert _breaks(run) == [] and all(not _head(r) for r in reqs)
    system = _text(reqs[0]["messages"][0])
    (old, new), = prompts.MEMORY_LAYOUT_APPEND_COMMIT_ROWS
    assert new.strip() in system and old.strip() not in system


def _first_entry(run) -> int:
    """The first request that shows level 1's entry, as a block or in the head."""
    return next(i for i, r in enumerate(run["requests"])
                if _head(r) or any(_blocks(m, "cleared_level") for m in r["messages"]))


def test_a_clear_staged_on_a_trim_request_folds_at_once(runs):
    # in the 30k window the turn that stages the clear also trims (as for 23 of
    # B's 110 commits): the block is folded before the request is sent
    run, base = runs["tight-append"], runs["tight"]
    reqs = run["requests"]
    first = _first_entry(run)
    assert first == _first_entry(base) and _head(reqs[first]) == _head(base["requests"][first])
    assert reqs[first]["evictions"] != reqs[first - 1]["evictions"]
    assert not any(_blocks(m, "cleared_level") for r in reqs for m in r["messages"])


@pytest.mark.parametrize("name", ["mid-append", "wide-append"])
def test_cleared_level_is_shown_once_with_the_entry_b_commits(runs, name):
    run, base = runs[name], runs["wide"]
    reqs = run["requests"]
    staged = _first_entry(run)
    block, = _blocks(reqs[staged]["messages"][-1], "cleared_level")
    b_head = next(h for h in map(_head, base["requests"]) if h)
    assert block == _entry(b_head, 1), "the entry text is B's, byte for byte"
    assert reqs[staged]["evictions"] == reqs[staged - 1]["evictions"]
    newest = _text(reqs[staged]["messages"][-1])
    if "<this_level>" in newest:
        assert newest.index("</cleared_level>") < newest.index("<this_level>")
        assert newest.rstrip().endswith("</this_level>")
    for i, request in enumerate(reqs):
        count = sum(len(_blocks(m, "cleared_level")) for m in request["messages"])
        if i < staged:
            assert count == 0
        elif request["evictions"] == reqs[staged]["evictions"] and not _head(request):
            assert count == 1, f"request {i}: the entry is in context exactly once until the fold"
        else:
            # folded at the next trim: in <previous_levels>, never again as a block
            assert count == 0 and _entry(_head(request), 1) == block, i
    if name == "mid-append":
        assert _head(reqs[-1]), "the 45k window trims after the clear"


def test_append_keeps_the_handover_unless_asked(runs):
    b, kept, dropped = runs["wide"], runs["wide-append"], runs["wide-append-no-handover"]
    assert len(b["handovers"]) >= 1
    assert len(kept["handovers"]) == len(b["handovers"])
    assert len(dropped["handovers"]) == len(b["handovers"]) - 1
    assert comparable_requests(dropped) == comparable_requests(kept)


def test_a_rolled_back_turn_resends_the_block(runs):
    run = runs["wide-append-fail"]
    assert len(run["failed"]) == 1 and run["levels_completed"] == 1
    failed = run["failed"][0]
    assert _blocks(failed["messages"][-1], "cleared_level")
    retry = run["requests"][failed["before_request"]]
    assert len(_blocks(retry["messages"][-1], "cleared_level")) == 1
    assert sum(len(_blocks(m, "cleared_level")) for m in retry["messages"]) == 1


def comparable_requests(run) -> list:
    return [{k: r[k] for k in ("messages", "tools", "prompt", "evictions")} for r in run["requests"]]


# ---------------------------------------------------------------- ARC3_MEM_TRIM_STRIP


def _this_levels(request) -> list[int]:
    return [int(b.split(" so far", 1)[0].split()[-1]) for m in request["messages"] if m["role"] == "user"
            for b in _blocks(m, "this_level")]


@pytest.mark.parametrize("name", ["tight-strip", "tight-append-strip"])
def test_trim_strip_acts_only_on_trim_requests(runs, name):
    run, base = runs[name], runs["tight" if name == "tight-strip" else "tight-append"]
    assert run["levels_completed"] == 1 and check_static_system(run) == []
    assert run["requests"][0]["messages"][0] == base["requests"][0]["messages"][0]
    reqs = run["requests"]
    for i in _breaks(run):
        assert reqs[i]["evictions"] != reqs[i - 1]["evictions"], f"request {i}: break without a trim"
    trims = [i for i in range(1, len(reqs)) if reqs[i]["evictions"] != reqs[i - 1]["evictions"]]
    assert trims
    for i in trims:
        levels = _this_levels(reqs[i])
        assert len(levels) == len(set(levels)), f"request {i}: one <this_level> per level after a trim"
        newest = reqs[i]["messages"][-1]
        if newest["role"] == "user" and "<this_level>" in _text(newest):
            assert _blocks(newest, "this_level"), "the current turn message keeps its block"
    # without the flag the same window keeps stale copies at its trims
    base_trims = [i for i in range(1, len(base["requests"]))
                  if base["requests"][i]["evictions"] != base["requests"][i - 1]["evictions"]]
    assert any(len(_this_levels(base["requests"][i])) > len(set(_this_levels(base["requests"][i])))
               for i in base_trims)
    # between trims nothing old is touched: each request extends the previous one
    for i in range(1, len(reqs)):
        if reqs[i]["evictions"] == reqs[i - 1]["evictions"]:
            assert reqs[i]["prompt"].startswith(reqs[i - 1]["prompt"]), i


# ---------------------------------------------------------------- ARC3_MEM_STUCK_FACT

COUNT_RE = re.compile(r"Harness count: you have written [\d.]+k tokens of reasoning and code on level (\d+)")


def _counts(message) -> list[int]:
    return [int(m.group(1)) for m in COUNT_RE.finditer(_text(message))]


@pytest.mark.parametrize("name", ["tight-stuck", "wide-stuck"])
def test_stuck_line_once_per_level_after_the_valid_actions_line(runs, name):
    run, base = runs[name], runs[name.split("-")[0]]
    assert run["levels_completed"] == 1 and check_static_system(run) == []
    reqs = run["requests"]
    carriers = [i for i, r in enumerate(reqs) if r["messages"][-1]["role"] == "user" and _counts(r["messages"][-1])]
    levels = [lv for i in carriers for lv in _counts(reqs[i]["messages"][-1])]
    assert sorted(levels) == [1, 2], "one count line per level"
    for i in carriers:
        text = _text(reqs[i]["messages"][-1])
        valid = re.search(r"Valid actions right now: [^\n]*\.\n", text)
        assert valid and COUNT_RE.match(text, valid.end()), "the line follows the valid-actions line"
        if "Only tool" in text:
            assert text.index("Harness count:") < text.index("Only tool"), "before the opener's static tail"
    # nothing else changes: the same breaks as without the flag, and every
    # other request extends the one before
    assert [r["evictions"] for r in reqs] == [r["evictions"] for r in base["requests"]]
    for i in range(1, len(reqs)):
        if reqs[i]["evictions"] == reqs[i - 1]["evictions"] and _head(reqs[i]) == _head(reqs[i - 1]):
            assert reqs[i]["prompt"].startswith(reqs[i - 1]["prompt"]), i
    # the line stays in history once sent (never re-sent, never edited)
    last = reqs[-1]["messages"]
    assert sum(len(_counts(m)) for m in last) <= 2


def test_stuck_line_is_resent_after_a_rolled_back_turn(runs):
    run = runs["wide-stuck-fail"]
    assert len(run["failed"]) == 1 and run["levels_completed"] == 1
    failed = run["failed"][0]
    level = _counts(failed["messages"][-1])
    assert level and _counts(run["requests"][failed["before_request"]]["messages"][-1]) == level


def test_stuck_line_default_threshold_is_not_reached_in_the_sim(runs):
    assert comparable_requests(runs["wide-stuck-default"]) == comparable_requests(runs["wide"])


# ---------------------------------------------------------------- the arms


@pytest.mark.parametrize("arm", MEMV2_ARMS)
def test_each_arm_plays_with_a_static_system_prompt(runs, arm):
    run = runs[f"arm-{arm}"]
    assert run["levels_completed"] == 1 and run["replies_used"] >= run["script_len"]
    assert check_static_system(run) == []
    reqs = run["requests"]
    for i in _breaks(run):
        assert reqs[i]["evictions"] != reqs[i - 1]["evictions"] or _head(reqs[i]) != _head(reqs[i - 1]), i
    if arm != "pf-keep-append-10x":
        assert reqs[0]["messages"][0] == runs["tight"]["requests"][0]["messages"][0]


def test_the_memv2_arm_shows_its_words(runs):
    run = runs["arm-pf-keep-memv2-10x"]
    heads = [h for h in map(_head, run["requests"]) if h]
    assert heads and heads[-1].startswith("<previous_levels>\nLevel 1: cleared. ")
    assert "19 actions spent in 2 attempts, 1 death. Winning route without loops (11 actions): " in heads[-1]
    assert re.search(r"click \(\d+,\d+\) [a-z]", heads[-1]), "a click named by its cell's colour"
    level2 = [_text(r["messages"][-1]) for r in run["requests"]
              if r["messages"][-1]["role"] == "user" and "You have completed the previous level" in _text(r["messages"][-1])]
    assert level2 and "`Level 1 recap: Goal: ... Rules: ...`" in level2[0]
    tails = [b for r in run["requests"] for m in r["messages"] if m["role"] == "user" for b in _blocks(m, "this_level")]
    assert any("(no colour count changed)" in b or "colour counts (cells may also" in b for b in tails)
    assert any(re.search(r"over 2 attempts, \d+ of them in this attempt \(attempt 2\)", b) for b in tails), \
        "the model's RESET and the death make later attempts"
