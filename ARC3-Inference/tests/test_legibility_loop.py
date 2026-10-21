"""ARC3_LEGIBILITY=1, and every legibility flag, through the real game loop under pf-keep-10x.

tests/anim_loop_sim.py (the solver session, ToolAgent.analyze(), the python
sandbox, the priority gate, a fake SGLang that renders the chat template)
with pf-keep-10x's env, a 30k window so history is trimmed, a model RESET,
memory tags, a level recap, and ANIM_SIM_LEGIBILITY=1: replies on level 2
that read a retained function's source, print a segmentation node, call the
runs helper, read the action numbering, ask for a not-kept animation, catch a
stale-state refusal and act again, click the border in a batch and diff it.
The same replies run with the umbrella off, explicitly off, on (the bundle)
and with every flag on ("all", the flag-only fixes included).

Checks:
- flags off (unset or every flag 0) the knob's run is identical either way;
- on, the game plays the same: the same engine actions, the same levels;
- on and all, the system prompt and tool schema are static for the run and
  are the flags-off ones with the enabled rows applied; the bundle leaves
  every flag-only text out;
- on and all, every request extends the previous one except where history
  was evicted or `<previous_levels>` was redrawn: no fix edits sent text;
- each fix's text reaches the model where its event happens (bundle fixes
  in the "on" run, flag-only fixes in the "all" run).
"""
from __future__ import annotations

import json
import re

import pytest

from inference.agent import legibility as lg
from inference.agent import prompts
from legibility_golden import loop_env
from test_anim_loop import check_static_system, comparable, run_sim, texts
from test_legibility import ALL_OFF, with_rows

KNOB = {"ANIM_SIM_LEGIBILITY": "1"}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("legibility-loop")
    return {
        "off": run_sim(tmp, "off", loop_env(**KNOB)),
        "explicit-off": run_sim(tmp, "explicit-off", loop_env(**KNOB, **ALL_OFF)),
        "on": run_sim(tmp, "on", loop_env(**KNOB, ARC3_LEGIBILITY="1")),
        "all": run_sim(tmp, "all", loop_env(**KNOB, ARC3_LEGIBILITY="1",
                                            **{name: "1" for name in lg.FLAG_ONLY})),
    }


def _head(request) -> str:
    messages = request["messages"]
    if len(messages) > 1 and messages[1]["role"] == "user":
        text = "".join(texts(messages[1]))
        if text.startswith("<previous_levels>"):
            return text
    return ""


def _tool_texts(run) -> list[str]:
    """Every tool result the model was sent, once each, in order."""
    seen: list[str] = []
    for request in run["requests"]:
        for message in request["messages"]:
            if message["role"] == "tool":
                text = "".join(texts(message))
                if text not in seen:
                    seen.append(text)
    return seen


def _user_texts(run) -> list[str]:
    seen: list[str] = []
    for request in run["requests"]:
        for message in request["messages"]:
            if message["role"] == "user":
                text = "\n".join(texts(message))
                if text not in seen:
                    seen.append(text)
    return seen


def _printed(run, key: str) -> list[str]:
    """The lines a probe printed, from the tool results' stdout."""
    out = []
    for text in _tool_texts(run):
        for match in re.finditer(r'"stdout": "((?:[^"\\]|\\.)*)"', text):
            for line in json.loads(f'"{match.group(1)}"').split("\n"):
                if line.startswith(key):
                    out.append(line)
    return out


def test_flags_off_explicitly_or_unset_send_the_same(runs):
    assert comparable(runs["explicit-off"]) == comparable(runs["off"])


@pytest.mark.parametrize("name", ["on", "all"])
def test_on_plays_the_same_game(runs, name):
    off, on = runs["off"], runs[name]
    for run in (off, on):
        assert run["replies_used"] >= run["script_len"]
        assert run["levels_completed"] == 1
    assert on["engine_actions"] == off["engine_actions"]
    assert on["actions"] == off["actions"]


@pytest.mark.parametrize("name,flags", [("on", lg.BUNDLE), ("all", lg.FLAGS)])
def test_on_system_prompt_and_tools_are_static_and_expected(runs, name, flags):
    off, on = runs["off"], runs[name]
    assert check_static_system(on) == [] and check_static_system(off) == []
    system_off = "".join(texts(off["requests"][0]["messages"][0]))
    system_on = "".join(texts(on["requests"][0]["messages"][0]))
    expected = system_off
    for flag in prompts.LEGIBILITY_SYSTEM_ROWS:
        if flag in flags:
            expected = with_rows(expected, flag)
    assert system_on == expected != system_off
    assert ("Letter case changes the colour" in system_on) == (name == "all")
    tool_off = off["requests"][0]["tools"][0]["function"]
    tool_on = on["requests"][0]["tools"][0]["function"]
    if name == "on":
        assert tool_on == tool_off
    else:
        assert tool_on == {**tool_off, "description": tool_off["description"].replace(
            *prompts.FRAME_RUNS_TOOL_ROW)}


def test_on_every_request_extends_except_at_trims_and_commits(runs):
    for name in ("off", "on", "all"):
        reqs = runs[name]["requests"]
        for i in range(1, len(reqs)):
            a, b = reqs[i - 1], reqs[i]
            if b["prompt"].startswith(a["prompt"]):
                continue
            assert b["evictions"] != a["evictions"] or _head(a) != _head(b), (
                f"{name} request {i}: a prefix break with no eviction and no head redraw")


def test_on_retained_source(runs):
    assert _printed(runs["off"], "SRC ") == ["SRC None"]
    assert _printed(runs["on"], "SRC ") == ["SRC def twice(x):"]  # the first printed line


def test_on_segmentation_runs_and_animation_probes(runs):
    assert _printed(runs["off"], "NODE ")[0].endswith("None")
    node = _printed(runs["on"], "NODE ")[0]
    assert node.startswith("NODE ['bbox', 'boundary',") and node.endswith("[5, 6, 5, 6]")
    assert _printed(runs["off"], "RUNS ") == _printed(runs["on"], "RUNS ") == ["RUNS None"]
    assert _printed(runs["all"], "RUNS ") == ["RUNS 8: W4-9"]
    assert _printed(runs["off"], "NOTKEPT ") == ["NOTKEPT True ['action_num', 'frames', 'timeline']"]
    assert _printed(runs["on"], "NOTKEPT ") == [
        "NOTKEPT False ['action', 'action_num', 'error', 'frames', 'timeline']"]


def test_on_stale_refusal_text_and_later_calls_refused(runs):
    for name, ending in (("off", "keep acting through no-ops.)"),
                         ("on", "itself was executed and counted as one action.)")):
        stale = _printed(runs[name], "STALE")
        assert len(stale) == 2 and all(line.endswith(ending) for line in stale), stale
        assert "STALE ran" not in stale
    # the SPACE executed and both later calls were refused: one action spent
    assert _printed(runs["on"], "STEP ") == _printed(runs["off"], "STEP ") == ["STEP 29"]


def test_all_border_click_and_diff_keys(runs):
    # the click's own tool result can be trimmed away before a request carries
    # it; the next opener's guard note and per-action trace always arrive
    clause = "1 cell changed in the 4-cell border, row 0, col 1"
    verdict = "1: MOUSE(row=0, col=1) -> NO-OP (executed, no change inside the board area"
    off = [t for t in [*_user_texts(runs["off"]), *_tool_texts(runs["off"])] if verdict in t]
    every = [t for t in [*_user_texts(runs["all"]), *_tool_texts(runs["all"])] if verdict in t]
    assert off and all(verdict + ")" in t and clause not in t for t in off)
    assert every and all(f"{verdict}; {clause})" in t for t in every)
    assert "same_colour_cells" not in _printed(runs["off"], "DIFF ")[0]
    assert "'same_colour_cells'" in _printed(runs["all"], "DIFF ")[0]
    # the no-op guard note in the next opener names the border cells itself
    assert any(f"executed but produced no change inside the board area; {clause}." in t
               for t in _user_texts(runs["all"]))


def test_all_action_numbering_holds_and_is_stated(runs):
    # current_frame.step counts the transitions, after a death, its automatic
    # RESET, a model RESET and a level change: action n is transitions[n-1]
    for name in ("off", "on", "all"):
        probes = _printed(runs[name], "IDX ")
        assert len(probes) == 2
        for line in probes:
            _, step, count, _ = line.split(" ", 3)
            assert step == count, line
    lines = 0
    for text in _user_texts(runs["all"]):
        match = re.search(r"(?:This was action (\d+)|These were actions (\d+)-(\d+)) of the game; "
                          r"(\d+) actions? ha(?:s|ve) been taken\.", text)
        if "Executed actions:" in text and "Executed actions: none." not in text:
            assert match, text[:300]
        if match:
            lines += 1
            taken = int(match.group(4))
            end = int(match.group(1) or match.group(3))
            assert end <= taken
            assert f"Current state: step {taken + 1}," in text
    assert lines >= 10


def _every_text(run):
    return [*_user_texts(run), *_tool_texts(run), "".join(texts(run["requests"][0]["messages"][0]))]


def test_on_turn_wording_and_reset_bar(runs):
    assert any("step/time budget" in t for t in _every_text(runs["off"]))
    for name in ("on", "all"):
        every = _every_text(runs[name])
        assert not any("step/time" in t or "a timer or remaining" in t for t in every)
        system = every[-1]
        assert system.count(prompts.RESET_BAR_FIRST_SENTENCE) == 1
        assert prompts.TURN_BASED_FACT in system
        # the death in the script: the game-over opener says the bar was restored
        assert any("restarted from its initial state, step bar included," in t
                   for t in _user_texts(runs[name]))


def test_all_digest_trailer(runs):
    digests = [t for t in _user_texts(runs["all"]) if "Animation digest" in t]
    assert digests and all(
        prompts.DIGEST_HEURISTIC_ROW[1] in t for t in digests
        if prompts.DIGEST_HEURISTIC_ROW[0].strip() in t)
    assert any(prompts.DIGEST_HEURISTIC_ROW[1] in t for t in digests)


def test_the_bundle_leaves_the_flag_only_texts_out(runs):
    every = _every_text(runs["on"])
    for text in ("of the game;", "same_colour_cells", "-cell border, row", "Positions are the (row, col)",
                 "Letter case changes the colour", ".runs(r0, r1, c0, c1)"):
        assert not any(text in t for t in every), text
