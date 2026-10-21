"""The ARC3_LEGIBILITY fixes (inference/agent/legibility.py), each flag on its own.

Flags off, the model is sent exactly what arm/pf-mem @ 1882bcb3 sent under the
pf-keep-10x env: the system prompt, three full turn openers, the sandbox
globals, and every request of the anim loop sim, tool outputs and tool schema
included (tests/fixtures/legibility_pf_keep_golden.json, see
tests/legibility_golden.py). That holds with the flags unset and with every
one set to 0.

Flag on, each must change only its own text: every system-prompt row's old
text occurs exactly once in the pf-keep-10x prompt, and the flag's prompt is
the flags-off prompt with its rows applied. The behaviour behind each text
(the bbox key, the runs helper, the falsy not-kept animation, ...) is tested
in-process here; tests/test_legibility_loop.py runs the umbrella through the
real game loop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inference.agent import legibility as lg  # noqa: E402
from inference.agent import prompts  # noqa: E402
from legibility_golden import FIXTURE, arm_env, collect, snapshot_hashes  # noqa: E402
from test_flags_off_identity import _snapshot  # noqa: E402

GOLDEN = json.loads(FIXTURE.read_text(encoding="utf-8"))
ALL_OFF = {name: "0" for name in (lg.UMBRELLA, *lg.FLAGS)}


@pytest.fixture(scope="module")
def shown():
    """flags_off_snapshot.py under pf-keep-10x: off, and each flag alone."""
    configs = {"off": {}, **{name: {name: "1"} for name in lg.FLAGS}}
    return {name: _snapshot(arm_env(**extra)) for name, extra in configs.items()}


# -- the switchboard ---------------------------------------------------------

def test_every_flag_is_off_by_default(monkeypatch):
    for name in (lg.UMBRELLA, *lg.FLAGS):
        monkeypatch.delenv(name, raising=False)
    assert lg.enabled_flags() == frozenset()


def test_umbrella_turns_on_the_bundle_only(monkeypatch):
    for name in lg.FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(lg.UMBRELLA, "1")
    assert lg.enabled_flags() == frozenset(lg.BUNDLE)
    assert lg.LEGEND_CASE_PAIRS not in lg.BUNDLE


def test_an_explicit_sub_flag_wins_over_the_umbrella(monkeypatch):
    for name in lg.FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(lg.UMBRELLA, "1")
    monkeypatch.setenv(lg.RETAINED_SOURCE, "0")
    monkeypatch.setenv(lg.DIFF_SAME_CELLS, "1")
    assert lg.enabled_flags() == frozenset(lg.BUNDLE) - {lg.RETAINED_SOURCE} | {lg.DIFF_SAME_CELLS}
    monkeypatch.setenv(lg.UMBRELLA, "0")
    monkeypatch.setenv(lg.SEG_BBOX, "1")
    assert lg.enabled_flags() == {lg.SEG_BBOX, lg.DIFF_SAME_CELLS}


def test_the_bundle_is_the_measured_harm_items():
    assert lg.BUNDLE == (lg.SEG_BBOX, lg.ANIM_SCHEMA, lg.RETAINED_SOURCE, lg.STALE_TRUTH,
                         lg.RESET_BAR_FACT, lg.TURN_BASED_FACT)
    assert set(lg.FLAG_ONLY) == {lg.DIFF_SAME_CELLS, lg.DIGEST_COORD_FACT, lg.EDGE_NOOP_FACT,
                                 lg.LEGEND_CASE_PAIRS, lg.ACTION_INDEX_FACT, lg.FRAME_RUNS}
    assert len(set(lg.FLAGS)) == len(lg.FLAGS) == 12


def test_apply_once_skips_a_row_that_is_missing_or_repeated(caplog):
    row = (("cat", "dog"),)
    assert prompts.apply_once("a cat sat", row) == "a dog sat"
    assert prompts.apply_once("a cow sat", row) == "a cow sat"
    assert prompts.apply_once("cat and cat", row) == "cat and cat"
    assert caplog.text.count("legibility row skipped") == 2
    # the first alternative that occurs once wins
    assert prompts.apply_once("one cow", (("cat", "dog"), ("cow", "ox"))) == "one ox"


def test_every_flag_has_a_place_in_the_row_order():
    assert set(prompts.LEGIBILITY_SYSTEM_ROWS) <= set(lg.FLAGS)


# -- flags off: byte-identical to arm/pf-mem @ 1882bcb3 ------------------------

def test_flags_unset_match_the_golden_snapshot(shown):
    assert snapshot_hashes(shown["off"]) == GOLDEN["snapshot"]


def test_flags_explicitly_off_match_the_golden(tmp_path):
    assert collect(tmp_path, **ALL_OFF) == {k: GOLDEN[k] for k in ("snapshot", "loop")}


def test_flags_unset_match_the_golden_loop(tmp_path):
    assert collect(tmp_path)["loop"] == GOLDEN["loop"]


# -- each flag alone changes only its own system-prompt text ------------------

def with_rows(prompt: str, name: str) -> str:
    for row in prompts.LEGIBILITY_SYSTEM_ROWS.get(name, ()):
        hits = [old for old, _ in row if prompt.count(old) == 1]
        assert hits, f"{name}: no alternative of {row[0][0][:60]!r} occurs exactly once"
        old, new = next(pair for pair in row if pair[0] == hits[0])
        prompt = prompt.replace(old, new)
    return prompt


@pytest.mark.parametrize("name", lg.FLAGS)
def test_each_flag_alone_changes_only_its_system_rows(shown, name):
    off, on = shown["off"]["system_prompt"], shown[name]["system_prompt"]
    assert on == with_rows(off, name)
    if prompts.LEGIBILITY_SYSTEM_ROWS.get(name):
        assert on != off  # guards against passing vacuously


# The openers each flag changes in flags_off_snapshot.py's three scenarios
# (checked word for word by that flag's own test); every other opener, and the
# sandbox globals, stay as flags off.
OPENER_CHANGES: dict[str, set[str]] = {
    lg.RESET_BAR_FACT: {"game_over"},
    lg.TURN_BASED_FACT: {"game_over"},
    lg.ACTION_INDEX_FACT: {"animated_noop", "game_over", "plain"},
}


@pytest.mark.parametrize("name", lg.FLAGS)
def test_each_flag_alone_leaves_other_openers_and_the_sandbox(shown, name):
    off, on = shown["off"], shown[name]
    assert on["sandbox"] == off["sandbox"]
    for opener in off["openers"]:
        if opener not in OPENER_CHANGES.get(name, set()):
            assert on["openers"][opener] == off["openers"][opener], opener


def test_all_flags_compose(monkeypatch):
    """All rows together equal the rows applied one flag at a time, in order."""
    from inference.agent import tool_agent as ta

    for key, value in arm_env().items():
        monkeypatch.setenv(key, value)
    for attr, key in (("_DIFF_IMAGE", "ARC3_DIFF_IMAGE"), ("_AUTO_FRAME_DIFF", "ARC3_AUTO_FRAME_DIFF")):
        monkeypatch.setattr(ta, attr, ta._get_env_bool(key, False))  # read at import time
    off = ta._build_system_prompt(tool_output_tokens=3072)
    for name in lg.FLAGS:
        monkeypatch.setenv(name, "1")
    on = ta._build_system_prompt(tool_output_tokens=3072)
    expected = off
    for name in prompts.LEGIBILITY_SYSTEM_ROWS:
        expected = with_rows(expected, name)
    assert on == expected


# -- shared helpers for the behaviour tests -----------------------------------

def _frame(grid, step=3, level=1):
    from inference.agent.runtime_state import Frame
    return Frame(grid=tuple(tuple(row) for row in grid), step=step, level=level)


def run_snippet(code: str, grid, *, previous=None, **flags) -> dict:
    """One python call in the real sandbox, with ``grid`` as current_frame."""
    from inference.agent import tool_agent as ta
    from inference.agent.python_tool_sandbox import run_sandboxed_python
    from inference.agent.runtime_state import HistoryEntry

    entries = []
    if previous is not None:
        entries.append(HistoryEntry(action="", frame=_frame(previous, step=0)))
    entries.append(HistoryEntry(action="UP" if previous is not None else "", frame=_frame(grid)))
    state = {
        "current_frame": ta._ascii_frame_view_payload(_frame(grid)),
        "history": ta._ascii_history_view_payload(entries),
        "valid_actions": ["UP"],
    }
    return run_sandboxed_python(code=code, timeout_seconds=20, initial_state=state,
                                action_handler=lambda actions, stale_after=None: {}, **flags)


def _grid(size=12, marks=None):
    rows = [[0] * size for _ in range(size)]
    for (r, c), value in (marks or {}).items():
        rows[r][c] = value
    return rows


# -- ARC3_SEG_BBOX -------------------------------------------------------------

def _shapes():
    """A filled rectangle, a one-cell-wide line, a single cell, an L and a ring."""
    marks = {}
    marks.update({(r, c): 9 for r in range(1, 4) for c in range(1, 5)})       # 3x4 rectangle
    marks.update({(6, c): 8 for c in range(1, 6)})                            # 1x5 line
    marks[(9, 9)] = 7                                                         # one cell
    marks.update({(r, 8): 6 for r in range(1, 5)} | {(4, c): 6 for c in range(8, 11)})  # an L
    return _grid(12, marks)


def _component(grid, seed):
    """The 4-connected same-value cells around ``seed``."""
    value, seen, stack = grid[seed[0]][seed[1]], {tuple(seed)}, [tuple(seed)]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (nr, nc) not in seen
                    and grid[nr][nc] == value):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def test_seg_bbox_is_the_inclusive_extent_and_the_last_key():
    import random

    from inference.utils.grid_utils import ARC_COLOR_CHARS
    from inference.utils.segmentation import segment_layer

    rng = random.Random(7)
    grids = [_shapes()] + [[[rng.choice((0, 0, 1, 2)) for _ in range(16)] for _ in range(16)]
                           for _ in range(20)]
    for grid in grids:
        plain = segment_layer(grid, ARC_COLOR_CHARS)
        boxed = segment_layer(grid, ARC_COLOR_CHARS, bbox=True)
        assert all("bbox" not in node for node in plain["nodes"])
        assert boxed["adjacency_list"] == plain["adjacency_list"]
        for a, b in zip(plain["nodes"], boxed["nodes"]):
            assert list(b) == [*a, "bbox"] and {k: b[k] for k in a} == a
            cells = _component(grid, a["boundary"][0])
            assert len(cells) == a["pixels"]
            assert b["bbox"] == [min(r for r, _ in cells), min(c for _, c in cells),
                                 max(r for r, _ in cells), max(c for _, c in cells)]
            points = a["boundary"]  # the same box as the corner points' min/max
            assert b["bbox"] == [min(r for r, _ in points), min(c for _, c in points),
                                 max(r for r, _ in points), max(c for _, c in points)]


def test_seg_bbox_prompt_counts_hold():
    """The corner counts the new `boundary` text states."""
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    from inference.utils.segmentation import segment_layer

    nodes = {node["color"]: node for node in segment_layer(_shapes(), ARC_COLOR_CHARS, bbox=True)["nodes"]}
    rect, line, cell = (nodes[ARC_COLOR_CHARS[v]] for v in (9, 8, 7))
    assert len(rect["boundary"]) == 4 and rect["bbox"] == [1, 1, 3, 4] and rect["pixels"] == 12
    assert len(line["boundary"]) == 2 and line["bbox"] == [6, 1, 6, 5]
    assert len(cell["boundary"]) == 1 and cell["bbox"] == [9, 9, 9, 9]


def test_seg_bbox_reaches_the_sandbox_only_when_on():
    code = "n = current_frame.segmentation['nodes'][1]\nprint(sorted(n)[:2], n.get('bbox'))"
    grid = _grid(12, {(2, 3): 9, (2, 4): 9, (3, 3): 9, (3, 4): 9})
    on = run_snippet(code, grid, seg_bbox=True)
    off = run_snippet(code, grid)
    assert on["stdout"].strip() == "['bbox', 'boundary'] [2, 3, 3, 4]", on
    assert off["stdout"].strip() == "['boundary', 'children'] None", off


def test_frame_diff_bbox_is_the_node_bbox():
    """The P:747 row says frame_diff's bbox is the same as a node's."""
    from inference.utils.frame_diff import compute_frame_diff
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    from inference.utils.segmentation import segment_layer

    before = _grid(12)
    after = _shapes()
    nodes = segment_layer(after, ARC_COLOR_CHARS, bbox=True)["nodes"]
    diff = compute_frame_diff(before, after, segment_layer(before, ARC_COLOR_CHARS)["nodes"], nodes)
    boxes = {tuple(node["bbox"]) for node in nodes}
    assert diff["appeared"] and all(tuple(entry["bbox"]) in boxes for entry in diff["appeared"])


# -- ARC3_STALE_TRUTH ----------------------------------------------------------

STALE_CATCH = ("(StaleStateActionError can be caught if a snippet has good reason to "
               "keep acting through no-ops.)")


def test_stale_detail_states_that_later_calls_are_refused(monkeypatch):
    from types import SimpleNamespace

    import arcengine
    from inference.framework import solver

    action = SimpleNamespace(id=arcengine.GameAction.ACTION1, data={})
    monkeypatch.delenv(lg.STALE_TRUTH, raising=False)
    monkeypatch.delenv(lg.UMBRELLA, raising=False)
    off = solver._stale_detail(action, "SPACE")
    assert off.endswith(STALE_CATCH) and off.count(STALE_CATCH) == 1
    monkeypatch.setenv(lg.UMBRELLA, "1")
    on = solver._stale_detail(action, "SPACE")
    assert on == off.replace(STALE_CATCH, prompts.stale_truth_line("SPACE"))
    assert "can be caught" not in on and "refused the same way" in on
    assert on.endswith("Your next python call can act again. SPACE itself was executed and "
                       "counted as one action.)")


def test_stale_refusal_text_holds_in_the_sandbox(tmp_path, monkeypatch):
    """What the new sentence states: catching does not lift the halt, later calls are
    refused and spend nothing, the next python call can act, the no-op was counted."""
    from test_animation_layers import _harness

    monkeypatch.setenv(lg.STALE_TRUTH, "1")
    session, agent = _harness(tmp_path, monkeypatch, stale_block=True)
    state = tmp_path / "state.json"
    code = (
        "action(['UP'])\n"                       # changes nothing: arms the latch
        "for name in (['UP'], ['RIGHT']):\n"
        "    try:\n"
        "        action(name)\n"
        "        print('ran', name)\n"
        "    except StaleStateActionError as e:\n"
        "        print('refused', name, str(e)[-60:])\n"
        "print('step', current_frame.step)\n"
    )
    first = json.loads(agent._run_python_tool(state, {"code": code}).content)
    out = first["stdout"]
    assert "ran" not in out and out.count("refused") == 2, out
    assert "UP itself was executed and counted as one action.)" in out
    assert "step 1" in out  # the no-op UP counted; the refused calls spent nothing
    second = json.loads(agent._run_python_tool(state, {"code": "r = action(['RIGHT'])\nprint('step', current_frame.step)"}).content)
    assert "step 2" in second["stdout"], second  # the next python call acts again


# -- ARC3_FRAME_RUNS -----------------------------------------------------------

def _runs_reference(ascii_text: str, r0: int, r1: int, c0: int, c1: int) -> str:
    rows = ascii_text.split("\n")
    out = []
    for r in range(r0, r1 + 1):
        parts, c = [], c0
        while c <= c1:
            start = c
            while c + 1 <= c1 and rows[r][c + 1] == rows[r][start]:
                c += 1
            parts.append(rows[r][start] + (str(start) if start == c else f"{start}-{c}"))
            c += 1
        out.append(f"{r}: " + " ".join(parts))
    return "\n".join(out)


def test_frame_runs_writes_each_run_with_inclusive_columns():
    from inference.utils.grid_utils import format_grid_ascii

    grid = _grid(12, {(2, 3): 9, (2, 4): 9, (2, 5): 12, (2, 7): 9, (3, 0): 5})
    previous = _grid(12, {(2, 3): 9})
    code = (
        "print(current_frame.runs(2, 3, 2, 8))\n"
        "print(previous_frame.runs(2, 2) == history[0].frame.runs(2, 2, 0, 11))\n"
        "print(current_frame.runs(10, 99, -5, 99).split(chr(10)))\n"
        "current_frame.runs(2, 2, 3, 5)\n"
    )
    out = run_snippet(code, grid, previous=previous, frame_runs=True)
    assert not out["error"], out
    lines = out["stdout"].split("\n")
    assert "\n".join(lines[:2]) == "2: W2 b3-4 O5 W6 b7 W8\n3: W2-8"
    assert "\n".join(lines[:2]) == _runs_reference(format_grid_ascii(grid), 2, 3, 2, 8)
    assert lines[2] == "True"
    assert lines[3] == "['10: W0-11', '11: W0-11']"  # clipped to the board
    assert lines[4] == "2: b3-4 O5"  # a snippet ending on runs(...) prints the lines


def test_frame_runs_absent_when_off():
    out = run_snippet("print(hasattr(current_frame, 'runs'))", _grid(12))
    assert out["stdout"].strip() == "False"


def test_frame_runs_tool_description(monkeypatch):
    from inference.agent import tool_agent as ta

    monkeypatch.delenv(lg.FRAME_RUNS, raising=False)
    monkeypatch.delenv(lg.UMBRELLA, raising=False)
    off = ta._python_tool_description()
    assert off.count(prompts.FRAME_RUNS_TOOL_ROW[0]) == 1
    monkeypatch.setenv(lg.FRAME_RUNS, "1")
    assert ta._python_tool_description() == off.replace(*prompts.FRAME_RUNS_TOOL_ROW)


# -- ARC3_ANIM_SCHEMA ----------------------------------------------------------

def _not_kept_payload(monkeypatch, on: bool) -> dict:
    from inference.agent import tool_agent as ta

    monkeypatch.delenv(lg.UMBRELLA, raising=False)
    monkeypatch.setenv(lg.ANIM_SCHEMA, "1" if on else "0")
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    agent._step_env_callback = lambda request: {"record": None, "ring": [182, 183, 208]}
    return agent._animation_ring_payload("for", {"action_num": 209})


def test_not_kept_animation_has_every_documented_key(monkeypatch):
    off = _not_kept_payload(monkeypatch, False)
    on = _not_kept_payload(monkeypatch, True)
    assert sorted(off) == ["action_num", "frames", "timeline"] and sorted(off["timeline"]) == ["error"]
    assert on == {
        "action_num": 209, "action": None, "frames": [],
        "timeline": {"action_num": 209, "action": None, "frames": 0, "steps": []},
        "error": off["timeline"]["error"],
    }
    assert "[182, 183, 208]" in on["error"]


PROBE = (
    "d = animation_for(209)\n"
    "print(bool(d), sorted(d), d.get('action'), len(d['frames']), bool(d['timeline']), "
    "d['timeline'].get('steps'))\n"
    "print('guard caught it' if not d or not d.get('timeline') else 'guard missed it')\n"
    "print(d['action'])\n"
)


def test_not_kept_animation_tests_false_in_the_sandbox(monkeypatch):
    payload = _not_kept_payload(monkeypatch, True)
    out = run_snippet(PROBE, _grid(12), animation_handler=lambda request: payload,
                      animation_ring=True, anim_schema=True)
    assert not out["error"], out
    assert out["stdout"].split("\n")[:3] == [
        "False ['action', 'action_num', 'error', 'frames', 'timeline'] None 0 False []",
        "guard caught it",
        "None",
    ]


def test_not_kept_animation_flags_off_is_the_old_truthy_dict(monkeypatch):
    payload = _not_kept_payload(monkeypatch, False)
    out = run_snippet(PROBE, _grid(12), animation_handler=lambda request: payload,
                      animation_ring=True)
    assert out["stdout"].startswith("True ['action_num', 'frames', 'timeline'] None 0 True None")
    assert "KeyError" in out["error"] and "'action'" in out["error"]


def test_timeline_changed_is_an_int():
    from inference.utils.animation import build_animation_view

    before = tuple(tuple(row) for row in _grid(8))
    flash = tuple(tuple(row) for row in _grid(8, {(2, 2): 9, (3, 3): 9}))
    view = build_animation_view({"chain": [before, flash, before], "action_display": "UP"})
    assert view["steps"] and all(type(step["changed"]) is int for step in view["steps"])


# -- ARC3_TURN_BASED_FACT ------------------------------------------------------

def test_turn_based_fact_extends_the_budget_truth_bullet(shown):
    on = shown[lg.TURN_BASED_FACT]["system_prompt"]
    bar = next(line for line in on.split("\n") if "is a step bar." in line)
    assert bar.endswith("not with wall-clock time." + prompts.TURN_BASED_FACT)
    assert "step/time" not in on and "remaining-action/time" not in on
    assert "a timer" not in on and "timer bar" not in on
    for text in ("including the step bar,", "(death, or the level's step budget ran out)",
                 "a bar at the grid border that shrinks as actions are taken",
                 "edge where a step bar usually sits", "only shrinks a step bar in those edge cells",
                 "only a progress or step bar moved"):
        assert on.count(text) == 1, text
    off = shown["off"]["system_prompt"]
    assert "step/time budget" in off and "a timer or remaining-steps bar" in off


def test_turn_based_fact_without_budget_truth_extends_franzens_bullet(monkeypatch):
    from inference.agent import tool_agent as ta

    for key, value in arm_env(ARC3_BUDGET_TRUTH="0").items():
        monkeypatch.setenv(key, value)
    for attr, key in (("_DIFF_IMAGE", "ARC3_DIFF_IMAGE"), ("_AUTO_FRAME_DIFF", "ARC3_AUTO_FRAME_DIFF")):
        monkeypatch.setattr(ta, attr, ta._get_env_bool(key, False))
    off = ta._build_system_prompt(tool_output_tokens=3072)
    monkeypatch.setenv(lg.TURN_BASED_FACT, "1")
    on = ta._build_system_prompt(tool_output_tokens=3072)
    assert on == with_rows(off, lg.TURN_BASED_FACT)
    assert "interacts with the puzzle mechanics." + prompts.TURN_BASED_FACT + "\n" in on


def test_turn_based_text_rows_each_match_their_source_once(monkeypatch):
    import inspect

    from inference.agent import tool_agent as ta
    from inference.framework import solver

    monkeypatch.delenv(lg.UMBRELLA, raising=False)
    monkeypatch.setenv(lg.TURN_BASED_FACT, "0")
    stop = ta._terminal_action_stop_detail("game_over")
    sources = {
        prompts.TURN_BASED_TEXT_ROWS[0][0]: stop,
        prompts.TURN_BASED_TEXT_ROWS[1][0]: inspect.getsource(ta.ToolAgent._build_user_prompt),
        prompts.TURN_BASED_TEXT_ROWS[2][0]: inspect.getsource(ta.ToolAgent._build_user_prompt),
        prompts.TURN_BASED_TEXT_ROWS[3][0]: inspect.getsource(solver._HarnessGameSession.step_env),
    }
    for old, source in sources.items():
        # the source spans string literals across lines; the old text never does
        assert source.count(old) == 1, old
    monkeypatch.setenv(lg.TURN_BASED_FACT, "1")
    assert ta._terminal_action_stop_detail("game_over") == stop.replace(
        "step/time budget", "step budget")
    assert solver._legible_text(stop) == ta._legible_text(stop) != stop


def test_turn_based_fact_rewords_the_game_over_opener(shown):
    off = shown["off"]["openers"]["game_over"]
    on = shown[lg.TURN_BASED_FACT]["openers"]["game_over"]
    old, new = prompts.TURN_BASED_TEXT_ROWS[1]
    texts_off = [p["text"] for p in off["content"] if p.get("type") == "text"]
    assert sum(t.count(old) for t in texts_off) == 1
    expected = {**off, "content": [
        {**p, "text": p["text"].replace(old, new)} if p.get("type") == "text" else p
        for p in off["content"]]}
    assert on == expected


# -- ARC3_DIGEST_COORD_FACT ----------------------------------------------------

def test_digest_heuristic_row_matches_the_digest_source_once():
    import inspect

    from inference.utils import animation

    source = inspect.getsource(animation.build_animation_digest)
    assert source.count(f'"{prompts.DIGEST_HEURISTIC_ROW[0]}"') == 1


def test_digest_sizes_are_rows_by_cols_and_positions_top_left():
    """What the new trailer sentence states about trail and tween numbers."""
    from inference.utils.animation import (
        _render_trail, classify_animation, describe_animation_type, overlay_trails)

    def board(top):
        # a 3-tall, 1-wide blue bar at (top, 5)
        return tuple(tuple(9 if (top <= r < top + 3 and c == 5) else 0 for c in range(16))
                     for r in range(16))

    chain = [board(2), board(4), board(6), board(8)]
    trail = overlay_trails(chain)[0]
    assert (trail["height"], trail["width"]) == (3, 1)
    assert _render_trail(trail, 8).startswith("- b 3x1 seen at (2,5)")
    kind = classify_animation(chain)
    assert kind["type"] == "tween" and kind["size"] == "3x1"
    assert describe_animation_type(kind).startswith("tween b 3x1 (2,5)>(8,5)")


# -- ARC3_DIFF_SAME_CELLS ------------------------------------------------------

def _diff(before, after, **kwargs):
    from inference.utils.frame_diff import compute_frame_diff
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    from inference.utils.segmentation import segment_layer

    return compute_frame_diff(before, after, segment_layer(before, ARC_COLOR_CHARS)["nodes"],
                              segment_layer(after, ARC_COLOR_CHARS)["nodes"], **kwargs)


def test_same_colour_cells_marks_a_merge_that_changed_no_cells():
    bars = {(2, c): 10 for c in range(1, 5)} | {(2, c): 10 for c in range(6, 11)}
    before, after = _grid(12, bars), _grid(12, bars | {(2, 5): 10})
    plain, marked = _diff(before, after), _diff(before, after, same_cells=True)
    # the left bar merged into the right one: "disappeared", yet no cell of it changed
    gone = marked["disappeared"]
    assert [entry["at"] for entry in gone] == [[2, 1]]
    assert gone[0]["same_colour_cells"] == gone[0]["pixels"] == 4
    def strip(diff):
        return {key: [{f: v for f, v in entry.items() if not f.startswith("same_")} for entry in value]
                if isinstance(value, list) else value for key, value in diff.items()}

    assert strip(marked) == plain  # nothing else differs


def test_same_colour_cells_is_zero_for_a_real_disappearance_and_candidates_count_the_hash():
    blocks = {(r, c): 9 for r in (4, 5) for c in (3, 4)}  # diagonal to the mover: touched
    before = _grid(12, blocks | {(r, c): 9 for r in (6, 7) for c in (5, 6)} | {(9, 9): 8})
    after = _grid(12, blocks | {(r, c): 9 for r in (6, 7) for c in (7, 8)})
    diff = _diff(before, after, same_cells=True)
    assert [(e["from"], e["to"], e["same_hash_candidates"]) for e in diff["moved"]] == [
        ([6, 5], [6, 7], 2)]
    assert [(e["color"], e["same_colour_cells"]) for e in diff["disappeared"]] == [("R", 0)]
    assert all("_seed" not in e for k in ("appeared", "disappeared") for e in diff[k])


def test_same_cells_reach_the_sandbox_frame_diff_only_when_on():
    before = _grid(12, {(2, c): 10 for c in range(1, 5)} | {(2, c): 10 for c in range(6, 11)})
    after = _grid(12, {(2, c): 10 for c in range(1, 11)})
    code = "d = frame_diff()\nprint([e.get('same_colour_cells') for e in d['disappeared']])"
    assert run_snippet(code, after, previous=before, diff_same_cells=True)["stdout"].strip() == "[4]"
    assert run_snippet(code, after, previous=before)["stdout"].strip() == "[None]"


# -- ARC3_EDGE_NOOP_FACT -------------------------------------------------------

def _border_script(action_name, data, board):
    """UP recolours two cells of the top border row for good (a bar tick).
    SPACE flashes a 2x2 block inside the board area and recolours (0, 9)."""
    from solver_fakes import frozen

    ticked = [list(line) for line in board]
    if action_name == "ACTION1":
        ticked[0][5] = ticked[0][6] = 12
        return [frozen(ticked)], {}
    if action_name == "ACTION5":
        ticked[0][9] = 12
        flash = [list(line) for line in board]
        for row in (7, 8):
            for col in (7, 8):
                flash[row][col] = 8
        return [frozen(flash), frozen(ticked)], {}
    return [board], {}


@pytest.mark.parametrize("on", [False, True])
def test_edge_noop_verdict_names_the_border_cells(tmp_path, monkeypatch, on):
    from inference.framework import solver as solver_module
    from test_animation_layers import _harness

    monkeypatch.setenv(lg.EDGE_NOOP_FACT, "1" if on else "0")
    monkeypatch.setattr(solver_module, "NEW_NOOP_WORDING", True)
    session, agent = _harness(tmp_path, monkeypatch, truth=True, script=_border_script)
    raw = session.step_env({"actions": [{"action": "UP"}, {"action": "UP"}, {"action": "SPACE"}]})
    clause = "2 cells changed in the 4-cell border, row 0, cols 5-6"
    assert raw["action_trace"][0] == (
        "1: UP -> NO-OP (executed, no change inside the board area"
        + (f"; {clause})" if on else ")"))
    # the second UP changed nothing anywhere: no clause either way
    assert raw["action_trace"][1] == "2: UP -> NO-OP (executed, no change inside the board area)"
    third = raw["action_trace"][2]
    assert third.startswith("3: SPACE -> ANIMATED (blink; board area unchanged; ")
    assert ("board area unchanged; 1 cell changed in the 4-cell border, row 0, col 9; "
            in third) == on
    compact = agent._compact_action_result(raw)
    assert "border_change" not in json.dumps(compact)
    assert all("border_change" not in entry.result for entry in session.history_entries)


def test_edge_noop_animated_message_counts_kept_border_cells(monkeypatch):
    from inference.agent import tool_agent as ta
    from inference.utils.animation import describe_animation

    summary = {"frames": 2, "transient_pixels": 4, "transient_bbox": [7, 7, 8, 8]}
    plain = describe_animation(summary, action_display="SPACE", gameplay_changed=False)
    kept = describe_animation(summary, action_display="SPACE", gameplay_changed=False,
                              border_cells=2)
    assert describe_animation(summary, action_display="SPACE", gameplay_changed=False,
                              border_cells=0) == plain
    assert kept == plain.replace(
        "around rows 7-8, cols 7-8. Whatever this action did",
        "around rows 7-8, cols 7-8, and 2 border cells changed and stayed changed. "
        "Whatever else this action did")
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    before = _grid(12)
    after = _grid(12, {(0, 5): 12, (0, 6): 12, (11, 0): 3})
    agent._last_animation_chain = [before, _grid(12, {(7, 7): 8}), after]
    monkeypatch.setenv("ARC3_NOOP_GUARD_BORDER", "4")
    assert agent._border_cells_kept() == 3
    agent._last_animation_chain = []
    assert agent._border_cells_kept() == 0


def test_edge_noop_leaves_the_memory_words(monkeypatch):
    """<this_level>'s effect words belong to memory-v2; this flag is tool text only."""
    from inference.agent import memory_layout as ml
    from inference.agent.memory_facts import ANIM, HUD

    monkeypatch.setenv(lg.EDGE_NOOP_FACT, "1")
    assert ml.counts_words([(ANIM, 1), (HUD, 2)]) == (
        "animated and changed nothing once, changed only the edge twice")


def test_edge_noop_guard_note_names_the_border_cells(tmp_path, monkeypatch):
    from inference.framework import solver as solver_module
    from test_animation_layers import _harness

    monkeypatch.setenv(lg.EDGE_NOOP_FACT, "1")
    monkeypatch.setattr(solver_module, "NEW_NOOP_WORDING", True)
    monkeypatch.setenv("ARC3_NEW_CHANGED_PROMPTS", "1")
    session, agent = _harness(tmp_path, monkeypatch, batch_block=True, script=_border_script)
    raw = session.step_env({"actions": [{"action": "UP"}, {"action": "RIGHT"}, {"action": "RIGHT"}]})
    assert raw["stop_reason"] == "no_op_action"
    compact = agent._compact_action_result(raw)
    clause = "2 cells changed in the 4-cell border, row 0, cols 5-6"
    assert compact["no_op_border"] == clause
    summary = agent._summarize_step_sequence([compact])
    assert summary["no_op_stops"][0]["border"] == clause
    from inference.agent.runtime_state import Frame
    prompt = agent._build_user_prompt(
        1, valid_actions=["UP", "RIGHT"], current_frame=Frame(grid=tuple(map(tuple, _grid(16))), step=1, level=1),
        history_entries=[], previous_step_summary=summary)
    assert (f"executed but produced no change inside the board area; {clause}. "
            "The remaining 2 action(s)") in prompt
    monkeypatch.setenv(lg.EDGE_NOOP_FACT, "0")
    raw_off = _harness(tmp_path / "off", monkeypatch, batch_block=True, script=_border_script)[0].step_env(
        {"actions": [{"action": "UP"}, {"action": "RIGHT"}, {"action": "RIGHT"}]})
    assert "no_op_border" not in raw_off


# -- ARC3_ACTION_INDEX_FACT ----------------------------------------------------

INDEX_LINES = {
    # flags_off_snapshot.py: action 7 animated; actions 10-11 died and the
    # automatic RESET was action 12; action 13 moved
    "animated_noop": "This was action 7 of the game; 7 actions have been taken.",
    "game_over": "These were actions 10-11 of the game; 12 actions have been taken.",
    "plain": "This was action 13 of the game; 13 actions have been taken.",
}


@pytest.mark.parametrize("opener", sorted(INDEX_LINES))
def test_action_index_line_follows_the_executed_actions(shown, opener):
    import re

    off = shown["off"]["openers"][opener]
    on = shown[lg.ACTION_INDEX_FACT]["openers"][opener]
    executed = re.compile(r"(Executed actions: [^\n]*\.\n)")
    parts = []
    for part in off["content"]:
        if part.get("type") == "text" and executed.search(part["text"]):
            assert len(executed.findall(part["text"])) == 1
            part = {**part, "text": executed.sub(
                lambda m: m.group(1) + INDEX_LINES[opener] + "\n", part["text"])}
        parts.append(part)
    assert on == {**off, "content": parts}
    text = "".join(p.get("text", "") for p in on["content"])
    # the next action's number is one past the actions taken
    taken = int(re.search(r"(\d+) actions? ha(?:ve|s) been taken", text).group(1))
    assert f"Current state: step {taken + 1}," in text


def test_action_index_line_wording():
    from inference.agent import tool_agent as ta

    assert ta._action_index_line({"start_action_num": 1, "end_action_num": 1}, 1) == (
        "This was action 1 of the game; 1 action has been taken.")
    assert ta._action_index_line({"start_action_num": 157, "end_action_num": 160}, 160) == (
        "These were actions 157-160 of the game; 160 actions have been taken.")
    assert ta._action_index_line({"start_action_num": None, "end_action_num": 3}, 3) == ""


def test_action_index_fact_without_the_ring_leaves_out_animation_for(monkeypatch):
    from inference.agent import tool_agent as ta

    for key, value in arm_env(ARC3_ACTION_INDEX_FACT="1").items():
        monkeypatch.setenv(key, value)
    assert "action n is `transitions[n-1]` and `animation_for(n)`;" in ta._build_system_prompt(
        tool_output_tokens=3072)
    monkeypatch.setenv("ARC3_ANIM_RING", "0")
    monkeypatch.setenv("ARC3_ANIM_DIGEST", "0")
    off = ta._build_system_prompt(tool_output_tokens=3072)
    assert "action n is `transitions[n-1]`; `current_frame.step`" in off
    assert "animation_for" not in off


def _numbering_script(action_name, data, board):
    """UP moves a dot one row; DOWN ends the attempt; LEFT completes the level;
    RESET restores the level's start (the dot at row 8, the level marker kept)."""
    from solver_fakes import frozen

    level = next((c for c in range(16) if board[0][c] == 3), 0)
    start = [[0] * 16 for _ in range(16)]
    start[0][level] = 3
    start[8][8] = 9
    if action_name == "RESET":
        return [frozen(start)], {}
    if action_name == "ACTION2":
        return [board], {"game_over": True}
    if action_name == "ACTION3":
        start[0][level], start[0][level + 1] = 0, 3
        return [frozen(start)], {"level_up": True}
    moved = [list(row) for row in board]
    row = next(r for r in range(16) if 9 in moved[r])
    moved[row][8], moved[row - 1][8] = 0, 9
    return [frozen(moved)], {}


def test_action_numbers_hold_across_resets_and_level_changes(tmp_path, monkeypatch):
    """What the fact states, on the real solver session and sandbox: after deaths,
    automatic and model RESETs and a level change, current_frame.step and every
    history frame's step count the actions taken, transitions[n-1] is action n,
    and the next opener's `Current state: step N` is one past them."""
    from solver_fakes import FakeGame, frozen, make_session

    from inference.agent import tool_agent as ta

    class Game(FakeGame):
        def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
            state = super().execute_action(action, generated_tokens, uncached_input_tokens)
            state.available_actions = [0, 1, 2, 3, 4, 5, 6]
            return state

    start = [[0] * 16 for _ in range(16)]
    start[0][0], start[8][8] = 3, 9
    game = Game(frozen(start), _numbering_script)
    game.current_state.available_actions = [0, 1, 2, 3, 4, 5, 6]
    session = make_session(tmp_path, game)
    session.seed_initial_history()
    session.write_runtime_state()
    played = []
    for batch in (["UP", "UP"], ["DOWN"], None, ["UP"], ["RESET"], ["UP", "LEFT"], ["UP"]):
        if batch is None:  # the play loop's automatic RESET after the death
            session._execute_auto_reset()
            played.append("RESET")
            continue
        raw = session.step_env({"actions": [{"action": name} for name in batch]})
        played += raw["executed_actions"]
    assert played == ["UP", "UP", "DOWN", "RESET", "UP", "RESET", "UP", "LEFT", "UP"]
    assert session.action_count == len(played) == 9

    monkeypatch.setenv(lg.ACTION_INDEX_FACT, "1")
    agent = ta.ToolAgent(model="test-model")
    agent._ensure_session(session.state_path)
    agent._step_env_callback = session.step_env
    code = ("print(current_frame.step, current_frame.level)\n"
            "print([(t.action, t.after_frame.step) for t in transitions])\n"
            "print([h.frame.step for h in history])")
    out = json.loads(agent._run_python_tool(session.state_path, {"code": code}).content)["stdout"]
    lines = out.strip().split("\n")
    assert lines[0] == "9 2"
    assert lines[1] == str([(name, n) for n, name in enumerate(played, 1)])
    assert lines[2] == str(list(range(0, 10)))

    frame = session.current_frame()
    summary = {"start_action_num": 8, "end_action_num": 9, "executed_count": 2,
               "executed_actions": ["LEFT", "UP"], "level": 2}
    prompt = agent._build_user_prompt(session.action_count, valid_actions=["UP"],
                                      current_frame=frame, history_entries=session.history_entries,
                                      previous_step_summary=summary)
    assert "These were actions 8-9 of the game; 9 actions have been taken." in prompt
    assert "Current state: step 10, level 2." in prompt


# -- ARC3_LEGEND_CASE_PAIRS ----------------------------------------------------

def test_case_pairs_match_the_legend():
    from inference.utils.grid_utils import ARC_COLOR_LEGEND

    legend = dict(item.split("=") for item in ARC_COLOR_LEGEND.split(", "))
    (old, new), = prompts.LEGIBILITY_SYSTEM_ROWS[lg.LEGEND_CASE_PAIRS][0]
    pairs = new[len(old) - 2:].split(": ", 1)[1].rstrip(".\n").split(", ")
    letters = set()
    for pair in pairs:
        for side in pair.split(" / "):
            letter, name = side.split(" ", 1)
            assert legend[letter] == name, side
            letters.add(letter)
    # every letter whose other case is also a colour, and no other
    assert letters == {c for c in legend if c.swapcase() in legend}


# -- ARC3_RETAINED_SOURCE ------------------------------------------------------

HELPER = "def double(x):\n    return 2 * x\n"


def test_retained_source_maps_each_retained_function_to_its_source():
    from inference.agent.python_tool_sandbox import run_sandboxed_python

    def run(**flags):
        return run_sandboxed_python(
            code="print(sorted(n for n in dir() if n == 'retained_source'))\n"
                 "print(retained_source if 'retained_source' in dir() else None, double(4))",
            timeout_seconds=20, initial_state={"current_frame": None, "history": [], "valid_actions": []},
            action_handler=lambda actions, stale_after=None: {}, **flags)

    on = run(kept_functions=[HELPER], retained_source=True)
    assert on["stdout"].split("\n")[:2] == ["['retained_source']", f"{{'double': {HELPER!r}}} 8"], on
    off = run(kept_functions=[HELPER])
    assert off["stdout"].split("\n")[:2] == ["[]", "None 8"], off
    # no retention, nothing to map: the global is absent even with the flag
    none = run(retained_source=True)
    assert none["stdout"].startswith("[]"), none


def test_retained_source_through_the_agent(tmp_path, monkeypatch):
    from test_animation_layers import _harness

    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", "1")
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS_SCOPE", "game")
    monkeypatch.setenv(lg.RETAINED_SOURCE, "1")
    _, agent = _harness(tmp_path, monkeypatch)
    state = tmp_path / "state.json"
    agent._run_python_tool(state, {"code": HELPER + "print(double(2))"})
    out = json.loads(agent._run_python_tool(state, {"code": "print(retained_source['double'])"}).content)
    assert out["stdout"].strip() == HELPER.strip(), out


def test_retained_source_bullet_needs_retention(monkeypatch):
    from inference.agent import tool_agent as ta

    for key, value in arm_env(ARC3_RETAINED_SOURCE="1").items():
        monkeypatch.setenv(key, value)
    assert "- `retained_source` maps the name" in ta._build_system_prompt(tool_output_tokens=3072)
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", "0")
    assert "retained_source" not in ta._build_system_prompt(tool_output_tokens=3072)


# -- ARC3_RESET_BAR_FACT -------------------------------------------------------

def test_reset_bar_fact_rewrites_the_reset_line_and_auto_reset(shown):
    off = shown["off"]["system_prompt"]
    on = shown[lg.RESET_BAR_FACT]["system_prompt"]
    reset = next(line for line in on.split("\n") if "`RESET`" in line and "When available" in line)
    assert reset.startswith("- " + prompts.RESET_BAR_FIRST_SENTENCE + " Use it to recover")
    assert ("RESET adds one action to the level's action count, which the score uses, as every "
            "action already spent does; it does not shorten the restored bar. RESET must be the "
            "first game action") in reset
    assert "usually restores" not in on and "usually restores" in off
    assert "auto-resets to its initial state, step bar included, and completed levels" in on


def test_reset_bar_fact_rewords_the_game_over_opener(shown):
    off = shown["off"]["openers"]["game_over"]
    on = shown[lg.RESET_BAR_FACT]["openers"]["game_over"]
    (old, new), = prompts.RESET_BAR_TEXT_ROWS
    expected = {**off, "content": [
        {**p, "text": p["text"].replace(old, new)} if p.get("type") == "text" else p
        for p in off["content"]]}
    assert on == expected != off


def test_reset_bar_and_turn_based_together(monkeypatch):
    """The RESET line is rewritten once, by the reset-bar row, and nothing is skipped."""
    import logging

    from inference.agent import tool_agent as ta

    for key, value in arm_env(ARC3_RESET_BAR_FACT="1", ARC3_TURN_BASED_FACT="1").items():
        monkeypatch.setenv(key, value)
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    prompts.log.addHandler(handler)
    try:
        on = ta._build_system_prompt(tool_output_tokens=3072)
    finally:
        prompts.log.removeHandler(handler)
    assert not [r for r in records if "skipped" in r.getMessage()]
    assert on.count(prompts.RESET_BAR_FIRST_SENTENCE) == 1
    assert "remaining-action" not in on and "step/time" not in on


def test_reset_restores_the_first_board_and_counts_one_action(tmp_path, monkeypatch):
    """The facts the new RESET sentences state, on the real solver session."""
    from solver_fakes import FakeGame, frozen, make_session

    start = _grid(16, {(0, c): 12 for c in range(16)} | {(8, 8): 9})

    def script(action_name, data, board):
        if action_name == "RESET":
            return [frozen(start)], {}
        moved = [list(row) for row in board]
        cells = [c for c in range(16) if moved[0][c] == 12]
        if cells:
            moved[0][cells[-1]] = 0  # one bar cell per action
        return [frozen(moved)], {}

    class Game(FakeGame):
        def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
            state = super().execute_action(action, generated_tokens, uncached_input_tokens)
            state.available_actions = [0, 1, 2, 3, 4, 5, 6]  # RESET too, as the engine lists it
            return state

    game = Game(frozen(start), script)
    game.current_state.available_actions = [0, 1, 2, 3, 4, 5, 6]
    session = make_session(tmp_path, game)
    session.seed_initial_history()
    for name in ("UP", "UP", "UP"):
        session.step_env({"actions": [{"action": name}]})
    assert session.history_entries[-1].frame.grid != frozen(start)
    raw = session.step_env({"actions": [{"action": "RESET"}]})
    assert session.history_entries[-1].frame.grid == frozen(start)  # bar included
    assert raw["action_num"] == 4 == session.action_count  # RESET counted as one action
