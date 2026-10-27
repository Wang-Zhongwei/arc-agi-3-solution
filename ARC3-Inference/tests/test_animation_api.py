"""The animation API names its counts as counts and lines its indexes up.

In porting-advaith v6 every int-as-list error in animation code was
`len(timeline['frames'])`, the empty timeline raised KeyError 'steps', and the
lazy timeline had no `.items()`. The prompt also said the frames belong to the
last action "that animated", while every executed action replaces them.

In v7 `history[i].animation` was only a summary, so an earlier action's frames
were gone, and `transitions[i].animation` did not exist (3 AttributeErrors).
Each action now keeps its own frames and timeline.
"""

import json
import random

from inference.agent import prompts, tool_agent
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.framework import solver as solver_module
from inference.utils.animation import (
    AnimationRecords,
    build_animation_frames,
    build_animation_view,
    summarize_animation,
)
from inference.utils.grid_utils import format_grid_ascii

BEFORE = ((0, 0), (0, 0))
CHAIN = [BEFORE, ((5, 0), (0, 0)), ((5, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (0, 7))]
RECORD = {"chain": CHAIN, "action_display": "SPACE", "step": 3, "level": 1}


def _cells(a, b):
    return {(r, c) for r in range(len(a)) for c in range(len(a[0])) if a[r][c] != b[r][c]}


def test_summary_counts_frames_as_frame_count():
    summary = summarize_animation(CHAIN)

    assert summary["frame_count"] == len(CHAIN) - 1
    assert "frames" not in summary


def test_timeline_frame_index_points_into_last_animation_frames():
    timeline = build_animation_view(RECORD)
    exposed = [frame["grid"] for frame in build_animation_frames(RECORD)]

    assert set(timeline) == {"action", "frame_count", "frame_diffs"}
    assert timeline["frame_count"] == len(exposed)
    assert [diff["frame_index"] for diff in timeline["frame_diffs"]] == [0, 2, 3]
    for diff in timeline["frame_diffs"]:
        index = diff["frame_index"]
        before = BEFORE if index == 0 else exposed[index - 1]
        assert diff["changed_count"] == len(_cells(before, exposed[index]))


def test_timeline_without_animation_keeps_its_keys():
    empty = {"action": None, "frame_count": 0, "frame_diffs": []}

    assert build_animation_view(None) == empty
    assert build_animation_view({"chain": [BEFORE, BEFORE]}) == empty


def _frame(grid):
    return {"ascii": format_grid_ascii([list(r) for r in grid]), "step": 0, "level": 1,
            "shape": [2, 2], "grid": [list(r) for r in grid]}


def _run(code, timeline, animation_handler=None):
    summary = summarize_animation(CHAIN)
    history = [
        {"action": "", "frame": _frame(BEFORE)},
        {"action": "SPACE", "frame": _frame(CHAIN[-1]),
         "result": {"action_num": 1, "animation": summary}},
        {"action": "UP", "frame": _frame(CHAIN[-1]), "result": {"action_num": 2, "executed": True}},
    ]
    return run_sandboxed_python(
        code=code, timeout_seconds=20,
        initial_state={"current_frame": _frame(CHAIN[-1]), "history": history},
        action_handler=lambda actions, stale_after=None: {"action_result": {}},
        animation_handler=animation_handler
        or (lambda request: [] if request.get("kind") == "frames" else timeline),
        kept_functions=[],
    )


def test_history_entries_carry_their_own_animation_summary():
    out = _run("print([h.animation for h in history])", build_animation_view(RECORD))

    assert out["stdout"].strip() == (
        f"[None, {summarize_animation(CHAIN)!r}, None]"
    ), out


def test_lazy_timeline_reads_like_a_dict_and_is_empty_without_animation():
    out = _run(
        "print(sorted(k for k, _ in last_animation_timeline.items()))\n"
        "print(last_animation_timeline['frame_diffs'], len(last_animation_frames))",
        {},
    )

    assert out["stdout"].splitlines() == ["['action', 'frame_count', 'frame_diffs']", "[] 0"], out


def test_history_and_transitions_read_their_own_actions_frames_once():
    record = {**RECORD, "step": 1}
    expected = [frame["ascii"] for frame in build_animation_frames(record)]
    requests = []

    def handler(request):
        requests.append(dict(request))
        # only action 1 animated; UP, the last executed action, did not
        own = record if request.get("action_num") == 1 else None
        return build_animation_frames(own) if request.get("kind") == "frames" else build_animation_view(own)

    out = _run(
        "a = history[1].animation\n"
        f"print(a['frame_count'], [f.ascii for f in a.frames] == {expected!r}, len(a.frames))\n"
        "print([d['frame_index'] for d in a.timeline['frame_diffs']], a.timeline['action'])\n"
        "print(transitions[0].animation is a, history[2].animation, transitions[1].animation)\n"
        "print(len(last_animation_frames))",
        None, handler,
    )

    assert out["stdout"].splitlines() == ["4 True 4", "[0, 2, 3] SPACE", "True None None", "0"], out
    # read three times, fetched once per kind
    assert sorted(r["kind"] for r in requests if r.get("action_num") == 1) == ["frames", "timeline"]


def test_records_unpack_exactly_what_was_packed():
    rng = random.Random(0)
    board = tuple(tuple(rng.choice((0, 0, 0, 5, 11)) for _ in range(64)) for _ in range(64))
    flash = tuple(tuple(15 if 20 <= r < 30 else cell for cell in row) for r, row in enumerate(board))
    chain = [board, flash, board]
    over_255 = [((0, 300), (1, 2)), ((0, 1), (2, 3))]
    ragged = [((0, 1), (1, 2)), ((0, 1), (2,))]
    records = AnimationRecords()
    records.add(7, {"chain": chain, "action_display": "UP", "step": 7, "level": 2})
    records.add(8, {"chain": over_255})
    records.add(9, {"chain": ragged})

    assert records.get(7) == {"chain": chain, "action_display": "UP", "step": 7, "level": 2}
    assert records.get(8)["chain"] == over_255
    assert records.get(9)["chain"] == ragged
    assert records.get(10) is None


def test_step_env_answers_for_the_action_it_is_asked_about(monkeypatch):
    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
    session = object.__new__(solver_module._HarnessGameSession)
    session.animation_record = {**RECORD, "step": 3}
    session.animation_records = AnimationRecords()
    session.animation_records.add(1, {**RECORD, "step": 1})

    assert session.step_env({"query": "animation"})["record"]["step"] == 3
    assert session.step_env({"query": "animation", "action_num": 1})["record"] == {**RECORD, "step": 1}
    assert session.step_env({"query": "animation", "action_num": 2})["record"] is None


def test_agent_serves_an_earlier_actions_frames_from_its_record(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC3_ANIMATION", "1")
    path = tmp_path / "state.json"
    record = {**RECORD, "step": 1}
    entries = [
        HistoryEntry("", Frame(grid=BEFORE, step=0, level=1)),
        HistoryEntry("SPACE", Frame(grid=CHAIN[-1], step=1, level=1),
                     result={"action_num": 1, "animation": summarize_animation(CHAIN)}),
        HistoryEntry("UP", Frame(grid=CHAIN[-1], step=2, level=1), result={"action_num": 2}),
    ]
    write_runtime_state(path, current_frame=entries[-1].frame, history=entries)
    records = AnimationRecords()
    records.add(1, record)

    def step_env(arguments):
        # UP, the last executed action, did not animate
        own = records.get(arguments["action_num"]) if "action_num" in arguments else None
        return {"executed": False, "query": "animation", "record": own}

    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._step_env_callback = step_env
    code = (
        "a = history[1].animation\n"
        "result = [len(last_animation_frames), [f.ascii for f in a.frames],\n"
        "          [d['frame_index'] for d in a.timeline['frame_diffs']]]\n"
    )
    payload = json.loads(agent._run_python_tool(path, {"code": code}).content)

    assert payload["result"] == [0, [f["ascii"] for f in build_animation_frames(record)], [0, 2, 3]], payload


def test_prompt_describes_the_record_as_it_behaves():
    text = prompts.ANIMATION_ADDENDUM + prompts.ANIMATION_ADDENDUM_TIMELINE

    assert "that animated" not in text
    assert "Every executed action except RESET replaces them" in text
    for name in ("`history[i].animation`", "`.frames`", "`history[i].animation.timeline`",
                 "`transitions[i].animation`"):
        assert name in text
    for name in ("`frame_count`", "`frame_diffs`", "`frame_index`", "`changed_count`"):
        assert name in text
    for old in ("`frames`", "`steps`", "`step`", "`changed`"):
        assert old not in text


def test_animation_reads_as_keys_and_as_attributes():
    # animation-history-frames v1: 7 x `h.animation.frame_count`, 1 x `h.animation['timeline']`
    def handler(request):
        own = RECORD if request.get("action_num") in (1, None) else None
        return build_animation_frames(own) if request.get("kind") == "frames" else build_animation_view(own)

    out = _run(
        "a = history[1].animation\n"
        "print(a.frame_count, a['frame_count'], len(a['frames']), len(a.get('frames')))\n"
        "tl = a['timeline']\n"
        "print([d.frame_index for d in tl.frame_diffs], tl.action, a.get('timeline')['frame_count'])\n"
        "print([d.changed_count for d in last_animation_timeline.frame_diffs], last_animation_timeline.frame_count)\n"
        "try:\n"
        "    a.nope\n"
        "except AttributeError as e:\n"
        "    print('AttributeError', 'frame_count' in str(e))\n",
        None, handler,
    )

    timeline = build_animation_view(RECORD)
    changed = [d["changed_count"] for d in timeline["frame_diffs"]]
    assert out["stdout"].splitlines() == [
        "4 4 4 4",
        "[0, 2, 3] SPACE 4",
        f"{changed} 4",
        "AttributeError True",
    ], out


def test_views_read_fields_as_keys_and_step_from_their_frame():
    # porting-advaith v3-v7: 4-9 x `history[i].step` a run (the step is the frame's)
    out = _run(
        "h, t = history[1], transitions[0]\n"
        "print(h['action'], h.step == h.frame.step == h['step'], h.level, t['action'], t.step == t.after_frame.step)\n"
        "print(h.result.action_num, h['result']['animation'].frame_count, current_frame['shape'], current_frame.get('nope'))\n"
        "for probe in (lambda: current_frame[0], lambda: list(current_frame), lambda: h['nope'], lambda: h['_grid']):\n"
        "    try:\n"
        "        probe()\n"
        "    except Exception as e:\n"
        "        print(type(e).__name__)\n",
        build_animation_view(RECORD),
    )

    assert out["stdout"].splitlines() == [
        "SPACE True 1 SPACE True",
        "1 4 (2, 2) None",
        "TypeError",
        "TypeError",
        "KeyError",
        "KeyError",
    ], out
