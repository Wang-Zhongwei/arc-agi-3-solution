"""Code that raised in porting-advaith v6 because it guessed a name or an access
style now reads, while names, printed values and the prompt stay v6's.

v6 had 266 runtime errors in 2,855 python calls. Each test below replays one
of its wrong guesses verbatim (game in the comment).
"""

import json

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.animation import build_animation_view, summarize_animation
from inference.utils.grid_utils import format_grid_ascii

BEFORE = ((0, 0), (0, 0))
CHAIN = [BEFORE, ((5, 0), (0, 0)), ((5, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (0, 7))]
RECORD = {"chain": CHAIN, "action_display": "SPACE"}
TIMELINE = build_animation_view(RECORD)
EMPTY_TIMELINE = build_animation_view(None)


def _frame(grid, step):
    return {"ascii": format_grid_ascii([list(r) for r in grid]), "step": step, "level": 1,
            "shape": [2, 2], "grid": [list(r) for r in grid]}


def _run(code, timeline=TIMELINE, action_result=None):
    result = {"action_num": 1, "executed": True, "action_display": "SPACE",
              "animation": summarize_animation(CHAIN)}
    history = [
        {"action": "", "frame": _frame(BEFORE, 0)},
        {"action": "SPACE", "frame": _frame(CHAIN[-1], 1), "result": result},
    ]
    out = run_sandboxed_python(
        code=code, timeout_seconds=20,
        initial_state={"current_frame": _frame(CHAIN[-1], 1), "history": history,
                       "last_action_call_result": result},
        action_handler=lambda actions, stale_after=None: {
            "action_result": action_result or {"executed": True, "action_display": "RIGHT",
                                               "gameplay_changed": True}},
        animation_handler=lambda request: [] if request.get("kind") == "frames" else timeline,
        kept_functions=[],
    )
    assert not out.get("error"), out
    return out["stdout"].splitlines()


def test_history_and_transition_entries_carry_their_frames_step_and_level():
    # dc22/m0r0/re86/sc25/sk48/vc33 `h.step`; lf52/ls20/wa30 `t.step`
    assert _run(
        "h, t = history[1], transitions[0]\n"
        "print(h.step, h.level, t.step, t.level, history[0].step)\n"
        "print(h['action'], h['step'], t['before_frame'].step, h.get('nope', '-'))"
    ) == ["1 1 1 1 0", "SPACE 1 0 -"]


def test_views_still_refuse_integer_keys_and_iteration():
    lines = _run(
        "for probe in (lambda: current_frame[0], lambda: list(current_frame),\n"
        "              lambda: history[1]['nope'], lambda: current_frame['_grid']):\n"
        "    try:\n"
        "        probe()\n"
        "    except Exception as e:\n"
        "        print(type(e).__name__)\n"
    )
    assert lines == ["TypeError", "TypeError", "KeyError", "KeyError"]


def test_frame_count_answers_len_and_prints_as_an_int():
    # bp35/s5i5/tn36 x2: `len(tl['frames'])`
    assert _run(
        "tl = last_animation_timeline\n"
        "print(tl['action'], len(tl['frames']), tl['frames'], tl['frames'] + 1)\n"
        "a = last_action_call_result['animation']\n"
        "print(len(a['frames']), a.frames, history[1].result.animation.frames)\n"
        "result = {'n': tl['frames']}\n"
    ) == ["SPACE 4 4 5", "4 4 4"]


def test_empty_timeline_reads_its_keys_and_prints_unchanged():
    # lf52 x2 / sc25 x2: `for s in tl['steps']` with no animation
    lines = _run(
        "tl = last_animation_timeline\n"
        "print(tl['steps'], len(tl['frames']), tl['action'], tl.steps)\n"
        "print(tl)\n",
        timeline=EMPTY_TIMELINE,
    )
    assert lines == ["[] 0 None []", repr(EMPTY_TIMELINE)]


def test_lazy_timeline_has_items_values_and_attributes():
    # dc22/ka59/tn36: `for k, v in tl.items()`
    assert _run(
        "tl = last_animation_timeline\n"
        "print(sorted(k for k, _ in tl.items()), len(list(tl.values())))\n"
        "print([s.step for s in tl.steps], [s['changed'] for s in tl['steps']])\n"
    ) == ["['action', 'frames', 'steps'] 3", "[1, 3, 4] [1, 1, 1]"]


def test_action_result_reads_action_and_attributes():
    # m0r0: `r = action(['RIGHT']); r['action']`
    lines = _run(
        "print(last_action_call_result['action'], last_action_call_result.executed)\n"
        "print(history[1].result['action'], transitions[0].result.action_num)\n"
        "r = action(['RIGHT'])\n"
        "print(r['action'], r.action, r.gameplay_changed, 'action' in r)\n"
        "print(sorted(r))\n"
    )
    assert lines == ["SPACE True", "SPACE 1", "RIGHT RIGHT True False",
                     "['action_display', 'executed', 'gameplay_changed']"]


def test_records_serialize_as_plain_json():
    out = run_sandboxed_python(
        code="result = [last_action_call_result, history[0].result, last_animation_timeline['steps']]",
        timeout_seconds=20,
        initial_state={"current_frame": _frame(CHAIN[-1], 1),
                       "history": [{"action": "SPACE", "frame": _frame(CHAIN[-1], 1),
                                    "result": {"animation": {"frames": 4}}}],
                       "last_action_call_result": {"animation": {"frames": 4}}},
        action_handler=lambda actions, stale_after=None: {"action_result": {}},
        animation_handler=lambda request: TIMELINE,
        kept_functions=[],
    )
    assert json.loads(json.dumps(out["result"])) == [
        {"animation": {"frames": 4}}, {"animation": {"frames": 4}}, TIMELINE["steps"]
    ], out
