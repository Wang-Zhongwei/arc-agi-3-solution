"""Code that raised in porting-advaith v6 because it guessed a name or an access
style now reads.

The animation fields carry v7's names (`frame_count`, `frame_diffs`,
`frame_index`, `changed_count`); v6's (`frames`, `steps`, `step`, `changed`)
still read as hidden aliases. Every view and record reads both as keys and as
attributes.

v6 had 266 runtime errors in 2,855 python calls. Each test below replays one
of its wrong guesses verbatim (game in the comment).
"""

import json

from inference.agent import prompts
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


def test_timeline_and_summary_use_the_new_names():
    assert set(TIMELINE) == {"action", "frame_count", "frame_diffs"}
    assert [d["frame_index"] for d in TIMELINE["frame_diffs"]] == [0, 2, 3]
    assert set(TIMELINE["frame_diffs"][0]) == {"frame_index", "changed_count", "bbox", "changes"}
    assert set(summarize_animation(CHAIN)) == {"frame_count", "transient_pixels", "transient_bbox"}
    assert EMPTY_TIMELINE == {"action": None, "frame_count": 0, "frame_diffs": []}


def test_prompt_names_only_the_new_fields():
    text = prompts.ANIMATION_ADDENDUM + prompts.ANIMATION_ADDENDUM_TIMELINE
    for name in ("`frame_count`", "`frame_diffs`", "`frame_index`", "`changed_count`"):
        assert name in text
    for old in ("`frames`", "`steps`", "`step`", "`changed`"):
        assert old not in text


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


def test_new_names_read_as_keys_and_attributes():
    assert _run(
        "tl = last_animation_timeline\n"
        "print(tl['frame_count'], tl.frame_count, tl.get('frame_count'), tl.action)\n"
        "print([d.frame_index for d in tl.frame_diffs], [d['changed_count'] for d in tl['frame_diffs']])\n"
        "a = last_action_call_result['animation']\n"
        "print(a.frame_count, a['transient_pixels'], history[1].result.animation.frame_count)\n"
        "print(len(tl.frame_count), len(tl.frame_diffs[0].changed_count))\n"
    ) == ["4 4 4 SPACE", "[0, 2, 3] [1, 1, 1]", "4 1 4", "4 1"]


def test_old_names_read_as_hidden_aliases():
    # bp35/s5i5/tn36 x2 `len(tl['frames'])`; lf52/sc25 `tl['steps']`, `s['step']`
    assert _run(
        "tl = last_animation_timeline\n"
        "print(tl['frames'], len(tl['frames']), tl.frames, tl.get('frames'))\n"
        "print([s['step'] for s in tl['steps']], [s.changed for s in tl.steps], tl.get('steps') is tl.frame_diffs)\n"
        "a = last_action_call_result['animation']\n"
        "print(len(a['frames']), a.frames, history[1].result.animation.frames)\n"
        "print('frames' in tl, sorted(tl), tl.get('nope', '-'))\n"
    ) == ["4 4 4 4", "[0, 2, 3] [1, 1, 1] True", "4 4 4",
          "False ['action', 'frame_count', 'frame_diffs'] -"]


def test_printing_shows_only_the_new_names():
    lines = _run("print(last_animation_timeline)\nprint(last_action_call_result['animation'])")
    assert lines == [repr(TIMELINE), repr(summarize_animation(CHAIN))]


def test_empty_timeline_reads_under_either_name():
    # lf52 x2 / sc25 x2: `for s in tl['steps']` with no animation
    for timeline in (EMPTY_TIMELINE, {}):
        assert _run(
            "tl = last_animation_timeline\n"
            "print(tl['steps'], tl.frame_diffs, len(tl['frames']), tl.frame_count, tl['action'])\n",
            timeline=timeline,
        ) == ["[] [] 0 0 None"]


def test_lazy_timeline_has_items_and_values():
    # dc22/ka59/tn36: `for k, v in tl.items()`
    assert _run(
        "tl = last_animation_timeline\n"
        "print(sorted(k for k, _ in tl.items()), len(list(tl.values())))\n"
    ) == ["['action', 'frame_count', 'frame_diffs'] 3"]


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
        code="result = [last_action_call_result, history[0].result, last_animation_timeline['frame_diffs']]",
        timeout_seconds=20,
        initial_state={"current_frame": _frame(CHAIN[-1], 1),
                       "history": [{"action": "SPACE", "frame": _frame(CHAIN[-1], 1),
                                    "result": {"animation": {"frame_count": 4}}}],
                       "last_action_call_result": {"animation": {"frame_count": 4}}},
        action_handler=lambda actions, stale_after=None: {"action_result": {}},
        animation_handler=lambda request: TIMELINE,
        kept_functions=[],
    )
    assert json.loads(json.dumps(out["result"])) == [
        {"animation": {"frame_count": 4}}, {"animation": {"frame_count": 4}}, TIMELINE["frame_diffs"]
    ], out
