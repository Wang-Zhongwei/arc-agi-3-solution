"""The animation API names its counts as counts and lines its indexes up.

In porting-advaith v6 every int-as-list error in animation code was
`len(timeline['frames'])`, the empty timeline raised KeyError 'steps', and the
lazy timeline had no `.items()`. The prompt also said the frames belong to the
last action "that animated", while every executed action replaces them.
"""

from inference.agent import prompts
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.animation import build_animation_frames, build_animation_view, summarize_animation
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


def _run(code, timeline):
    summary = summarize_animation(CHAIN)
    history = [
        {"action": "", "frame": _frame(BEFORE)},
        {"action": "SPACE", "frame": _frame(CHAIN[-1]), "result": {"animation": summary}},
        {"action": "UP", "frame": _frame(CHAIN[-1]), "result": {"executed": True}},
    ]
    return run_sandboxed_python(
        code=code, timeout_seconds=20,
        initial_state={"current_frame": _frame(CHAIN[-1]), "history": history},
        action_handler=lambda actions, stale_after=None: {"action_result": {}},
        animation_handler=lambda request: [] if request.get("kind") == "frames" else timeline,
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


def test_prompt_describes_the_record_as_it_behaves():
    text = prompts.ANIMATION_ADDENDUM + prompts.ANIMATION_ADDENDUM_TIMELINE

    assert "that animated" not in text
    assert "Every executed action replaces them" in text
    assert "`history[i].animation`" in text
    for name in ("`frame_count`", "`frame_diffs`", "`frame_index`", "`changed_count`"):
        assert name in text
    for old in ("`frames`", "`steps`", "`step`", "`changed`"):
        assert old not in text
