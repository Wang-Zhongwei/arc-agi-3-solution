"""A stale-state refusal stops the snippet at the first refused action.

The latch clears only when an executed action changes the board, so after one
refusal every later action() in the snippet would be refused too. The refusal
is a BaseException: a probe loop's `except Exception:` cannot swallow it, the
model is told once, and the snippet's retained functions still come back.
"""

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.framework.solver import _stale_detail
from inference.utils.grid_utils import format_grid_ascii

GRID = [[0, 0], [0, 9]]
FRAME = {"ascii": format_grid_ascii(GRID), "step": 0, "level": 1, "shape": [2, 2], "grid": GRID}

# The probe sweep models write: the first click is a no-op, the rest would be refused.
SWEEP = """
def probe_cells():
    return [(0, 0), (0, 1), (1, 0), (1, 1)]

for (r, c) in probe_cells():
    try:
        res = action([{'action': 'MOUSE', 'row': r, 'col': c}])
        print((r, c), 'noop', res.get('no_op'))
    except Exception as e:
        print((r, c), 'caught', type(e).__name__)
print('after the loop')
"""


def _run(code):
    calls = []

    def handler(actions, stale_after=None):
        calls.append((actions, stale_after))
        if stale_after:
            return {"action_result": {
                "executed": False, "stop_reason": "stale_state",
                "stop_detail": "MOUSE was not executed (stale state).",
            }}
        return {"action_result": {
            "executed": True, "no_op": True, "gameplay_changed": False,
            "executed_actions": ["MOUSE(row=0, col=0)"],
        }}

    out = run_sandboxed_python(
        code=code, timeout_seconds=20,
        initial_state={"current_frame": FRAME, "history": [{"action": "", "frame": FRAME}]},
        action_handler=handler, kept_functions=[],
    )
    return out, calls


def test_caught_probe_loop_stops_at_the_first_refusal():
    out, calls = _run(SWEEP)

    # one executed no-op, one refused call; the loop's other probes never run
    assert len(calls) == 2
    assert calls[0][1] is None and calls[1][1]
    assert out["error"].count("StaleStateActionError") == 1, out["error"]
    assert "(0, 0) noop True" in out["stdout"]
    assert "caught" not in out["stdout"] and "after the loop" not in out["stdout"]
    # the refusal reaches the model, and the sandbox did not crash
    assert "stale state" in out["error"]
    assert "exited unexpectedly" not in out["error"]


def test_functions_defined_before_the_refusal_are_retained():
    out, _ = _run(SWEEP)

    assert [f["name"] for f in out["keepable_functions"]] == ["probe_cells"]


def test_refusal_text_no_longer_invites_catching_it():
    class _Id:
        name = "ACTION1"

    class _Action:
        id = _Id()
        data = {}

    detail = _stale_detail(_Action(), "UP")
    assert "can be caught" not in detail
    assert "good reason" not in detail
    assert detail.endswith("then act on what you find.")
