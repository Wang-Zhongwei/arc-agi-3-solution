"""Per-transition animation access: transitions[i].animation_frames / _timeline."""

import json

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.framework import solver
from inference.utils.animation import (
    build_animation_frames,
    build_animation_view,
    pack_chain,
    unpack_chain,
)


def grid(fill, size=3):
    return tuple(tuple(fill for _ in range(size)) for _ in range(size))


# step 1 animated 0 -> 5 -> 0 -> 1; step 2 did not animate; step 3 animated
# 1 -> 7 -> 2; step 4 is an automatic RESET after it.
CHAINS = {
    1: [grid(0), grid(5), grid(0), grid(1)],
    3: [grid(1), grid(7), grid(2)],
}


def frame_payload(fill, step):
    g = grid(fill)
    return {
        "ascii": "\n".join("".join(str(c) for c in row) for row in g),
        "step": step,
        "level": 1,
        "shape": [3, 3],
        "grid": [list(row) for row in g],
    }


def entry(action, fill, step, *, animated=False, name=None):
    result = {"action_num": step, "action_name": name or action}
    if animated:
        result["animation"] = {"frames": len(CHAINS[step]) - 1, "transient_pixels": 9}
    return {"action": action, "frame": frame_payload(fill, step), "result": result}


HISTORY = [
    {"action": "", "frame": frame_payload(0, 0), "result": {}},
    entry("UP", 1, 1, animated=True, name="ACTION1"),
    entry("DOWN", 1, 2, name="ACTION2"),
    entry("LEFT", 2, 3, animated=True, name="ACTION3"),
    entry("RESET", 0, 4),
]


def run(code, records, requests, history=HISTORY):
    def handle(request):
        requests.append(dict(request))
        step = request.get("step")
        record = records.get(step) if step is not None else None
        if request.get("kind") == "frames":
            return build_animation_frames(record)
        return build_animation_view(record)

    result = run_sandboxed_python(
        code=code,
        timeout_seconds=20,
        initial_state={
            "current_frame": history[-1]["frame"],
            "history": history,
            "valid_actions": ["UP"],
        },
        action_handler=lambda *a, **k: {},
        animation_handler=handle,
    )
    assert not result.get("error"), result
    return json.loads(result["stdout"].strip().splitlines()[-1])


def records_for(steps):
    return {
        step: {"action_display": "x", "chain": CHAINS[step], "summary": {}, "step": step, "level": 1}
        for step in steps
    }


def test_pack_chain_round_trips():
    chain = CHAINS[1]
    packed = pack_chain(chain)
    assert all(isinstance(row, bytes) for row in packed[0])
    assert unpack_chain(packed) == chain


def test_every_transition_reads_its_own_animation():
    requests = []
    out = run(
        "import json\n"
        "print(json.dumps({\n"
        "  'lens': [len(t.animation_frames) for t in transitions],\n"
        "  'first': [f._grid[0][0] for f in transitions[0].animation_frames],\n"
        "  'timeline_frames': transitions[0].animation_timeline['frames'],\n"
        "  'last': [f._grid[0][0] for f in last_animation_frames],\n"
        "  'last_tl': last_animation_timeline['frames'],\n"
        "}))\n",
        records_for([1, 3]),
        requests,
    )
    assert out["lens"] == [3, 0, 2, 0]
    assert out["first"] == [5, 0, 1]
    assert out["timeline_frames"] == 3
    # after RESET, last_* is the action before it, not the reset
    assert out["last"] == [7, 2]
    assert out["last_tl"] == 2
    # still actions and the RESET never cross the pipe; repeats are cached
    assert sorted({(r["kind"], r["step"]) for r in requests}) == [
        ("frames", 1), ("frames", 3), ("timeline", 1), ("timeline", 3),
    ]
    assert len(requests) == 4


def test_unkept_record_reads_empty_on_transition():
    # Without ARC3_ANIMATION_HISTORY only the latest record exists.
    requests = []
    out = run(
        "import json\n"
        "print(json.dumps({\n"
        "  'old_frames': len(transitions[0].animation_frames),\n"
        "  'old_tl': transitions[0].animation_timeline,\n"
        "  'last': len(last_animation_frames),\n"
        "}))\n",
        records_for([3]),
        requests,
    )
    assert out == {"old_frames": 0, "old_tl": {}, "last": 2}


def test_last_timeline_keeps_error_when_last_action_did_not_animate():
    out = run(
        "import json\n"
        "print(json.dumps({'tl': last_animation_timeline.get('error'),"
        " 'frames': len(last_animation_frames)}))\n",
        records_for([]),
        [],
        history=HISTORY[:3],  # ends on the still DOWN
    )
    assert out["frames"] == 0
    assert out["tl"].startswith("No animation is available")


def test_solver_query_returns_record_for_step(monkeypatch):
    monkeypatch.setattr(solver, "ANIMATION_ENABLED", True)
    session = object.__new__(solver._HarnessGameSession)
    session.animation_records = {
        1: {"chain": pack_chain(CHAINS[1]), "step": 1, "summary": {}},
    }
    hit = session.step_env({"query": "animation", "step": 1})
    assert hit["executed"] is False
    assert hit["record"]["chain"] == CHAINS[1]
    assert session.step_env({"query": "animation", "step": 2})["record"] is None
    assert session.step_env({"query": "animation", "step": None})["record"] is None
    assert session.step_env({"query": "animation", "step": "x"})["record"] is None
