import pytest

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.grid_utils import format_grid_ascii
from inference.utils.retained_functions import prepare_functions


def state(step):
    def frame(index):
        grid = [[index]]
        return {"grid": grid, "ascii": format_grid_ascii(grid), "shape": [1, 1],
                "step": index, "level": 1}

    return {
        "current_frame": frame(step),
        "history": [{"action": f"ACTION{i}" if i else "", "frame": frame(i)}
                    for i in range(step + 1)],
        "valid_actions": [f"ACTION{step + 1}"],
        "last_action_call_result": {"step": step},
        "death_ledger": {"attempts": list(range(step))},
    }


def run(code, *, step=0, kept_functions=None):
    def action_handler(actions, **kwargs):
        nonlocal step
        step += 1
        return {"action_result": {"executed": len(actions)}, "state": state(step)}

    response = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=state(step),
        action_handler=action_handler, kept_functions=kept_functions,
    )
    assert response["error"] == "", response
    return response


def test_hud_default_reads_latest_frame_after_each_action():
    response = run("""
def hud(f=current_frame):
    return f.step, f.ascii
readings = [hud()]
for _ in range(2):
    action(['ACTION1'])
    readings.append(hud())
result = readings
""")
    assert response["result"] == [[i, format_grid_ascii([[i]])] for i in range(3)]


@pytest.mark.parametrize("name", [
    "current_frame", "latest_frame", "previous_frame", "last_action_frame",
    "history", "transitions", "last_transition", "last_action", "valid_actions",
    "last_action_call_result", "death_ledger",
])
def test_positional_and_keyword_defaults_follow_runtime_state(name):
    # previous_frame/last_transition/last_action begin as None, then become available.
    response = run(f"""
def captured(value={name}, /, *, other={name}):
    return value, other
for _ in range(2):
    action(['ACTION1'])
    value, other = captured()
    assert value is {name} and other is {name}
result = True
""")
    assert response["result"] is True


def test_literal_defaults_and_explicit_snapshots_stay_fixed():
    response = run("""
def live(f=previous_frame):
    return f.step
def historical(f=history[-2].frame):
    return f.step
def cropped(f=current_frame[:, :]):
    return f.step
def copied(actions=list(valid_actions)):
    return actions
def literals(nothing=None, action_name='ACTION1', *, value=None):
    return nothing, action_name, value
saved = previous_frame
action(['ACTION1'])
action(['ACTION1'])
result = [live(), historical(), cropped(), copied(), literals(), saved.step]
""", step=1)
    assert response["result"] == [2, 0, 1, ["ACTION2"], [None, "ACTION1", None], 0]


def test_runtime_defaults_survive_function_retention_and_refresh_again():
    first = run("""
def hud(f=current_frame, *, before=previous_frame, actions=valid_actions):
    return f.step, None if before is None else before.step, actions
result = hud()
""", kept_functions=[])
    assert first["retention_rejected"] == []
    assert first["result"] == [0, None, ["ACTION1"]]
    sources = [entry["source"] for entry in first["keepable_functions"]]
    assert len(sources) == 1

    second = run("""
readings = [hud()]
action(['ACTION3'])
readings.append(hud())
result = readings
""", step=2, kept_functions=sources)
    assert second["retention_rejected"] == []
    assert second["result"] == [[2, 1, ["ACTION3"]], [3, 2, ["ACTION4"]]]


@pytest.mark.parametrize("default", [
    "action(['ACTION1'])", "current_frame.step", "current_frame[:, :]",
    "snippet_value", "len(history)",
])
def test_retention_does_not_replay_default_expressions(default):
    kept, rejected = prepare_functions(
        [f"def hud(f={default}):\n    return f"],
        {"action", "current_frame", "history", "len", "snippet_value"},
    )
    assert kept == {}
    assert len(rejected) == 1
    assert "defaults" in rejected[0]["reason"]


def test_retention_requires_runtime_default_binding_to_be_available():
    kept, rejected = prepare_functions(["def hud(f=previous_frame):\n    return f"], set())
    assert kept == {}
    assert rejected[0]["missing_dependencies"] == ["previous_frame"]
