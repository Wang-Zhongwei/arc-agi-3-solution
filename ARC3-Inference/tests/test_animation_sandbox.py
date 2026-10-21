"""animation_for / animations in the python sandbox, end to end through a fake game."""
from __future__ import annotations

import pytest

from inference.agent import tool_agent
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.framework import solver as solver_module

from solver_fakes import FakeGame, blank, flash_script, frozen, make_session


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_RING_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_DIGEST_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_RING_SIZE", 8)
    monkeypatch.setattr(solver_module, "BATCH_NOOP_BLOCK", False)
    monkeypatch.setattr(solver_module, "STALE_STATE_BLOCK", False)
    monkeypatch.setenv("ARC3_ANIMATION", "1")
    monkeypatch.setenv("ARC3_ANIM_DIGEST", "1")
    monkeypatch.setenv("ARC3_ANIM_RING", "8")
    board = blank()
    board[3][3] = 9
    game_session = make_session(tmp_path, FakeGame(frozen(board), flash_script))
    game_session.step_env({"actions": [
        {"action": "SPACE"}, {"action": "MOUSE", "row": 6, "col": 6}, {"action": "RIGHT"},
    ]})
    return game_session


@pytest.fixture
def agent(session):
    instance = tool_agent.ToolAgent(model="test-model")
    instance._step_env_callback = session.step_env
    return instance


def test_ring_payload_for_and_listing(agent, session):
    listing = agent._animation_ring_payload("ring", {})
    assert [entry["action_num"] for entry in listing] == [1, 2]
    assert set(listing[0]) == {"action_num", "action", "step_count", "transient_pixels", "level"}
    payload = agent._animation_ring_payload("for", {"action_num": 1})
    assert payload["action"] == "SPACE"
    assert len(payload["frames"]) == 2  # S1, S2; the board before is not a frame
    assert payload["timeline"]["steps"][0]["step"] == 1
    missing = agent._animation_ring_payload("for", {"action_num": 3})
    assert missing["frames"] == [] and "No animation is kept for action 3" in (
        missing["timeline"]["error"]
    )


def _handler_for(agent):
    def handler(request):
        kind = str(request.get("kind") or "")
        if kind in ("for", "ring"):
            return agent._animation_ring_payload(kind, request)
        return {} if kind == "timeline" else []
    return handler


_PROBE = """
names = sorted(n for n in dir() if not n.startswith('__'))
print('animation_for' in names, 'animations' in names)
print([a['action_num'] for a in animations], len(animations))
first = animation_for(1)
print(first['action'], len(first['frames']), first['frames'][0].shape, first['timeline']['frames'])
print(type(first['frames'][0]).__name__)
print(sorted(k for k, _ in last_animation_timeline.items()), list(last_animation_timeline.values()))
"""


def test_sandbox_exposes_ring_with_digest(agent):
    result = run_sandboxed_python(
        code=_PROBE,
        timeout_seconds=20,
        initial_state={"current_frame": None, "history": [], "valid_actions": []},
        action_handler=lambda actions, stale_after=None: {},
        animation_handler=_handler_for(agent),
        animation_ring=True,
    )
    assert not result.get("error"), result
    lines = result["stdout"].splitlines()
    assert lines[0] == "True True"
    assert lines[1] == "[1, 2] 2"
    assert lines[2] == "SPACE 2 (16, 16) 2"
    assert lines[3] == "FrameView"
    assert lines[4] == "[] []"


def test_sandbox_without_digest_has_no_ring_or_items(agent):
    code = (
        "names = sorted(n for n in dir() if not n.startswith('__'))\n"
        "print('animation_for' in names, 'animations' in names)\n"
        "print(hasattr(last_animation_timeline, 'items'), hasattr(last_animation_timeline, 'values'))\n"
    )
    result = run_sandboxed_python(
        code=code,
        timeout_seconds=20,
        initial_state={"current_frame": None, "history": [], "valid_actions": []},
        action_handler=lambda actions, stale_after=None: {},
        animation_handler=_handler_for(agent),
    )
    assert result["stdout"].splitlines() == ["False False", "False False"]


def test_compact_result_carries_batch_animations(agent, session):
    raw = session.step_env({"actions": [{"action": "SPACE"}, {"action": "UP"}]})
    compact = agent._compact_action_result(raw)
    assert [entry["action_display"] for entry in compact["animations"]] == ["SPACE"]
    assert "animation_records" not in compact


def test_compact_result_leaves_out_a_lone_last_animation(agent, session):
    raw = session.step_env({"actions": [{"action": "UP"}, {"action": "SPACE"}]})
    compact = agent._compact_action_result(raw)
    # `animation` already says it: the only animated action was the last one
    assert compact["animation"] and "animations" not in compact


def test_compact_result_caps_animations_after_the_stop_keys(agent, session, monkeypatch):
    # the flashing clicks pass the no-op stop; the inert UP after them does not
    monkeypatch.setattr(solver_module, "BATCH_NOOP_BLOCK", True)
    monkeypatch.setattr(solver_module, "ANIM_NOOP_EXEMPT", True)
    clicks = [{"action": "MOUSE", "row": 5, "col": col} for col in range(4, 15)]
    raw = session.step_env({"actions": [*clicks, {"action": "UP"}, {"action": "UP"}]})
    assert raw["stop_reason"] == "no_op_action" and len(raw["animations"]) == 11
    compact = agent._compact_action_result(raw)
    keys = list(compact)
    assert keys[-2:] == ["animations", "animations_omitted"]
    assert keys.index("stop_reason") < keys.index("animations")
    assert keys.index("time_remaining_seconds") < keys.index("animations")
    assert [entry["action_display"] for entry in compact["animations"]] == [
        f"MOUSE(row=5, col={col})" for col in range(7, 15)
    ]
    assert compact["animations_omitted"] == 3
