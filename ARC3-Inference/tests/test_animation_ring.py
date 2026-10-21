"""The solver side of the animation digest: ring, query and batch payload."""
from __future__ import annotations

import pytest

from inference.framework import solver as solver_module
from inference.utils.animation import ANIMATION_RING_MAX_FRAMES, decode_chain

from solver_fakes import FakeGame, blank, flash_script, frozen, make_session


@pytest.fixture
def digest_on(monkeypatch):
    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_RING_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_DIGEST_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_RING_SIZE", 3)
    monkeypatch.setattr(solver_module, "BATCH_NOOP_BLOCK", False)
    monkeypatch.setattr(solver_module, "STALE_STATE_BLOCK", False)


def _start():
    board = blank()
    board[3][3] = 9
    return frozen(board)


def test_ring_keeps_last_n_animated_actions(tmp_path, digest_on):
    session = make_session(tmp_path, FakeGame(_start(), flash_script))
    for col in (5, 6, 7, 8):
        session.step_env({"actions": [{"action": "MOUSE", "row": 5, "col": col}]})
    session.step_env({"actions": [{"action": "RIGHT"}]})  # does not animate
    ring = list(session.animation_ring)
    assert [record["action_num"] for record in ring] == [2, 3, 4]
    # with the ring on, a non-animated action leaves the record alone: it is
    # the newest ring entry, with its full chain
    assert session.animation_record["step"] == ring[-1]["action_num"] == 4
    record = ring[-1]
    assert record["action_display"] == "MOUSE(row=5, col=8)"
    assert record["summary"]["transient_pixels"] == 1
    assert record["gameplay_changed"] is False
    assert record["level"] == 1
    assert isinstance(record["frames"][0][0], str)  # hex rows
    # same mechanic at another cell: same signature
    assert len({item["signature"] for item in ring}) == 1


def test_query_by_action_num_costs_no_action(tmp_path, digest_on):
    session = make_session(tmp_path, FakeGame(_start(), flash_script))
    session.step_env({"actions": [{"action": "SPACE"}]})
    session.step_env({"actions": [{"action": "MOUSE", "row": 2, "col": 2}]})
    before = session.action_count
    reply = session.step_env({"query": "animation", "action_num": 1})
    assert session.action_count == before
    assert reply["executed"] is False
    record = reply["record"]
    assert record["action_display"] == "SPACE"
    assert len(record["chain"]) == 3 and record["step"] == 1
    # the plain query still answers with the last record, as before
    plain = session.step_env({"query": "animation"})
    assert plain["record"]["action_display"] == "MOUSE(row=2, col=2)"
    assert session.step_env({"query": "animation", "action_num": 99})["record"] is None
    listing = session.step_env({"query": "animation", "ring": True})["records"]
    assert listing == [
        {"action_num": 1, "action": "SPACE", "step_count": 2, "transient_pixels": 4, "level": 1},
        {"action_num": 2, "action": "MOUSE(row=2, col=2)", "step_count": 2,
         "transient_pixels": 1, "level": 1},
    ]
    assert session.action_count == before


def test_batch_payload_lists_every_animated_action(tmp_path, digest_on):
    session = make_session(tmp_path, FakeGame(_start(), flash_script))
    payload = session.step_env({"actions": [
        {"action": "SPACE"}, {"action": "RIGHT"},
        {"action": "MOUSE", "row": 1, "col": 1}, {"action": "RIGHT"},
    ]})
    assert payload["executed_count"] == 4
    assert [item["action_num"] for item in payload["animations"]] == [1, 3]
    assert payload["animations"][0] == {
        "action_num": 1,
        "action_display": "SPACE",
        "animation": {"frames": 2, "transient_pixels": 4, "transient_bbox": [7, 7, 8, 8]},
    }
    assert [record["action_num"] for record in payload["animation_records"]] == [1, 3]
    assert "animation_ring_record" not in payload
    assert "animation" not in payload  # the last action did not animate


def test_flags_off_payload_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_RING_ENABLED", False)
    monkeypatch.setattr(solver_module, "ANIM_DIGEST_ENABLED", False)
    session = make_session(tmp_path, FakeGame(_start(), flash_script))
    payload = session.step_env({"actions": [{"action": "SPACE"}, {"action": "SPACE"}]})
    for key in ("animations", "animation_records", "animation_ring_record"):
        assert key not in payload
    assert len(session.animation_ring) == 0
    # with the digest off, action_num is ignored exactly as before
    reply = session.step_env({"query": "animation", "action_num": 1})
    assert set(reply) == {"executed", "query", "record"}


def test_stored_frames_are_capped(tmp_path, digest_on):
    def long_script(action_name, data, board):
        frames = []
        for step in range(60):
            grid = [list(line) for line in board]
            grid[step % 16][(step * 7) % 16] = 8
            frames.append(frozen(grid))
        frames.append(board)
        return frames, {}

    session = make_session(tmp_path, FakeGame(_start(), long_script))
    session.step_env({"actions": [{"action": "SPACE"}]})
    record = session.animation_ring[-1]
    assert record["summary"]["frames"] == 61
    assert len(record["frames"]) == ANIMATION_RING_MAX_FRAMES
    full = session.animation_record["chain"]
    stored = decode_chain(record["frames"])
    assert stored[0] == full[0] and stored[-1] == full[-1]
    looked_up = session.step_env({"query": "animation", "action_num": 1})["record"]
    assert looked_up["subsampled_from"] == 61


@pytest.mark.parametrize("exempt", [False, True])
def test_noop_exempt_lets_a_hiding_animation_through(tmp_path, monkeypatch, exempt):
    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_RING_ENABLED", False)
    monkeypatch.setattr(solver_module, "ANIM_DIGEST_ENABLED", False)
    monkeypatch.setattr(solver_module, "BATCH_NOOP_BLOCK", True)
    monkeypatch.setattr(solver_module, "STALE_STATE_BLOCK", False)
    monkeypatch.setattr(solver_module, "NOOP_GUARD_BORDER_WIDTH", 4)
    monkeypatch.setattr(solver_module, "ANIM_NOOP_EXEMPT", exempt)
    session = make_session(tmp_path, FakeGame(_start(), flash_script))
    # SPACE flashes and reverts (board area unchanged); UP changes nothing
    payload = session.step_env({"actions": [
        {"action": "SPACE"}, {"action": "SPACE"}, {"action": "UP"}, {"action": "SPACE"},
    ]})
    if exempt:
        # both flashes run; the inert UP is still stopped
        assert payload["executed_actions"] == ["SPACE", "SPACE", "UP"]
        assert payload["no_op_action"] == "UP"
    else:
        assert payload["executed_actions"] == ["SPACE"]
        assert payload["no_op_action"] == "SPACE"
    assert payload["stop_reason"] == "no_op_action"
