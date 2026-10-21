"""The opener's Animation digest, driven through a real python tool call."""
from __future__ import annotations

import pytest

from inference.agent import tool_agent
from inference.agent.runtime_state import load_runtime_state
from inference.framework import solver as solver_module

from solver_fakes import FakeGame, blank, flash_script, frozen, make_session

DIGEST_ENV = {
    "ARC3_ANIMATION": "1",
    "ARC3_ANIM_DIGEST": "1",
    "ARC3_ANIM_RING": "8",
}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_RING_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_DIGEST_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_RING_SIZE", 8)
    monkeypatch.setattr(solver_module, "BATCH_NOOP_BLOCK", False)
    monkeypatch.setattr(solver_module, "STALE_STATE_BLOCK", False)
    for key in ("ARC3_ANIM_DIGEST_GATE", "ARC3_ANIM_DIGEST_CHARS", "ARC3_ANIM_DIGEST_GAME_CHARS",
                "ARC3_ANIM_KEYFRAMES", "MULTIMODAL_CONTEXT", "ARC3_ANIMATION_TIMELINE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in DIGEST_ENV.items():
        monkeypatch.setenv(key, value)
    board = blank()
    board[3][3] = 9
    session = make_session(tmp_path, FakeGame(frozen(board), flash_script))
    session.seed_initial_history()
    session.write_runtime_state()
    agent = tool_agent.ToolAgent(model="test-model")
    agent._ensure_session(session.state_path)
    agent._step_env_callback = session.step_env
    agent._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]
    return session, agent


def _turn(harness, code: str) -> str:
    """Run one python call, then build the next turn's opener."""
    session, agent = harness
    dispatch = agent._run_python_tool(session.state_path, {"code": code})
    assert dispatch.step_executed, dispatch.content
    current_frame, history = load_runtime_state(session.state_path)
    return agent._build_user_prompt(
        session.action_count,
        valid_actions=["UP", "SPACE", "MOUSE"],
        current_frame=current_frame,
        history_entries=history,
        previous_step_summary=agent._last_step_summary,
    )


def _digest(prompt: str) -> list[str]:
    lines = prompt.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Animation digest")), None)
    if start is None:
        return []
    end = start
    while end + 1 < len(lines) and not lines[end + 1].startswith(("You are still", "You have")):
        end += 1
    return lines[start:end + 1]


def test_digest_follows_the_animation_line(harness):
    prompt = _turn(harness, "action([{'action': 'MOUSE', 'row': 6, 'col': 6}])")
    lines = prompt.splitlines()
    animation_line = next(i for i, line in enumerate(lines) if "animated over" in line)
    assert lines[animation_line + 1].startswith(
        "Animation digest (novel): action 1 `MOUSE(row=6, col=6)`: blink x1; 2 states "
        "(S0 = board before; S2 = current board). Board area unchanged. "
        "1 cell changed and changed back."
    )
    assert "- R 1x1 seen at (6,6); step S1" in lines
    assert "last_animation_timeline" in _digest(prompt)[-1]


def test_batch_digest_names_each_action_and_repeats_shrink(harness):
    prompt = _turn(harness, "action(['SPACE', {'action': 'MOUSE', 'row': 6, 'col': 6}, 'RIGHT'])")
    digest = _digest(prompt)
    assert digest[0] == "Animation digest (novel), 2 of 2 animated actions in the sequence:"
    assert any(line.startswith("action 1 `SPACE`: blink x1; 2 states") for line in digest)
    assert any(line.startswith("action 2 `MOUSE(row=6, col=6)`: blink x1; 2 states") for line in digest)
    # the same mechanics again, at another cell: one pointer line each
    prompt = _turn(harness, "action([{'action': 'MOUSE', 'row': 7, 'col': 9}])")
    assert _digest(prompt) == [
        "Animation digest: action 4 `MOUSE(row=7, col=9)` repeats the animation of action 2 "
        "(same signature); `animation_for(4)`."
    ]


def test_seen_signatures_reset_on_level_change(harness):
    _turn(harness, "action(['SPACE'])")
    assert _digest(_turn(harness, "action(['SPACE'])"))[0].startswith(
        "Animation digest: action 2 `SPACE` repeats"
    )
    _turn(harness, "action(['LEFT'])")  # level 2, no animation
    prompt = _turn(harness, "action(['SPACE'])")
    assert _digest(prompt)[0].startswith("Animation digest (novel): action 4 `SPACE`")


def test_level_completing_batch_does_not_file_old_level_signatures(harness):
    _, agent = harness
    # LEFT completes the level: SPACE's animation belongs to level 1
    _turn(harness, "action(['SPACE', 'LEFT'])")
    assert agent._anim_seen_level == 2
    assert agent._anim_seen_sigs == {}
    prompt = _turn(harness, "action(['SPACE'])")
    assert _digest(prompt)[0].startswith("Animation digest (novel): action 3 `SPACE`")


def test_animated_level_completing_action_is_filed_under_its_own_level(harness):
    session, agent = harness
    completed = []

    def script(action_name, data, board):
        if action_name == "ACTION3":
            # LEFT flashes like SPACE; only the first one completes the level
            frames, _ = flash_script("ACTION5", data, board)
            flags = {} if completed else {"level_up": True}
            completed.append(True)
            return frames, flags
        return flash_script(action_name, data, board)

    session.game._script = script
    _turn(harness, "action(['LEFT'])")  # ran on level 1, completed it
    listing = session.step_env({"query": "animation", "ring": True})["records"]
    assert [(entry["action_num"], entry["level"]) for entry in listing] == [(1, 1)]
    assert session._ring_lookup(1)["level"] == 1
    assert agent._anim_seen_level == 2
    assert agent._anim_seen_sigs == {}
    # the same animation on level 2 is new there, not a pointer to level 1
    prompt = _turn(harness, "action(['LEFT'])")
    assert _digest(prompt)[0].startswith("Animation digest (novel): action 2 `LEFT`")
    listing = session.step_env({"query": "animation", "ring": True})["records"]
    assert [(entry["action_num"], entry["level"]) for entry in listing] == [(1, 1), (2, 2)]


def test_rebuilt_opener_repeats_without_recharging(harness):
    session, agent = harness
    first = _turn(harness, "action(['SPACE'])")
    charged = agent._anim_game_chars
    current_frame, history = load_runtime_state(session.state_path)
    again = agent._build_user_prompt(
        session.action_count, valid_actions=["UP", "SPACE", "MOUSE"],
        current_frame=current_frame, history_entries=history,
        previous_step_summary=agent._last_step_summary,
    )
    assert _digest(again) == _digest(first)
    assert agent._anim_game_chars == charged > 0


def test_game_cap_and_digest_budget(harness, monkeypatch):
    _, agent = harness
    monkeypatch.setenv("ARC3_ANIM_DIGEST_CHARS", "300")
    prompt = _turn(harness, "action(['SPACE'])")
    digest = "\n".join(_digest(prompt))
    assert 0 < len(digest) <= 300
    assert agent._anim_game_chars == len(digest)
    # what is left of the game cap is below one header: no digest, but the
    # old opener line still fires
    monkeypatch.setenv("ARC3_ANIM_DIGEST_GAME_CHARS", str(len(digest) + 100))
    prompt = _turn(harness, "action([{'action': 'MOUSE', 'row': 6, 'col': 6}])")
    assert _digest(prompt) == []
    assert "animated over" in prompt
    # with room left, the digest is cut to the remainder
    monkeypatch.setenv("ARC3_ANIM_DIGEST_GAME_CHARS", str(len(digest) + 200))
    prompt = _turn(harness, "action([{'action': 'MOUSE', 'row': 5, 'col': 9}])")
    assert 0 < len("\n".join(_digest(prompt))) <= 200


def test_invalid_gate_fails_at_startup(harness, monkeypatch):
    monkeypatch.setenv("ARC3_ANIM_DIGEST_GATE", "bogus")
    with pytest.raises(ValueError):
        tool_agent.ToolAgent(model="test-model")


def test_game_over_digest_sits_in_the_fatal_block(harness):
    prompt = _turn(harness, "action(['DOWN'])")
    lines = prompt.splitlines()
    death_line = next(i for i, line in enumerate(lines) if line.startswith("That fatal action"))
    assert lines[death_line + 1].startswith("Animation digest (novel): action 1 `DOWN`")
    assert "S2 = death frame" in lines[death_line + 1]
    assert "GAME OVER occurred" in "\n".join(lines[:death_line])


def test_action_result_lists_batch_animations(harness):
    session, agent = harness
    dispatch = agent._run_python_tool(
        session.state_path, {"code": "r = action(['SPACE', 'RIGHT'])\nprint(r['animations'])"}
    )
    assert "'action_display': 'SPACE'" in dispatch.content
    assert agent._last_step_summary["animations"][0]["action_num"] == 1


def _message(harness, code: str, monkeypatch, **env):
    session, agent = harness
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    prompt = _turn(harness, code)
    current_frame, _ = load_runtime_state(session.state_path)
    return prompt, agent._build_user_message(prompt, current_frame)


SWEEP = "action([{'action': 'MOUSE', 'row': 12, 'col': 4}])"


def test_keyframes_follow_the_board_image(harness, monkeypatch):
    prompt, message = _message(
        harness, SWEEP, monkeypatch,
        MULTIMODAL_CONTEXT="current_grid", MULTIMODAL_UPSCALE="10", ARC3_ANIM_KEYFRAMES="3",
    )
    assert "Animation digest (novel): action 1 `MOUSE(row=12, col=4)`: 6 states" in prompt
    content = message["content"]
    kinds = [part["type"] for part in content]
    assert kinds == ["text", "image_url", "text", "image_url", "image_url", "image_url"]
    assert content[0]["text"].endswith("Current grid image:")
    assert content[2]["text"] == (
        "Animation keyframes of action 1 `MOUSE(row=12, col=4)`: the real boards "
        "S1, S3, S5 of S0-S6, oldest first (S0 = board before, S6 = board it settled on; "
        "the digest's step numbers)."
    )
    # board at 10x, keyframes at min(10, 4) = 4x
    assert tool_agent._png_dimensions(content[1]["image_url"]["url"]) == (160, 160)
    for part in content[3:]:
        assert tool_agent._png_dimensions(part["image_url"]["url"]) == (64, 64)
    # priced by the same estimator as every other image
    _, image_tokens = tool_agent._split_messages_for_estimate([message])
    assert image_tokens == sum(
        tool_agent._image_part_tokens(part) for part in content if part["type"] == "image_url"
    )
    assert image_tokens > tool_agent._image_part_tokens(content[1])


def test_keyframe_upscale_override(harness, monkeypatch):
    _, message = _message(
        harness, SWEEP, monkeypatch,
        MULTIMODAL_CONTEXT="current_grid", ARC3_ANIM_KEYFRAMES="1",
        ARC3_ANIM_KEYFRAMES_UPSCALE="2",
    )
    images = [part for part in message["content"] if part["type"] == "image_url"]
    assert len(images) == 2
    assert tool_agent._png_dimensions(images[1]["image_url"]["url"]) == (32, 32)


def test_no_keyframes_without_images_or_for_a_repeat(harness, monkeypatch):
    _, message = _message(harness, SWEEP, monkeypatch, ARC3_ANIM_KEYFRAMES="3")
    assert isinstance(message["content"], str)  # images off: text only, as before
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    prompt, message = _message(harness, "action([{'action': 'MOUSE', 'row': 12, 'col': 6}])",
                               monkeypatch)
    assert "repeats the animation of action 1" in prompt
    assert [part["type"] for part in message["content"]] == ["text", "image_url"]


def test_step_summary_keeps_every_animation_the_result_caps(harness):
    session, agent = harness
    clicks = ", ".join(f"{{'action': 'MOUSE', 'row': 5, 'col': {col}}}" for col in range(4, 14))
    dispatch = agent._run_python_tool(
        session.state_path, {"code": f"r = action([{clicks}, 'RIGHT'])\nprint(len(r['animations']), r['animations_omitted'])"}
    )
    assert "8 2" in dispatch.content
    assert [entry["action_num"] for entry in agent._last_step_summary["animations"]] == list(
        range(1, 11)
    )


def test_digest_names_animation_for_only_inside_the_ring(harness, monkeypatch):
    session, _ = harness
    monkeypatch.setattr(session, "animation_ring", type(session.animation_ring)(maxlen=1))
    prompt = _turn(harness, "action(['SPACE', {'action': 'MOUSE', 'row': 12, 'col': 5}])")
    digest = _digest(prompt)
    assert "animation_for(n)" not in "\n".join(digest)
    assert digest[-1].endswith(
        "Exact: `animation_for(2)['timeline']`; S<k> = `animation_for(2)['frames'][k-1]`. "
        "Action 1's frames are no longer kept."
    )
