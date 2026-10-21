"""The layered animation flags (docs: research/archive/reports/animation-combo-design.md).

L1 ARC3_ANIM_TRUTH, L2 ARC3_ANIM_RING, L3 ARC3_ANIM_DIGEST; each is off by
default and each layer is tested here on the real solver session and the real
python sandbox (tests/solver_fakes.FakeGame drives the engine side).
"""
from __future__ import annotations

import pytest

from inference.agent import prompts, tool_agent
from inference.framework import solver as solver_module


ANIM_ENV_KEYS = (
    "ARC3_ANIMATION", "ARC3_ANIM_TRUTH", "ARC3_ANIM_RING", "ARC3_ANIM_DIGEST",
    "ARC3_ANIM_DIGEST_GATE", "ARC3_ANIM_DIGEST_CHARS", "ARC3_ANIM_DIGEST_GAME_CHARS",
    "ARC3_ANIM_KEYFRAMES", "ARC3_ANIM_NOOP_EXEMPT", "ARC3_ANIMATION_TIMELINE",
    "MULTIMODAL_CONTEXT",
)


@pytest.fixture
def clean_env(monkeypatch):
    for key in ANIM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# ---------------------------------------------------------------- flag split

def test_digest_without_ring_fails_at_startup(clean_env):
    clean_env.setenv("ARC3_ANIMATION", "1")
    clean_env.setenv("ARC3_ANIM_DIGEST", "1")
    with pytest.raises(ValueError, match="ARC3_ANIM_RING"):
        tool_agent.ToolAgent(model="test-model")


def test_ring_alone_adds_only_the_ring_addendum(clean_env):
    clean_env.setenv("ARC3_ANIMATION", "1")
    base = tool_agent._build_system_prompt(tool_output_tokens=3072)
    clean_env.setenv("ARC3_ANIM_RING", "8")
    ring = tool_agent._build_system_prompt(tool_output_tokens=3072)
    assert prompts.ANIMATION_RING_ADDENDUM in ring
    assert prompts.ANIMATION_DIGEST_ADDENDUM not in ring
    assert ring.replace(prompts.ANIMATION_RING_ADDENDUM, "", 1) == base


def test_ring_zero_is_off(clean_env):
    clean_env.setenv("ARC3_ANIMATION", "1")
    clean_env.setenv("ARC3_ANIM_RING", "0")
    assert not tool_agent._anim_ring_enabled()
    clean_env.setenv("ARC3_ANIM_RING", "8")
    assert tool_agent._anim_ring_enabled()
    clean_env.setenv("ARC3_ANIMATION", "0")
    assert not tool_agent._anim_ring_enabled()


def test_solver_ring_default_is_off():
    # read at import; the test process does not set ARC3_ANIM_RING
    assert solver_module.ANIM_RING_SIZE == 0 or "ARC3_ANIM_RING" in __import__("os").environ


# ---------------------------------------------------------------- L2: ring

from inference.agent.runtime_state import load_runtime_state  # noqa: E402

from solver_fakes import FakeGame, blank, flash_script, frozen, make_session  # noqa: E402


def _start_board():
    board = blank()
    board[6][5] = 9  # inside the 4-cell border, so moving it is a gameplay change
    return frozen(board)


def _harness(tmp_path, monkeypatch, *, ring: int = 0, truth: bool = False, digest: bool = False,
             batch_block: bool = False, stale_block: bool = False, script=flash_script):
    """A real solver session and ToolAgent wired as solver.py wires them."""
    for key in ANIM_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ARC3_ANIMATION", "1")
    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
    monkeypatch.setattr(solver_module, "ANIM_RING_SIZE", ring)
    monkeypatch.setattr(solver_module, "ANIM_RING_ENABLED", ring > 0)
    monkeypatch.setattr(solver_module, "ANIM_DIGEST_ENABLED", ring > 0 and digest)
    if hasattr(solver_module, "ANIM_TRUTH"):
        monkeypatch.setattr(solver_module, "ANIM_TRUTH", truth)
    monkeypatch.setattr(solver_module, "BATCH_NOOP_BLOCK", batch_block)
    monkeypatch.setattr(solver_module, "STALE_STATE_BLOCK", stale_block)
    monkeypatch.setattr(solver_module, "NOOP_GUARD_BORDER_WIDTH", 4)
    monkeypatch.setattr(solver_module, "GUARDS_FROM_LEVEL", 1)
    if ring:
        monkeypatch.setenv("ARC3_ANIM_RING", str(ring))
    if truth:
        monkeypatch.setenv("ARC3_ANIM_TRUTH", "1")
    if digest:
        monkeypatch.setenv("ARC3_ANIM_DIGEST", "1")
    session = make_session(tmp_path, FakeGame(_start_board(), script))
    session.seed_initial_history()
    session.write_runtime_state()
    agent = tool_agent.ToolAgent(model="test-model")
    agent._ensure_session(session.state_path)
    agent._step_env_callback = session.step_env
    agent._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]
    return session, agent


def _python(harness, code: str) -> str:
    """One real python tool call; returns its stdout."""
    import json as _json
    session, agent = harness
    dispatch = agent._run_python_tool(session.state_path, {"code": code})
    payload = _json.loads(dispatch.content)
    return str(payload.get("stdout") or "") + str(payload.get("stderr") or "") + str(payload.get("error") or "")


def _opener(harness) -> str:
    session, agent = harness
    current_frame, history = load_runtime_state(session.state_path)
    return agent._build_user_prompt(
        session.action_count, valid_actions=["UP", "SPACE", "MOUSE"], current_frame=current_frame,
        history_entries=history, previous_step_summary=agent._last_step_summary,
    )


@pytest.mark.parametrize("ring", [0, 8])
def test_last_animation_follows_the_newest_animated_action(tmp_path, monkeypatch, ring):
    harness = _harness(tmp_path, monkeypatch, ring=ring)
    _python(harness, "action([{'action': 'SPACE'}])")  # animates, hides 4 cells
    _python(harness, "action([{'action': 'RIGHT'}])")  # moves, does not animate
    out = _python(harness, "print(len(last_animation_frames)); print(last_animation_timeline)")
    if ring:
        assert out.splitlines()[0] == "2"
        assert "'action_num': 1" in out and "'action': 'SPACE'" in out
    else:
        assert out.splitlines()[0] == "0"
        assert "the last executed action returned a single frame" in out


def test_animation_for_two_calls_ago_and_eviction(tmp_path, monkeypatch):
    harness = _harness(tmp_path, monkeypatch, ring=8)
    _python(harness, "action([{'action': 'MOUSE', 'row': 6, 'col': 6}])")  # action 1
    _python(harness, "action([{'action': 'UP'}])")  # action 2, inert
    out = _python(harness, "a = animation_for(1); print(a['action'], len(a['frames']), a['timeline']['action_num'])")
    assert out.strip() == "MOUSE(row=6, col=6) 2 1"
    for col in range(4, 12):  # 8 more animated actions: action 1 falls out of the ring
        _python(harness, f"action([{{'action': 'MOUSE', 'row': 5, 'col': {col}}}])")
    out = _python(harness, "a = animation_for(1); print(a['frames'], a['timeline']['error'])")
    assert "No animation is kept for action 1" in out
    assert "[3, 4, 5, 6, 7, 8, 9, 10]" in out


def test_frame_index_matches_timeline_step(tmp_path, monkeypatch):
    harness = _harness(tmp_path, monkeypatch, ring=8)
    _python(harness, "action([{'action': 'MOUSE', 'row': 12, 'col': 3}])")  # 1x2 sweep, 6 states
    out = _python(harness, (
        "a = animation_for(1)\n"
        "ok = []\n"
        "for step in a['timeline']['steps']:\n"
        "    k = step['step']\n"
        "    before = a['frames'][k-2].ascii if k >= 2 else previous_frame.ascii\n"
        "    after = a['frames'][k-1].ascii\n"
        "    n = sum(1 for x, y in zip(before, after) if x != y)\n"
        "    ok.append(n == step['changed'])\n"
        "print(len(a['frames']), all(ok), len(ok))\n"
    ))
    assert out.strip() == "6 True 6"


def test_death_animation_readable_after_auto_reset(tmp_path, monkeypatch):
    harness = _harness(tmp_path, monkeypatch, ring=8)
    session, _ = harness
    _python(harness, "action([{'action': 'DOWN'}])")  # flashes a bar, then game over
    session._execute_auto_reset()
    out = _python(harness, (
        "print(len(last_animation_frames), last_animation_timeline['action'])\n"
        "a = animation_for(1); print(a['action'], len(a['frames']))\n"
    ))
    assert out.splitlines() == ["2 DOWN", "DOWN 2"]


def _tween_script(action_name, data, board):
    """RIGHT moves the 9-dot one cell over two frames, stretching then
    settling: every cell changes once, so nothing is hidden (a motion-only
    animation). SPACE flashes and reverts (hides 4 cells)."""
    if action_name == "ACTION4":
        (row, col), = [(r, c) for r in range(16) for c in range(15) if board[r][c] == 9]
        stretch = [list(line) for line in board]
        stretch[row][col + 1] = 9
        settled = [list(line) for line in board]
        settled[row][col], settled[row][col + 1] = 0, 9
        return [frozen(stretch), frozen(settled)], {}
    return flash_script(action_name, data, board)


def test_batch_animations_list_only_hiding_entries(tmp_path, monkeypatch):
    session, agent = _harness(tmp_path, monkeypatch, ring=8, script=_tween_script)
    raw = session.step_env({"actions": [{"action": "RIGHT"}, {"action": "RIGHT"}, {"action": "RIGHT"}]})
    compact = agent._compact_action_result(raw)
    assert raw["animation"]["transient_pixels"] == 0
    for key in ("animations", "animations_unlisted", "animations_omitted"):
        assert key not in compact  # a tween-only batch adds no key
    raw = session.step_env({"actions": [
        {"action": "SPACE"}, {"action": "RIGHT"}, {"action": "SPACE"}, {"action": "RIGHT"},
    ]})
    compact = agent._compact_action_result(raw)
    assert [e["action_display"] for e in compact["animations"]] == ["SPACE", "SPACE"]
    assert compact["animations_unlisted"] == 2
    assert list(compact)[-2:] == ["animations", "animations_unlisted"]
    # the ring itself still keeps every animated action
    assert [r["action_display"] for r in session.animation_ring][-4:] == ["SPACE", "RIGHT", "SPACE", "RIGHT"]


# ---------------------------------------------------------------- L1: truth

@pytest.mark.parametrize("truth,ring", [(False, 0), (True, 0), (True, 8)])
def test_echo_says_animated_for_a_hiding_no_op(tmp_path, monkeypatch, truth, ring):
    monkeypatch.setattr(solver_module, "ACTION_ECHO", True)
    monkeypatch.setattr(solver_module, "NEW_NOOP_WORDING", True)
    harness = _harness(tmp_path, monkeypatch, truth=truth, ring=ring)
    out = _python(harness, "r = action([{'action': 'SPACE'}])")
    echo = out.splitlines()[0]
    if not truth:
        assert echo == "[action] SPACE -> NO-OP (executed, no change inside the board area)"
    elif ring:
        assert echo == ("[action] SPACE -> ANIMATED (blink; board area unchanged; 2 frames, "
                        "4 cells changed and changed back - see animation_for(1))")
    else:
        assert echo == ("[action] SPACE -> ANIMATED (blink; board area unchanged; 2 frames, "
                        "4 cells changed and changed back - see last_animation_timeline)")


def test_a_real_no_op_and_a_tween_stay_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_module, "ACTION_ECHO", True)
    monkeypatch.setattr(solver_module, "NEW_NOOP_WORDING", True)
    harness = _harness(tmp_path, monkeypatch, truth=True, ring=8)
    assert _python(harness, "action([{'action': 'UP'}])").splitlines()[0].endswith(
        "-> NO-OP (executed, no change inside the board area)")

    def border_tween(action_name, data, board):
        # a motion-only animation in the border: board area unchanged, nothing hidden
        stretch = [list(line) for line in board]
        stretch[0][1] = 7
        settled = [list(line) for line in stretch]
        settled[0][0] = 7
        return [frozen(stretch), frozen(settled)], {}

    harness2 = _harness(tmp_path / "b", monkeypatch, truth=True, ring=8, script=border_tween)
    out = _python(harness2, "r = action([{'action': 'UP'}]); print(r['animation'])")
    assert "NO-OP" in out.splitlines()[0] and "'transient_pixels': 0" in out


def test_batch_trace_uses_the_same_word(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_module, "NEW_NOOP_WORDING", True)
    session, agent = _harness(tmp_path, monkeypatch, truth=True, ring=0)
    raw = session.step_env({"actions": [{"action": "SPACE"}, {"action": "SPACE"}, {"action": "UP"}]})
    trace = raw["action_trace"]
    # without the ring only the last action's frames are kept, so no pointer here
    assert trace[0] == ("1: SPACE -> ANIMATED (blink; board area unchanged; 2 frames, "
                        "4 cells changed and changed back)")
    assert trace[2] == "3: UP -> NO-OP (executed, no change inside the board area)"
    assert raw["gameplay_changed_per_action"] == [False, False, False]  # unchanged facts
    assert "animated_noop_per_action" not in agent._compact_action_result(raw)
    assert "animation_type" not in agent._compact_action_result(raw)


@pytest.mark.parametrize("truth", [False, True])
def test_batch_stop_passes_animated_and_stops_on_a_real_no_op(tmp_path, monkeypatch, truth):
    session, _ = _harness(tmp_path, monkeypatch, truth=truth, batch_block=True)
    raw = session.step_env({"actions": [
        {"action": "SPACE"}, {"action": "SPACE"}, {"action": "UP"}, {"action": "SPACE"},
    ]})
    assert raw["stop_reason"] == "no_op_action"
    if truth:
        assert raw["executed_actions"] == ["SPACE", "SPACE", "UP"]
        assert raw["no_op_action"] == "UP"
    else:
        assert raw["executed_actions"] == ["SPACE"]


@pytest.mark.parametrize("truth", [False, True])
def test_known_noop_guard_does_not_file_animated_actions(tmp_path, monkeypatch, truth):
    monkeypatch.setenv("ARC3_NOOP_REPEAT_GUARD", "1")
    session, agent = _harness(tmp_path, monkeypatch, truth=truth)
    assert agent._noop_repeat_guard is not None
    session.action_guard_hook = agent._action_guard_hook
    _python((session, agent), "action([{'action': 'SPACE'}])")
    _python((session, agent), "action([{'action': 'UP'}, {'action': 'SPACE'}])")
    filed = {sig for entry in agent._noop_repeat_guard._states.values() for sig in entry}
    assert "UP" in filed
    assert ("SPACE" in filed) is (not truth)


def test_truth_adds_one_static_clause(clean_env):
    clean_env.setenv("ARC3_ANIMATION", "1")
    base = tool_agent._build_system_prompt(tool_output_tokens=3072)
    clean_env.setenv("ARC3_ANIM_TRUTH", "1")
    truth = tool_agent._build_system_prompt(tool_output_tokens=3072)
    assert truth.replace(prompts.ANIMATION_TRUTH_ADDENDUM, "", 1) == base


@pytest.mark.parametrize("truth", [False, True])
def test_stale_latch_not_armed_after_animated(tmp_path, monkeypatch, truth):
    harness = _harness(tmp_path, monkeypatch, truth=truth, stale_block=True)
    out = _python(harness, (
        "action([{'action': 'SPACE'}])\n"
        "try:\n"
        "    action([{'action': 'SPACE'}])\n"
        "    print('second ran')\n"
        "except StaleStateActionError as exc:\n"
        "    print('stale')\n"
    ))
    assert ("second ran" in out) is truth and ("stale" in out) is (not truth)


def test_stale_latch_still_armed_after_a_real_no_op(tmp_path, monkeypatch):
    harness = _harness(tmp_path, monkeypatch, truth=True, stale_block=True)
    out = _python(harness, (
        "action([{'action': 'UP'}])\n"
        "try:\n"
        "    action([{'action': 'UP'}])\n"
        "    print('second ran')\n"
        "except StaleStateActionError:\n"
        "    print('stale')\n"
    ))
    assert "stale" in out and "second ran" not in out


def test_ring_addendum_names_the_record_keys():
    text = prompts.ANIMATION_RING_ADDENDUM
    for key in ("action_num", "step_count", "transient_pixels", "transient_bbox", "frames[k-1]",
                "animation_for(action_num)", "costs no game action"):
        assert key in text


def test_layer_addenda_order(clean_env):
    clean_env.setenv("ARC3_ANIMATION", "1")
    for key, value in (("ARC3_ANIM_TRUTH", "1"), ("ARC3_ANIM_RING", "8"), ("ARC3_ANIM_DIGEST", "1")):
        clean_env.setenv(key, value)
    prompt = tool_agent._build_system_prompt(tool_output_tokens=3072)
    positions = [prompt.index(text) for text in (
        prompts.ANIMATION_TRUTH_ADDENDUM, prompts.ANIMATION_RING_ADDENDUM,
        prompts.ANIMATION_DIGEST_ADDENDUM)]
    assert positions == sorted(positions)


# ---------------------------------------------------------------- L3: digest

from inference.utils.animation import classify_animation, describe_animation_type  # noqa: E402


def _g(marks, n=16):
    rows = [[0] * n for _ in range(n)]
    for (r, c), v in marks.items():
        rows[r][c] = v
    return tuple(tuple(row) for row in rows)


def _block(r, c, color=8):
    return _g({(r, c): color, (r + 1, c): color, (r, c + 1): color, (r + 1, c + 1): color})


def test_classifier_types():
    x, y = _g({(8, 8): 9}), _g({(8, 8): 9, (9, 9): 8})
    assert describe_animation_type(classify_animation([x, y, x])) == "blink x1"
    assert describe_animation_type(classify_animation([x, y, x, y, x])) == "blink x2"
    slide = [_block(6, 4), _block(6, 6), _block(6, 8), _block(6, 10)]
    assert describe_animation_type(classify_animation(slide, 4)) == "tween R 2x2 (6,4)>(6,10) d(+0,+6)"
    turn = [_block(6, 4), _block(6, 6), _block(8, 6), _block(8, 9)]
    assert describe_animation_type(classify_animation(turn, 4)) == "path R 2x2 (6,4)>(6,6)>(8,6)>(8,9)"
    # a highlight sweeping over identical tokens and reverting: it is seen at
    # positions (path), never reported as a token that moved (issue #41's case)
    tokens = {(10, c): 3 for c in range(4, 14)}
    sweep = [_g(tokens)]
    for c in (4, 6, 8, 10):
        sweep.append(_g({**tokens, (10, c): 14, (10, c + 1): 14}))
    sweep.append(_g(tokens))
    assert classify_animation(sweep, 0)["type"] == "path"
    # a HUD tick in the border does not turn a flash into "other"
    hud = [_g({(0, 0): 1}), _g({(0, 1): 1, (8, 8): 5}), _g({(0, 2): 1})]
    assert describe_animation_type(classify_animation(hud, 4)) == "blink x1"
    assert classify_animation([x, y]) == {}


def test_classifier_host_cost():
    import random
    import time
    rng = random.Random(0)
    chain = [tuple(tuple(rng.randrange(10) for _ in range(64)) for _ in range(64)) for _ in range(111)]
    start = time.perf_counter()
    classify_animation(chain, 4)
    assert time.perf_counter() - start < 1.0  # measured about 0.07 s; host budget is the 30 s tool call


def test_digest_header_is_typed_and_capped(tmp_path, monkeypatch):
    harness = _harness(tmp_path, monkeypatch, ring=8, truth=True, digest=True)
    _python(harness, "action([{'action': 'SPACE'}])")
    prompt = _opener(harness)
    digest = [line for line in prompt.splitlines() if line.startswith("Animation digest")]
    assert digest and digest[0].startswith(
        "Animation digest (novel): action 1 `SPACE`: blink x1; 2 states (S0 = board before; S2 = current board).")
    assert tool_agent._anim_digest_chars() == 600


def test_digest_never_exceeds_the_default_cap(tmp_path, monkeypatch):
    harness = _harness(tmp_path, monkeypatch, ring=8, digest=True)
    _python(harness, "action([" + ", ".join(
        f"{{'action': 'MOUSE', 'row': {r}, 'col': {c}}}" for r in (5, 6, 7, 8) for c in (5, 7, 9, 11)
    ) + ", {'action': 'MOUSE', 'row': 12, 'col': 4}])")
    prompt = _opener(harness)
    lines = prompt.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Animation digest"))
    end = start
    while end + 1 < len(lines) and (lines[end + 1].startswith(("- ", "action ", "(+", "Trails"))):
        end += 1
    block = "\n".join(lines[start:end + 1])
    assert len(block) <= 600
