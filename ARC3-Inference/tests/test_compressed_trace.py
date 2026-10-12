"""ARC3_COMPRESSED_TRACE_BATCHES: boards and action batches the history window dropped."""

from inference.agent import compressed_trace as ct
from inference.agent import tool_agent
from inference.agent.compressed_trace import Batch
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state


def _grid(value):
    return tuple((value,) * 64 for _ in range(64))


def _entry(step, level, action="", value=None, **result):
    return HistoryEntry(action=action, frame=Frame(grid=_grid(step if value is None else value), step=step, level=level), result=result)


def _history():
    """Level 1 cleared at step 3, a game over at 6, the automatic RESET at 7."""
    return [
        _entry(0, 1, value=0),
        _entry(1, 1, "LEFT"),
        _entry(2, 1, "LEFT"),
        _entry(3, 2, "UP", value=12, level_completed=True),  # level 2 start board
        _entry(4, 2, "MOUSE(row=1, col=2)"),
        _entry(5, 2, "MOUSE(row=1, col=2)"),
        _entry(6, 2, "DOWN", game_over=True),
        _entry(7, 2, "RESET", value=12, automatic=True),      # same as the level 2 start
        _entry(8, 2, "RIGHT"),
        _entry(9, 2, "RIGHT"),
    ]


_RECORDED = [Batch(1, 3), Batch(4, 6), Batch(8, 9, "no_op_action", 2)]
_SOLVED = {3: _grid(15)}


def _image(frame):
    return {"type": "image_url", "image_url": {"url": f"board:{frame.step}:{frame.grid[0][0]}"}}


def _render(content):
    """Text with each image shown as <image step:first-cell>."""
    out = []
    for part in content:
        if part["type"] == "text":
            out.append(part["text"])
        else:
            out.append("<image " + part["image_url"]["url"].split(":", 1)[1] + ">")
    return "\n".join(out)


def test_run_length_merges_only_equal_neighbours():
    assert ct.run_length(["LEFT", "LEFT", "UP", "LEFT"]) == ["LEFT×2", "UP", "LEFT"]
    assert ct.run_length(["MOUSE(row=1, col=2)", "MOUSE(row=1, col=2)", "MOUSE(row=1, col=3)"]) == [
        "MOUSE(row=1, col=2)×2", "MOUSE(row=1, col=3)",
    ]


def test_segment_uses_recorded_batches_and_fills_the_automatic_reset():
    assert ct.segment_batches(_history(), _RECORDED, 9) == [
        Batch(1, 3), Batch(4, 6), Batch(7, 7, "automatic_reset"), Batch(8, 9, "no_op_action", 2),
    ]


def test_unrecorded_steps_split_at_level_up_and_game_over():
    assert ct.segment_batches(_history(), [], 9) == [
        Batch(1, 3), Batch(4, 6), Batch(7, 7, "automatic_reset"), Batch(8, 9),
    ]


def test_render_chains_boards_and_actions():
    message = ct.build_trace_message(_history(), _RECORDED, _SOLVED, 9, 30, _image)
    assert message[ct.TRACE_MESSAGE_KEY] == 9
    assert _render(message["content"]) == "\n".join([
        "Earlier play, compressed: the boards and the actions between them, oldest first. "
        "Your reasoning and tool output are not shown. Step n is `history[n]`; "
        "`history[0]` is the starting board. A solved-level board is the only board here "
        "that is not in `history`.",
        "",
        "Level 1. Starting board, step 0:",
        "<image 0:0>",
        "→ steps 1–3: LEFT×2, UP · LEVEL 1 CLEARED",
        "Board as level 1 was solved (shown during step 3, not in history):",
        "<image 3:15>",
        "Level 2. Board after step 3, the level 2 start:",
        "<image 3:12>",
        "→ steps 4–6: MOUSE(row=1, col=2)×2, DOWN · GAME OVER",
        "Board after step 6, the game-over board:",
        "<image 6:6>",
        "→ step 7: RESET (automatic), level 2 restarts",
        "Board after step 7: same as after step 3.",
        "→ steps 8–9: RIGHT×2 (stopped: no change, +2 skipped)",
        "The board after step 9 is the first board in the conversation below.",
    ])


def test_cap_keeps_the_newest_batches_and_starts_from_the_board_before_them():
    message = ct.build_trace_message(_history(), _RECORDED, _SOLVED, 9, 2, _image)
    text = _render(message["content"])
    assert "Level 2. Board after step 6:\n<image 6:6>\n→ step 7: RESET (automatic)" in text
    assert "solved" not in text  # no level-up within the two batches kept


def test_no_trace_before_any_trim_or_when_indexing_breaks():
    assert ct.build_trace_message(_history(), _RECORDED, _SOLVED, 0, 30, _image) is None
    assert ct.build_trace_message(_history(), _RECORDED, _SOLVED, 9, 0, _image) is None
    broken = _history()
    broken[4] = _entry(5, 2, "MOUSE(row=1, col=2)")  # history[4] claims step 5
    assert ct.build_trace_message(broken, _RECORDED, _SOLVED, 9, 30, _image) is None


def test_long_batches_keep_twelve_runs():
    names = [f"MOUSE(row={i}, col=0)" for i in range(15)]
    assert ct._actions_text(names).endswith("MOUSE(row=11, col=0), … (+3 actions)")


def test_missing_solved_board_and_images_off():
    message = ct.build_trace_message(_history(), _RECORDED, {}, 9, 30, lambda frame: None)
    text = _render(message["content"])
    assert "Board as level 1 was solved: unavailable." in text
    assert "Level 2 starts after step 3." in text
    assert "<image" not in text


# --- agent wiring -------------------------------------------------------------


def _agent(monkeypatch, tmp_path, batches="30"):
    monkeypatch.setenv("ARC3_COMPRESSED_TRACE_BATCHES", batches)
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("MULTIMODAL_UPSCALE", "10")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    state = tmp_path / "state.json"
    agent._ensure_session(state)
    history = _history()
    write_runtime_state(state, current_frame=history[-1].frame, history=history)
    agent._trace_state_path = state
    agent._trace_batches = list(_RECORDED)
    agent._trace_solved_boards = dict(_SOLVED)
    return agent


def _turn(agent, step, text, size=400):
    """An opener, reply and tool result appended at `step`, tagged as in a real turn."""
    agent._trace_step_now = step
    messages = []
    for message in (
        {"role": "user", "content": text},
        {"role": "assistant", "content": "x" * size},
        {"role": "tool", "tool_call_id": "t", "content": "y" * size},
    ):
        agent._append_context_message(messages, message)
    return messages


def test_trim_pins_the_trace_after_the_system_prompt(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    first = _turn(agent, 0, "first", size=6_000)  # larger than the trace that replaces it
    history = [*first, *_turn(agent, 6, "second"), *_turn(agent, 9, "third")]
    system = {"role": "system", "content": "sys"}
    untrimmed = agent._trim_messages_for_context([system, *history])
    assert untrimmed == [system, *history]  # window still starts at step 0: no trace

    full = agent._estimate_request_input_tokens([system, *history])
    agent._context_budget_tokens = full - 10  # over budget by one turn
    trimmed = agent._trim_messages_for_context([system, *history])
    assert trimmed[0] is system
    assert trimmed[1][ct.TRACE_MESSAGE_KEY] == 6  # the window now starts at the step-6 opener
    assert trimmed[2]["content"] == "second"

    again = agent._trim_messages_for_context(trimmed)
    assert again[1] is trimmed[1]  # cached: identical message between trims


def test_persisted_history_never_stores_the_trace(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, 0, "first", size=6_000), *_turn(agent, 6, "second")]
    system = {"role": "system", "content": "sys"}
    agent._context_budget_tokens = agent._estimate_request_input_tokens([system, *history]) - 10
    stored = agent._persistent_history_messages([system, *history])
    assert stored and not any(ct.TRACE_MESSAGE_KEY in m for m in stored)
    assert stored[0]["content"] == "second"


def test_trim_drops_more_when_the_rebuilt_trace_does_not_fit(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    # "first" is smaller than the step-6 trace, so dropping it alone does not fit;
    # dropping "second" too does, with the step-9 trace
    history = [*_turn(agent, 0, "first"), *_turn(agent, 6, "second", size=4_500), *_turn(agent, 9, "third")]
    system = {"role": "system", "content": "sys"}
    agent._context_budget_tokens = agent._estimate_request_input_tokens([system, *history]) - 10
    trimmed = agent._trim_messages_for_context([system, *history])
    assert trimmed[1][ct.TRACE_MESSAGE_KEY] == 9
    assert trimmed[2]["content"] == "third"
    assert agent._estimate_request_input_tokens(trimmed) <= agent._context_budget_tokens


def test_private_keys_never_reach_the_server(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = _turn(agent, 6, "opener")
    trace = ct.build_trace_message(_history(), _RECORDED, _SOLVED, 6, 30, _image)
    wire = tool_agent._strip_control_keys([trace, *history])
    assert all(not any(k.startswith("_arc3_") for k in m) for m in wire)


def test_record_batch_from_the_solver_payload(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    agent._trace_batches = []
    agent._record_trace_batch({
        "executed": True, "action_num": 12, "executed_count": 3,
        "stop_reason": "level_completed", "level_end_grid": _grid(7),
    })
    agent._record_trace_batch({"executed": False, "action_num": 12})
    assert agent._trace_batches == [Batch(10, 12, "level_completed", 0)]
    assert agent._trace_solved_boards[12] == _grid(7)
    assert agent._trace_step_now == 12


def test_disabled_leaves_messages_untouched(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path, batches="0")
    history = _turn(agent, 6, "opener")
    assert not any(ct.STEP_KEY in m for m in history)
    system = {"role": "system", "content": "sys"}
    agent._record_trace_batch({"executed": True, "action_num": 12, "executed_count": 1})
    assert agent._trace_batches == list(_RECORDED)  # unchanged: nothing recorded when off
    assert agent._trim_messages_for_context([system, *history]) == [system, *history]


# --- level-up snippets (ARC3_COMPRESSED_TRACE_LEVELUP_FRAMES) ---------------------


def _three_levels():
    """Level 1 won at step 4, level 2 at step 7; level 3 is being played."""
    return [
        _entry(0, 1, value=0),
        _entry(1, 1, "RIGHT"),
        _entry(2, 1, "RIGHT"),
        _entry(3, 1, "UP"),
        _entry(4, 2, "UP", value=12, level_completed=True),
        _entry(5, 2, "LEFT"),
        _entry(6, 2, "LEFT"),
        _entry(7, 3, "DOWN", value=13, level_completed=True),
        _entry(8, 3, "RIGHT"),
        _entry(9, 3, "RIGHT"),
        _entry(10, 3, "UP"),
    ]


_THREE_BATCHES = [Batch(1, 1), Batch(2, 2), Batch(3, 4), Batch(5, 5), Batch(6, 7), Batch(8, 8), Batch(9, 10)]
_THREE_SOLVED = {4: _grid(14), 7: _grid(15)}


def _three(max_batches, frames):
    message = ct.build_trace_message(
        _three_levels(), _THREE_BATCHES, _THREE_SOLVED, 10, max_batches, _image, levelup_frames=frames,
    )
    return _render(message["content"]).split("\n", 2)[2]  # drop the header and blank line


def test_leading_gap_line_names_the_steps_left_out():
    message = ct.build_trace_message(_history(), _RECORDED, _SOLVED, 9, 2, _image)
    text = _render(message["content"])
    assert text.split("\n")[0].endswith(' A line "... steps a–b ..." marks steps left out.')
    assert text.split("\n")[2:4] == ["... steps 1–6 ...", "Level 2. Board after step 6:"]


def test_snippets_are_prepended_and_contiguous_parts_join():
    assert _three(2, 2) == "\n".join([
        "... step 1 ...",
        "Level 1. Board after step 1:",
        "<image 1:1>",
        "→ step 2: RIGHT",
        "Board after step 2:",
        "<image 2:2>",
        "→ steps 3–4: UP×2 · LEVEL 1 CLEARED",
        "Board as level 1 was solved (shown during step 4, not in history):",
        "<image 4:14>",
        "Level 2. Board after step 4, the level 2 start:",
        "<image 4:12>",
        "→ step 5: LEFT",
        "Board after step 5:",
        "<image 5:5>",
        "→ steps 6–7: LEFT, DOWN · LEVEL 2 CLEARED",
        "Board as level 2 was solved (shown during step 7, not in history):",
        "<image 7:15>",
        "Level 3. Board after step 7, the level 3 start:",
        "<image 7:13>",
        "→ step 8: RIGHT",
        "Board after step 8:",
        "<image 8:8>",
        "→ steps 9–10: RIGHT, UP",
        "The board after step 10 is the first board in the conversation below.",
    ])


def test_gap_between_a_snippet_and_the_recent_chain():
    text = _three(1, 2)
    assert "Level 3 starts after step 7.\n... step 8 ...\nLevel 3. Board after step 8:\n<image 8:8>" in text
    assert text.endswith("→ steps 9–10: RIGHT, UP\nThe board after step 10 is the first board in the conversation below.")


def test_level_up_inside_the_recent_chain_gets_no_snippet():
    text = _three(4, 2)
    assert text.count("LEVEL 1 CLEARED") == 1 and text.count("LEVEL 2 CLEARED") == 1
    assert text.startswith("... step 1 ...\nLevel 1. Board after step 1:")  # the level 1 snippet
    assert "Level 2. Board after step 4, the level 2 start:" in text  # chain opens on level 2


def test_snippet_shorter_than_k_keeps_the_whole_level():
    text = _three(1, 5)
    assert text.startswith("Level 1. Starting board, step 0:\n<image 0:0>\n→ step 1: RIGHT")


def test_frames_zero_matches_the_plain_trace():
    plain = ct.build_trace_message(_three_levels(), _THREE_BATCHES, _THREE_SOLVED, 10, 2, _image)
    zero = ct.build_trace_message(_three_levels(), _THREE_BATCHES, _THREE_SOLVED, 10, 2, _image, levelup_frames=0)
    assert plain == zero
    assert _render(plain["content"]).split("\n")[2] == "... steps 1–7 ..."


def test_agent_cache_follows_the_levelup_flag(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path, batches="1")
    history = _turn(agent, 9, "opener")
    without = agent._compressed_trace_messages(history)[0]
    monkeypatch.setenv("ARC3_COMPRESSED_TRACE_LEVELUP_FRAMES", "2")
    with_snippets = agent._compressed_trace_messages(history)[0]
    assert without is not with_snippets
    assert "LEVEL 1 CLEARED" not in tool_agent._message_display_text(without)
    assert "LEVEL 1 CLEARED" in tool_agent._message_display_text(with_snippets)
