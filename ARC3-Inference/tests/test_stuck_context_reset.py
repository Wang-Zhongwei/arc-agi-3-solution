"""ARC3_STUCK_CONTEXT_RESET: a trim with no level-up since the previous trim empties the context."""

from inference.agent import tool_agent

NOTICE = tool_agent._STUCK_CONTEXT_RESET_NOTICE
SYSTEM = {"role": "system", "content": "sys"}


def _agent(monkeypatch, tmp_path, enabled=True):
    monkeypatch.setenv("ARC3_STUCK_CONTEXT_RESET", "1" if enabled else "0")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.delenv("ARC3_CONTEXT_DRAIN_TOKENS", raising=False)
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    return agent


def _turn(agent, text, size=400):
    messages = []
    for message in (
        {"role": "user", "content": text},
        {"role": "assistant", "content": "x" * size},
        {"role": "tool", "tool_call_id": "t", "content": "y" * size},
    ):
        agent._append_context_message(messages, message)
    return messages


def _over_budget_by_one_turn(agent, history):
    agent._context_budget_tokens = agent._estimate_request_input_tokens([SYSTEM, *history]) - 10


def _contents(messages):
    return [m["content"] if isinstance(m["content"], str) else m["content"][0]["text"] for m in messages]


def test_off_keeps_the_oldest_first_drain(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path, enabled=False)
    history = [*_turn(agent, "first"), *_turn(agent, "second"), *_turn(agent, "third")]
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    assert _contents(trimmed)[1] == "second"
    assert agent._context_reset_log == ""


def test_request_trim_while_stuck_keeps_only_the_current_turn(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first"), *_turn(agent, "second"), *_turn(agent, "third")]
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    assert trimmed[0] is SYSTEM
    assert len(trimmed) == 4  # system + the third turn's opener, reply, tool result
    assert trimmed[1]["content"] == NOTICE + "\n\nthird"
    assert trimmed[2:] == history[-2:]
    assert history[-3]["content"] == "third"  # the stored message is not mutated
    assert "stuck_reset=1" in agent._pending_trim_reasons[-1]
    assert agent._context_was_trimmed
    notes = []
    agent._log_context_reset(lambda label, text: notes.append((label, text)))
    assert notes == [("ANALYZER STATUS", "context reset (request): no level-up since the previous trim; history messages 9 -> 3")]


def test_a_level_up_since_the_last_trim_drains_normally_once(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    agent._last_step_summary = {"level": 2, "level_transition": True}
    agent._update_summarized_knowledge_from_step_summary()
    assert agent._levelup_since_trim
    history = [*_turn(agent, "first"), *_turn(agent, "second"), *_turn(agent, "third")]
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    assert _contents(trimmed)[1:] == ["second", "x" * 400, "y" * 400, "third", "x" * 400, "y" * 400]
    assert not agent._levelup_since_trim  # the trim consumed it
    history = [*trimmed[1:], *_turn(agent, "fourth")]
    _over_budget_by_one_turn(agent, history)
    again = agent._trim_messages_for_context([SYSTEM, *history])
    assert again[1]["content"] == NOTICE + "\n\nfourth"
    assert len(again) == 4


def test_no_trim_no_reset(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first"), *_turn(agent, "second")]
    assert agent._trim_messages_for_context([SYSTEM, *history]) == [SYSTEM, *history]
    assert agent._persistent_history_messages([SYSTEM, *history]) == history
    assert agent._context_reset_notice == ""
    assert not agent._context_was_trimmed


def test_a_nudge_is_not_a_turn_opener(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    current = _turn(agent, "third")
    nudge = tool_agent._mark_control_message({"role": "user", "content": "You have not acted yet."}, "nudge")
    history = [*_turn(agent, "first"), *_turn(agent, "second"), *current, nudge,
               {"role": "assistant", "content": "z" * 400}]
    _over_budget_by_one_turn(agent, history)
    trimmed = agent._trim_messages_for_context([SYSTEM, *history])
    assert trimmed[1]["content"] == NOTICE + "\n\nthird"
    assert trimmed[-2] is nudge


def test_a_drain_that_lands_on_the_current_turn_still_notices(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first"), *_turn(agent, "second", size=8_000)]
    # below the second turn alone: the drain stops at its opener, which is never dropped
    agent._context_budget_tokens = agent._estimate_request_input_tokens([SYSTEM, *history[-1:]]) + 5
    trimmed = agent._trim_messages_for_context([SYSTEM, *history], preserve_recent=1)
    assert trimmed[1]["content"] == NOTICE + "\n\nsecond"
    assert trimmed[2:] == history[-2:]


def test_end_of_turn_reset_carries_nothing_and_notices_once(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first"), *_turn(agent, "second"), *_turn(agent, "third")]
    _over_budget_by_one_turn(agent, history)
    stored = agent._persistent_history_messages([SYSTEM, *history])
    assert stored == []
    assert "stuck_reset=1" in agent._pending_trim_reasons[-1]
    assert agent._context_reset_log.startswith("context reset (end of turn)")
    opener = agent._build_user_message("The code executed 1 action.", None)
    assert opener["content"].startswith(NOTICE + "\n\nThe code executed 1 action.")
    assert NOTICE not in agent._build_user_message("next", None)["content"]


def test_end_of_turn_reset_before_a_resumption_keeps_its_turn(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first"), *_turn(agent, "second"), *_turn(agent, "third")]
    _over_budget_by_one_turn(agent, history)
    agent._resume_after_yield = True
    stored = agent._persistent_history_messages([SYSTEM, *history])
    assert _contents(stored) == ["third", "x" * 400, "y" * 400]


def test_end_of_turn_reset_after_a_level_up_drains_normally(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    agent._levelup_since_trim = True
    history = [*_turn(agent, "first"), *_turn(agent, "second"), *_turn(agent, "third")]
    _over_budget_by_one_turn(agent, history)
    stored = agent._persistent_history_messages([SYSTEM, *history])
    assert _contents(stored)[0] == "second"
    assert agent._context_reset_notice == ""
    assert not agent._levelup_since_trim


def test_turn_cap_trim_also_resets(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC3_HISTORY_ASSISTANT_TURNS", "2")
    agent = _agent(monkeypatch, tmp_path)
    history = [*_turn(agent, "first"), *_turn(agent, "second"), *_turn(agent, "third")]
    assert agent._persistent_history_messages([SYSTEM, *history]) == []
    assert agent._context_reset_notice == NOTICE


def test_a_new_game_clears_the_state(monkeypatch, tmp_path):
    agent = _agent(monkeypatch, tmp_path)
    agent._levelup_since_trim = True
    agent._context_reset_notice = NOTICE
    agent._ensure_session(tmp_path / "other" / "state.json")
    assert not agent._levelup_since_trim
    assert agent._context_reset_notice == ""


def test_a_level_up_through_the_sandbox_marks_progress(monkeypatch, tmp_path):
    from inference.agent.runtime_state import Frame, HistoryEntry, discard_runtime_state, write_runtime_state

    agent = _agent(monkeypatch, tmp_path)
    path = tmp_path / "state.json"
    entries = [HistoryEntry("", Frame(grid=((0,) * 8,) * 8, step=0, level=1))]
    write_runtime_state(path, current_frame=entries[-1].frame, history=entries)

    def step_env(arguments):
        step = entries[-1].frame.step + 1
        completed = step == 2
        level = entries[-1].frame.level + int(completed)
        result = {"action_name": "ACTION4", "action_num": step, "level": level,
                  "level_completed": completed, "state": "NOT_FINISHED"}
        entries.append(HistoryEntry("RIGHT", Frame(grid=((step,) * 8,) * 8, step=step, level=level), result=result))
        write_runtime_state(path, current_frame=entries[-1].frame, history=entries)
        return {"executed": True, "executed_count": 1, "action_display": "RIGHT", "valid_actions": ["RIGHT"],
                "gameplay_changed": True, "board_changed": True, **result}

    agent._current_valid_actions = ["RIGHT"]
    agent._step_env_callback = step_env
    agent._run_python_tool(path, {"code": "action(['RIGHT'])"})
    assert not agent._levelup_since_trim
    agent._run_python_tool(path, {"code": "action(['RIGHT'])"})  # completes level 1
    assert agent._levelup_since_trim
    discard_runtime_state(path)


def _play(tmp_path, monkeypatch, turns, level_up_at=None):
    """Real analyze() turns against a stub model that acts once per turn.

    The estimate is 1000 tokens per message and the budget 7500, so the third
    turn's request (8 messages) is the first one that has to be trimmed.
    """
    import json
    from inference.agent.runtime_state import Frame, HistoryEntry, discard_runtime_state, write_runtime_state

    monkeypatch.setenv("MULTIMODAL_CONTEXT", "none")
    agent = _agent(monkeypatch, tmp_path)
    state = tmp_path / "artifacts" / "g_tool_runtime_state.json"
    state.parent.mkdir(parents=True)
    entries = [HistoryEntry("", Frame(grid=((0,) * 8,) * 8, step=0, level=1))]
    write_runtime_state(state, current_frame=entries[-1].frame, history=entries, flush=True)

    def step_env(arguments):
        step = entries[-1].frame.step + 1
        completed = step == level_up_at
        level = entries[-1].frame.level + int(completed)
        result = {"action_name": "ACTION4", "action_num": step, "level": level,
                  "level_completed": completed, "state": "NOT_FINISHED"}
        entries.append(HistoryEntry("RIGHT", Frame(grid=((step,) * 8,) * 8, step=step, level=level), result=result))
        write_runtime_state(state, current_frame=entries[-1].frame, history=entries, flush=True)
        return {"executed": True, "executed_count": 1, "action_display": "RIGHT", "valid_actions": ["RIGHT"],
                "gameplay_changed": True, "board_changed": True, **result}

    sent = []

    def fake_chat(messages, **kwargs):
        sent.append(json.loads(json.dumps(messages)))
        call = {"id": f"c{len(sent)}", "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": "action(['RIGHT'])"})}}
        return tool_agent._ChatCompletionResult(
            message={"role": "assistant", "content": "", "tool_calls": [call]}, finish_reason="tool_calls",
            usage={"prompt_tokens": 100, "completion_tokens": 5},
        )

    monkeypatch.setattr(agent, "_chat_completion", fake_chat)
    monkeypatch.setattr(agent, "_estimate_request_input_tokens", lambda messages, tools=None: 1000 * len(messages))
    agent._context_budget_tokens = 7500
    transcript = tmp_path / "transcripts" / "g.txt"
    transcript.parent.mkdir()
    for turn in range(1, turns + 1):
        agent.analyze(state, turn, valid_actions=["RIGHT"], step_env=step_env,
                      transcript_path=transcript, analysis_step=turn)
    discard_runtime_state(state)
    return sent, transcript.read_text()


def _first_user_text(request):
    content = request[1]["content"]
    return content if isinstance(content, str) else content[0]["text"]


def test_analyze_stuck_turn_three_sends_only_its_own_opener(tmp_path, monkeypatch):
    sent, transcript = _play(tmp_path, monkeypatch, turns=4)
    assert [len(r) for r in sent] == [2, 5, 2, 5]  # turn 3 reset; turn 4 sees turn 3 only
    assert _first_user_text(sent[2]).startswith(NOTICE)
    assert NOTICE not in json_dumps(sent[1]) and NOTICE in _first_user_text(sent[3])  # kept opener
    assert "context reset (request): no level-up since the previous trim; history messages 7 -> 1" in transcript


def test_analyze_level_up_defers_the_reset_by_one_trim(tmp_path, monkeypatch):
    sent, transcript = _play(tmp_path, monkeypatch, turns=4, level_up_at=2)
    assert [len(r) for r in sent] == [2, 5, 5, 2]  # turn 3 drains normally, turn 4 resets
    assert NOTICE not in json_dumps(sent[2])
    assert _first_user_text(sent[3]).startswith(NOTICE)
    assert transcript.count("context reset (") == 1


def json_dumps(value):
    import json
    return json.dumps(value)
