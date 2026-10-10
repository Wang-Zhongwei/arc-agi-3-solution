"""Early-level reminders: budget, lifecycle, and actual request delivery."""

import json

import pytest
import requests

from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state


THRESHOLD = 32768
MARKER = "Progress check:"


def frame(level=1, step=0):
    return Frame(grid=((0, 0), (0, 1)), step=step, level=level)


@pytest.fixture
def setup_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC3_EARLY_LEVEL_RECOVERY_TOKENS", str(THRESHOLD))
    monkeypatch.setenv("ARC3_SUMMARY_INTERVAL_TOKENS", "0")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setenv("ARC3_PRUNE_CONTROL_CONTEXT", "1")
    agent = tool_agent.ToolAgent(model="test-model", provider="vllm", base_url="http://unused/v1")
    agent._tool_steps = 1
    agent._yield_seconds = None
    agent._yield_tokens = None
    state = tmp_path / "state.json"
    write_runtime_state(state, current_frame=frame(), history=[HistoryEntry("INIT", frame())])
    agent._ensure_session(state)
    sent = []

    def completion(messages, **kwargs):
        sent.append(json.loads(json.dumps(messages)))
        return tool_agent._ChatCompletionResult(
            message={"content": "I will compare the observed transitions."},
            usage={"prompt_tokens": 100000, "completion_tokens": 100},
            finish_reason="stop",
        )

    monkeypatch.setattr(agent, "_chat_completion", completion)
    return agent, state, sent


def analyze(agent, state, **kwargs):
    return agent.analyze(state_path=state, action_num=0, valid_actions=["UP"], **kwargs)


def opener(messages):
    return tool_agent._message_display_text(messages[-1])


@pytest.mark.parametrize("setting", [None, "0", "-1", "invalid", "1000001"])
def test_disabled_or_below_threshold_leaves_prompts_identical(setup_agent, monkeypatch, setting):
    agent, _, _ = setup_agent
    agent._session_generated_tokens = 1000000
    monkeypatch.delenv("ARC3_EARLY_LEVEL_RECOVERY_TOKENS")
    baseline = agent._build_user_prompt(0, valid_actions=["UP"], current_frame=frame())
    system = agent._system_prompt
    if setting is not None:
        monkeypatch.setenv("ARC3_EARLY_LEVEL_RECOVERY_TOKENS", setting)
    assert agent._build_user_prompt(0, valid_actions=["UP"], current_frame=frame()) == baseline
    other = tool_agent.ToolAgent(model="test-model", provider="vllm", base_url="http://unused/v1")
    assert other._system_prompt == system


@pytest.mark.parametrize("level,tokens,expected", [
    (1, THRESHOLD - 1, False), (1, THRESHOLD, True),
    (2, THRESHOLD, True), (3, THRESHOLD * 10, False), (0, THRESHOLD, False),
])
def test_threshold_and_early_levels_only(setup_agent, level, tokens, expected):
    agent, _, _ = setup_agent
    agent._session_generated_tokens = tokens
    assert bool(agent._early_level_recovery_prompt(frame(level))) is expected
    assert agent._early_level_recovery_shown_levels == set()  # rendering is pure


def test_generated_budget_excludes_inputs_and_includes_summary_output(setup_agent):
    agent, _, _ = setup_agent
    agent._accumulate_usage_tokens({"prompt_tokens": 900000, "total_tokens": 900000})
    agent._accumulate_usage_tokens(None)
    assert not agent._early_level_recovery_prompt(frame())
    agent._accumulate_usage_tokens({
        "completion_tokens": THRESHOLD - 100,
        "completion_tokens_details": {"reasoning_tokens": THRESHOLD - 200},
    })
    assert not agent._early_level_recovery_prompt(frame())
    agent._accumulate_usage_tokens({"output_tokens": 100}, count_toward_turn=False)
    assert MARKER in agent._early_level_recovery_prompt(frame())


@pytest.mark.parametrize("event", [{"game_over": True}, {"manual_reset": True}, {}])
def test_death_reset_or_inspection_does_not_restart_budget(setup_agent, event):
    agent, _, _ = setup_agent
    agent._session_generated_tokens = THRESHOLD
    agent._last_step_summary = {"level": 1, **event}
    agent._update_summarized_knowledge_from_step_summary()
    assert MARKER in agent._early_level_recovery_prompt(frame())


def test_level_up_and_new_game_restart_budget(setup_agent):
    agent, state, _ = setup_agent
    agent._session_generated_tokens = THRESHOLD + 100
    agent._early_level_recovery_shown_levels.add(1)
    agent._last_step_summary = {"level_transition": True, "level": 2}
    agent._update_summarized_knowledge_from_step_summary()
    assert not agent._early_level_recovery_prompt(frame(2))
    agent._accumulate_usage_tokens({"generated_tokens": THRESHOLD})
    assert MARKER in agent._early_level_recovery_prompt(frame(2))
    assert not agent._early_level_recovery_prompt(frame(1))
    agent._ensure_session(state.parent / "another_game" / "state.json")
    assert agent._early_level_recovery_shown_levels == set()
    assert not agent._early_level_recovery_prompt(frame())
    agent._accumulate_usage_tokens({"completion_tokens": THRESHOLD})
    assert MARKER in agent._early_level_recovery_prompt(frame())


def test_missing_frame_or_finished_run_gets_no_reminder(setup_agent):
    agent, _, _ = setup_agent
    agent._session_generated_tokens = THRESHOLD
    assert not agent._early_level_recovery_prompt(None)
    agent._last_step_summary = {"run_complete": True}
    assert not agent._early_level_recovery_prompt(frame())


@pytest.mark.parametrize("mode", ["full", "short", "state_only"])
def test_reminder_is_sent_once_in_every_opener_mode(setup_agent, monkeypatch, mode):
    agent, state, sent = setup_agent
    monkeypatch.setenv("ARC3_YIELD_RESUME_PROMPT", mode)
    agent._session_generated_tokens = THRESHOLD
    agent._resume_after_yield = mode != "full"
    assert analyze(agent, state) is not None
    assert opener(sent[0]).count(MARKER) == 1
    assert "32,768 generated tokens on level 1" in opener(sent[0])
    assert agent._early_level_recovery_shown_levels == {1}
    agent._resume_after_yield = mode != "full"
    assert analyze(agent, state) is not None
    assert MARKER not in opener(sent[1])
    # Context eviction does not renew the one-shot allowance.
    agent._history_messages = []
    assert analyze(agent, state) is not None
    assert MARKER not in opener(sent[2])


def test_crossing_threshold_on_yield_is_seen_at_next_opener(setup_agent, monkeypatch):
    agent, state, sent = setup_agent
    monkeypatch.setenv("ARC3_YIELD_RESUME_PROMPT", "short")
    agent._session_generated_tokens = THRESHOLD - 50
    agent._tool_steps = None
    agent._yield_tokens = 100
    assert analyze(agent, state) is not None
    assert MARKER not in opener(sent[0])
    assert agent._early_level_recovery_shown_levels == set()
    assert agent._resume_after_yield
    assert analyze(agent, state) is not None
    assert MARKER in opener(sent[1])


def test_failed_request_does_not_consume_reminder(setup_agent, monkeypatch):
    agent, state, sent = setup_agent
    agent._session_generated_tokens = THRESHOLD
    completion = agent._chat_completion

    def fail(messages, **kwargs):
        raise requests.ConnectionError("test endpoint unavailable")

    monkeypatch.setattr(agent, "_chat_completion", fail)
    analyze(agent, state)
    assert agent._early_level_recovery_shown_levels == set()
    assert agent._history_messages == []
    monkeypatch.setattr(agent, "_chat_completion", completion)
    assert analyze(agent, state) is not None
    assert MARKER in opener(sent[0])


def test_stop_before_request_does_not_consume_reminder(setup_agent):
    agent, state, sent = setup_agent
    agent._session_generated_tokens = THRESHOLD
    analyze(agent, state, should_stop=lambda: True)
    assert sent == []
    assert agent._early_level_recovery_shown_levels == set()
    analyze(agent, state)
    assert MARKER in opener(sent[0])


def test_unused_actions_use_execution_history_across_death_and_level_up(setup_agent, monkeypatch):
    agent, _, _ = setup_agent
    monkeypatch.setenv("EXPOSE_UNDO", "on")
    agent._session_generated_tokens = THRESHOLD
    history = [
        HistoryEntry("INIT", frame(1, 0)),
        HistoryEntry("ACTION1", frame(1, 1)),
        HistoryEntry("MOUSE(row=1, col=1)", frame(1, 2)),
        HistoryEntry("RESET", frame(1, 3), {"automatic": True}),
        HistoryEntry("RIGHT", frame(2, 4)),  # completing level 1, not tried on 2
        HistoryEntry("LEFT", frame(2, 5)),
    ]
    actions = ["UP", "LEFT", "RIGHT", "SPACE", "MOUSE", "UNDO", "RESET"]
    text = agent._early_level_recovery_prompt(frame(1), valid_actions=actions, history_entries=history)
    assert "Action types not yet tried on this level: LEFT, SPACE, UNDO." in text
    text = agent._early_level_recovery_prompt(frame(2), valid_actions=actions, history_entries=history)
    assert "Action types not yet tried on this level: UP, RIGHT, SPACE, MOUSE, UNDO." in text
    assert "does not cover every target or interaction" in text


@pytest.mark.parametrize("mode", ["full", "short", "state_only"])
def test_unused_actions_reach_every_opener_without_counting_skipped_actions(setup_agent, monkeypatch, mode):
    agent, state, sent = setup_agent
    monkeypatch.setenv("ARC3_YIELD_RESUME_PROMPT", mode)
    agent._session_generated_tokens = THRESHOLD
    history = [HistoryEntry("INIT", frame()), HistoryEntry("UP", frame(step=1))]
    write_runtime_state(state, current_frame=frame(step=1), history=history)
    agent._last_step_summary = {"executed_actions": ["UP"], "skipped_actions": ["SPACE"]}
    agent._resume_after_yield = mode != "full"
    agent.analyze(state_path=state, action_num=1, valid_actions=["UP", "SPACE"])
    assert "Action types not yet tried on this level: SPACE." in opener(sent[0])
    assert "Action types not yet tried on this level: UP" not in opener(sent[0])
    # Model context eviction has no effect on the runtime execution history.
    agent._history_messages = []
    text = agent._early_level_recovery_prompt(frame(), valid_actions=["UP"], history_entries=history)
    assert not text  # successful delivery is still one-shot


def test_all_actions_tried_does_not_claim_all_interactions_explored(setup_agent):
    agent, _, _ = setup_agent
    agent._session_generated_tokens = THRESHOLD
    history = [HistoryEntry("INIT", frame()), HistoryEntry("MOUSE(row=0, col=0)", frame(step=1))]
    text = agent._early_level_recovery_prompt(frame(), valid_actions=["MOUSE"], history_entries=history)
    assert "All currently valid action types have been tried" in text
    assert "does not cover every target or interaction" in text
