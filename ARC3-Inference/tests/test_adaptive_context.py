import pickle

import pytest

from inference.agent.adaptive_context import AdaptiveContext
from inference.agent import tool_agent


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("ARC3_ADAPTIVE_CONTEXT", "1")
    for name, value in (("NORMAL_LOW", 32), ("NORMAL_HIGH", 64),
                        ("EXPANDED_LOW", 96), ("EXPANDED_HIGH", 128)):
        monkeypatch.setenv(f"ARC3_CONTEXT_{name}", str(value))
    obj = tool_agent.ToolAgent(model="flashnext", provider="vllm")
    # Exact, small costs let the integration tests exercise whole exchanges.
    monkeypatch.setattr(obj, "_estimate_request_input_tokens",
                        lambda messages, **kwargs: sum(m.get("cost", 0) for m in messages))
    return obj


def history(blocks):
    result = [{"role": "system", "content": "rules", "cost": 4}]
    for i in range(blocks):
        result.extend([
            {"role": "user", "content": f"board {i}", "cost": 4},
            {"role": "assistant", "tool_calls": [{"id": str(i), "type": "function",
                "function": {"name": "python", "arguments": "{}"}}], "cost": 2},
            {"role": "tool", "tool_call_id": str(i), "content": "result", "cost": 2},
        ])
    return result


def assert_valid_history(messages):
    assert messages[1]["role"] == "user"
    for i, message in enumerate(messages):
        if message["role"] == "tool":
            assert messages[i - 1]["tool_calls"][0]["id"] == message["tool_call_id"]


def test_first_stall_expands_without_eviction_or_handover(agent, monkeypatch):
    messages = history(8)  # 68 tokens, beyond normal high water.
    trimmed = agent._trim_messages_for_context(messages)
    assert trimmed == messages
    assert agent._adaptive_context.expanded
    assert not agent._context_was_trimmed
    monkeypatch.setattr(agent, "_handover_yield_ucb", lambda *args: pytest.fail("unexpected yield"))
    agent._maybe_handover()


def test_recent_progress_keeps_small_range_and_whole_exchanges(agent):
    agent._adaptive_context.record_progress(1)
    trimmed = agent._trim_messages_for_context(history(8))
    assert 24 < agent._estimate_request_input_tokens(trimmed) <= 32
    assert not agent._adaptive_context.expanded
    assert agent._context_was_trimmed
    assert_valid_history(trimmed)


def test_expanded_stall_trims_to_96_and_yields_once(agent, monkeypatch):
    monkeypatch.setenv("ARC3_SCHEDULER", "ucb")
    seen = []
    monkeypatch.setattr(agent, "_handover_yield_ucb", lambda *args: seen.append(args))
    trimmed = agent._trim_messages_for_context(history(16))  # 132 -> 92
    assert 88 < agent._estimate_request_input_tokens(trimmed) <= 96
    assert agent._adaptive_context.expanded
    agent._maybe_handover()
    agent._maybe_handover()
    assert len(seen) == 1
    assert_valid_history(trimmed)


def test_expanded_success_waits_until_next_trim_even_after_another_stall(agent):
    messages = history(8)
    agent._trim_messages_for_context(messages)
    agent._last_step_summary = {"level": 2, "level_transition": True}
    agent._update_summarized_knowledge_from_step_summary()
    assert agent._adaptive_context.reset_pending
    # No immediate eviction or smaller ceiling after progress.
    assert agent._trim_messages_for_context(messages) == messages
    assert not agent._context_was_trimmed
    agent._adaptive_context.record_append(1000)
    trimmed = agent._trim_messages_for_context(history(16))
    assert agent._estimate_request_input_tokens(trimmed) <= 32
    assert not agent._adaptive_context.expanded
    assert not agent._adaptive_context.reset_pending
    assert agent._context_was_trimmed


def test_64k_lookback_survives_trimming_and_repeated_progress_reports(agent):
    policy = agent._adaptive_context
    policy.record_progress(1)
    policy.record_append(63)
    policy.record_progress(1)
    assert policy.last_progress_tokens == 0  # No credit for the same completion.
    agent._trim_messages_for_context(history(8))
    assert not policy.expanded
    policy.record_append(1)
    agent._trim_messages_for_context(history(8))
    assert policy.expanded


def test_counter_counts_appends_only_and_survives_pickle(agent):
    messages = []
    for message in history(2)[1:]:
        agent._append_context_message(messages, message)
    policy = agent._adaptive_context
    assert policy.appended_tokens == 16
    for _ in range(3):
        agent._estimate_request_input_tokens(messages)
    assert policy.appended_tokens == 16
    policy.expanded = True
    policy.record_progress(2)
    restored = pickle.loads(pickle.dumps(policy))
    assert restored == policy
    restored.record_append(1000)
    assert restored.limits(132, 140, force_drain=False) == (128, 32)


def test_counter_uses_real_reasoning_and_image_estimator(agent, monkeypatch):
    monkeypatch.delattr(agent, "_estimate_request_input_tokens")
    monkeypatch.setenv("ARC3_IMAGE_TOKENS_FLAT", "1000")
    message = {"role": "assistant", "reasoning_content": "investigate " * 100,
               "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}
    expected = agent._estimate_request_input_tokens([message]) - agent._estimate_request_input_tokens([])
    agent._append_context_message([], message)
    assert expected > 1000
    assert agent._adaptive_context.appended_tokens == expected


def test_assistant_turn_cap_cannot_preempt_adaptive_token_boundary(agent, monkeypatch):
    monkeypatch.setenv("ARC3_HISTORY_ASSISTANT_TURNS", "1")
    monkeypatch.setenv("ARC3_HISTORY_TURN_DRAIN", "1")
    messages = history(4)
    assert agent._persistent_history_messages(messages) == messages[1:]
    assert not agent._context_was_trimmed


@pytest.mark.parametrize("expanded,success,target", [(False, False, 32), (True, False, 96), (True, True, 32)])
def test_overflow_forces_drain_without_expanding(agent, expanded, success, target):
    policy = agent._adaptive_context
    policy.expanded = expanded
    if success:
        policy.record_progress(1)
    messages = history(14 if expanded else 7)  # Below the active ceiling.
    trimmed = agent._trim_messages_for_context(messages, force_drain=True, extra_safety_tokens=1)
    assert len(trimmed) < len(messages)
    assert agent._estimate_request_input_tokens(trimmed) <= target
    assert policy.expanded == (expanded and not success)


def test_session_reset_clears_adaptive_state(agent, tmp_path):
    agent._adaptive_context.expanded = True
    agent._adaptive_context.record_append(123)
    agent._ensure_session(tmp_path / "state.json")
    assert not agent._adaptive_context.expanded
    assert agent._adaptive_context.appended_tokens == 0


def test_agent_resume_preserves_pending_reset(agent, monkeypatch, tmp_path):
    monkeypatch.delattr(agent, "_estimate_request_input_tokens")
    state_path = tmp_path / "state.json"
    agent._ensure_session(state_path)
    agent._adaptive_context.expanded = True
    agent._adaptive_context.record_append(123)
    agent._adaptive_context.record_progress(2)
    resumed = pickle.loads(pickle.dumps(agent))
    resumed._ensure_session(state_path)
    assert resumed._adaptive_context == agent._adaptive_context
    assert resumed._adaptive_context.reset_pending


def test_legacy_trim_is_unchanged_when_disabled(agent, monkeypatch):
    monkeypatch.delenv("ARC3_ADAPTIVE_CONTEXT")
    del agent._adaptive_context
    agent._context_budget_tokens = 64
    monkeypatch.setenv("ARC3_CONTEXT_DRAIN_TOKENS", "16")
    trimmed = agent._trim_messages_for_context(history(8))
    assert 40 < agent._estimate_request_input_tokens(trimmed) <= 48
    assert agent._adaptive_context is None


def test_requires_output_headroom_and_ordered_thresholds(monkeypatch):
    monkeypatch.setenv("ARC3_ADAPTIVE_CONTEXT", "1")
    with pytest.raises(ValueError, match="after reserving output"):
        AdaptiveContext.from_env(131072 - 12288 - 512)
    assert AdaptiveContext.from_env(147456 - 12288 - 512).expanded_high == 131072
    monkeypatch.setenv("ARC3_CONTEXT_NORMAL_LOW", "65536")
    with pytest.raises(ValueError, match="normal_low"):
        AdaptiveContext.from_env(140000)
