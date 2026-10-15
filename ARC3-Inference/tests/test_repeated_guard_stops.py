"""A refusal collected on every action() call of a turn is reported once, with its count."""

from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry


def test_identical_refusals_become_one_note_with_a_count(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    executed = {
        "executed": True, "executed_count": 3, "action_num": 40, "action_display": "MOUSE(row=58, col=47)",
        "executed_actions": ["RESET", "MOUSE(row=36, col=39)", "MOUSE(row=58, col=47)"],
    }
    refusal = {
        "executed": False, "stop_reason": "stale_state", "blocked_action": "MOUSE(row=33, col=39)",
        "skipped_actions": ["MOUSE(row=33, col=39)"],
    }
    summary = agent._summarize_step_sequence([executed, dict(refusal), dict(refusal), dict(refusal)])
    assert len(summary["guard_stops"]) == 1 and summary["guard_stops"][0]["count"] == 3
    current = Frame(grid=tuple((0,) * 64 for _ in range(64)), step=40, level=2)
    entries = [HistoryEntry("INIT", Frame(grid=current.grid, step=37, level=2)), HistoryEntry("MOUSE", current)]
    agent._auto_diff_prev_frame = entries[0].frame
    prompt = agent._build_user_prompt(current.step, valid_actions=["MOUSE"], current_frame=current,
                                      history_entries=entries, previous_step_summary=summary)
    assert prompt.count("the sequence was stopped before 'MOUSE(row=33, col=39)'") == 1
    assert "This stop was hit 3 times in that turn." in prompt


def test_distinct_stops_stay_separate():
    stops = [{"kind": "stale_state", "action": "A", "skipped": ["A"]},
             {"kind": "stale_state", "action": "B", "skipped": ["B"]},
             {"kind": "stale_state", "action": "A", "skipped": ["A"]}]
    merged = tool_agent._merge_repeated_stops(stops)
    assert [(s["action"], s["count"]) for s in merged] == [("A", 2), ("B", 1)]
