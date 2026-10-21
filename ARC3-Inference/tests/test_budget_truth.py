"""ARC3_BUDGET_TRUTH: the run clock hidden from the model, truthful bar wording."""
from __future__ import annotations

import pytest

from inference.agent import prompts
from inference.agent import tool_agent as ta
from notebook_env import notebook_env

PAYLOAD = {
    "executed": True, "action_num": 3, "level": 1, "score": 0, "state": "NOT_FINISHED",
    "valid_actions": ["UP"], "board_changed": True, "gameplay_changed": False, "done": False,
    "level_completed": False, "game_over": False, "run_complete": False, "action_display": "UP",
    "run_elapsed_seconds": 1234.5, "time_remaining_seconds": 5965.5,
}


@pytest.fixture
def kaggle_env(monkeypatch):
    for key, value in notebook_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("ARC3_BUDGET_TRUTH", raising=False)
    return monkeypatch


def system_prompt() -> str:
    return ta._build_system_prompt(tool_output_tokens=3072)


def tools() -> list[dict]:
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    agent._ensure_session = lambda state_path: None
    return agent._tools(None)


def test_every_replacement_matches_the_kaggle_prompt_once(kaggle_env):
    """Guards against drift: a reworded source line would make the flag inert."""
    prompt = system_prompt()
    for old, _ in prompts.BUDGET_TRUTH_BAR_REPLACEMENTS + prompts.BUDGET_TRUTH_PERSISTENT_REPLACEMENTS:
        assert prompt.count(old) == 1, old
    assert ta._EPHEMERAL_TOOL_DESCRIPTION_OLD in ta._PYTHON_TOOL_DESCRIPTION


def test_flag_off_leaves_prompt_and_tools_alone(kaggle_env):
    off_prompt, off_tools = system_prompt(), tools()
    kaggle_env.setenv("ARC3_BUDGET_TRUTH", "0")
    assert system_prompt() == off_prompt
    assert tools() == off_tools
    function = off_tools[0]["function"]
    assert function["description"] == ta._PYTHON_TOOL_DESCRIPTION
    assert function["parameters"]["properties"]["code"]["description"] == (
        "Python code to run. The snippet is ephemeral and is not saved across tool calls.")


def test_flag_on_system_prompt(kaggle_env):
    off_prompt = system_prompt()
    kaggle_env.setenv("ARC3_BUDGET_TRUTH", "1")
    on_prompt = system_prompt()
    for phrase in ("do not get distracted by it", "DON'T DO THIS!",
                   "HUD elements such as countdown bars", "HUD/timer state", "ephemeral"):
        assert phrase in off_prompt
        assert phrase not in on_prompt
    assert "is a step bar. It usually counts down a per-level action budget" in on_prompt
    assert "only HUD elements.\n" in on_prompt
    assert "classify it as HUD state, not as an object to click through segment by segment.\n" in on_prompt
    assert len(on_prompt) < len(off_prompt)
    # nothing else moved: undoing the substitutions gives the default prompt back
    undone = on_prompt
    for old, new in prompts.BUDGET_TRUTH_BAR_REPLACEMENTS + prompts.BUDGET_TRUTH_PERSISTENT_REPLACEMENTS:
        undone = undone.replace(new, old)
    assert undone == off_prompt


def test_flag_on_tools_drop_ephemeral_only_with_persistent_functions(kaggle_env):
    kaggle_env.setenv("ARC3_BUDGET_TRUTH", "1")
    function = tools()[0]["function"]
    assert "ephemeral" not in function["description"]
    assert function["description"].startswith("Run one Python snippet against preloaded")
    assert function["parameters"]["properties"]["code"]["description"] == "Python code to run."

    kaggle_env.setenv("ARC3_PERSISTENT_FUNCTIONS", "0")
    function = tools()[0]["function"]
    assert function["description"] == ta._PYTHON_TOOL_DESCRIPTION
    assert "ephemeral" in function["parameters"]["properties"]["code"]["description"]
    prompt = system_prompt()
    assert "call it with one ephemeral `code` string" in prompt  # true without retention
    assert "do not get distracted by it" not in prompt  # the bar wording does not depend on it


def test_compact_result_hides_clock_but_keeps_elapsed(kaggle_env):
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    agent._game_elapsed_seconds = None
    off = agent._compact_action_result(dict(PAYLOAD))
    assert off["run_elapsed_seconds"] == 1234.5 and off["time_remaining_seconds"] == 5965.5

    kaggle_env.setenv("ARC3_BUDGET_TRUTH", "1")
    agent._game_elapsed_seconds = None
    on = agent._compact_action_result(dict(PAYLOAD))
    assert "run_elapsed_seconds" not in on and "time_remaining_seconds" not in on
    assert {k: v for k, v in off.items() if k not in ta._RUN_CLOCK_KEYS} == on
    assert agent._game_elapsed_seconds == 1234.5  # the endgame phase still reads it


def test_endgame_phase_still_sees_the_clock(kaggle_env):
    kaggle_env.setenv("ARC3_BUDGET_TRUTH", "1")
    kaggle_env.setenv("ARC3_ENDGAME_START_MINUTES", "20")
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    agent._game_elapsed_seconds = None
    agent._compact_action_result({**PAYLOAD, "run_elapsed_seconds": 1300.0})
    assert agent._in_endgame()
