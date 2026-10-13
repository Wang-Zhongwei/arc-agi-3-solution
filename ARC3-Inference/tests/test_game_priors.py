from inference.agent import tool_agent
from inference.agent.prompts import GAME_PRIORS_ADDENDUM


def test_game_priors_off_by_default(monkeypatch):
    monkeypatch.delenv("ARC3_GAME_PRIORS", raising=False)
    assert "Priors and probing" not in tool_agent._build_system_prompt(tool_output_tokens=3072)


def test_game_priors_is_the_last_section(monkeypatch):
    monkeypatch.delenv("ARC3_GAME_PRIORS", raising=False)
    base = tool_agent._build_system_prompt(tool_output_tokens=3072)
    monkeypatch.setenv("ARC3_GAME_PRIORS", "1")
    assert tool_agent._build_system_prompt(tool_output_tokens=3072) == base + GAME_PRIORS_ADDENDUM
