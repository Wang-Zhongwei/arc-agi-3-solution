"""ARC3_ACTION_ALIASES: U/D/L/R accepted as the four arrows (arm/stall-fix-tool).

Run U lost ~73 min on m0r0 L3 to a BFS executor that called action(['L']):
the host refused the name and nothing ran. With the flag on the four letters
resolve to UP/DOWN/LEFT/RIGHT; nothing advertises them and S stays unknown.
"""
from __future__ import annotations

import pytest

from inference.agent import action_names
from inference.agent.action_names import to_engine_action, to_model_action, to_model_actions


@pytest.fixture
def aliases(monkeypatch):
    monkeypatch.setenv("ARC3_ACTION_ALIASES", "1")
    return monkeypatch


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("ARC3_ACTION_ALIASES", raising=False)
    for letter in ("U", "D", "L", "R", "l", " r "):
        assert to_engine_action(letter) is None
    monkeypatch.setenv("ARC3_ACTION_ALIASES", "0")
    assert to_engine_action("L") is None


def test_letters_resolve_to_the_arrows(aliases):
    assert [to_engine_action(x) for x in ("U", "D", "L", "R")] == [
        "ACTION1", "ACTION2", "ACTION3", "ACTION4",
    ]
    assert to_engine_action(" l ") == "ACTION3"
    assert to_engine_action("r") == "ACTION4"


def test_nothing_else_is_aliased(aliases):
    for name in ("S", "s", "M", "X", "LL", "UR", "", None, "Z"):
        assert to_engine_action(name) is None
    # the full names and engine names are unchanged
    assert to_engine_action("LEFT") == "ACTION3"
    assert to_engine_action("action6") == "ACTION6"
    assert to_engine_action("RESET") == "RESET"


def test_advertised_names_do_not_change(aliases):
    # model-facing names come from to_model_action(s), which the flag leaves alone
    engine = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "RESET"]
    assert to_model_actions(engine) == ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE", "RESET"]
    assert to_model_action("L") == "L"
    assert set(action_names.MODEL_TO_ENGINE_ACTION) == {
        "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE", "RESET",
    }


def test_valid_action_line_is_unchanged(aliases):
    from inference.agent import tool_agent as ta

    names = ["RESET", "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]
    assert ta._normalize_valid_actions(names) == names
    assert ta._format_valid_action_line(names) == "RESET, UP, DOWN, LEFT, RIGHT, SPACE, MOUSE"


def test_solver_executes_an_aliased_letter_under_its_full_name(tmp_path, aliases):
    from test_animation_layers import _harness, _python

    harness = _harness(tmp_path, aliases)
    out = _python(harness, "r = action(['r']); print(r['executed'], r['executed_actions'])")
    assert out.strip() == "True ['RIGHT']"
    session, _ = harness
    assert [entry.action for entry in session.history_entries[1:]] == ["RIGHT"]
