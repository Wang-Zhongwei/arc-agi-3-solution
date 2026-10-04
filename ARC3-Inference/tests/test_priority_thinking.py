import pytest

from inference.agent.scheduling.priority import PrioritySnapshot, priority_value, thinking_multiplier
from inference.agent.tool_agent import (
    _priority_refresh_enabled, _priority_thinking_reference, _usage_reasoning_tokens,
)


def test_multiplier_shape():
    assert thinking_multiplier(1024, 1024) == 1.0
    assert thinking_multiplier(256, 1024) == pytest.approx(2.0)
    assert thinking_multiplier(4096, 1024) == pytest.approx(0.5)
    assert thinking_multiplier(512, 1024) == pytest.approx(2 ** 0.5)
    assert thinking_multiplier(0, 1024) == 2.0         # clipped
    assert thinking_multiplier(100_000, 1024) == 0.5   # clipped


def test_multiplier_off_or_unknown():
    assert thinking_multiplier(None, 1024) == 1.0
    assert thinking_multiplier(float("nan"), 1024) == 1.0
    assert thinking_multiplier(100, 0.0) == 1.0


def test_priority_value_scales_both_phases():
    base = PrioritySnapshot(3, 40, 30000.0)
    short = PrioritySnapshot(3, 40, 30000.0, thinking_tokens=256)
    long = PrioritySnapshot(3, 40, 30000.0, thinking_tokens=4096)
    for endgame in (False, True):
        v = priority_value(base, endgame=endgame, thinking_reference=1024)
        assert priority_value(short, endgame=endgame, thinking_reference=1024) == pytest.approx(2 * v, abs=2)
        assert priority_value(long, endgame=endgame, thinking_reference=1024) == pytest.approx(v / 2, abs=2)
        # off by default, even when the snapshot carries thinking tokens
        assert priority_value(short, endgame=endgame) == v


def test_usage_reasoning_tokens():
    assert _usage_reasoning_tokens({"reasoning_tokens": 812}) == 812
    assert _usage_reasoning_tokens({"completion_tokens_details": {"reasoning_tokens": 7}}) == 7
    assert _usage_reasoning_tokens({"completion_tokens": 50}) is None


def test_env_flag(monkeypatch):
    for name in ("ARC3_PRIORITY_REFRESH_QUEUE", "ARC3_PRIORITY_PACE", "ARC3_PRIORITY_TAIL_FADE",
                 "ARC3_SCHEDULER", "ARC3_PRIORITY_THINKING_REF"):
        monkeypatch.delenv(name, raising=False)
    assert _priority_thinking_reference() == 0.0
    assert not _priority_refresh_enabled()
    monkeypatch.setenv("ARC3_PRIORITY_THINKING_REF", "1024")
    assert _priority_thinking_reference() == 1024.0
    assert _priority_refresh_enabled()
    monkeypatch.setenv("ARC3_PRIORITY_THINKING_REF", "-1")
    with pytest.raises(ValueError):
        _priority_thinking_reference()
