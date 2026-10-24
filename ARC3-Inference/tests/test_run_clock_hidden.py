"""The run clock is not shown to the model.

`_compact_action_result` builds what `action(...)` returns and what
`last_action_call_result` holds. It leaves out run_elapsed_seconds and
time_remaining_seconds, and still reads them for the harness's own deadline.
"""

import time
from types import SimpleNamespace

from inference.agent.tool_agent import ToolAgent


def test_compact_result_hides_the_clock_but_the_harness_still_reads_it():
    agent = SimpleNamespace(_game_elapsed_seconds=None, _game_deadline=None)
    payload = {
        "executed": True,
        "action_num": 3,
        "level": 1,
        "run_elapsed_seconds": 1013.4,
        "time_remaining_seconds": 30906.6,
    }

    before = time.monotonic()
    compact = ToolAgent._compact_action_result(agent, payload)

    assert "run_elapsed_seconds" not in compact
    assert "time_remaining_seconds" not in compact
    assert compact["executed"] is True and compact["action_num"] == 3
    assert agent._game_elapsed_seconds == 1013.4
    assert before + 30906.6 <= agent._game_deadline <= time.monotonic() + 30906.6
