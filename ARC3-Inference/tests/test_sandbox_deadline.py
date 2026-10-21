"""The python tool's deadline and what survives a kill, through the real sandbox process (#4)."""
from __future__ import annotations

import re
import time

from inference.agent.python_tool_sandbox import (
    FORWARDED_HEAD_CHARS,
    FORWARDED_TAIL_CHARS,
    _ForwardedOutput,
    run_sandboxed_python,
)


def _action_handler(*, sleep=0.0, calls=None):
    def handler(actions, stale_after=None):
        if calls is not None:
            calls.append(list(actions))
        if sleep:
            time.sleep(sleep)
        return {
            "action_result": {
                "executed": True,
                "executed_count": len(actions),
                "executed_actions": [item["action"] for item in actions],
                "gameplay_changed": True,
            },
            "state": {},
        }

    return handler


def _run(code, *, timeout_seconds=2, **kwargs):
    kwargs.setdefault("action_handler", _action_handler())
    return run_sandboxed_python(
        code=code,
        timeout_seconds=timeout_seconds,
        initial_state={},
        **kwargs,
    )


def test_normal_result_is_unchanged():
    result = _run(
        'print("before")\n'
        'action("UP")\n'
        'action(["LEFT", "RIGHT"])\n'
        'print("after")\n'
        '1 + 1\n'
    )
    assert result == {
        "stdout": "before\nafter\n2\n",
        "result": 2,
        "error": "",
        "action_results": [
            {"executed": True, "executed_count": 1, "executed_actions": ["UP"],
             "gameplay_changed": True},
            {"executed": True, "executed_count": 2, "executed_actions": ["LEFT", "RIGHT"],
             "gameplay_changed": True},
        ],
        "keepable_functions": [],
    }


def test_error_result_is_unchanged():
    result = _run('print("x")\naction("UP")\nraise ValueError("bad")\n')
    assert result["stdout"] == "x\n"
    assert result["error"].endswith("ValueError: bad")
    assert len(result["action_results"]) == 1
    assert "keepable_functions" in result


def test_timeout_keeps_output_and_reports_executed_actions():
    # prints before, between and after the actions, then never finishes: all
    # of it comes back, and the error says the actions were already spent
    result = _run(
        'print("before")\n'
        'action("UP")\n'
        'action(["LEFT", "RIGHT"])\n'
        'print("after")\n'
        'while True:\n'
        '    pass\n'
    )
    assert result["error"] == "Tool timed out after 2s (3 actions already executed)"
    assert result["stdout"] == "before\nafter\n"
    assert [item["executed_count"] for item in result["action_results"]] == [1, 2]
    # a call that never answered reports no retention outcome
    assert "keepable_functions" not in result


def test_timeout_without_actions_keeps_the_plain_message():
    result = _run('print("thinking")\nwhile True:\n    pass\n')
    assert result["error"] == "Tool timed out after 2s"
    assert result["stdout"] == "thinking\n"
    assert result["action_results"] == []


def test_crash_keeps_output_forwarded_before_it():
    # BaseException is not in the sandbox builtins, and nothing in the child
    # catches it, so this takes the process down like a real crash would
    result = _run(
        'print("x")\n'
        'action("UP")\n'
        'raise Exception.__mro__[1]("boom")\n'
    )
    assert result["error"] == "Sandbox process exited unexpectedly (1 action already executed)."
    assert result["stdout"] == "x\n"
    assert len(result["action_results"]) == 1


def test_host_action_and_animation_time_is_not_charged():
    # 8 x (0.15 s action + 0.15 s animation) = 2.4 s of host time against a
    # 2 s budget: under wall-clock billing this snippet timed out
    calls = []

    def animation_handler(request):
        time.sleep(0.15)
        return []

    result = _run(
        'for _ in range(8):\n'
        '    action("UP")\n'
        '    len(last_animation_frames)\n'
        'print("done")\n',
        timeout_seconds=2,
        wall_cap_seconds=20,
        action_handler=_action_handler(sleep=0.15, calls=calls),
        animation_handler=animation_handler,
    )
    assert result["error"] == ""
    assert result["stdout"] == "done\n"
    assert len(calls) == 8


def test_snippet_time_still_runs_out_after_host_time():
    # the deadline is pushed back by host time, not removed
    started = time.monotonic()
    result = _run(
        'action("UP")\nwhile True:\n    pass\n',
        timeout_seconds=1,
        action_handler=_action_handler(sleep=0.5),
    )
    elapsed = time.monotonic() - started
    assert result["error"] == "Tool timed out after 1s (1 action already executed)"
    assert elapsed >= 1.5


def test_wall_cap_ends_a_runaway_action_loop():
    # almost all of this loop's time is host time, so the snippet's own 1 s
    # budget never runs out; only the default cap (3 x the timeout) stops it
    calls = []
    started = time.monotonic()
    result = _run(
        'print("looping")\nwhile True:\n    action("UP")\n',
        timeout_seconds=1,
        action_handler=_action_handler(sleep=0.2, calls=calls),
    )
    elapsed = time.monotonic() - started
    assert 3 <= elapsed < 10
    count = len(result["action_results"])
    assert count == len(calls) >= 5
    assert result["error"].startswith("Tool timed out after 3s in total, ")
    assert result["error"].endswith(f"s of it executing actions ({count} actions already executed)")
    assert result["stdout"] == "looping\n"


def test_line_printed_before_a_call_that_holds_the_interpreter_survives():
    # the pump thread cannot run during this match, so only a line forwarded
    # when it was printed reaches the host before the kill
    result = _run(
        'print("trying regex")\n'
        'import re\n'
        're.match(r"(a+)+$", "a" * 40 + "b")\n',
        timeout_seconds=1,
    )
    assert result["error"] == "Tool timed out after 1s"
    assert result["stdout"] == "trying regex\n"


def test_line_printed_after_an_action_survives_a_call_that_holds_the_interpreter():
    result = _run(
        'action("UP")\nprint("after action")\nsum(range(10 ** 12))\n',
        timeout_seconds=1,
    )
    assert result["error"] == "Tool timed out after 1s (1 action already executed)"
    assert result["stdout"] == "after action\n"


def test_killed_print_flood_keeps_only_its_first_and_latest_output():
    result = _run(
        'i = 0\nwhile True:\n    print(f"line {i} " + "x" * 80)\n    i += 1\n',
        timeout_seconds=1,
    )
    assert result["error"] == "Tool timed out after 1s"
    stdout = result["stdout"]
    assert stdout.startswith("line 0 xxx")
    head, marker, tail = stdout.partition(" chars omitted] ...\n")
    assert marker
    assert len(head) < FORWARDED_HEAD_CHARS + 30
    assert 0 < len(tail) <= FORWARDED_TAIL_CHARS
    # the tail is the end of the flood, far past where the head stopped
    latest = max(int(number) for number in re.findall(r"line (\d+) ", tail))
    assert latest > 2 * FORWARDED_HEAD_CHARS // 90


def test_finished_call_still_reports_all_of_a_long_output():
    result = _run('for i in range(5000):\n    print("x" * 80)\n', timeout_seconds=5)
    assert result["error"] == ""
    assert result["stdout"] == ("x" * 80 + "\n") * 5000


def test_invalid_reply_keeps_output_and_reports_executed_actions():
    # the snippet writes past the protocol straight to the host's pipe
    result = _run(
        'print("x")\n'
        'action("UP")\n'
        'action.__globals__["HOST_STDOUT"].write("not json\\n")\n'
        'action.__globals__["HOST_STDOUT"].flush()\n'
        'while True:\n'
        '    pass\n'
    )
    assert result["error"] == (
        "Sandbox process returned an invalid response (1 action already executed)."
    )
    assert result["stdout"] == "x\n"
    assert "keepable_functions" not in result


def test_unknown_reply_keeps_output():
    result = _run(
        'print("x")\n'
        'action.__globals__["_send"]({"type": "bogus"})\n'
        'while True:\n'
        '    pass\n'
    )
    assert result["error"] == "Sandbox process returned an unknown message type."
    assert result["stdout"] == "x\n"


def test_forwarded_output_is_whole_until_it_outgrows_head_and_tail():
    output = _ForwardedOutput()
    output.add("a" * FORWARDED_HEAD_CHARS)
    output.add("b" * FORWARDED_TAIL_CHARS)
    assert output.text() == "a" * FORWARDED_HEAD_CHARS + "b" * FORWARDED_TAIL_CHARS
    output.add("c" * 10)
    assert output.text() == (
        "a" * FORWARDED_HEAD_CHARS
        + "\n... [10 chars omitted] ...\n"
        + "b" * (FORWARDED_TAIL_CHARS - 10)
        + "c" * 10
    )


def test_forwarded_output_starts_a_new_tail_after_a_child_skip():
    output = _ForwardedOutput()
    output.add("a" * FORWARDED_HEAD_CHARS)
    output.add("b" * 100)
    # the child left 50 characters out between the b's and the c's
    output.add("c" * 20, skipped=50)
    assert output.text() == "a" * FORWARDED_HEAD_CHARS + "\n... [150 chars omitted] ...\n" + "c" * 20
