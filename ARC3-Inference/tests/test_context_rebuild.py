"""context_rebuild turns a transcript back into the messages of each request.

The window is the suffix of the message stream that the logged
`history_messages` count keeps; a turn that keeps exactly its starting count
was rolled back; a trim inside a turn starts at the first request over budget.
"""

import json

from inference.rft.context_rebuild import rebuild


def meta(*calls):
    lines = ["finish_reason: tool_calls", "served_by: (unknown)", f"tool_call_count: {len(calls)}"]
    if calls:
        lines += ["raw_tool_calls:", json.dumps([
            {"id": i, "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": c})}}
            for i, c in calls], indent=2)]
    return "\n".join(lines)


def section(label, text):
    return f"[{label}]\n{text.strip()}\n\n"


def step(n, user, replies, kept, budget=100000, extra=""):
    body = section("SYSTEM PROMPT", "system") + section("USER PROMPT", user)
    for i, (thinking, code, result) in enumerate(replies):
        call_id = f"c{n}-{i}"
        body += section("MODEL RESPONSE META", meta((call_id, code)))
        body += section("THINKING", thinking)
        body += section("TOOL CALL: python", f"<tool_call>\n<function=python>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>")
        body += section("TOOL RESULT: python", result)
    body += extra
    body += section("ANALYZER STATUS", f"context_budget_tokens: {budget}\nhistory_messages: {kept}")
    return f"--- analysis_step={n} | action={n} | 00:00:00 | tool-agent ---\n" + body


def roles(request):
    return [m["role"] for m in request.messages]


def test_requests_carry_history_and_turn_messages():
    game = rebuild(step(1, "frame 1\n[image attached]", [("look", "print(1)", "1")], kept=3)
                   + step(2, "frame 2", [("act", "action(['UP'])", "[action] UP -> effect")], kept=6))
    first, second = game.requests
    assert roles(first) == ["system", "user"]
    assert first.messages[1]["content"][1]["type"] == "image_url"
    assert first.reply["reasoning"] == "look"
    assert roles(second) == ["system", "user", "assistant", "tool", "user"]
    assert json.loads(second.messages[3]["content"])["stdout"] == "1"


def test_trim_keeps_the_logged_suffix():
    game = rebuild(step(1, "a", [("t1", "x=1", "ok")], kept=3)
                   + step(2, "b", [("t2", "x=2", "ok")], kept=6)
                   + step(3, "c", [("t3", "x=3", "ok")], kept=3)
                   + step(4, "d", [("t4", "x=4", "ok")], kept=6))
    third, fourth = game.requests[2:]
    assert [m.get("reasoning") for m in third.messages if m["role"] == "assistant"] == ["t1", "t2"]
    assert [m.get("reasoning") for m in fourth.messages if m["role"] == "assistant"] == ["t3"]


def test_rolled_back_turn_leaves_history_unchanged():
    game = rebuild(step(1, "a", [("t1", "x=1", "ok")], kept=3)
                   + step(1, "resume", [("dropped", "x=2", "ok")], kept=3)
                   + step(2, "b", [("t2", "x=3", "ok")], kept=6))
    last = game.requests[2]
    assert "dropped" not in [m.get("reasoning") for m in last.messages]
    assert len(last.messages) == 1 + 3 + 1


def test_trim_inside_a_turn_starts_at_the_first_request_over_budget():
    big = "y" * 6000
    game = rebuild(step(1, "a", [("t1", "x=1", "ok")], kept=3, budget=10 ** 6)
                   + step(2, "b", [("small", "x=2", big), ("again", "x=3", "ok")], kept=5, budget=1500))
    first, second = game.requests[1:]
    assert len(first.messages) == 1 + 3 + 1  # under budget: full history
    assert roles(second) == ["system", "user", "assistant", "tool"]  # trimmed to this turn


def test_vc_sampling_replacement_swaps_the_reply():
    replaced = (section("ANALYZER STATUS", "vc_sampling: replaced reply (simulator 0.0 -> 1.0)")
                + section("THINKING", "chosen")
                + section("TOOL CALL: python", "<tool_call>\n<function=python>\n<parameter=code>\nx=9\n</parameter>\n</function>\n</tool_call>")
                + section("TOOL RESULT: python", "9"))
    game = rebuild(step(1, "a", [("candidate", "x=1", "1")], kept=3, extra=replaced)
                   + step(2, "b", [("next", "x=2", "2")], kept=6))
    assert game.requests[0].reply["reasoning"] == "chosen"
    history = game.requests[1].messages
    assert [m.get("reasoning") for m in history if m["role"] == "assistant"] == ["chosen"]
    assert json.loads(history[2]["tool_calls"][0]["function"]["arguments"]) == {"code": "x=9"}
