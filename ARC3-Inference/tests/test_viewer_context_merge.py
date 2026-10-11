from viewer.data import _extract_context, _normalize_event


def _image_prompt(text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "text", "text": "Current grid image:"},
            {"type": "image_url", "image_url": {"url": "<image>"}},
        ],
    }


def _tool_call(call_id: str, code: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "hidden",
        "tool_calls": [{"id": call_id, "function": {"name": "python", "arguments": f'{{"code": "{code}"}}'}}],
    }


def _tool_result(call_id: str, stdout: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": f'{{"stdout": "{stdout}"}}'}


def test_current_turn_stays_after_earlier_history_when_prompt_has_image() -> None:
    snapshot = {
        "messages": [
            {"role": "system", "content": "System"},
            _image_prompt("Current state: step 1, level 1."),
            _tool_call("a", "print(1)"),
            _tool_result("a", "1"),
            _image_prompt("Current state: step 35, level 3."),
            _tool_call("b", "print(2)"),
            _tool_result("b", "2"),
        ]
    }
    event = _normalize_event(
        {
            "transcript": (
                "[USER PROMPT]\nCurrent state: step 35, level 3.\n\nCurrent grid image:\n\n[image attached]\n"
                "[THINKING]\nLevel 3 reasoning\n"
                "[TOOL CALL: python]\n<tool_call>\n<function=python>\n<parameter=code>\nprint(2)\n</parameter>\n"
                "</function>\n</tool_call>\n"
                "[TOOL RESULT: python]\n2\n"
                "[THINKING]\nFinal reasoning\n"
            )
        }
    )

    context = _extract_context([event], include_system_prompt=False, request_snapshot=snapshot)
    order = [(section["label"], section["content"].splitlines()[0]) for section in context["sections"]]

    assert order == [
        ("USER PROMPT", "Current state: step 1, level 1."),
        ("TOOL CALL: python", "<tool_call>"),
        ("TOOL RESULT: python", "1"),
        ("USER PROMPT", "Current state: step 35, level 3."),
        ("THINKING", "Level 3 reasoning"),
        ("TOOL CALL: python", "<tool_call>"),
        ("TOOL RESULT: python", "2"),
        ("THINKING", "Final reasoning"),
    ]


def test_unmatched_transcript_follows_request_history() -> None:
    snapshot = {"messages": [{"role": "user", "content": "Earlier prompt"}]}
    event = _normalize_event({"transcript": "[THINKING]\nNew reasoning\n"})

    context = _extract_context([event], include_system_prompt=False, request_snapshot=snapshot)

    assert [section["label"] for section in context["sections"]] == ["USER PROMPT", "THINKING"]
