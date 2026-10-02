import pytest

from viewer.data import _normalize_event


@pytest.mark.parametrize("status", ["", "False", "True"])
@pytest.mark.parametrize("response", ["", "[ASSISTANT]\nAnswer\n"])
def test_reasoning_rendering_matches_recorded_history(status: str, response: str) -> None:
    transcript = "[THINKING]\nReasoning\n" + response + "[TOOL CALL: python]\nprint(1)\n"
    if status:
        transcript += f"[ANALYZER STATUS]\nreasoning_as_content: {status}\n"
    event = _normalize_event({"transcript": transcript})
    sections = event["transcript_sections"]
    assert event["transcript"] == transcript
    if status == "True":
        assert sections[0]["label"] == "ASSISTANT"
        assert sections[0]["content"] == ("Reasoning\n\nAnswer" if response else "Reasoning")
        assert sections[1]["label"] == "TOOL CALL: python"
    else:
        assert sections[0]["label"] == "THINKING"
        assert sections[0]["content"] == "Reasoning"
        if response:
            assert sections[1]["label"] == "ASSISTANT"


def test_assistant_text_cannot_enable_reasoning_conversion() -> None:
    event = _normalize_event({"transcript": "[THINKING]\nReasoning\n[ASSISTANT]\nreasoning_as_content: True"})
    assert event["transcript_sections"][0]["label"] == "THINKING"
