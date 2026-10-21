"""Unit tests for ARC3_OPENER_DEDUP (tool_agent._dedup_opening_message and friends)."""
from __future__ import annotations

import copy

import pytest

from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame

BLOCK = "Only tool: `python`. It receives things.\nKeep tool output compact.\nWhen calling `python`, emit the call."
PROMPT = "Current state: step 4, level 1.\nValid actions right now: UP, DOWN.\n" + BLOCK
RETAINED_FNS = "\n\nYour retained functions from earlier Python calls in this game: f(x)."
BOARD = "data:image/png;base64,AAAA"
OTHER = "data:image/png;base64,BBBB"
CAPTION = ta._RESUME_IMAGE_OMITTED_CAPTION


def image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def opener(prompt=PROMPT, url=BOARD, extra=""):
    return {"role": "user", "content": [{"type": "text", "text": prompt + extra + "\n\nCurrent grid image:"},
                                        image(url)]}


def carrier():
    return opener()


def dedup(message, retained, modes=("static", "resume_image"), prompt=PROMPT, block=BLOCK, resumed=None):
    return ta._dedup_opening_message(message, retained, modes=frozenset(modes), user_prompt=prompt,
                                     static_block=block, resumed_image_url=resumed)


@pytest.mark.parametrize("raw,expected", [
    ("", set()), ("0", set()), ("off", set()), ("none", set()),
    ("1", {"static", "resume_image"}), ("all", {"static", "resume_image"}), ("on", {"static", "resume_image"}),
    ("static", {"static"}), ("resume_image", {"resume_image"}),
    ("static,resume_image", {"static", "resume_image"}), ("static+resume_image", {"static", "resume_image"}),
    ("bogus", set()), ("STATIC, bogus", {"static"}),
])
def test_modes_parse(monkeypatch, raw, expected):
    monkeypatch.setenv("ARC3_OPENER_DEDUP", raw)
    assert ta._opener_dedup_modes() == frozenset(expected)


def test_modes_default_off(monkeypatch):
    monkeypatch.delenv("ARC3_OPENER_DEDUP", raising=False)
    assert ta._opener_dedup_modes() == frozenset()


def test_static_block_removed_exactly_when_retained_carries_it():
    msg = opener(extra=RETAINED_FNS)
    before = copy.deepcopy(msg)
    out = dedup(msg, [{"role": "assistant", "content": "x"}, carrier()], modes=("static",))
    assert msg == before, "the original message must not be mutated"
    text = out["content"][0]["text"]
    assert text == PROMPT[: -len("\n" + BLOCK)] + RETAINED_FNS + "\n\nCurrent grid image:"
    assert out["content"][1:] == msg["content"][1:]
    assert BLOCK not in text and "Valid actions right now: UP, DOWN." in text


def test_static_block_kept_without_a_carrier():
    msg = opener()
    assert dedup(msg, [], modes=("static",)) is msg
    # a nudge holds part of the guidance but not the block: not a carrier
    nudge = {"role": "user", "content": "You have not acted yet. When calling `python`, emit the call."}
    assert dedup(msg, [nudge], modes=("static",)) is msg
    # a retained opener with a different block variant is not a carrier of this one
    variant = BLOCK.replace("compact", "short")
    assert dedup(msg, [opener(prompt=PROMPT.replace(BLOCK, variant))], modes=("static",)) is msg
    # the system prompt is not passed in as retained, so it never counts


def test_a_longer_variant_containing_the_block_word_for_word_carries_it():
    longer = opener(prompt=PROMPT + "\nIf you use MOUSE, include integer row and col arguments.")
    out = dedup(opener(), [longer], modes=("static",))
    assert BLOCK not in out["content"][0]["text"]


def test_static_needs_the_prompt_to_end_with_the_block():
    msg = opener()
    assert dedup(msg, [carrier()], modes=("static",), prompt=PROMPT + "\nmore") is msg
    assert dedup(msg, [carrier()], modes=("static",), block=None) is msg
    assert dedup(msg, [carrier()], modes=("static",), prompt=None) is msg       # short/state_only forms


def test_static_on_a_text_only_opener():
    msg = {"role": "user", "content": PROMPT + RETAINED_FNS}
    out = dedup(msg, [carrier()], modes=("static",))
    assert out["content"] == PROMPT[: -len("\n" + BLOCK)] + RETAINED_FNS


def test_resume_image_dropped_only_when_the_last_image_is_the_same_board():
    msg = opener()
    out = dedup(msg, [opener(), {"role": "assistant", "content": "x"}], modes=("resume_image",), resumed=BOARD)
    assert out["content"] == PROMPT + "\n\n" + CAPTION
    # most recent retained image is a different board: keep
    assert dedup(msg, [opener(url=OTHER)], modes=("resume_image",), resumed=BOARD) is msg
    # the same board further back but a different one since: keep
    assert dedup(msg, [opener(), opener(url=OTHER)], modes=("resume_image",), resumed=BOARD) is msg
    # no retained image at all: keep
    assert dedup(msg, [{"role": "user", "content": "text"}], modes=("resume_image",), resumed=BOARD) is msg
    # a fresh turn (no resumed image) never drops
    assert dedup(msg, [opener()], modes=("resume_image",), resumed=None) is msg


def test_resume_image_matches_inside_a_multi_image_message():
    last = opener()
    last["content"] += [{"type": "text", "text": "Diff image"}, image(OTHER)]
    out = dedup(opener(), [last], modes=("resume_image",), resumed=BOARD)
    assert out["content"] == PROMPT + "\n\n" + CAPTION


def test_resume_image_left_alone_when_the_opener_has_more_images():
    msg = opener()
    msg["content"] += [{"type": "text", "text": "Diff image"}, image(OTHER)]
    assert dedup(msg, [opener()], modes=("resume_image",), resumed=BOARD) is msg
    # or when its image is not the resumed board
    assert dedup(opener(url=OTHER), [opener(url=OTHER)], modes=("resume_image",), resumed=BOARD) is not None
    assert dedup(opener(url=OTHER), [opener(url=OTHER)], modes=("resume_image",), resumed=BOARD)["content"][1] \
        == image(OTHER)


def test_both_parts_together_and_control_tag_kept():
    msg = ta._mark_control_message(opener(extra=RETAINED_FNS), "resume")
    out = dedup(msg, [carrier()], resumed=BOARD)
    assert out["content"] == PROMPT[: -len("\n" + BLOCK)] + RETAINED_FNS + "\n\n" + CAPTION
    assert out[ta._CONTROL_MESSAGE_KEY] == "resume"


def test_no_modes_is_the_identity():
    msg = opener()
    assert dedup(msg, [carrier()], modes=(), resumed=BOARD) is msg


def frame(level=1, step=3):
    grid = tuple(tuple(5 for _ in range(8)) for _ in range(8))
    return Frame(grid=grid, step=step, level=level)


@pytest.fixture
def agent():
    return ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")


def test_prompt_builder_records_its_static_tail(agent):
    prompt = agent._build_user_prompt(3, valid_actions=["UP", "DOWN", "MOUSE"], current_frame=frame())
    block = agent._opener_static_block
    assert block and block.startswith("Only tool: `python`.")
    assert prompt.endswith("\n" + block)
    assert ta.TOOL_CALL_FORMAT_GUIDANCE in block
    assert block.endswith("If you use MOUSE, include integer row and col arguments.")
    head = prompt[: -len("\n" + block)]
    assert "Valid actions right now:" in head and "Only tool" not in head


def test_static_tail_is_identical_across_turns(agent):
    a = agent._build_user_prompt(3, valid_actions=["UP"], current_frame=frame(step=3))
    block_a = agent._opener_static_block
    b = agent._build_user_prompt(9, valid_actions=["UP"], current_frame=frame(step=9, level=2))
    assert agent._opener_static_block == block_a and a != b
    agent._build_user_prompt(0, valid_actions=["UP"], current_frame=frame(step=0))
    assert agent._opener_static_block != block_a          # the first turn's "Ground yourself" variant


def test_no_static_tail_recorded_with_a_world_model_block(agent, monkeypatch):
    monkeypatch.setattr(agent, "_summarized_knowledge_lines", lambda: ["World model: walls block."])
    agent._build_user_prompt(3, valid_actions=["UP"], current_frame=frame())
    assert agent._opener_static_block is None
