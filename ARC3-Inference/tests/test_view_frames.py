"""Historical frame inspection is always available and never executes an action."""

import base64
import io
import json

import pytest
from PIL import Image

from inference.agent import tool_agent, vision_context
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state


def _frame(step, color=0, level=2):
    return Frame(grid=tuple((color,) * 8 for _ in range(8)), step=step, level=level)


# Deliberately separate history indices from action steps.
HISTORY = [
    HistoryEntry("INIT", _frame(100, 0, level=1)),
    HistoryEntry("LEFT", _frame(101, 1), {"level_completed": True}),
    HistoryEntry("UP", _frame(102, 2), {"game_over": True}),
    HistoryEntry("RESET", _frame(103, 3), {"automatic": True}),
    *[HistoryEntry("RIGHT", _frame(100 + i, i)) for i in range(4, 10)],
]


def _state(entries=HISTORY):
    return {
        "current_frame": tool_agent._ascii_frame_view_payload(entries[-1].frame),
        "history": tool_agent._ascii_history_view_payload(entries),
        "valid_actions": ["LEFT"],
        "last_action_call_result": {},
    }


def _forbid_action(*args, **kwargs):
    pytest.fail("Viewing history must not invoke the action handler.")


def _run(code, **kwargs):
    return run_sandboxed_python(
        code=code,
        timeout_seconds=kwargs.pop("timeout_seconds", 20),
        initial_state=kwargs.pop("initial_state", _state()),
        action_handler=kwargs.pop("action_handler", _forbid_action),
        **kwargs,
    )


def _image(part):
    data = part["image_url"]["url"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data)))


@pytest.mark.parametrize("indices, expected", [
    ("1", [1]), ("-1", [9]), ("range(1, 4)", [1, 2, 3]),
])
def test_single_negative_and_range_indices_are_absolute(indices, expected):
    out = _run(f"view_frames({indices})")
    assert out["error"] == "", out
    assert out["viewed_frames"] == [{"index": index, "bbox": None} for index in expected]
    assert out["action_results"] == []
    for index in expected:
        assert f"history[{index}]" in out["stdout"]


def test_labels_identify_transition_death_and_reset():
    out = _run("view_frames([1, 2, 3])")
    assert out["error"] == "", out
    assert "board after LEFT, level 2, step 101" in out["stdout"]
    assert "level_completed" in out["stdout"]
    assert "board after UP, level 2, step 102" in out["stdout"]
    assert "game_over" in out["stdout"]
    assert "board after RESET, level 2, step 103" in out["stdout"]


@pytest.mark.parametrize("indices", ["True", "1.5", "'1'", "[0, 10]", "[-11]"])
def test_invalid_indices_queue_no_images(indices):
    out = _run(f"view_frames({indices})")
    assert out["error"], out
    assert out["viewed_frames"] == []
    assert out["action_results"] == []


@pytest.mark.parametrize("bbox", [
    "(0, 1, 2)", "(True, 0, 2, 2)", "(0, 0, 2.5, 2)",
    "(3, 0, 2, 2)", "(0, 3, 2, 2)", "(-1, 0, 2, 2)", "(0, 0, 8, 2)",
])
def test_invalid_bbox_is_rejected_without_queuing_a_partial_view(bbox):
    out = _run(f"view_frames([0, 1], bbox={bbox})")
    assert "bbox" in out["error"], out
    assert out["viewed_frames"] == []


def test_crop_bounds_keep_global_row_and_column_coordinates():
    out = _run("view_frames(-1, bbox=(1, 2, 3, 5))")
    assert out["error"] == "", out
    assert out["viewed_frames"] == [{"index": 9, "bbox": [1, 2, 3, 5]}]
    assert "history[9]" in out["stdout"]
    assert "bbox=(1, 2, 3, 5)" in out["stdout"]


def test_identical_views_are_deduplicated_but_distinct_crops_are_kept():
    out = _run(
        "view_frames([0, 0])\n"
        "view_frames(0, bbox=(0, 0, 1, 1))\n"
        "view_frames(0, bbox=[0, 0, 1, 1])\n"
        "view_frames(0, bbox=(2, 2, 3, 3))"
    )
    assert out["error"] == "", out
    assert out["viewed_frames"] == [
        {"index": 0, "bbox": None},
        {"index": 0, "bbox": [0, 0, 1, 1]},
        {"index": 0, "bbox": [2, 2, 3, 3]},
    ]


def test_default_four_image_cap_spans_all_helper_calls_in_a_snippet():
    out = _run("view_frames([0, 0, 1])\nview_frames([1, 2, 3, 4])")
    assert out["error"] == "", out
    assert out["viewed_frames"] == [{"index": index, "bbox": None} for index in range(4)]
    assert "skipped" in out["stdout"] and "images per python call" in out["stdout"]


def test_helper_remains_callable_when_invocation_image_allowance_is_exhausted():
    out = _run("view_frames(1)", view_frames_limit=0)
    assert out["error"] == "", out
    assert out["viewed_frames"] == []
    assert "skipped" in out["stdout"]


def test_caught_index_error_leaves_prior_views_and_allows_another_view():
    out = _run(
        "view_frames(0)\n"
        "try:\n    view_frames([1, 999])\n"
        "except IndexError:\n    pass\n"
        "view_frames(2)"
    )
    assert out["error"] == "", out
    assert out["viewed_frames"] == [{"index": 0, "bbox": None}, {"index": 2, "bbox": None}]


@pytest.mark.parametrize("failure", ["view_frames(999)", "raise ValueError('after viewing')"])
def test_uncaught_error_preserves_successfully_queued_views(failure):
    out = _run("view_frames(1)\n" + failure)
    assert out["error"], out
    assert out["viewed_frames"] == [{"index": 1, "bbox": None}]


def test_timeout_preserves_successfully_queued_views():
    out = _run("view_frames(1)\nwhile True:\n    pass", timeout_seconds=1)
    assert "timed out" in out["error"], out
    assert out["viewed_frames"] == [{"index": 1, "bbox": None}]


def test_viewing_after_action_uses_refreshed_history():
    entries = list(HISTORY[:1])
    actions_seen = []

    def action_handler(actions, **kwargs):
        actions_seen.extend(actions)
        entries.append(HistoryEntry("LEFT", _frame(101, 1), {"level_completed": True}))
        return {
            "action_result": {"executed": True, "level_completed": True},
            "state": _state(entries),
        }

    out = _run(
        "view_frames(-1)\naction('LEFT')\nview_frames(-1)",
        initial_state=_state(entries),
        action_handler=action_handler,
    )
    assert out["error"] == "", out
    assert actions_seen == [{"action": "LEFT"}]
    assert out["viewed_frames"] == [{"index": 0, "bbox": None}, {"index": 1, "bbox": None}]
    assert "level_completed" in out["stdout"]


def test_host_renders_inclusive_crops_and_absolute_history_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "")
    state_path = tmp_path / "state.json"
    write_runtime_state(state_path, current_frame=HISTORY[-1].frame, history=HISTORY)
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://localhost/v1", provider="vllm")
    rendered = agent._view_frames_images(state_path, [
        {"index": 1, "bbox": None}, {"index": 2, "bbox": [1, 2, 3, 5]},
    ])
    assert len(rendered) == 2
    assert "history[1]" in rendered[0][0] and "step 101" in rendered[0][0]
    assert "history[2]" in rendered[1][0] and "bbox=(1, 2, 3, 5)" in rendered[1][0]
    assert "inclusive global coordinates" in rendered[1][0]
    # Only views that have margins mention them.
    assert "padding" not in rendered[0][0]
    assert "blue outer margins are padding" in rendered[1][0]
    assert _image(rendered[0][1]).size == (256, 256)
    cropped = _image(rendered[1][1])
    assert cropped.size == (256, 256)
    # The 3-by-4 crop is enlarged 64 times and centered vertically.
    assert cropped.getpixel((0, 0)) == vision_context.VIEW_PADDING_RGB
    assert cropped.getpixel((0, 32)) == vision_context.ARC_COLOR_MAP[2]
    assert cropped.getpixel((255, 223)) == vision_context.ARC_COLOR_MAP[2]
    assert cropped.getpixel((255, 224)) == vision_context.VIEW_PADDING_RGB
    assert all(tool_agent._image_part_tokens(part) == 66 for _, part in rendered)



def _budgeted_image(frame, *, bbox=None, upscale=4):
    part = {"type": "image_url", "image_url": {"url": vision_context.frame_to_png_data_url(
        frame, upscale=upscale, bbox=bbox, visual_token_limit=64,
    )}}
    return _image(part), part


def _assert_64_visual_tokens(image, part):
    width, height = image.size
    assert width % 32 == 0 and height % 32 == 0
    assert (width // 32) * (height // 32) == 64
    # The two boundary markers are separate from the requested visual budget.
    assert tool_agent._image_part_tokens(part) == 66


def test_bbox_increases_cell_clarity_without_increasing_visual_tokens(monkeypatch):
    monkeypatch.delenv("ARC3_IMAGE_TOKENS_FLAT", raising=False)
    frame = Frame(grid=tuple((6,) + (9,) * 63 for _ in range(64)), step=0, level=1)
    full, full_part = _budgeted_image(frame)
    crop, crop_part = _budgeted_image(frame, bbox=(0, 0, 15, 15))
    assert full.size == crop.size == (256, 256)
    _assert_64_visual_tokens(full, full_part)
    _assert_64_visual_tokens(crop, crop_part)
    # Each original cell occupies four pixels; focusing on 16 cells raises
    # that to sixteen, retaining exact palette boundaries in the same budget.
    assert full.getpixel((3, 0)) == vision_context.ARC_COLOR_MAP[6]
    assert full.getpixel((4, 0)) == vision_context.ARC_COLOR_MAP[9]
    assert crop.getpixel((15, 0)) == vision_context.ARC_COLOR_MAP[6]
    assert crop.getpixel((16, 0)) == vision_context.ARC_COLOR_MAP[9]


@pytest.mark.parametrize("rows, cols, size, scale, offset", [
    (16, 64, (512, 128), 8, (0, 0)),
    (3, 57, (1024, 64), 17, (27, 6)),
    (1, 64, (2048, 32), 32, (0, 0)),
])
def test_rectangular_views_fit_the_budget_without_stretching_cells(
    monkeypatch, rows, cols, size, scale, offset,
):
    monkeypatch.delenv("ARC3_IMAGE_TOKENS_FLAT", raising=False)
    frame = Frame(grid=tuple((6,) + (9,) * (cols - 1) for _ in range(rows)), step=0, level=1)
    rendered, part = _budgeted_image(frame)
    assert rendered.size == size
    _assert_64_visual_tokens(rendered, part)
    left, top = offset
    assert rendered.getpixel((left, top)) == vision_context.ARC_COLOR_MAP[6]
    assert rendered.getpixel((left + scale - 1, top + scale - 1)) == vision_context.ARC_COLOR_MAP[6]
    assert rendered.getpixel((left + scale, top)) == vision_context.ARC_COLOR_MAP[9]
    assert rendered.getpixel((left + cols * scale - 1, top + rows * scale - 1)) == vision_context.ARC_COLOR_MAP[9]
    if left:
        assert rendered.getpixel((left - 1, top)) == vision_context.VIEW_PADDING_RGB
    if top:
        assert rendered.getpixel((left, top - 1)) == vision_context.VIEW_PADDING_RGB


@pytest.mark.parametrize("upscale, resample", [(1, "nearest"), (128, "bilinear"), (16, "lanczos")])
def test_budgeted_views_ignore_ordinary_multimodal_scale_and_interpolation(monkeypatch, upscale, resample):
    monkeypatch.setenv("MULTIMODAL_UPSCALE", str(upscale))
    monkeypatch.setenv("MULTIMODAL_RESAMPLE", resample)
    frame = Frame(grid=((6, 9), (9, 6)), step=0, level=1)
    rendered, part = _budgeted_image(frame, upscale=upscale)
    assert rendered.size == (256, 256)
    _assert_64_visual_tokens(rendered, part)
    assert {color for _, color in rendered.getcolors(maxcolors=256)} == {vision_context.ARC_COLOR_MAP[6], vision_context.ARC_COLOR_MAP[9]}
    assert rendered.getpixel((127, 127)) == vision_context.ARC_COLOR_MAP[6]
    assert rendered.getpixel((128, 127)) == vision_context.ARC_COLOR_MAP[9]


def test_oversized_view_downsamples_with_nearest_neighbor_to_stay_in_budget(monkeypatch):
    monkeypatch.delenv("ARC3_IMAGE_TOKENS_FLAT", raising=False)
    monkeypatch.setenv("MULTIMODAL_RESAMPLE", "bicubic")
    grid = tuple(tuple(6 if col < 256 else 9 for col in range(512)) for _ in range(512))
    rendered, part = _budgeted_image(Frame(grid=grid, step=0, level=1))
    assert rendered.size == (256, 256)
    _assert_64_visual_tokens(rendered, part)
    assert {color for _, color in rendered.getcolors(maxcolors=256)} == {vision_context.ARC_COLOR_MAP[6], vision_context.ARC_COLOR_MAP[9]}
    assert rendered.getpixel((127, 128)) == vision_context.ARC_COLOR_MAP[6]
    assert rendered.getpixel((128, 128)) == vision_context.ARC_COLOR_MAP[9]


def test_ragged_view_retains_empty_cell_fill_within_the_visual_budget(monkeypatch):
    monkeypatch.delenv("ARC3_IMAGE_TOKENS_FLAT", raising=False)
    rendered, part = _budgeted_image(Frame(grid=((6, 9), (6,)), step=0, level=1))
    assert rendered.size == (256, 256)
    _assert_64_visual_tokens(rendered, part)
    assert rendered.getpixel((200, 50)) == vision_context.ARC_COLOR_MAP[9]
    assert rendered.getpixel((200, 200)) == vision_context.ARC_COLOR_MAP[0]


def test_visual_budget_is_opt_in_for_ordinary_frame_images(monkeypatch):
    monkeypatch.setenv("MULTIMODAL_UPSCALE", "6")
    frame = Frame(grid=tuple((6, 9, 6, 9) for _ in range(3)), step=0, level=1)
    ordinary = {"type": "image_url", "image_url": {"url": vision_context.frame_to_png_data_url(frame)}}
    explicit = {"type": "image_url", "image_url": {"url": vision_context.frame_to_png_data_url(frame, upscale=4)}}
    focused, focused_part = _budgeted_image(frame)
    assert _image(ordinary).size == (24, 18)
    assert _image(explicit).size == (16, 12)
    assert focused.size == (256, 256)
    _assert_64_visual_tokens(focused, focused_part)

def test_system_prompt_always_documents_single_index_and_crop(monkeypatch):
    monkeypatch.delenv("ARC3_VIEW_FRAMES", raising=False)
    monkeypatch.delenv("ARC3_HISTORY_REVISIT_HINT", raising=False)
    prompt = tool_agent._build_system_prompt(tool_output_tokens=1024)
    assert "view_frames(indices, bbox=None)" in prompt
    assert "inclusive" in prompt and "single" in prompt
    assert "4 images" in prompt and "8" in prompt
    assert "64 visual tokens" in prompt
    assert "focus" in prompt and "more clearly" in prompt


def _tool_calls(*codes):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": f"call-{index}", "type": "function", "function": {
                "name": "python", "arguments": json.dumps({"code": code}),
            }}
            for index, code in enumerate(codes)
        ],
    }


def test_images_follow_the_complete_tool_group_and_invocation_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC3_ANIMATION", "0")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    state_path = tmp_path / "state.json"
    write_runtime_state(state_path, current_frame=HISTORY[-1].frame, history=HISTORY)
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://localhost/v1", provider="vllm")
    agent._yield_seconds = agent._yield_tokens = None
    agent._tool_steps = 4
    agent._context_budget_tokens = 100_000
    replies = iter([
        _tool_calls("view_frames(range(4))", "view_frames(range(4, 8))"),
        _tool_calls("view_frames([8, 9])"),
        _tool_calls("action('LEFT')"),
    ])
    requests = []

    def chat(messages, **kwargs):
        requests.append(json.loads(json.dumps(messages)))
        return tool_agent._ChatCompletionResult(message=next(replies), finish_reason="tool_calls")

    monkeypatch.setattr(agent, "_chat_completion", chat)
    result = agent.analyze(
        state_path, action_num=109, analysis_step=1, valid_actions=["LEFT"],
        step_env=lambda arguments: {"executed": True, "executed_count": 1, "level": 2},
    )
    assert result is not None and result.step_executed
    second = requests[1]
    first_tool_index = next(i for i, message in enumerate(second) if message["role"] == "tool")
    assert [message["role"] for message in second[first_tool_index:first_tool_index + 3]] == [
        "tool", "tool", "user",
    ]
    image_message = second[first_tool_index + 2]
    assert sum(part["type"] == "image_url" for part in image_message["content"]) == 8
    captions = [part["text"] for part in image_message["content"] if part["type"] == "text"]
    for index in range(8):
        assert any(f"history[{index}]" in caption for caption in captions)
    # The second model reply has exhausted this invocation's image allowance.
    third = requests[2]
    last_tool_index = max(i for i, message in enumerate(third) if message["role"] == "tool")
    assert not any(
        part.get("type") == "image_url"
        for message in third[last_tool_index + 1:]
        for part in message["content"] if isinstance(message["content"], list)
    )
    # A new analyzer invocation receives a fresh allowance even though the
    # preceding invocation's images are still in conversation history.
    replies = iter([_tool_calls("view_frames([8, 9])"), _tool_calls("action('LEFT')")])
    next_result = agent.analyze(
        state_path, action_num=110, analysis_step=2, valid_actions=["LEFT"],
        step_env=lambda arguments: {"executed": True, "executed_count": 1, "level": 2},
    )
    assert next_result is not None and next_result.step_executed
    refreshed = requests[4]
    last_tool_index = max(i for i, message in enumerate(refreshed) if message["role"] == "tool")
    new_images = refreshed[last_tool_index + 1]
    assert new_images["role"] == "user"
    assert sum(part["type"] == "image_url" for part in new_images["content"]) == 2


@pytest.mark.parametrize("fail_first_delivery", [False, True])
@pytest.mark.parametrize("trim_before_delivery", [False, True])
def test_unseen_view_survives_image_history_eviction_until_delivery(
    monkeypatch, tmp_path, fail_first_delivery, trim_before_delivery,
):
    monkeypatch.setenv("ARC3_ANIMATION", "0")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setenv("ARC3_HISTORY_IMAGE_KEEP", "0")
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "")
    state_path = tmp_path / "state.json"
    write_runtime_state(state_path, current_frame=HISTORY[-1].frame, history=HISTORY)
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://localhost/v1", provider="vllm")
    agent._yield_seconds = agent._yield_tokens = None
    agent._context_budget_tokens = 100_000
    requests = []

    def chat(messages, **kwargs):
        requests.append(json.loads(json.dumps(messages)))
        if fail_first_delivery and len(requests) == 2:
            raise tool_agent.requests.RequestException("delivery failed before a response")
        code = "view_frames(0); action('LEFT')" if len(requests) == 1 else "action('LEFT')"
        return tool_agent._ChatCompletionResult(message=_tool_calls(code), finish_reason="tool_calls")

    def requested_images(messages):
        return [
            part
            for message in messages
            if isinstance(message.get("content"), list)
            and any("history[0]: board after INIT" in part.get("text", "")
                    for part in message["content"])
            for part in message["content"] if part.get("type") == "image_url"
        ]

    def analyze(step):
        return agent.analyze(
            state_path, action_num=109 + step, analysis_step=step,
            valid_actions=["LEFT"],
            step_env=lambda arguments: {"executed": True, "executed_count": 1, "level": 2},
        )

    monkeypatch.setattr(agent, "_chat_completion", chat)
    first = analyze(1)
    assert first is not None and first.step_executed
    # view_frames ran in the snippet that ended the invocation. Its image has
    # never been in a model request, and normal history retention omits it.
    assert requested_images(requests[0]) == []
    assert requested_images(agent._history_messages) == []
    if trim_before_delivery:
        # Force actual context eviction before delivering the queued image.
        # The fresh state prompt must survive alongside it, even though the
        # pair exceeds this deliberately tiny test budget.
        agent._context_budget_tokens = 1
    second = analyze(2)
    assert len(requested_images(requests[1])) == 1
    # The restored view replaces its image-stripped persisted copy.
    assert sum(
        "history[0]: board after INIT" in part.get("text", "")
        for message in requests[1] if isinstance(message.get("content"), list)
        for part in message["content"]
    ) == 1
    assert _image(requested_images(requests[1])[0]).size == (256, 256)
    if trim_before_delivery:
        delivered_text = "\n".join(
            tool_agent._message_display_text(message) for message in requests[1]
        )
        assert "Current state: step 112, level 2." in delivered_text
        assert "Valid actions right now: LEFT" in delivered_text
        assert "Current state: step 111, level 2." not in delivered_text
    if fail_first_delivery:
        assert second is not None and second.retryable_failure and not second.step_executed
        second = analyze(3)
        assert len(requested_images(requests[2])) == 1
    assert second is not None and second.step_executed
    # Once a response confirms delivery, the history image policy applies
    # normally; the pending queue must not reattach the view indefinitely.
    third = analyze(4)
    assert third is not None and third.step_executed
    assert requested_images(requests[-1]) == []
