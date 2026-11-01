"""Competition-mode leanness: with debug artifacts off a turn writes nothing and
skips work only the logs read, while the model sees exactly the same request."""

import base64
import io
import json

import pytest
from PIL import Image

from inference.agent import tool_agent, vision_context
from inference.agent.runtime_state import Frame, HistoryEntry, discard_runtime_state, write_runtime_state


def _grid(value, row, size=64):
    return tuple(tuple(value if r == row else 0 for _ in range(size)) for r in range(size))


def _reference_png(grid, scale):
    """The renderer as it was before caching."""
    rows, cols = len(grid), max(len(row) for row in grid)
    image = Image.new("RGB", (cols, rows), vision_context.ARC_COLOR_MAP[0])
    pixels = image.load()
    for r, row in enumerate(grid):
        for c in range(cols):
            pixels[c, r] = vision_context.ARC_COLOR_MAP.get(int(row[c]), vision_context.ARC_COLOR_MAP[0])
    if scale > 1:
        image = image.resize((cols * scale, rows * scale), vision_context.multimodal_resample())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.parametrize("scale", [1, 10, 12])
def test_cached_png_is_byte_identical_and_reused(scale):
    vision_context._grid_png_data_url.cache_clear()
    grid = tuple(tuple((r * 7 + c * 3) % 16 for c in range(64)) for r in range(64))
    first = vision_context.frame_to_png_data_url(Frame(grid=grid, step=1, level=1), upscale=scale)
    assert first == _reference_png(grid, scale)
    # same board as a list-of-lists, another step: served from the cache
    again = vision_context.frame_to_png_data_url(Frame(grid=[list(row) for row in grid], step=9, level=2), upscale=scale)
    assert again is first
    assert vision_context._grid_png_data_url.cache_info().hits == 1


def test_unchanged_board_renders_no_diff_image(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    monkeypatch.setattr(tool_agent, "_DIFF_IMAGE", True)
    calls = []
    real = vision_context.diff_image_part
    monkeypatch.setattr(vision_context, "diff_image_part", lambda a, b: calls.append(1) or real(a, b))
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._ensure_session(tmp_path / "state.json")
    frame = Frame(grid=_grid(3, 10), step=4, level=1)
    agent._auto_diff_prev_frame = frame
    summary = {"executed_count": 1, "executed_actions": ["MOUSE(row=1, col=1)"]}
    agent._build_user_prompt(5, valid_actions=["MOUSE"], current_frame=Frame(grid=frame.grid, step=5, level=1),
                             history_entries=[HistoryEntry("MOUSE", frame)], previous_step_summary=summary)
    assert calls == [] and agent._pending_diff_image is None


def _run_turn(tmp_path, monkeypatch, debug):
    monkeypatch.setenv("ARC3_DEBUG_ARTIFACTS", debug)
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", "off")
    run = tmp_path / debug
    (run / "artifacts").mkdir(parents=True)
    state = run / "artifacts" / "g_tool_runtime_state.json"
    frame = Frame(grid=_grid(5, 30), step=0, level=1)
    write_runtime_state(state, current_frame=frame, history=[HistoryEntry("", frame)], flush=True)
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    sent = []

    def fake_chat(messages, **kwargs):
        sent.append(json.dumps(messages, sort_keys=True))
        if len(sent) > 3:
            raise RuntimeError("stop the test turn")
        return tool_agent._ChatCompletionResult(
            message={"role": "assistant", "content": "thinking only"}, finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 5},
        )

    monkeypatch.setattr(agent, "_chat_completion", fake_chat)
    transcript = run / "transcripts" / "g.txt"
    transcript.parent.mkdir()
    try:
        agent.analyze(state, 1, valid_actions=["MOUSE"], transcript_path=transcript, analysis_step=1)
    except RuntimeError:
        pass
    discard_runtime_state(state)
    return sent, transcript, run


def test_debug_off_turn_writes_nothing_and_sends_the_same_request(tmp_path, monkeypatch):
    sent_on, transcript_on, _ = _run_turn(tmp_path, monkeypatch, "on")
    sent_off, transcript_off, run_off = _run_turn(tmp_path, monkeypatch, "off")
    assert sent_on and sent_on[0] == sent_off[0]
    assert transcript_on.exists() and "analysis_step=1" in transcript_on.read_text()
    assert not transcript_off.exists()
    assert not (run_off / "prompts").exists()
