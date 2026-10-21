"""Optional multimodal context helpers for ARC analyzer prompts.

Token-saving design (batched perception):

* Instead of attaching one full-board PNG to every request, the analyzer
  attaches a single composite image that tiles up to ``MULTIMODAL_BATCH_FRAMES``
  recent frames (oldest to newest, left to right). One image therefore covers a
  whole action sequence, and the model sees before/after transitions directly.
* Images are attached only every ``MULTIMODAL_IMAGE_EVERY`` requests (default 1)
  and are skipped when the board has not changed since the last image.
* Images are stripped from persisted history messages unless
  ``MULTIMODAL_KEEP_HISTORY_IMAGES`` is set, so a request never carries more than
  one image.
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from inference.agent.runtime_state import Frame


ARC_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (204, 204, 204),
    2: (153, 153, 153),
    3: (102, 102, 102),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (229, 58, 163),
    7: (255, 123, 204),
    8: (249, 60, 49),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 133, 27),
    13: (146, 18, 49),
    14: (79, 204, 48),
    15: (163, 86, 214),
}

_TILE_GAP_PX = 6
_LABEL_BAR_PX = 14
_GAP_COLOR = (64, 64, 64)
_LABEL_TEXT_COLOR = (255, 255, 255)
HISTORY_IMAGE_PLACEHOLDER = "[grid image omitted from history to save tokens; only the newest turn carries an image]"


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def multimodal_context() -> str:
    return os.environ.get("MULTIMODAL_CONTEXT", "").strip().lower()


def current_grid_image_enabled() -> bool:
    return multimodal_context() == "current_grid"


def current_grid_image_upscale() -> int:
    return _env_int("MULTIMODAL_UPSCALE", 16, minimum=1)


def batch_max_frames() -> int:
    """Maximum number of frames tiled into one composite image (1 = current frame only)."""
    return _env_int("MULTIMODAL_BATCH_FRAMES", 4, minimum=1)


def image_every_n_requests() -> int:
    """Attach an image only every N analyzer requests (level changes always attach one)."""
    return _env_int("MULTIMODAL_IMAGE_EVERY", 1, minimum=1)


def keep_history_images() -> bool:
    return _env_bool("MULTIMODAL_KEEP_HISTORY_IMAGES", False)


def skip_unchanged_images() -> bool:
    return _env_bool("MULTIMODAL_SKIP_UNCHANGED", True)


def _render_grid(grid: tuple[tuple[int, ...], ...], *, scale: int) -> Image.Image:
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    if rows <= 0 or cols <= 0:
        raise ValueError("Cannot render an empty grid as an image.")
    image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
    pixels = image.load()
    for row_idx, row in enumerate(grid):
        for col_idx in range(cols):
            value = row[col_idx] if col_idx < len(row) else 0
            pixels[col_idx, row_idx] = ARC_COLOR_MAP.get(int(value), ARC_COLOR_MAP[0])
    if scale > 1:
        image = image.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)
    return image


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def frame_to_png_data_url(frame: Frame, *, upscale: int | None = None) -> str:
    scale = current_grid_image_upscale() if upscale is None else max(1, int(upscale))
    return _encode_png(_render_grid(frame.grid, scale=scale))


@dataclass(frozen=True)
class LabeledFrame:
    label: str
    frame: Frame


def _label_font() -> Any:
    try:
        return ImageFont.load_default(size=11)
    except TypeError:  # pragma: no cover - older Pillow without size kwarg
        return ImageFont.load_default()


def frames_to_batched_png_data_url(
    frames: list[LabeledFrame],
    *,
    upscale: int | None = None,
) -> str:
    """Tile several frames side by side (oldest left, newest right) into one PNG."""
    if not frames:
        raise ValueError("At least one frame is required for a batched image.")
    scale = current_grid_image_upscale() if upscale is None else max(1, int(upscale))
    tiles = [_render_grid(item.frame.grid, scale=scale) for item in frames]
    if len(tiles) == 1 and not frames[0].label:
        return _encode_png(tiles[0])

    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    total_w = len(tiles) * tile_w + (len(tiles) + 1) * _TILE_GAP_PX
    total_h = tile_h + _LABEL_BAR_PX + 2 * _TILE_GAP_PX
    canvas = Image.new("RGB", (total_w, total_h), _GAP_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = _label_font()
    x = _TILE_GAP_PX
    for item, tile in zip(frames, tiles):
        if item.label:
            draw.text((x + 2, 1), item.label[:40], fill=_LABEL_TEXT_COLOR, font=font)
        canvas.paste(tile, (x, _LABEL_BAR_PX + _TILE_GAP_PX))
        x += tile_w + _TILE_GAP_PX
    return _encode_png(canvas)


def select_batch_frames(
    candidates: list[LabeledFrame],
    *,
    max_frames: int,
) -> list[LabeledFrame]:
    """Keep at most ``max_frames`` frames: always the first and last, evenly spaced between."""
    if max_frames <= 0 or len(candidates) <= max_frames:
        return list(candidates)
    if max_frames == 1:
        return [candidates[-1]]
    last_index = len(candidates) - 1
    picks = sorted({round(i * last_index / (max_frames - 1)) for i in range(max_frames)})
    return [candidates[i] for i in picks]


def image_part_from_data_url(data_url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url}}


def current_grid_image_part(frame: Frame | None) -> dict[str, Any] | None:
    """Single-frame image part (kept for callers that do not batch)."""
    if frame is None or not current_grid_image_enabled():
        return None
    return image_part_from_data_url(frame_to_png_data_url(frame))


def strip_image_parts(message: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``message`` with image parts replaced by a short text note."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    text_parts: list[str] = []
    had_image = False
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))
        elif item.get("type") in {"image_url", "image", "input_image"}:
            had_image = True
    if not had_image:
        return message
    text = "\n".join(part for part in text_parts if part).rstrip()
    text = text.removesuffix("Current grid image:").rstrip()
    stripped = dict(message)
    stripped["content"] = f"{text}\n\n{HISTORY_IMAGE_PLACEHOLDER}" if text else HISTORY_IMAGE_PLACEHOLDER
    return stripped


def strip_history_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if keep_history_images():
        return list(messages)
    return [strip_image_parts(message) for message in messages]
