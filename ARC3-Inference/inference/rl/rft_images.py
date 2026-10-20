"""The images of a rebuilt request, drawn again from the run's saved frames.

Transcripts keep only `[image attached]` markers. Each marker follows a label
line ("Current grid image:", "Diff image (step 4 -> 5): ...", ...) that names
what the harness drew, and `intermediate_states.pkl` keeps every frame of every
action (`states[n]` is the game after action n, `states[0]` the first board), so
each image can be drawn again with the run's own settings (`MULTIMODAL_*` in
`taaf_setup_env.json`). A label this module does not know gives None, and the
caller drops the sample.

All requests of an analyzer turn see the board after the actions that ran
before it: a turn's actions run when it ends. The transcript header's `action`
is that count + 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from PIL import Image

# vision_context.ARC_COLOR_MAP and DIFF_UNCHANGED_RGB; identical in every run snapshot (checked 10-20)
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}
UNCHANGED = (16, 24, 48)
DEFAULT_UPSCALE = 16  # vision_context's default when MULTIMODAL_UPSCALE is unset

Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Scales:
    board: int
    animation: int
    intermediate: int

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "Scales":
        def scale(key, default):
            raw = str(env.get(key, "")).strip()
            return max(1, int(raw)) if raw else default
        board = scale("MULTIMODAL_UPSCALE", DEFAULT_UPSCALE)
        return cls(board, scale("MULTIMODAL_ANIMATION_UPSCALE", board), scale("MULTIMODAL_INTERMEDIATE_UPSCALE", board))


def grid_of(frame: Any) -> Grid:
    rows = frame.tolist() if hasattr(frame, "tolist") else frame
    return tuple(tuple(int(c) for c in row) for row in rows)


class Frames:
    """The boards and animation frames of one game run.

    `boards[n]` is the board after action n (0 = the first board), from the events file, which is
    written as the game runs. `frames[n]` lists what action n returned, from intermediate_states.pkl;
    its arrays were saved by reference and some were later overwritten (bp35 in
    thinking-budget-and-user-prompt-trim v02: 14 of 134 actions), so an action's frames are used
    only when their last one equals the event board."""

    def __init__(self, frames: Sequence[Sequence[Grid]], boards: dict[int, Grid] | None = None):
        self.frames = [list(f) for f in frames]
        self.boards = boards if boards is not None else {n: f[-1] for n, f in enumerate(self.frames) if f}

    @classmethod
    def from_run(cls, states: Sequence[Any], events: Iterable[dict[str, Any]]) -> "Frames":
        boards = {}
        for e in events:
            if e.get("type") == "initial":
                boards[0] = grid_of(e["board"])
            elif e.get("type") == "action":
                boards[e["action_num"]] = grid_of(e["board"])
        return cls([[grid_of(f) for f in s.raw.frame] for s in states], boards)

    def after(self, n: int) -> Grid:
        return self.boards[n]

    def trusted(self, n: int) -> bool:
        """Whether action n's saved frames can be used: every saved board of the game matches its event
        board (in bp35 and lf52 some never do, so their other frames may be overwritten too)."""
        if not hasattr(self, "_all_match"):
            self._all_match = all(self.frames[k][-1] == board for k, board in self.boards.items()
                                  if k < len(self.frames) and self.frames[k])
        return self._all_match and 0 <= n < len(self.frames) and bool(self.frames[n])

    def chain(self, n: int) -> list[Grid] | None:
        """Action n's animation chain as solver._animation_chain builds it: the board before, then
        each distinct state (runs of equal grids collapsed)."""
        if n < 1 or not self.trusted(n):
            return None
        out = [self.after(n - 1)]
        for grid in self.frames[n]:
            if out[-1] != grid:
                out.append(grid)
        return out

    def level_end(self, n: int) -> Grid | None:
        """solver._level_end_grid: frame[-2] of a level-completing action, unless it is the all-green
        placeholder TAAF pads a lone transition frame with."""
        if not self.trusted(n):
            return None
        frames = self.frames[n]
        if len(frames) < 2 or {c for row in frames[-2] for c in row} == {14}:
            return None
        return frames[-2]


def draw(grid: Grid, scale: int, diff_from: Grid | None = None) -> Image.Image:
    """vision_context's rendering: palette colors, nearest-neighbour upscale; with `diff_from`,
    unchanged cells in the off-palette navy."""
    rows, cols = len(grid), max(len(r) for r in grid)
    image = Image.new("RGB", (cols, rows), UNCHANGED if diff_from is not None else PALETTE[0])
    pixels = image.load()
    for y, row in enumerate(grid):
        for x in range(cols):
            value = row[x] if x < len(row) else 0
            if diff_from is not None:
                before = diff_from[y][x] if x < len(diff_from[y]) else 0
                if before == value:
                    continue
            pixels[x, y] = PALETTE.get(int(value), PALETTE[0])
    if scale > 1:
        image = image.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)
    return image


STEP_DIFF = re.compile(r"^Diff image \(step (\d+) -> (\d+)\)")
HISTORY_DIFF = re.compile(r"^(?:Diff image|Fatal step diff image) \(history\[(\d+)\]\.frame -> history\[(\d+)\]\.frame\)")
HISTORY_FRAME = re.compile(r"^history\[(\d+)\]\.frame")
ANIMATION_FRAME = re.compile(r"^last_animation_frames\[(\d+)\]:")
BOARD_AFTER = re.compile(r"^(?:Level \d+\. )?Board after step (\d+)")
SOLVED = re.compile(r"^Board as level \d+ was solved \(shown during step (\d+)")
CURRENT = ("Current grid image", "Current board", "This is the CURRENT board")


def resolve(label: str, before: int, frames: Frames, scales: Scales, completed: Sequence[int]) -> Image.Image | None:
    """The image a label names, for a message built when `before` actions had run.
    `completed` lists the level-completing action numbers in order."""
    label = label.strip()
    if label.startswith(CURRENT) or label.endswith("this is the CURRENT board to act on."):
        return draw(frames.after(before), scales.board)
    if (m := STEP_DIFF.match(label)) or (m := HISTORY_DIFF.match(label)):
        a, b = int(m[1]), int(m[2])
        return draw(frames.after(b), scales.board, diff_from=frames.after(a))
    if label.startswith("Last level-complete image"):
        done = [n for n in completed if n <= before]
        grid = frames.level_end(done[-1]) if done else None
        return draw(grid, scales.board) if grid is not None else None
    if (m := ANIMATION_FRAME.match(label)):
        chain = frames.chain(before)
        index = int(m[1]) + 1
        return draw(chain[index], scales.animation) if chain and index < len(chain) - 1 else None
    if label.startswith("Death frame image"):
        return draw(frames.after(before - 1), scales.board)
    if label.startswith("Fatal step diff image (last alive frame -> death frame)"):
        return draw(frames.after(before - 1), scales.board, diff_from=frames.after(before - 2))
    if (m := HISTORY_FRAME.match(label)):
        scale = scales.board if "DEATH board" in label else scales.intermediate
        return draw(frames.after(int(m[1])), scale)
    if label.startswith("Starting board, step 0"):
        return draw(frames.after(0), scales.board)
    if (m := BOARD_AFTER.match(label)):
        return draw(frames.after(int(m[1])), scales.board)
    if (m := SOLVED.match(label)):
        grid = frames.level_end(int(m[1]))
        return draw(grid, scales.board) if grid is not None else None
    return None


def label_before(text: str) -> str:
    """The label of an image: the last non-empty line of the text before its marker."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def message_images(message: dict[str, Any], before: int, frames: Frames, scales: Scales,
                   completed: Sequence[int]) -> list[Image.Image | None]:
    """One image (or None) per image part of a rebuilt user message, in order."""
    out, previous = [], ""
    for part in message.get("content") or []:
        if part.get("type") == "image_url":
            out.append(resolve(label_before(previous), before, frames, scales, completed))
            previous = ""
        else:
            previous = part.get("text") or ""
    return out
