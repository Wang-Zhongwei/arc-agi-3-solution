"""Deterministic exploration phase for the tool agent.

Before the model plays a level it first *explores* it without any reasoning:

1. ``plan_probes`` builds a fixed, deterministic list of probes from the valid
   actions: every non-mouse button once (in a canonical order, optionally repeated),
   then ``MOUSE`` clicks on up to ``max_mouse_targets`` distinct objects found by
   connected-component segmentation of the board (one click per distinct
   colour+shape, spread across the board in reading order).
2. ``ExplorationRunner`` executes the probes one at a time through the solver's
   ``step_env`` and, for every probe, sends the *describer* a single image that
   tiles the before and after frames -- and nothing else (no world model, no
   history, thinking disabled) -- asking only "what changed?".
   When ``reset_between_probes`` is on (default), the level is RESET before a probe whenever
   an earlier probe left the board different from the level's initial state, so every
   before/after pair measures one input applied to the same clean starting state.
3. Once every probe has run, one *synthesis* call receives the action -> observation
   log (plus the previous level's hypothesis, if any) and writes the initial world
   model, goal model, action model, open questions and plan. The tool agent seeds its
   carried world model with that hypothesis before the first real model turn.

All knobs are environment-configurable (see ``ExplorationConfig``); the Makefile maps
them from the ``exploration`` block of the config JSON.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.vision_context import (
    LabeledFrame,
    frames_to_batched_png_data_url,
    image_part_from_data_url,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
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


@dataclass(frozen=True)
class ExplorationConfig:
    enabled: bool = True
    # "table": press each button once (no resets, no model calls) and hand the model a measured probe table.
    # "full": the older phase with resets, mouse probes, a describer call per probe and a synthesis call.
    mode: str = "table"
    # Explore at the start of every level (True) or only the first level seen (False).
    every_level: bool = True
    # How many times each non-mouse button is pressed in a row (each press is its own probe).
    button_repeats: int = 1
    # MOUSE probes: click at most this many distinct objects, ignoring components smaller than
    # ``mouse_min_pixels`` cells and the largest component (assumed background).
    max_mouse_targets: int = 8
    mouse_min_pixels: int = 1
    # Skip components larger than this fraction of the board (panels/arenas) and strips spanning >= 90% of a side.
    mouse_max_fraction: float = 0.10
    # Hard cap on probes per level (buttons + clicks).
    max_probes: int = 24
    # RESET the level before a probe when an earlier probe changed the board, so each probe starts clean.
    reset_between_probes: bool = True
    # Describer call (one per probe that changed the board): image only, thinking off.
    describer_max_output: int = 400
    describer_thinking: bool = False
    # Tell the describer which input was made between the two frames (it still gets no other context).
    show_action: bool = True
    # Give the describer the measured diff (cell count and row/col range) so distances are not guessed from pixels.
    describer_show_diff: bool = True
    # Skip the describer when the board did not change at all (the observation is recorded as no-change).
    skip_unchanged: bool = True
    # Synthesis call (once per explored level).
    synthesis_max_output: int = 2048
    synthesis_thinking: bool = False
    synthesis_attach_image: bool = True
    # How many analyzer prompts the observation table stays visible after exploration.
    note_turns: int = 3

    @classmethod
    def from_env(cls) -> "ExplorationConfig":
        return cls(
            enabled=_env_bool("EXPLORATION_ENABLED", True),
            mode=(os.environ.get("EXPLORATION_MODE", "table").strip().lower() or "table"),
            every_level=_env_bool("EXPLORATION_EVERY_LEVEL", True),
            button_repeats=_env_int("EXPLORATION_BUTTON_REPEATS", 1, minimum=1),
            max_mouse_targets=_env_int("EXPLORATION_MAX_MOUSE_TARGETS", 8, minimum=0),
            mouse_min_pixels=_env_int("EXPLORATION_MOUSE_MIN_PIXELS", 1, minimum=1),
            mouse_max_fraction=max(0.0, min(1.0, _env_float("EXPLORATION_MOUSE_MAX_FRACTION", 0.10))),
            max_probes=_env_int("EXPLORATION_MAX_PROBES", 24, minimum=1),
            reset_between_probes=_env_bool("EXPLORATION_RESET_BETWEEN_PROBES", True),
            describer_max_output=_env_int("EXPLORATION_DESCRIBER_MAX_OUTPUT", 400, minimum=64),
            describer_thinking=_env_bool("EXPLORATION_DESCRIBER_THINKING", False),
            show_action=_env_bool("EXPLORATION_SHOW_ACTION", True),
            describer_show_diff=_env_bool("EXPLORATION_DESCRIBER_SHOW_DIFF", True),
            skip_unchanged=_env_bool("EXPLORATION_SKIP_UNCHANGED", True),
            synthesis_max_output=_env_int("EXPLORATION_SYNTHESIS_MAX_OUTPUT", 2048, minimum=256),
            synthesis_thinking=_env_bool("EXPLORATION_SYNTHESIS_THINKING", False),
            synthesis_attach_image=_env_bool("EXPLORATION_SYNTHESIS_ATTACH_IMAGE", True),
            note_turns=_env_int("EXPLORATION_NOTE_TURNS", 3, minimum=0),
        )


# ---------------------------------------------------------------------------
# Probe planning
# ---------------------------------------------------------------------------

BUTTON_ORDER = ("UP", "DOWN", "LEFT", "RIGHT", "SPACE")
MOUSE_PLACEHOLDER = "__MOUSE_TARGETS__"


@dataclass(frozen=True)
class Probe:
    action: dict[str, Any]
    label: str
    # Placeholder probes expand into concrete MOUSE clicks when reached, using the board at that time.
    placeholder: bool = False


@dataclass(frozen=True)
class GridObject:
    color: int
    cells: tuple[tuple[int, int], ...]
    top_left: tuple[int, int]

    @property
    def pixels(self) -> int:
        return len(self.cells)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        return min(rows), min(cols), max(rows), max(cols)

    @property
    def shape_key(self) -> tuple[int, tuple[tuple[int, int], ...]]:
        r0, c0, _, _ = self.bbox
        return self.color, tuple(sorted((r - r0, c - c0) for r, c in self.cells))

    def click_target(self) -> tuple[int, int]:
        """The object's cell closest to its bounding-box centre (always inside the object)."""
        r0, c0, r1, c1 = self.bbox
        centre_r = (r0 + r1) / 2.0
        centre_c = (c0 + c1) / 2.0
        return min(self.cells, key=lambda rc: ((rc[0] - centre_r) ** 2 + (rc[1] - centre_c) ** 2, rc))


def grid_objects(grid: tuple[tuple[int, ...], ...]) -> list[GridObject]:
    """4-connected same-colour components in reading order of their top-left cell."""
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    if height == 0 or width == 0:
        return []
    seen = [[False] * width for _ in range(height)]
    objects: list[GridObject] = []
    for sr in range(height):
        for sc in range(width):
            if seen[sr][sc]:
                continue
            value = grid[sr][sc] if sc < len(grid[sr]) else 0
            stack = [(sr, sc)]
            seen[sr][sc] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr][nc]:
                        neighbour = grid[nr][nc] if nc < len(grid[nr]) else 0
                        if neighbour == value:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            objects.append(GridObject(color=int(value), cells=tuple(sorted(cells)), top_left=(sr, sc)))
    return objects


def _spread_picks(items: list[Any], limit: int) -> list[Any]:
    """Keep at most ``limit`` items, evenly spaced over the list (always first and last)."""
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    last = len(items) - 1
    picks = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    return [items[i] for i in picks]


def plan_mouse_targets(
    frame: Frame | None,
    *,
    max_targets: int,
    min_pixels: int,
    max_fraction: float = 1.0,
) -> list[Probe]:
    """Deterministic click targets: one per distinct (colour, shape).

    Skipped: the largest component (background), components smaller than ``min_pixels``,
    components covering more than ``max_fraction`` of the board (arenas, panels), and strips
    spanning at least 90% of the board's width or height (HUD bars, borders).
    """
    if frame is None or max_targets <= 0:
        return []
    objects = grid_objects(frame.grid)
    if not objects:
        return []
    rows, cols = frame.shape
    total_cells = max(1, rows * cols)
    background = max(objects, key=lambda obj: (obj.pixels, -obj.top_left[0], -obj.top_left[1]))
    seen_shapes: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    candidates: list[GridObject] = []
    for obj in objects:
        if obj is background or obj.pixels < min_pixels:
            continue
        if obj.pixels > max_fraction * total_cells:
            continue
        r0, c0, r1, c1 = obj.bbox
        if (c1 - c0 + 1) >= 0.9 * cols or (r1 - r0 + 1) >= 0.9 * rows:
            continue
        key = obj.shape_key
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        candidates.append(obj)
    probes: list[Probe] = []
    for obj in _spread_picks(candidates, max_targets):
        row, col = obj.click_target()
        r0, c0, r1, c1 = obj.bbox
        color_char = ARC_COLOR_CHARS[max(0, min(15, obj.color))]
        label = (
            f"MOUSE(row={row}, col={col}) on {color_char}-coloured object "
            f"{r1 - r0 + 1}x{c1 - c0 + 1} at rows {r0}-{r1}, cols {c0}-{c1} ({obj.pixels} cells)"
        )
        probes.append(Probe(action={"action": "MOUSE", "row": row, "col": col}, label=label))
    return probes


def object_diff_summary(before: tuple[tuple[int, ...], ...], after: tuple[tuple[int, ...], ...], *, max_items: int = 4) -> str:
    """Deterministic object-level description of a transition: what moved, appeared, disappeared."""
    summary = diff_summary(before, after)
    if not summary.get("changed_cells"):
        return "no change"
    rows = len(before); cols = max((len(r) for r in before), default=0)
    big = 0.3 * rows * cols
    # Objects this large are backgrounds/arenas: they get a new shape whenever something moves over them,
    # which would otherwise show up as a spurious "appeared + disappeared" pair.
    reshaped_ok = 0.08 * rows * cols
    def objects(grid):
        objs = [o for o in grid_objects(grid) if o.pixels <= big]
        keyed: dict[tuple, list[GridObject]] = {}
        for o in objs:
            keyed.setdefault(o.shape_key, []).append(o)
        return keyed
    b, a = objects(before), objects(after)
    items: list[tuple[int, str]] = []
    def label(o: GridObject) -> str:
        r0, c0, r1, c1 = o.bbox
        return f"{ARC_COLOR_CHARS[max(0, min(15, o.color))]} {r1 - r0 + 1}x{c1 - c0 + 1}"
    for key in set(b) | set(a):
        bl, al = b.get(key, []), a.get(key, [])
        if bl and al:
            # Pair each before-object with the nearest unmatched after-object (stationary objects pair at distance 0).
            remaining = list(al)
            pairs: list[tuple[GridObject, GridObject]] = []
            for x in sorted(bl, key=lambda o: o.top_left):
                if not remaining:
                    break
                y = min(remaining, key=lambda o: abs(o.top_left[0] - x.top_left[0]) + abs(o.top_left[1] - x.top_left[1]))
                remaining.remove(y)
                pairs.append((x, y))
            moved = [(x, y) for x, y in pairs if x.top_left != y.top_left]
            for x, y in moved[:2]:
                dr, dc = y.top_left[0] - x.top_left[0], y.top_left[1] - x.top_left[1]
                items.append((x.pixels, f"{label(x)} at r{x.bbox[0]} c{x.bbox[1]} moved ({dr:+d},{dc:+d})"))
            if len(al) > len(bl):
                items.append((al[0].pixels, f"{label(al[0])} appeared x{len(al) - len(bl)}"))
            elif len(bl) > len(al):
                items.append((bl[0].pixels, f"{label(bl[0])} disappeared x{len(bl) - len(al)}"))
        elif bl:
            if bl[0].pixels > reshaped_ok:
                continue
            items.append((bl[0].pixels, f"{label(bl[0])} at r{bl[0].bbox[0]} c{bl[0].bbox[1]} disappeared" + (f" x{len(bl)}" if len(bl) > 1 else "")))
        else:
            if al[0].pixels > reshaped_ok:
                continue
            items.append((al[0].pixels, f"{label(al[0])} at r{al[0].bbox[0]} c{al[0].bbox[1]} appeared" + (f" x{len(al)}" if len(al) > 1 else "")))
    # A shape that "disappeared" and an identically labelled one that "appeared" at the same place is one
    # object whose outline changed (something moved across it): report it once, as reshaped.
    texts = [t for _, t in items]
    collapsed: list[tuple[int, str]] = []
    seen_reshaped: set[str] = set()
    for pixels, text in items:
        base = text.replace(" disappeared", "").replace(" appeared", "")
        if " disappeared" in text or " appeared" in text:
            other = text.replace(" disappeared", " appeared") if " disappeared" in text else text.replace(" appeared", " disappeared")
            if other in texts:
                if base not in seen_reshaped:
                    seen_reshaped.add(base)
                    collapsed.append((pixels, f"{base} reshaped"))
                continue
        collapsed.append((pixels, text))
    items = collapsed
    items.sort(key=lambda t: -t[0])
    head = f"{summary['changed_cells']} cells changed (rows {summary['row_range'][0]}-{summary['row_range'][1]}, cols {summary['col_range'][0]}-{summary['col_range'][1]})"
    if not items:
        return head + "; " + ", ".join(summary.get("top_transitions", []))
    return head + "; " + "; ".join(text for _, text in items[:max_items])


def plan_probes(valid_actions: list[str], config: ExplorationConfig) -> list[Probe]:
    """Buttons first (canonical order, then any other named action), then a MOUSE placeholder (full mode only)."""
    names: list[str] = []
    for value in valid_actions or []:
        name = str(value or "").strip().upper()
        if name and name not in names and name != "RESET":
            names.append(name)
    ordered = [name for name in BUTTON_ORDER if name in names]
    ordered.extend(name for name in names if name not in BUTTON_ORDER and name != "MOUSE")
    probes: list[Probe] = []
    for name in ordered:
        for repeat in range(config.button_repeats):
            suffix = f" (press {repeat + 1}/{config.button_repeats})" if config.button_repeats > 1 else ""
            probes.append(Probe(action={"action": name}, label=f"{name}{suffix}"))
    if "MOUSE" in names and config.max_mouse_targets > 0 and config.mode != "table":
        probes.append(Probe(action={"action": "MOUSE"}, label=MOUSE_PLACEHOLDER, placeholder=True))
    return probes[: config.max_probes] if len(probes) > config.max_probes else probes


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def diff_summary(before: tuple[tuple[int, ...], ...], after: tuple[tuple[int, ...], ...]) -> dict[str, Any]:
    """Cheap numeric summary of a grid change (for the transcript and the synthesis prompt)."""
    changed: list[tuple[int, int]] = []
    transitions: Counter[tuple[int, int]] = Counter()
    rows = max(len(before), len(after))
    for r in range(rows):
        row_before = before[r] if r < len(before) else ()
        row_after = after[r] if r < len(after) else ()
        cols = max(len(row_before), len(row_after))
        for c in range(cols):
            a = row_before[c] if c < len(row_before) else 0
            b = row_after[c] if c < len(row_after) else 0
            if a != b:
                changed.append((r, c))
                transitions[(int(a), int(b))] += 1
    if not changed:
        return {"changed_cells": 0}
    return {
        "changed_cells": len(changed),
        "row_range": (min(r for r, _ in changed), max(r for r, _ in changed)),
        "col_range": (min(c for _, c in changed), max(c for _, c in changed)),
        "top_transitions": [
            f"{ARC_COLOR_CHARS[max(0, min(15, a))]}->{ARC_COLOR_CHARS[max(0, min(15, b))]}x{n}"
            for (a, b), n in transitions.most_common(3)
        ],
    }


def describe_diff_summary(summary: dict[str, Any]) -> str:
    count = int(summary.get("changed_cells", 0) or 0)
    if count <= 0:
        return "no cell changed"
    rows = summary.get("row_range")
    cols = summary.get("col_range")
    parts = [f"{count} cells changed"]
    if rows and cols:
        parts.append(f"within rows {rows[0]}-{rows[1]}, cols {cols[0]}-{cols[1]}")
    top = summary.get("top_transitions")
    if top:
        parts.append(f"colour transitions {', '.join(top)}")
    return " ".join(parts)


@dataclass
class Observation:
    index: int
    label: str
    executed: bool
    board_changed: bool
    before_step: int
    after_step: int
    description: str
    diff: dict[str, Any] = field(default_factory=dict)
    game_over: bool = False
    level_completed: bool = False
    run_complete: bool = False
    error: str = ""

    def line(self) -> str:
        outcome: list[str] = []
        if not self.executed:
            outcome.append(f"not executed ({self.error or 'unknown error'})")
        else:
            outcome.append(f"board changed: {'yes' if self.board_changed else 'no'} (measured: {describe_diff_summary(self.diff)})")
            if self.level_completed:
                outcome.append("LEVEL COMPLETED")
            if self.game_over:
                outcome.append("GAME OVER (level was reset)")
            if self.run_complete:
                outcome.append("RUN COMPLETE")
        text = f"Probe {self.index}: {self.label} -> {'; '.join(outcome)}."
        if self.description:
            text += f" Observed: {self.description}"
        return text


@dataclass
class ExplorationRun:
    level: int
    probes: list[Probe]
    next_index: int = 0
    observations: list[Observation] = field(default_factory=list)
    complete: bool = False
    aborted_reason: str | None = None
    hypothesis: dict[str, str] = field(default_factory=dict)
    game_overs: int = 0
    # Board at the start of the level; probes are measured against it when reset_between_probes is on.
    initial_grid: tuple[tuple[int, ...], ...] | None = None
    resets: int = 0

    @property
    def finished(self) -> bool:
        return self.complete or self.aborted_reason is not None


@dataclass
class ExplorationTurnResult:
    executed_payloads: list[dict[str, Any]]
    finished: bool
    hypothesis: dict[str, str] | None
    observations: list[Observation]
    level: int


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HYPOTHESIS_LABELS = [
    "World model",
    "Goal model",
    "Action model",
    "Open questions",
    "Plan",
    "New on this level",
]


def build_describer_messages(
    *,
    system_prompt: str,
    before: Frame,
    after: Frame,
    action_label: str | None,
    measured: str | None = None,
) -> list[dict[str, Any]]:
    data_url = frames_to_batched_png_data_url(
        [LabeledFrame("before", before), LabeledFrame("after", after)]
    )
    lines = ["Two tiles of the same grid: LEFT is before, RIGHT is after exactly one input."]
    if action_label:
        lines.append(f"The input was: {action_label}.")
    if measured:
        lines.append(f"Measured by the harness (exact): {measured}. Use these numbers for distances and locations.")
    lines.append("Describe precisely what changed between the tiles.")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "\n".join(lines)},
                image_part_from_data_url(data_url),
            ],
        },
    ]


def build_synthesis_messages(
    *,
    system_prompt: str,
    level: int,
    valid_actions: list[str],
    observations: list[Observation],
    previous_hypothesis: dict[str, str] | None,
    image_part: dict[str, Any] | None,
    reset_between_probes: bool = False,
) -> list[dict[str, Any]]:
    changed = sum(1 for obs in observations if obs.executed and obs.board_changed)
    lines = [
        f"Exploration log for level {level}: {len(observations)} deterministic probes were executed in order, "
        f"{changed} of them changed the board. Each line is one input, the harness's exact measurement of the change "
        "(cell count, row/col range, colour transitions), and an unbiased description written by an observer who was "
        "shown only the two frames. The measurement is authoritative: when the prose disagrees with it on distance, "
        "size or location, trust the measurement (e.g. a change spanning 12 rows is a 6-row move of a 6-row object, "
        "whatever the prose says).",
    ]
    if reset_between_probes:
        lines.append(
            "Every probe was applied to the level's initial state: whenever an earlier probe had changed the board, "
            "the level was RESET before the next probe, so the effects below are independent, not cumulative."
        )
    lines += [
        f"Valid inputs on this level: {', '.join(valid_actions) if valid_actions else 'unknown'}.",
        "",
        *[obs.line() for obs in observations],
    ]
    if previous_hypothesis:
        lines.extend(
            [
                "",
                "Hypothesis carried from the previous level (mechanics often persist; layouts and goals may change):",
                *[f"- {label}: {value}" for label, value in previous_hypothesis.items() if value],
            ]
        )
    lines.extend(
        [
            "",
            "Write the initial hypothesis for this level. Respond with exactly these labeled sections, each label on its own line:",
            "World model: what the board contains (objects by colour/shape/position, HUD elements such as edge bars, player-like objects if any)",
            "Goal model: the most likely goal and 1-2 alternatives, each with the evidence for it",
            "Action model: one line per input -- what it does, whether it changed the board, and what it seems to affect",
            "Open questions: what the probes did not settle and the cheapest test for each",
            "Plan: the first few concrete actions to try next and why",
        ]
    )
    if previous_hypothesis:
        lines.append("New on this level: what differs from the previous level's hypothesis (or 'nothing obvious')")
    text = "\n".join(lines)
    content: Any = text
    if image_part is not None:
        content = [
            {"type": "text", "text": f"{text}\n\nImage: the level's first frame (left) and the board after the probes (right)."},
            image_part,
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def probe_table_lines(run: ExplorationRun) -> list[str]:
    """Compact, measured probe table (table mode): one line per button, no prose."""
    lines = [f"Probe table for level {run.level} (harness pressed each button once from the level start, in order, no resets; effects measured, not interpreted):"]
    for obs in run.observations:
        if not obs.executed:
            lines.append(f"- {obs.label}: not executed ({obs.error or 'error'})")
            continue
        lines.append(f"- {obs.label}: {obs.description or describe_diff_summary(obs.diff)}" + (" [GAME_OVER]" if obs.game_over else "") + (" [LEVEL COMPLETED]" if obs.level_completed else ""))
    return lines


def exploration_prompt_lines(run: ExplorationRun, *, reset_between_probes: bool = False) -> list[str]:
    """Lines injected into the analyzer's user prompt after the exploration phase."""
    header = (
        f"Exploration phase on level {run.level}: {len(run.observations)} deterministic probes were executed "
        "before your first turn (each button once, then clicks on distinct objects). Observed effects:"
    )
    lines = [header]
    if reset_between_probes:
        lines.append(
            f"- Each probe started from the level's initial state (the level was RESET {run.resets} time(s) between probes), "
            "so effects are independent of each other; the board you see now is the state after the last probe."
        )
    for obs in run.observations:
        lines.append(f"- {obs.line()}")
    if run.aborted_reason:
        lines.append(f"- Exploration stopped early: {run.aborted_reason}.")
    lines.append(
        "- The working world model above was synthesized from these probes; test its open questions before committing to a plan."
    )
    return lines


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(part for part in parts if part).strip()
    return ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExplorationRunner:
    """Owns the per-level exploration state and runs probes through ``step_env``."""

    def __init__(
        self,
        *,
        config: ExplorationConfig,
        describer_system_prompt: str,
        synthesis_system_prompt: str,
        chat_completion: Callable[..., Any],
        extract_blocks: Callable[[str, list[str]], dict[str, str]],
        on_usage: Callable[[dict[str, Any] | None], None] | None = None,
    ) -> None:
        self.config = config
        self._describer_system_prompt = describer_system_prompt
        self._synthesis_system_prompt = synthesis_system_prompt
        self._chat_completion = chat_completion
        self._extract_blocks = extract_blocks
        self._on_usage = on_usage
        self.run: ExplorationRun | None = None
        self.explored_levels: set[int] = set()
        self.last_hypothesis: dict[str, str] = {}
        self.note_turns_left = 0

    def reset(self) -> None:
        self.run = None
        self.explored_levels = set()
        self.last_hypothesis = {}
        self.note_turns_left = 0

    # -- planning -----------------------------------------------------------

    def should_explore(self, level: int) -> bool:
        if not self.config.enabled:
            return False
        if self.run is not None and self.run.level == level:
            return not self.run.finished
        if level in self.explored_levels:
            return False
        if self.explored_levels and not self.config.every_level:
            return False
        return True

    def start_level(self, level: int, valid_actions: list[str], initial_frame: Frame | None = None) -> ExplorationRun:
        self.run = ExplorationRun(
            level=level,
            probes=plan_probes(valid_actions, self.config),
            initial_grid=initial_frame.grid if initial_frame is not None else None,
        )
        self.explored_levels.add(level)
        return self.run

    def _maybe_reset(
        self,
        run: ExplorationRun,
        *,
        load_state: Callable[[], tuple[Frame | None, list[HistoryEntry]]],
        step_env: Callable[[dict[str, Any]], dict[str, Any]],
        append_transcript: Callable[[str, str], None],
        executed_payloads: list[dict[str, Any]],
        reason: str,
    ) -> None:
        """RESET the level if the board no longer matches the level's initial state."""
        if not self.config.reset_between_probes or run.initial_grid is None or self.config.mode == "table":
            return
        frame, _ = load_state()
        if frame is None or frame.grid == run.initial_grid:
            return
        payload = step_env({"actions": [{"action": "RESET"}]})
        executed = isinstance(payload, dict) and bool(payload.get("executed"))
        if executed:
            executed_payloads.append(payload)
            run.resets += 1
        after, _ = load_state()
        restored = after is not None and after.grid == run.initial_grid
        append_transcript(
            "EXPLORATION RESET",
            f"before {reason}: RESET executed={executed} board_restored_to_initial={restored}"
            + ("" if executed else f" error={payload.get('error') if isinstance(payload, dict) else payload}"),
        )

    # -- model calls --------------------------------------------------------

    def describe(
        self, *, before: Frame, after: Frame, action_label: str, measured: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        messages = build_describer_messages(
            system_prompt=self._describer_system_prompt,
            before=before,
            after=after,
            action_label=action_label if self.config.show_action else None,
            measured=measured if self.config.describer_show_diff else None,
        )
        result = self._chat_completion(
            messages,
            tools=None,
            max_tokens=self.config.describer_max_output,
            thinking=self.config.describer_thinking,
        )
        message = result.message if hasattr(result, "message") else {}
        content = _message_text(message.get("content", ""))
        reasoning = _message_text(message.get("reasoning") or message.get("reasoning_content") or "")
        if not content and reasoning:
            content = reasoning
        usage = getattr(result, "usage", None)
        if self._on_usage is not None:
            self._on_usage(usage)
        meta = {"finish_reason": getattr(result, "finish_reason", ""), "usage": usage, "reasoning": reasoning}
        return " ".join(content.split()), meta

    def synthesize(
        self,
        *,
        run: ExplorationRun,
        valid_actions: list[str],
        first_frame: Frame | None,
        current_frame: Frame | None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        image_part = None
        if self.config.synthesis_attach_image and current_frame is not None:
            tiles = []
            if first_frame is not None and first_frame.grid != current_frame.grid:
                tiles.append(LabeledFrame(f"level {run.level} start", first_frame))
            tiles.append(LabeledFrame("after probes", current_frame))
            image_part = image_part_from_data_url(frames_to_batched_png_data_url(tiles))
        messages = build_synthesis_messages(
            system_prompt=self._synthesis_system_prompt,
            level=run.level,
            valid_actions=valid_actions,
            observations=run.observations,
            previous_hypothesis=self.last_hypothesis or None,
            image_part=image_part,
            reset_between_probes=self.config.reset_between_probes,
        )
        result = self._chat_completion(
            messages,
            tools=None,
            max_tokens=self.config.synthesis_max_output,
            thinking=self.config.synthesis_thinking,
        )
        message = result.message if hasattr(result, "message") else {}
        content = _message_text(message.get("content", ""))
        reasoning = _message_text(message.get("reasoning") or message.get("reasoning_content") or "")
        usage = getattr(result, "usage", None)
        if self._on_usage is not None:
            self._on_usage(usage)
        hypothesis = self._extract_blocks(content, HYPOTHESIS_LABELS) if content else {}
        if not hypothesis and reasoning:
            hypothesis = self._extract_blocks(reasoning, HYPOTHESIS_LABELS)
        hypothesis = {label: value for label, value in hypothesis.items() if value}
        meta = {
            "finish_reason": getattr(result, "finish_reason", ""),
            "usage": usage,
            "content": content,
            "reasoning": reasoning,
        }
        return hypothesis, meta

    # -- execution ----------------------------------------------------------

    def run_turn(
        self,
        *,
        level: int,
        valid_actions: list[str],
        load_state: Callable[[], tuple[Frame | None, list[HistoryEntry]]],
        step_env: Callable[[dict[str, Any]], dict[str, Any]],
        append_transcript: Callable[[str, str], None],
        should_stop: Callable[[], bool] | None = None,
        note: Callable[[str], None] | None = None,
    ) -> ExplorationTurnResult | None:
        """Execute as many pending probes as possible. Returns None when nothing is pending."""
        if not self.should_explore(level):
            return None
        run = self.run if self.run is not None and self.run.level == level else None
        if run is None:
            initial_frame, _ = load_state()
            run = self.start_level(level, valid_actions, initial_frame)
            append_transcript(
                "EXPLORATION START",
                f"level={level} probes={len(run.probes)} valid_actions={', '.join(valid_actions) or 'unknown'}\n"
                + "\n".join(f"{i + 1}. {probe.label}" for i, probe in enumerate(run.probes)),
            )
        executed_payloads: list[dict[str, Any]] = []
        current_frame, _ = load_state()
        first_frame = current_frame
        history_start = current_frame

        while run.next_index < len(run.probes):
            if should_stop is not None and should_stop():
                append_transcript("EXPLORATION STATUS", "stopped: solver requested stop; remaining probes resume next turn.")
                break
            probe = run.probes[run.next_index]
            if probe.placeholder:
                self._maybe_reset(
                    run, load_state=load_state, step_env=step_env, append_transcript=append_transcript,
                    executed_payloads=executed_payloads, reason="planning MOUSE targets",
                )
                before_frame, _ = load_state()
                targets = plan_mouse_targets(
                    before_frame,
                    max_targets=self.config.max_mouse_targets,
                    min_pixels=self.config.mouse_min_pixels,
                    max_fraction=self.config.mouse_max_fraction,
                )
                remaining_budget = max(0, self.config.max_probes - (len(run.probes) - 1))
                targets = targets[:remaining_budget]
                run.probes[run.next_index : run.next_index + 1] = targets
                append_transcript(
                    "EXPLORATION MOUSE TARGETS",
                    "\n".join(f"{run.next_index + i + 1}. {t.label}" for i, t in enumerate(targets)) or "(no clickable objects found)",
                )
                continue

            index = run.next_index + 1
            if note is not None:
                note(f"exploration probe {index}/{len(run.probes)}: {probe.label}")
            self._maybe_reset(
                run, load_state=load_state, step_env=step_env, append_transcript=append_transcript,
                executed_payloads=executed_payloads, reason=f"probe {index} ({probe.label})",
            )
            before_frame, _ = load_state()
            payload = step_env({"actions": [dict(probe.action)]})
            if not isinstance(payload, dict):
                payload = {"executed": False, "error": "step_env returned no payload"}
            after_frame, _ = load_state()
            run.next_index += 1

            executed = bool(payload.get("executed"))
            if executed:
                executed_payloads.append(payload)
            observation = Observation(
                index=index,
                label=probe.label,
                executed=executed,
                board_changed=bool(payload.get("board_changed")),
                before_step=before_frame.step if before_frame is not None else 0,
                after_step=after_frame.step if after_frame is not None else 0,
                description="",
                game_over=bool(payload.get("game_over")),
                level_completed=bool(payload.get("level_completed")),
                run_complete=bool(payload.get("run_complete")),
                error=str(payload.get("error") or ""),
            )
            if executed and before_frame is not None and after_frame is not None:
                observation.diff = diff_summary(before_frame.grid, after_frame.grid)
                observation.board_changed = observation.board_changed or observation.diff.get("changed_cells", 0) > 0
            append_transcript(
                "EXPLORATION PROBE",
                f"probe={index}/{len(run.probes)} action={probe.label} executed={executed} "
                f"board_changed={observation.board_changed} diff={describe_diff_summary(observation.diff) if executed else 'n/a'} "
                f"level_completed={observation.level_completed} game_over={observation.game_over} run_complete={observation.run_complete}"
                + (f" error={observation.error}" if observation.error else ""),
            )
            need_description = executed and before_frame is not None and after_frame is not None
            if need_description and self.config.mode == "table":
                observation.description = object_diff_summary(before_frame.grid, after_frame.grid)
                need_description = False
            if need_description and not observation.board_changed and self.config.skip_unchanged:
                observation.description = "No visible change."
                need_description = False
            if need_description:
                try:
                    description, meta = self.describe(
                        before=before_frame,
                        after=after_frame,
                        action_label=probe.label,
                        measured=describe_diff_summary(observation.diff),
                    )
                except Exception as exc:  # the describer must never break exploration
                    description, meta = f"(describer error: {exc})", {}
                observation.description = description or "(empty description)"
                extra = f" finish_reason={meta.get('finish_reason')}" if meta.get("finish_reason") else ""
                append_transcript("EXPLORATION OBSERVATION", f"probe={index} {probe.label}{extra}\n{observation.description}")
            run.observations.append(observation)

            if observation.run_complete:
                run.aborted_reason = "the run completed during exploration"
                break
            if observation.level_completed:
                run.aborted_reason = f"probe {index} ({probe.label}) completed the level"
                break
            if observation.game_over:
                run.game_overs += 1
                append_transcript(
                    "EXPLORATION STATUS",
                    f"probe {index} ({probe.label}) reached GAME_OVER; the runner auto-resets and exploration resumes next turn.",
                )
                break
            if not executed and payload.get("valid_actions") == []:
                run.aborted_reason = "the environment reported a terminal state"
                break

        hypothesis: dict[str, str] | None = None
        finished = False
        if run.aborted_reason is not None or run.next_index >= len(run.probes):
            if self.config.mode == "table":
                run.hypothesis = {}
            elif run.aborted_reason is None and not run.complete:
                current_frame, _ = load_state()
                if note is not None:
                    note("exploration synthesis running")
                try:
                    hypothesis, meta = self.synthesize(
                        run=run,
                        valid_actions=valid_actions,
                        first_frame=first_frame if first_frame is not None else history_start,
                        current_frame=current_frame,
                    )
                except Exception as exc:  # never break the main turn
                    hypothesis, meta = {}, {"content": "", "reasoning": "", "finish_reason": f"error: {exc}"}
                if meta.get("reasoning"):
                    append_transcript("EXPLORATION SYNTHESIS THINKING", str(meta["reasoning"])[:4000])
                append_transcript(
                    "EXPLORATION HYPOTHESIS",
                    str(meta.get("content") or "")
                    or f"(no hypothesis parsed; finish_reason={meta.get('finish_reason') or '(empty)'})",
                )
                run.hypothesis = dict(hypothesis or {})
                if run.hypothesis:
                    self.last_hypothesis = dict(run.hypothesis)
            elif run.aborted_reason is not None:
                append_transcript("EXPLORATION STATUS", f"aborted: {run.aborted_reason}")
            run.complete = True
            finished = True
            self.note_turns_left = self.config.note_turns
            append_transcript(
                "EXPLORATION STATUS",
                f"finished level={run.level} probes_executed={len(run.observations)} "
                f"board_changes={sum(1 for o in run.observations if o.board_changed)} resets={run.resets} game_overs={run.game_overs} "
                f"hypothesis_sections={len(run.hypothesis)}",
            )
        return ExplorationTurnResult(
            executed_payloads=executed_payloads,
            finished=finished,
            hypothesis=hypothesis,
            observations=list(run.observations),
            level=run.level,
        )

    def prompt_lines(self) -> list[str]:
        if self.run is None or not self.run.complete or self.note_turns_left <= 0 or not self.run.observations:
            return []
        if self.config.mode == "table":
            return probe_table_lines(self.run)
        return exploration_prompt_lines(self.run, reset_between_probes=self.config.reset_between_probes)

    def probe_facts(self) -> dict[str, str]:
        """Probe results as short facts (persist in the harness memory)."""
        if self.run is None or not self.run.observations:
            return {}
        return {f"probe {obs.label}": (obs.description or describe_diff_summary(obs.diff))[:160] for obs in self.run.observations if obs.executed}
