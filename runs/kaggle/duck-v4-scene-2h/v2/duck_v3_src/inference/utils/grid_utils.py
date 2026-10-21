"""Grid formatting helpers."""
from __future__ import annotations

from typing import Sequence

ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
ARC_COLOR_LEGEND = (
    "W=white, w=light gray, g=gray, G=dark gray, c=charcoal, B=black, "
    "M=magenta, P=pink, R=red, b=blue, S=sky blue, Y=yellow, O=orange, "
    "r=dark red, N=light green, p=purple"
)


def format_grid_rle(grid: Sequence[Sequence[int]]) -> str:
    """Whole-board text that fits a tool budget: one line per row as colour-letter run lengths, identical
    consecutive rows folded ('rows 12-41: ...'). 64x64 boards come out at roughly 250-650 tokens."""
    if not grid:
        return "(empty grid)"
    def rle(row):
        out = []
        prev = None; run = 0
        for v in row:
            ch = ARC_COLOR_CHARS[max(0, min(15, int(v)))]
            if ch == prev:
                run += 1
            else:
                if prev is not None:
                    out.append(f"{prev}{run}")
                prev, run = ch, 1
        if prev is not None:
            out.append(f"{prev}{run}")
        return " ".join(out)
    lines = []
    start = 0
    encoded = [rle(row) for row in grid]
    while start < len(encoded):
        end = start
        while end + 1 < len(encoded) and encoded[end + 1] == encoded[start]:
            end += 1
        label = f"row {start}" if end == start else f"rows {start}-{end}"
        lines.append(f"{label}: {encoded[start]}")
        start = end + 1
    return "\n".join(lines)


def format_grid_ascii(grid: Sequence[Sequence[int]]) -> str:
    if not grid:
        return "(empty grid)"
    lines = []
    for row in grid:
        chars = []
        for v in row:
            chars.append(ARC_COLOR_CHARS[max(0, min(15, int(v)))])
        lines.append("".join(chars))
    return "\n".join(lines)
