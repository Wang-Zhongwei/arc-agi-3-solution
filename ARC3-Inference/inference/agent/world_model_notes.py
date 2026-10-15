"""World-model notes: `<world_model>` updates the model emits, kept as a file.

The model emits a `<world_model>...</world_model>` block whenever its
understanding of the game changes, in its thinking or in its reply. The harness
maintains `world_model.md` from these updates: each completion contributes at
most one, and every line of it is tagged with the level and action count it was
written at (`[L2 s57] ...`), so the model can filter the file by tag. The file
is never pushed back into the prompt; the model reads it from Python with
`open('world_model.md')` when it wants to.

Extraction rules:
- Only complete blocks count. A block cut off by the output cap has no closing
  tag and is ignored, so a truncated draft never enters the file.
- A backticked `<world_model>` is a mention, not a block, and blocks inside a
  `<tool_call>` (code that happens to contain the tag) are ignored.
- Thinking often drafts a block and revises it a few lines later, so the LAST
  block wins. A block in the reply is the considered one and beats any in the
  thinking.
- A completion without a block leaves the file unchanged, and an update
  identical to the previous one is not repeated.
"""

from __future__ import annotations

import re

NOTES_FILENAME = "world_model.md"

_BLOCK_RE = re.compile(r"(?<!`)<world_model>(.*?)</world_model>", flags=re.DOTALL | re.IGNORECASE)
_OPEN_TAG_RE = re.compile(r"(?<!`)<world_model>", flags=re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", flags=re.DOTALL | re.IGNORECASE)


def _last_block(text: str) -> str:
    if not text or "<world_model>" not in text.lower():
        return ""
    text = _TOOL_CALL_RE.sub("", text)
    last = ""
    for match in _BLOCK_RE.finditer(text):
        # an earlier opener inside the span (a mention or an unclosed draft) is
        # not part of this block: keep only what follows the last opener
        body = _OPEN_TAG_RE.split(match.group(1))[-1].strip()
        if body:
            last = body
    return last


def extract_world_model(reasoning: str, content: str) -> str:
    """The update this completion contributes, or "" for none."""
    return _last_block(content) or _last_block(reasoning)


def render_notes(notes: list[tuple[int, int, str]]) -> str:
    """`world_model.md`: every non-blank line tagged `[L<level> s<step>]`."""
    lines = [
        f"[L{int(level)} s{int(step)}] {line.strip()}"
        for level, step, body in notes
        for line in body.splitlines()
        if line.strip()
    ]
    return "\n".join(lines) + "\n" if lines else ""
