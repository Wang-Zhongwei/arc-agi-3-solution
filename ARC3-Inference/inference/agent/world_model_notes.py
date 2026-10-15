"""World-model notes: `<world_model>` blocks the model writes, kept as a log.

The model may write a `<world_model>...</world_model>` block whenever its
understanding of the game changes, in its thinking or in its reply. Each
completion contributes at most one block, which is appended to a per-game log
tagged with the level and action count it was written at (`[L2 s57] ...`). The
log is never pushed back into the prompt; the model reads it from Python as
`world_model_notes` when it wants to.

Extraction rules:
- Only complete blocks count. A block cut off by the output cap has no closing
  tag and is ignored, so a truncated draft never enters the log.
- The opening tag must start a line. A passing mention such as "update my
  <world_model> later" or a backticked `<world_model>` is not a block.
- Blocks inside a `<tool_call>` (code that happens to contain the tag) are
  ignored.
- Thinking often drafts a block and revises it a few lines later, so the LAST
  block wins. A block in the reply is the considered one and beats any in the
  thinking.
- A completion without a block leaves the log unchanged, and a block identical
  to the previous entry is not repeated.
"""

from __future__ import annotations

import re

_BLOCK_RE = re.compile(
    r"^[ \t]*<world_model>(.*?)</world_model>",
    flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_OPEN_TAG_RE = re.compile(r"<world_model>", flags=re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", flags=re.DOTALL | re.IGNORECASE)


def _last_block(text: str) -> str:
    if not text or "<world_model>" not in text.lower():
        return ""
    text = _TOOL_CALL_RE.sub("", text)
    last = ""
    for match in _BLOCK_RE.finditer(text):
        # a stray opening tag inside the span (an earlier unclosed draft) is
        # not part of this block: keep only what follows the last opener
        body = _OPEN_TAG_RE.split(match.group(1))[-1].strip()
        if body:
            last = body
    return last


def extract_world_model(reasoning: str, content: str) -> str:
    """The block this completion contributes, or "" for none."""
    return _last_block(content) or _last_block(reasoning)


def format_note(level: int, step: int, body: str) -> str:
    return f"[L{int(level)} s{int(step)}] {body.strip()}"


def note_body(note: str) -> str:
    return note.split("] ", 1)[1] if note.startswith("[L") and "] " in note else note


def render_notes(notes: list[str]) -> str:
    return "\n\n".join(notes)
