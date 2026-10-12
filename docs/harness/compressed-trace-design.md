# Compressed play trace before the history window

Status: implemented behind `ARC3_COMPRESSED_TRACE_BATCHES` (default 0, off), 2026-10-12.
Code: `ARC3-Inference/inference/agent/compressed_trace.py`, wired in `tool_agent.py`;
tests: `ARC3-Inference/tests/test_compressed_trace.py`. Not yet run on Kaggle.

## Idea

Today the prompt is the system prompt plus an oscillating window of full
history: it grows to about 118k tokens, then drains back to about 58k
(`ARC3_CONTEXT_DRAIN_TOKENS=59392`). Everything older is gone from the
conversation, though it stays in the runtime `history` list.

The change keeps the window as it is and adds one fixed-size block in front of
it: the most recent play that the window has dropped, reduced to what was done
and what the board looked like. Reasoning, tool code and tool output are left
out.

```
┌──────────────────────────────┐
│ system prompt (unchanged)    │
├──────────────────────────────┤
│ compressed trace, ≤30 batches│  ← new, ~13k tokens, rebuilt only at a trim
├──────────────────────────────┤
│ full history window          │  ← unchanged oscillating window
│ (58k → 118k, then drain)     │
└──────────────────────────────┘
```

## Indexing

Checked in `inference/framework/solver.py` and on the events of
`level-up-board` v01:

- `history[0]` is the starting board, seeded with an empty action. It is
  step 0.
- Every executed action appends one entry, including the automatic `RESET`
  after a game over. Refused or skipped actions append nothing.
- So **step n is `history[n]`**, n ≥ 1 counts the game's executed actions
  from 1, and `current_frame.step` is the latest step (`len(history) - 1`).
  The frame's step comes from the engine's action count, the same number as
  `action_num` in the events. The implementation checks
  `history[n].frame.step == n` up to the window start and leaves the trace out
  if it fails, rather than quote steps that point at the wrong board.
- Ranges in the trace are inclusive: `steps 112–117` is six actions,
  `history[112]` to `history[117]`.

## Format: board, actions, board, actions

The trace is a chain of boards and the action batches between them. Each
board is both the result of the actions above it and the start of the actions
below it, so the model reads it as before/after transitions:

```
board → actions → board → actions → … → actions → (first board of the window)
```

- **One action line per batch.** The agent emits actions one `action([...])`
  call at a time, so steps come in runs. An action line covers one call: its
  step range and its actions.
- **Run-length encoding**, within one batch only: consecutive equal actions
  merge, `LEFT, LEFT, LEFT` → `LEFT×3`. `MOUSE` merges only with the same
  target. Action names are copied from `history[n].action`, the same text the
  turn opener's "Executed actions:" line uses (`UP`, `UNDO`,
  `MOUSE(row=21, col=31)`), so the trace adds no new notation. Batches are never
  merged with each other.
- **Long batches** keep the first 12 runs and end with `… (+k actions)`; the
  step range still covers all of them.
- **Every image has a literal text label before it,** such as `Board after step
  117:`. The model sees no alt text, so the label is the only way to tie an
  image to a step.
- **Levels** appear where they change: the first board says which level it is,
  and a level-up starts the next level's section.

## Boundaries

The solver stops a batch at a level-up, a game over or the end of the run
(`solver.py`, `step_env` loop), so those always end an action line. Rules:

1. **Level-up.** The winning action is recorded with the *new* level, and its
   `history` frame is already the next level's starting board (events: action
   #28 `UP` reads level 2 with `level_completed=True`). The trace puts the
   action in the level it cleared, marks it `LEVEL k CLEARED`, then shows two
   boards: the board as the level was solved, and the next level's start
   (`history[n].frame`).
2. **The solved board is not in `history`.** The engine renders it and the next
   level's first board within the one winning action, and only the animation
   chain holds it (`_level_end_grid`: `raw.frame[-2]`). The solver hands it to
   the agent once in the raw payload, where `ARC3_LEVELUP_BOARD_IMAGE` uses and
   clears it. For the trace, the agent keeps it per level when the action
   executes. Its label says it is not in `history`. When TAAF returns no real
   board (a lone transition frame), the line says `solved board unavailable`.
3. **Game over.** The fatal action ends its line, marked `GAME OVER`; the board
   after it is the death board, `history[k].frame`.
4. **Automatic RESET** is its own action line, `RESET (automatic)`. The board
   after it starts the next batch, so it stays in the chain; when it is
   identical to that level's start board already shown in the block, the label
   says so instead of repeating the image (402 tokens saved).
5. **Batch stopped early.** With `ARC3_BATCH_NOOP_BLOCK=1` the solver also
   stops a batch when an action changes nothing inside the board. The line lists
   only executed actions and ends `(stopped: no change, +k skipped)`.
6. **Trim edge.** Trims remove whole turns, so the window starts at a turn
   boundary. The first retained turn's opener echoes the last trimmed batch and
   attaches the board after it. So the trace ends with that batch's action line
   and no image: "The board after step s is the first board in the conversation
   below." The two must list the same actions.

## Example

What the model would see after a trim. Steps, actions and levels are
illustrative; `<image>` is a board image as the harness renders it today (64 ×
64 cells at `MULTIMODAL_UPSCALE=10`). Everything else is literal prompt text.

```text
[system]
You are a coding agent solving a grid-based puzzle game.
… (system prompt unchanged) …

[user]
Earlier play, compressed: the boards and the actions between them, oldest
first. Your reasoning and tool output are not shown. Step n is `history[n]`;
`history[0]` is the starting board. A solved-level board is the only board
here that is not in `history`.

Level 2. Board after step 111:
<image>
→ steps 112–117: RIGHT×4, UP×2
Board after step 117:
<image>
→ steps 118–120: MOUSE(row=21, col=31)×2, DOWN · LEVEL 2 CLEARED
Board as level 2 was solved (shown during step 120, not in history):
<image>
Level 3. Board after step 120, the level 3 start:
<image>
→ steps 121–130: LEFT×5, RIGHT×5
Board after step 130:
<image>
→ steps 131–133: UP×2, LEFT · GAME OVER
Board after step 133, the game-over board:
<image>
→ step 134: RESET (automatic), level 3 restarts
Board after step 134: same as after step 120.
→ steps 135–141: MOUSE(row=9, col=40), MOUSE(row=9, col=44), DOWN×5
Board after step 141:
<image>
…
→ steps 168–170: RIGHT×2, UP
The board after step 170 is the first board in the conversation below.

[user]   ← first retained turn of the window, unchanged
The code executed 3 actions in the previous sequence.
Executed actions: RIGHT, RIGHT, UP.
… (current board image: the board after step 170) …
```

## Size

| Part | Tokens | Count |
|---|---:|---|
| Board image at upscale 10 (`_IMAGE_TOKENS_FALLBACK`) | 402 | one per batch: the starting board, then the board after every batch but the last; +1 per level-up, −1 per restart identical to the level start |
| Action line and image label, e.g. `→ steps 135–141: MOUSE(row=9, col=40), MOUSE(row=9, col=44), DOWN×5` | about 30–50 | one per batch |
| **30 batches** | **about 13k** | |
| Board as ASCII text instead (median of 5,655 frames, Qwen3.8 tokenizer) | 1,617 | four times an image, so not used |

The server limits only video per prompt (`--limit-mm-per-prompt
{"video": 0}`), and history already carries an image on most turns, so about
30 more images are within limits. The block counts toward the context; with
the same 118k trim point, the window loses about 13k of room.

## When the block changes

Only at a trim. At each drain the harness takes the batches that just left the
window, keeps the last 30 batches (the ones directly before the new window
start), and rebuilds the block. Between trims the block is frozen, so the
prompt stays append-only and the prefix cache stays hot, as it does today. A
trim already forces one full prefill, so rebuilding then costs nothing extra.
Before the first trim the block is absent.

Implementation notes:

- Every message entering the conversation is tagged with the latest executed
  step (private key `_arc3_step`). After any trim the retained history starts
  with a user message, and its tag is the last step the trace covers.
- The trace is a user message with private key `_arc3_trace`, inserted at
  index 1. The trimmer never evicts it and stored history never contains it;
  it is rebuilt from the runtime `history` file and cached on its last step.
  Both keys are stripped before a request is sent and from token estimates.
- The eviction loop prices the trace for where history started. When the
  rebuilt trace would push the request over budget (the first trim goes from no
  trace to a full one), the trimmer drops further blocks.
- Batches are recorded from the solver's batch payload in the `action()`
  handler (`action_num`, `executed_count`, `stop_reason`, `skipped_actions`),
  with the solved board from `level_end_grid`. Steps no batch covers, the
  automatic RESET or play from before recording, become batches of their own.
- With images off the trace keeps its text lines only.
- *N* counts `action()` calls, not steps. On `level-up-board` v01 bp35, which
  mostly sends one action per call, 30 batches covered 34 steps.

## Open choices

1. **Cap:** 30 batches (about 13k tokens) or 15 (about 6.5k).
2. **Earlier levels:** keep batches from levels before the current one, or
   only the current level's. Keeping them shows how the previous level was won.
3. **Always keep the last level-up,** even when it is older than the last 30
   batches.
4. **Trim point:** keep 118k (the window shrinks by about 13k) or raise it to
   about 131k (needs that much room under the server's 147k limit).

Flag: `ARC3_COMPRESSED_TRACE_BATCHES` (0 = off, default 0). Choices 2–4 are
implemented as: earlier levels kept, no special keep for the last level-up,
trim point unchanged.
