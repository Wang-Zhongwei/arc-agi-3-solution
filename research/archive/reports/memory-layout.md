# memory-layout: the memory in every request, in plain words

Branch `arm/memory-main` (PR #148), on top of the memory audit
([memory-main.md](memory-main.md), Audit). Arm `mem-layout-10x`.

## The request (2026-10-10)

The user, after the audit: "i think the memory ledger should be pushed EVERY model call, not only
when window needs to be trimmer (i.e should fit both, and due to loss in the middle, the history of
the previous levels should be kept first and then whatever the model has done prior). also the
formatting of the memory ledger is not legible that well."

What the model saw before, in the runs (mem-v2-10x):
- the memory came only after a trim, as a `<memory>` card glued to the head of the oldest surviving
  turn message, so it sat next to that turn's stale `Current state: step 150, level 4` line while
  the card said level 4 was solved and gave the actions for level 5 (ar25 in mem-v2-10x);
- its lines were compressed codes: `<solved>L1: 17 acts [U L D11 L4], spent 25, 0 deaths</solved>`,
  `UP: not tried this level | game moved 5, changed 17, no change 1`, `MOUSE@R`, `W B S Y`;
- the model's own notes were nearly gone (58 tags in user-combo-10x v1, 4 in level-path-10x).

## What changed (all behind flags, off by default)

| Flag | What it does |
|---|---|
| `ARC3_MEM_EVERY_CALL` (needs `ARC3_MEM_SOLVED`) | The memory goes into every request, in two places. `<previous_levels>` is its own user message right after the system prompt: one frozen entry per cleared level (actions taken and spent, deaths, the winning route, the model's recap, up to 4 of its notes on the level). `<this_level>` is the last text of each full turn message: the current level's action and click effects, untried actions, the newest two GAME OVERs, the Seen-before line, and the model's notes on this level, capped at 1,100 chars. The trim-only card, the per-turn Seen-before line and the `<memory_update>` lines are not sent. |
| `ARC3_MEM_LEGIBLE` | Every memory line in plain words: labels, full action names, `x` counts, `click (row, col)`, colour names, one fact per line. Applies to both blocks and, without the layout, to the card, delta and Seen-before lines. |
| `ARC3_LEVEL_COMPACT` (needs the layout) | A level clear is a compaction point: when the cleared level is committed, its turns are dropped except the last 2 (`ARC3_LEVEL_COMPACT_KEEP`), in the same step, and what is left is drained to the drain target if it is above it. |
| `ARC3_MEM_DEBRIEF` (fix) | The recap must be about the cleared level (the review's 14 of 83 wrong-level and 4 non-recap captures). |

### Why two places, and when each one changes

The cache rule on this model: every request must extend the previous one byte for byte, and a break
re-prefills the game's context (~50k tokens at ~12k tok/s). So the two blocks have different jobs:

- `<previous_levels>` is first, as the user asked ("the history of the previous levels should be
  kept first"). Anything at the front breaks the prefix whenever it changes, so it changes only at
  a **commit** (a cleared level is frozen into it) and at a **trim** (the prefix is broken anyway):
  - a level is committed at the first turn start after its recap is settled: the recap is read from
    the first two reasonings on the next level, so that is usually the second turn on the new
    level; at a trim every pending level is committed;
  - with `ARC3_LEVEL_COMPACT` the commit drops the cleared level's turns in the same step, so one
    clear is one re-prefill, and that re-prefill is of a short context;
  - a redraw is priced like a trim (`_context_was_trimmed`): the slot is handed over, and the #58
    eviction guard counts it as a real prefix break.
- `<this_level>` changes every turn, so it goes where change is free: the end of the newest turn
  message. It is built once per turn and never edited, so later requests extend it. It is also the
  position the "lost in the middle" concern asks for.

The model's note channel is back: notes appear in `<this_level>` while it plays a level and move to
`<previous_levels>` when the level is cleared; open hypotheses are kept until confirmed or refuted;
a note is labelled with the level it names (in tn36, "... completes level 1" written on level 2
went to level 2's entry before this); the system prompt says where notes appear and gives the
open-hypothesis form.

### Arm level-debrief (PR #156)

`<previous_levels>` reads `arm/level-debrief`'s pinned debrief (`MemoryLedger.debriefs`, GOAL and
MECHANICS) as `Goal:` and `Rules:` when that branch is merged, and with `ARC3_LEVEL_DEBRIEF` on a
commit waits for the debrief (or two turns). The branch is not merged here (it conflicts with this
branch in `memory_ledger.py`, `tool_agent.py` and `arms.json`); the seam is tested with a pinned
debrief set by hand. After a merge, its extra debrief request must also send `<previous_levels>`
(`_with_mem_head`), as the rolling summary request now does.

## Cost, measured on logged runs

`layout_replay.py` (session scratchpad, `memlayout/`) replays a run's logged requests and replies:
the FactRecorder is fed the logged frames, the ledger absorbs the logged replies (reasoning first),
commits and compactions run as `_mem_layout_turn_start` does, and each request is rebuilt (old card,
deltas and Seen-before lines removed; head and blocks added; compacted turns dropped) and rendered
through the Qwen3.8 chat template next to the logged one. Prefill is the tokens after the shared
prefix with the previous request (Qwen tokenizer, plus the logged image tokens per image). The logged
replies were written without the layout, so this measures cost and placement, not behaviour; the
logged runs' own trims are kept, so breaks are an upper bound.

| Replay | Requests | Breaks/game, logged | Breaks/game, layout | Commits | Prefill, logged | Prefill, layout | Extra prefill | Head tokens mean / max | `<this_level>` tokens mean / max | Memory tokens in context per request | Prompt mean, logged / layout |
|---|---|---|---|---|---|---|---|---|---|---|---|
| user-combo-10x v1, layout with compaction | 2,777 | 5.7 | 8.3 | 104 | 12.39M | 12.79M | +3.2% (+15.9k, +1.3 s per game) | 359 / 1,041 | 152 / 344 | 1,501 | 54.9k / 47.8k |
| user-combo-10x v1, layout without compaction | 2,777 | 5.7 | 9.5 | 104 | 12.39M | 17.54M | +41.5% (+205.9k, +17.2 s per game) | 359 / 1,041 | 152 / 344 | 1,501 | 54.9k / 56.8k |
| mem-v2-10x, layout with compaction | 2,780 | 5.8 | 8.0 | 95 | 12.48M | 12.79M | +2.5% (+12.5k, +1.0 s per game) | 300 / 1,051 | 155 / 340 | 1,500 | 54.7k / 48.0k |

- Every layout break is at a logged trim or a commit (0 unexplained of 208 and 201). 18 and 20
  commits fell on a logged trim; 29 and 24 logged trims became unnecessary after compaction.
- Without compaction every clear re-prefills the whole context: +41.5%. With it the layout costs
  +2.5-3.2% prefill, about 1 s of prefill per game, and the mean prompt is 13% shorter.
- Tokens added per request: the head is cached after its first request; each new turn adds one
  `<this_level>` (152 tokens on average) to the prefill. In context, head plus the retained turns'
  blocks average 1.5k tokens.
- The scripted game loop (`anim_loop_sim`, one clear in a 30k window): 3 breaks against 2 (the two
  trims plus the commit), prefill 27.7k against 24.5k tokens.

## What the model sees, before and after (verbatim)

Before: the logged mem-v2-10x requests. After: the same requests rebuilt by the replay.

**(a) sp80, request 26: the first turn message on level 3 after level 2 is committed.** Before (as mem-v2-10x sent it): the oldest kept message starts with the card from the last trim, drawn on level 2, glued to a level-2 turn message ("Current state: step 14, level 2") while the game is on level 3:

```text
<memory>
  <solved>L1: 7 acts [L D R4 S], spent 11, 0 deaths</solved>
  <actions level="2">
  UP: no interior change on 1 press across 1 board
  DOWN: not tried this level | game moved 1
  LEFT: not tried this level | game moved 2
  RIGHT: not tried this level | game moved 5
  SPACE: no interior change on 2 presses across 1 board | game clear 1, anim only 4
  MOUSE@R 1 target: changed 1; median 128 cells; e.g. (17,12) changed
  colours on the board not clicked this level: M b Y
  </actions>
</memory>
The code executed 1 action in the previous sequence.
Executed actions: SPACE.
[... rest of that old level-2 turn message ...]
```

After (the same request under mem-layout-10x): message 1 is the head, message 2 the oldest kept turn (one of level 2's last two):

```text
<previous_levels>
Level 1: cleared in 11 actions (11 spent, no deaths). Route: LEFT, DOWN, SPACE, RIGHT x5, SPACE, LEFT, SPACE.
  Your recap: Level 1 solved! The win was: streams into both cup holes (fluid filled both cups). Level 1 solution: paddle left edge at 24 so streams land at 20-23 and 44-47.
Level 2: cleared in 11 actions (11 spent, no deaths). Route: SPACE x2, click (17, 12), UP, RIGHT x3, click (37, 30), RIGHT x2, SPACE.
  Your recap: Key mechanic recap: - MOUSE on a red block swaps it with the blue paddle (the clicked red becomes the movable blue; the old blue becomes red/static in place). - Movement: LEFT/RIGHT only (UP was a no-op on level 2).
</previous_levels>
```
```text
The code executed 1 action in the previous sequence.
Executed actions: UP.
Note: a batched sequence was stopped by the no-op guard: action 'UP' (position 1 in that batch) executed but produced no change inside the board area. The remaining  [...]
```

and the turn message ends with:

```text
<this_level>
Level 3 so far: 1 action, attempt 1, no deaths.
SPACE (1 press on this level): the board inside the edge never changed, on 1 board.
Not tried on this level: UP, DOWN, LEFT, RIGHT.
Colours on the board not clicked on this level: magenta, red, blue, yellow.
</this_level>
```

**(b) ar25, request 34: mid-level 3, right after a trim.** Before (mem-v2-10x): the trim glues the card to the oldest surviving turn message, a level-2 turn; the card's `<actions>` table is level 3's, the state line under it is level 2's:

```text
<memory>
  <solved>L1: 17 acts [U L D11 L4], spent 25, 0 deaths</solved>
  <solved>L2: 16 acts [R2 S D8 S L4], spent 23, 0 deaths | how: SPACE (dots off) then LEFT×4 to move the line from center 43 to 31.</solved>
  <actions level="3">
  UP: moved 3, changed 4 | game moved 4, changed 4, no change 1
  DOWN: not tried this level | game moved 19, no change 1
  LEFT: moved 3, changed 8, no change 1 | game clear 2, moved 10, changed 13, no change 1
  RIGHT: moved 5, changed 13 | game moved 10, changed 15
  SPACE: changed 1 | game changed 5
  not tried yet this game: UNDO
  colours on the board not clicked this level: W B S Y
  </actions>
</memory>
The code executed 7 actions in the previous sequence.
Executed actions: DOWN, DOWN, DOWN, DOWN, DOWN, DOWN, DOWN.
You are still on the same level.
Current state: step 44, level 2.
Valid actions right now: RESET, UP, DOWN, LEFT, RIGHT, SPACE, MOUSE, UNDO.
[...]
```

After: the head is its own message; the old turn follows it unchanged; level 3's facts are at the end of the newest turn message instead:

```text
<previous_levels>
Level 1: cleared in 17 actions (25 spent, no deaths). Route: UP, LEFT, DOWN x11, LEFT x4.
Level 2: cleared in 16 actions (23 spent, no deaths). Route: RIGHT x2, SPACE, DOWN x8, SPACE, LEFT x4.
</previous_levels>
```
```text
<this_level>
Level 3 so far: 39 actions, attempt 2, no deaths.
UP (7 presses on this level): moved cells 3 times, changed cell colours 4 times.
LEFT (12 presses on this level): moved cells 3 times, changed cell colours 8 times, did nothing once.
RIGHT (18 presses on this level): moved cells 5 times, changed cell colours 13 times.
SPACE (1 press on this level): changed cell colours once.
Not tried on this level: DOWN, UNDO.
Colours on the board not clicked on this level: white, black, sky blue, yellow.
</this_level>
```

**(c) lf52, request 74: the end of a turn message on level 4.** Before (mem-v2-10x; this message carries no memory lines):

```text
Executed actions: MOUSE(row=25, col=13), MOUSE(row=25, col=25), MOUSE(row=25, col=25), MOUSE(row=25, col=37), MOUSE(row=25, col=37), MOUSE(row=25, col=55).
You are still on the same level.
Current state: step 202, level 4.
Valid actions right now: RESET, UP, DOWN, LEFT, RIGHT, MOUSE, UNDO.

Your retained functions from earlier Python calls in this game: grid_state(fr), cell_rc(r, c), cc(r, c), bar(fr), ppos(fr), ypos(fr), bbox(b), beads(fr), step(a), snap(tag), nbr(p, allc), ccells(fr). These are your previous definitions, available to call directly. Their presence does not mean they are correct; revise them if new evidence contradicts their assumptions.

Current grid image:
Diff image (step 195 -> 201): cells that changed since your previous analyzer turn in their new true color; unchanged cells dimmed to dark navy (an off-palette color).
```

After: the same message ends with

```text
<this_level>
Level 4 so far: 7 actions, attempt 1, no deaths.
Clicks on light green cells (3 clicks on 3 cells): changed cell colours 3 times. Last: (25, 37) changed cell colours.
Clicks on light gray cells (2 clicks on 2 cells): changed cell colours twice. Last: (25, 37) changed cell colours.
Clicks on orange cells (1 click on 1 cell): changed cell colours once. Last: (25, 55) changed cell colours.
Clicks on pink cells (1 click on 1 cell): changed only the edge once. Last: (25, 19) changed only the edge.
Not tried on this level: UP, DOWN, LEFT, RIGHT, UNDO.
Colours on the board not clicked on this level: white, black, blue, yellow, purple.
The current board was seen before on this level: after action 5 of this attempt. From it: click (25, 37) reached the board after action 6. Not tried from it: UP, DOWN, LEFT, RIGHT, UNDO.
Your note (world model): Peg-jumping puzzle: click a peg then a highlighted landing cell; pegs can jump onto a bead's dock cell and ride it; boards scroll as beads move.
</this_level>
```


## Recap precision (ARC3_MEM_DEBRIEF)

Replayed on level-path-10x, mem-v2-10x and user-combo-10x v1: 22, 26 and 25 recaps on 96, 100 and
113 level-ups, in 17, 19 and 17 of 25 games (before the fix: 24, 28 and 31, of which 14 named another
level and 4 were no recap). The 5 kept recaps whose first number is not the cleared level all name
it second ("Level 4 solved L3 with ...", "Level 2 solved... no wait, LEVEL 1 COMPLETED!").

## Tests

- `test_memory_layout.py`: the plain-words renderers and both blocks (the user's example line
  verbatim, caps, compression order).
- `test_memory_layout_ledger.py`: notes per level and by named level, open hypotheses, commits,
  a late recap, level-debrief's fields, the legible card, both system-prompt texts, the turn helpers.
- `test_memory_debrief.py`: the review's wrong-level and non-recap cases from the logs.
- `test_memory_layout_loop.py` (real game loop, Qwen chat template, 30k window): the new knobs are
  inert without the layout (byte-identical requests); every request extends the previous one except
  at trims and commits; the head is its own message after the system prompt in every request from
  the commit on, never next to a state line or a `<memory>` card; every full turn message ends with
  a capped `<this_level>` about its own level; no codes in legible blocks; compaction keeps 2 turns
  of the cleared level, is one break, and no break follows within 3 requests.
- `eviction_sim` (#58 guard): `combo-mem-layout-10x` and its no-compaction variant: no violations.
- Flags off: `test_memory_main_loop.py`'s identity test against main still passes, so
  user-combo-10x's requests are byte-identical. `kaggle/arms.json`'s existing arms are unchanged.
  Notebook fingerprints include the code patch, so any arm built from this branch differs from its
  recorded run's (user-combo-10x builds as 74a49e07f25e70c0 here, 4b7ea0b46b56742c before this
  work, 11d2c46673d51188 on main today; recorded 399abe34aa243a07).

## The arm

`mem-layout-10x` = `mem-v2-fixed-10x` + `ARC3_MEM_EVERY_CALL=1`, `ARC3_MEM_LEGIBLE=1`,
`ARC3_LEVEL_COMPACT=1`; 121 min, all games, user-combo-10x's serve settings. Built locally with
`--new-batch` (`16e52136558ae40f`, f99bca7a), not pushed.

Screens for its first full run, before any score claim:
- model tags at or above user-combo-10x v1's 1.1% of messages (mem-v2-10x 0.22%);
- a recap on at least 15 of 25 games' entries;
- prefix breaks per game near 8 and cache hit within 2 points of user-combo-10x;
- then paired repeats against user-combo-10x with `make significance`.

## Not done here

- The sandbox bug from the audit: a malformed action (`step({'action':'MOUSE',...})` in ka59)
  returns `{'ok': True, 'gp': None}` and runs nothing. It changes main's behaviour, so it is its own
  arm.
- Merging `arm/level-debrief`, `arm/probe-lap` and `arm/fable-facts-main` (known conflicts in
  `tool_agent.py` and `kaggle/arms.json`), and the 3-hour run the user asked for.
