# Prompt-requested emissions do not work in this harness

2026-09-26. Four arms, one conclusion: **asking this model to emit something in
its reply does not produce it.** The finding is mechanical, not a matter of
wording, and it invalidates the whole class of prompt features.

Counts separate the *instruction* from the *model's output*, because a pattern
in the instruction text is otherwise counted as compliance. That error is how
`1,794 <level_note>` and `937 Predict:` were once read as the model complying
when the real figures were ~14 and 13; see the correction in
[memory-ledger-run-0923.md](memory-ledger-run-0923.md).

| feature | prompts carrying the ask | the model wrote | arm |
| --- | --- | --- | --- |
| memory tags (`<world_model>` etc.) | every turn, 950 turns | **~14** | `wang-intel-memory-fixed` |
| `Predict:` line, asked every turn | 924 | **13** | `e1-predict-uniform` |
| `<scratchpad>` | 997 | **1** | `g1-scratchpad2h` |
| spend-actions (behaviour, not a tag) | 1,022 | no behaviour change | `b4-scoring-guide` |

## Why: there is no prose channel to write into

Measured over 1,935 model replies in `e1`:

- **100%** carry a tool call.
- **53%** have *completely empty* assistant content. Median content length: **0
  characters**; mean 70.

The harness forces a python call every turn, so the model's whole output is a
`<think>` trace plus a tool call. Every prompt-requested tag or line has been
asking it to write somewhere it does not write. `_absorb_memory` scans the trace
as well as the content, so both channels were available -- and both are empty of
tags.

## What this explains

- The memory ledger's win came mostly from `note_think()` **auto-distilling**
  level notes out of the trace with no tag required (34 of b1's 48 ledger
  entries), i.e. from code, not from compliance.
- `a3-thinking` and `a4-scratchpad` were dead for a different reason (an env key
  that never reached the notebook), but would likely have been null anyway.
- `b1-thinking-real` was null for a third reason: the addendum restated the
  per-turn protocol's own instruction, so both arms ran effectively the same
  prompt. Rebuilt in `5ebc306` to replace rather than duplicate, and to ask for
  a countable emission -- which then went uncomplied-with, as above.

## What to do instead

1. **A required tool-call argument.** Add e.g. `predict` to the python tool
   schema. The model cannot make a legal call without filling it, so the API
   enforces what prose cannot, and it feeds the predictive-hypothesis loop
   directly.
2. **Code that reads the trace.** `note_think()` is the existing proof.
3. **Stop spending run slots on prompt arms.** Four independent confirmations is
   enough.

## Measurement rules this session added

- Count a pattern in the prompt region and the model region **separately**. A
  single total is almost always the instruction echoing.
- A feature that is re-rendered every turn inflates any whole-transcript count
  by roughly the number of turns remaining. Count **distinct** entries.
- Verify an arm can *vary* something before running it: no line of a new
  addendum may already appear in the prompt (now a test), and every policy must
  be in the routing set (now a test).
- Rank on turn-matched groups and levels-per-turn, never the bare mean: under
  UCB a single game handed 7x the turns can carry the whole mean (`b1`: +91.67
  of its +4.10 came from `ar25` alone; corr(time-share change, score change) =
  0.78).
