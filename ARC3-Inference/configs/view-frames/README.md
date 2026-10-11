# History frame viewer

First test of the always-available `view_frames(indices, bbox=None)` helper.
It accepts one history index or an iterable, with an optional inclusive crop.
Each image uses 64 visual tokens: a full 64×64 board uses upscale 4, and crops
are enlarged within the same budget. Odd-sized crops have clearly labelled blue
padding. Four images are allowed per Python call and eight per analyzer
invocation. The prompt encourages bbox crops to focus on objects or see details
more clearly.

`v1.json` copies [lorddraft-baseline-no-io/v1.json](../lorddraft-baseline-no-io/v1.json)
on `allocation-vc-one-wave` at `415b436`, changing only the notebook/source-dataset
names and enabling request logs. It keeps all 25 public games, one pass, a
120-minute public soft runtime, and the existing model, generation settings,
context policy, action guards, priority allocation, and 14-sequence serving
setup. Current-board upscale remains 10, tool output remains 3,072 tokens, and
animation images remain off. The notebook envelope includes the normal ten
minutes of overhead; private rerun settings remain 532 minutes in one wave.

Destination: `jonathanwang2022/view-frames`, source dataset
`jonathanwang2022/view-frames-src`. Kaggle assigns the notebook version.

The [recovery design](../../../docs/harness/stall-recovery-design.md) remains a
proposal for later experiments. This test adds the viewer without enabling the
proposed stall recovery reminders or thinking/action nudges. Existing baseline
transition-reminder code and settings are preserved.

The branch was rebased onto the user's allocation baseline and moved into the
current repository checkout. Local validation: **264 passed, 59 skipped** across
the viewer, transition reminders, level-up boards, intermediate images, frame
diffs, animation descriptions, sandbox modules, serving handoff, and public
budget suites. The 42 viewer tests all passed; compatibility skips follow the
branch's existing test policy. Python syntax, local documentation links,
config comparison, and `git diff --check` passed. The config differs from the
allocation baseline only in the three destination names and request logging.
