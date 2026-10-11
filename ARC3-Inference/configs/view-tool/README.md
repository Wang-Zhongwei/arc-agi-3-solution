# View tool

First test of `ARC3_VIEW_TOOL`. The agent gets `view(indices, bbox=None)`, which
shows one or more `history` boards as images, optionally cropped. Once conversation
has been trimmed, a note after the system prompt names the `history` range the
trimmed conversation covered and suggests `view` with a bbox for key moments such
as level-ups or game overs. See [CONFIGURATION.md](../../CONFIGURATION.md#view-tool).

`v1.json` copies [lorddraft-baseline-no-io/v1.json](../lorddraft-baseline-no-io/v1.json)
and changes only the notebook/source-dataset names, request logging (on), and
`ARC3_VIEW_TOOL=1`. It keeps all 25 public games, one pass, a 120-minute public
runtime, and the baseline's model, serving, context, guards, and allocation.
Transition reminders (PR #30, reverted) are not part of this branch.

Destination: `jonathanwang2022/view-tool`, source dataset
`jonathanwang2022/view-tool-src`. Kaggle assigns the notebook version.

[Version 1](https://www.kaggle.com/code/jonathanwang2022/view-tool) was pushed
2026-10-11 from commit `f7722e4`; the bundle carries `ARC3_VIEW_TOOL=1` and its
harness files match that commit. Status at push: `RUNNING`, not yet evaluated.
