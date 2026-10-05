# Looking back at level-ups and game overs

Status: proposed. Runtime unchanged.

Make past outcomes easy to revisit. The agent can decide when a previous success or failure is useful and how to investigate it. A brief mention in the API reference is enough; avoid prescribing a review routine.

`history` and `transitions` already retain the full game after older conversation turns are dropped. Use those existing lists. Separate `level_ups` and `gameovers` globals are optional convenience, not required for this proposal.

## Prompt hint

Add one sentence beside the existing history description:

> `history` and `transitions` RETAIN earlier actions and boards even after conversation trimming, so you can REVIST what led to a level-up or game over.

Leave the choice of when to look back, which events to inspect, and what to infer to the agent.

## Viewing past frames

Add a Python helper, `view_frames(indices)`, to display selected frames from `history` as images. For example, `view_frames([40, 41, 42])` shows those three recorded boards, labeled with their action and level.

Its API description can be one sentence:

> `view_frames(indices)` displays past boards by `history` index; index 0 is the initial board, and index n is the board after action n.

Use the existing image rendering and context-retention paths. Keep image cost modest; choose rendering and limits during implementation to fit the current infrastructure.

## Implementation notes

Preserve the existing history and transition semantics. Level completion includes the final win. A normal level-up's after-frame belongs to the next level; a game-over transition records the fatal action, with automatic RESET following separately. Keep these details in the API reference where needed.

Check that frame selection and labels are correct, including level boundaries and death followed by RESET, and that historical images reach the model through the existing image path.

Relevant code: [prompts](../../ARC3-Inference/inference/agent/prompts.py), [sandbox](../../ARC3-Inference/inference/agent/python_tool_sandbox.py), and [tool agent](../../ARC3-Inference/inference/agent/tool_agent.py).

Compare the hint alone and the hint with `view_frames` against the baseline, following the [A/B guide](../rules/t-test.md). Start with bp35, then broaden to the full game set. Measure score and context cost; more history lookups are not themselves a success criterion.