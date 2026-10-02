# Harness Engineer

## Objective

Give the model accurate, compact observations and dependable tools so it can investigate mechanics, act, and reuse its work with less confusion and overhead.

## Responsibilities

- Specify action and batch-result semantics, including executed/skipped actions, stop reasons, death, reset, and level transitions.
- Expose board frames, differences, and intermediate animations as data the model can inspect and compute over.
- Prototype higher-level geometric signals while keeping the underlying evidence accessible.
- Specify Python and workspace persistence across tool calls, process restarts, level transitions, and game boundaries.
- Evaluate action guards and output formatting without obscuring valid game behavior.

Extend familiar APIs where practical. Document compatibility before changing an existing interface. Strategy owns when and how the agent uses the tools; Harness owns their behavior and correctness.
