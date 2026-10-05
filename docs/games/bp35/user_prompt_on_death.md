```
GAME OVER occurred during the previous sequence. This means the attempt FAILED (it does NOT mean the run is finished). The game has already been automatically RESET: the current level restarted from its initial state and previously completed levels are kept. A game over is caused either by the specific action taken (e.g., moving onto a hazard cell) or by a step/time budget running out, which is usually indicated by a bar at the grid border that shrinks with each action. Before acting again, diagnose the cause by inspecting the board right before the death. `history[-1].action` is 'RESET', `history[-2].action` is the fatal action, `history[-2].frame` is the death frame (the board after the fatal action), and `history[-3].frame` is the board immediately before the fatal action — check what cell the fatal move targeted there, and inspect the budget bar in the death frame. BAR RULE: the bar shrinking with each action is NORMAL — a slightly shorter bar is NOT evidence of the cause of death. The budget caused the death ONLY if the bar is fully (or almost fully) depleted in the death frame (`history[-2].frame`). If it IS depleted: you must reach the goal in fewer actions, or look for a way to restore the bar (e.g., collecting an item that refills it). If the bar clearly had budget remaining: the fatal action itself caused the death (hazard or forbidden move) — do NOT retry the same path; a DIFFERENT approach must be tried. Record the diagnosed cause in your world model.
That fatal action animated over 27 frames, and inside it 1968 cells changed and changed back around rows 0-62, cols 1-62. Those cells are in neither `history[-3].frame` nor `history[-2].frame` - read `last_animation_timeline` to see what happened between them.
The game over occurred immediately after action 'MOUSE(row=33, col=39)'. Apply the BAR RULE above to decide: bar fully depleted in the death frame means budget death (shorter route or bar restore needed); bar with budget remaining means this action itself was fatal and a DIFFERENT approach is required.
Per-action trace of the previous sequence: 1: MOUSE(row=33, col=39) -> GAME OVER (attempt ended immediately after this action).
Do NOT re-submit this same series of moves: the game is deterministic, so replaying the sequence that just killed you will kill you again. Change the plan BEFORE the fatal point - a different route, action, or timing. If you believe the same moves 'should' work, that belief is exactly what this death falsified; record the correction in your world model instead of retesting it.
```


new prompt
```
GAME OVER: the attempt failed. The game automatically RESET the current level; completed levels are kept.

Before acting, diagnose the death:
- `history[-1].action` is RESET.
- `history[-2].action` is the fatal action; `history[-2].frame` is the death frame.
- `history[-3].frame` is the board before the fatal action. 

BAR RULE: the border budget bar normally shrinks with each action. Attribute death to the budget ONLY if the bar is fully or almost depleted in the death frame. Then use fewer actions or find a way to refill it. If budget clearly remains, then the last action must trigger some mechanism that leads to death. What is that mechanism, and how to avoid it?

Previous sequence: 1: MOUSE(row=33, col=39) -> GAME OVER.
The fatal action animated over 27 frames. 1968 cells changed more than once around rows 0-62, cols 1-62. Inspect `last_animation_frames` or `last_animation_timeline`.
```
