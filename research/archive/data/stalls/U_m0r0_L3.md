# m0r0 L3 in run U: a silent tooling failure, then starvation

## 1. Level and fast route
Two sky-blue 4x4 squares move together; the right one's LEFT/RIGHT is mirrored. Three small blue 2x2 dots are solid. Clicking a dot selects it (dot yellow, squares grey); arrows then move only that dot (not mirrored, 4-px steps). Clicking a square returns control to the pair. The dot at the centre cell (2,6) plugs the only corridor between the halves; another dot blocks the right square's route. Win: park the dots, reselect the squares, merge them (both into one cell).
uc3 cleared it in 24 min / 29 requests / 37k tokens: probe arrows; click dot (32,40) and steer; click a square; park the three dots in dead ends (2,3), (8,1), (4,10) with click+steer; click a square; 25-move merge by BFS. uc3 i=47 "the yellow marker MOVED! ... So the clicked marker became a controllable object!"; i=50 "The b at (3,7) blocks the only connection. I must move it away by clicking it and steering it."
The fast runs had no insight U lacked: U had the same model by i=61-64 and a valid plan at i=67.

## 2. Timeline in U (1259→6782 s = 21.0→113.0 min, requests 46-111, 137.7k tokens)
- P1 i=46-61 (21-26 min, 18k): productive. i=61 "MAJOR discovery: The YELLOW token moved! ... the token at (2,6) blocks the only [crossing]".
- P2 i=62-71 (26-39 min, 31k): productive but slow. i=64 clicking a square restores control. i=67 a 51-action plan (park A, B, C, then a 24-move merge), 4 turn-budget yields. Simulated under the true rules: it merges on its 51st action.
- P3 i=72-80 (39-62 min, 29k): phantom no-ops. The executor called `action(['L'])` (BFS alphabet). Host returned `{'executed': False, 'error': "Unknown action at index 1: 'L'"}`; nothing raised, no `[action]` echo; the verifier printed `MISMATCH`. i=72 "token A did NOT move from (2,7) to (2,6)" (LEFT never pressed). i=77 same bug with 'R'; i=78 "the token at (1,7) can't move right either!" → "midline gate" theory, a 12,288-token cut-off at i=76, two python errors. Starvation gap 1: i=79-80, 929 s for 2.5k tokens.
- P4 i=81-101 (62-92 min, 48k): false rule + joint BFS. '#'/'.' notation mix-up; i=84 invented "token moves only if the LEFT slider's target in that direction is also free" (fits the phantom data); staged 6-object BFS with 3 python bugs and a HUD-bar detour. Starvation gap 2: i=88-89, 988 s for 2.5k tokens.
- P5 i=102-110 (92-97 min, 12k): same bug a third time. i=103 "The LEFT press did NOTHING! ... if EITHER slider is blocked, NEITHER moves". i=109 printed `last_action_call_result`; i=110 "The exec_plan call FAILED: `Unknown action at index 1: 'L'` ... the LEFT was never executed at all! ... My model is fine!"
- P6 i=110-111: 26 then 27 verified actions. Starvation gap 3: 944 s for a 539-token request before the final batch. Cleared at action 139.
All three gaps started on the first request after a context trim (messages 55→22, 44→20, 74→45; cold ~40k prefill) while other games ran ~1.8k tok/game/min. Gaps total 48 min = 52% of the stall's wall time.

## 3. Decisive gap
Not a missing mechanic: its action names were silently dropped. Evidence in context: the turn message "The code executed 1 action ... Executed actions: MOUSE(row=15, col=35)" for a 6-action snippet; `<this_level>` kept "LEFT (2 presses)" unchanged; the i=80 history shows no LEFT at (2,7). It read its own `MISMATCH` as game behaviour and built two false rules. Once the error dict was printed (i=109) it fixed the names in one request.
Elsewhere: comb hit the identical bug on this level (i=85, 'U') and fixed it in one request because its code printed the result; ms1 on L5 ("Nothing was executed ... redo with full names"). Visible `Unknown action` also in re86 (U), tu93 (Ucore), tu93 and wa30 (comb), re86 (uc3) — only cases where the model printed the result.

## 4. Classification (137.7k tokens)
(c) execution/tooling ~64% (P3-P5, 88.6k tokens, ~58 wall-min): silent unknown action names; the visible symptom is (b) (midline-gate and coupling rules held against history, both from phantom evidence). (e) starvation 0 tokens, 48 wall-min (all after the bug began). Normal discovery/planning ~35% (P1-P2, 48.6k; fast runs used 37-87k). (a)/(d) <5%. (f) none.

## 5. Intervention (fires at i=71)
- Fix 1, fail loudly: in the sandbox `action()` (`python_tool_sandbox.py`), when the host's `action_result` has `executed: False` and an `error`, print an echo line and raise. Echo: `[action] 'L' -> NOT EXECUTED: unknown action name`. Exception: `InvalidActionError: 'L' is not an action name; nothing was executed. Valid names: UP, DOWN, LEFT, RIGHT, SPACE, MOUSE, RESET.`
- Fix 2, aliases: `U/D/L/R -> UP/DOWN/LEFT/RIGHT` in `MODEL_TO_ENGINE_ACTION` (`action_names.py`). Don't alias `S` (U used it for "select squares").
- Either fix at i=71: the 51-action plan runs at once; L3 clears ~39-40 min instead of 113 (~73 min, ~89k tokens saved), and the three cold-cache gaps don't happen.
- Not covered by existing facts (game-state facts; this is a tool-contract fact).
- Secondary: every 15-min stall began on the first request after a trim; check whether trimmed cold-prefix requests of a stalled game land at the back of the scheduler queue.
Files: ARC3-Inference/inference/agent/python_tool_sandbox.py (`action()` ~598-690 never checks `error`); inference/framework/solver.py 1043-1051 (builds the "Unknown action" payload); inference/agent/action_names.py.
