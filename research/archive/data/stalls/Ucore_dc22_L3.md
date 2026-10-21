# dc22 L3 stall, Ucore (mem-unified-core-10x)

## 1. Level and winning route
- Green 2x2 knob moves 2 cells per arrow; checkerboard "ghost" cells are walls. Plugs: red rotates the red bars about the dark-red pivot; blue swaps 1 solid blue square for 3 ghost slots; magenta EXCHANGES the two look-alike P/M 2x2 tokens (room A (16,6), room D (46,12)); if the knob stands on one token it is carried to the other. Stepping on the purple T in the isolated bottom cluster spawns a purple plug at (45,52), which fills the purple corridors and opens the way to yellow.
- Route (U, uc2, ms1, uc1, A): blue, red, walk to the pivot, red, blue, walk left, blue, UP x4 onto token A, magenta teleports the knob to (46,12), UP UP RIGHT RIGHT onto the T, purple plug, red, ~15 moves to yellow.
- Every fast run found the teleport by re-clicking magenta while standing on token A. U req 57 "Let me test the magenta plug again right now (1 action) and see if its effect differs now that the pink core is collected." → req 58 "WOW! … The magenta plug click TELEPORTED THE PLAYER!"; uc2 req 61 "maybe the magenta button rotates the purple bars only when the ball is standing on a socket?!"; ms1 req 55 "earlier … looked like a rotation of each 2x2 token in place… But maybe it was actually a SWAP."

## 2. Timeline (L3 36.8 min → deadline; requests 48–109; 145k tokens; 59 actions)
| Requests | Wall (min) | Tokens | What happened |
|---|---|---|---|
| 48–51 | 37–38 | 6k | Probed the 3 plugs. |
| 52–60 | 38–45 | 26k | Labelled magenta useless; poked inert cells. |
| 61–68 | 46–50 | 14k | Reached token A, then left it. |
| 69–91 | 51–95 | 52k | Knob parked on token A, analysis only. |
| 92–109 | 95–121 | 47k | Walked off the token; deadline. |
- 52–60: harness `frame_diff` printed the magenta click as eight 1-pixel moves ("P from [16,6] to [16,7]"); read as in-place rotation. req 52 "rotating the 2x2 token in place. That's cosmetic-ish"; req 58 "Cosmetic/decoy." Then walked onto the purple ghost, clicked the ghost and T (no-ops), connectivity theories.
- 61–68: reached token A at action 155; called it "walkable pink floor… maybe decoys"; stepped off. Right idea, wrong plug: req 62 "If the knob could TELEPORT between blue squares (ride the blue plug)…"; rode blue, died (death reverts), req 66 "So riding is not the mechanic."
- 69–91: on token A for 44 wall-min (25 a starvation gap), almost no actions; python-only: dither-parity x3, HUD bar x4, L2 socket replays, a reachability BFS rebuilt 4x with 3 bugs, re-clicking inert cells (T x3). req 91 finally queued the winning test ("test the M plug click while standing on the pink tile… If the knob gets 'carried' → interesting") as the 4th of 4 probes; the 1st was a no-op, so the stale-state latch refused the other three. req 92 "The subsequent probes were halted…"; never re-issued.
- 92–109: req 96 walked off; planned a 45-action sweep of unvisited cells (BFS bug again); re-checked the action budget until the deadline; a second starvation gap of 13 min.

## 3. Decisive gap
At req 52 one click changed two distant look-alike tokens at once; a swap looks like each token rotating in place, and the harness diff presented in-place pixel moves. Nothing contradicted "cosmetic" without a test on a token; Ucore's two magenta clicks were at knob (18,30) and (32,22); its one ride test used blue and was over-generalised. Never cleared.

## 4. Classification
(a) ~40% swap misread (diff output reinforced it); (d) ~25% repeated inert clicks/BFS/budget/L2 replays; (c) ~20% latch dropped 9 probe actions in 6 snippets incl. the decisive click, BFS bugs 76–81 and 99–101, 2 tracebacks; (f) ~15% linking swap→teleport still needs a hypothesis; (e) 38 of 84 wall-min with no request.

## 5. Interventions
A. Exchange fact (primary; moved-objects family): after a non-arrow action, group changed cells into 8-connected clusters; for each pair of equal-shape clusters whose before-contents share the colour multiset (>= 2 colours), test after(A)==before(B) and after(B)==before(A). Text: "MOUSE(row=36, col=51) exchanged two look-alike 2x2 patches: rows 16-17 cols 6-7 now hold what rows 46-47 cols 12-13 held before, and vice versa." Fires at req 52 (after action 119) and 58 (after 134). Companion when the player's footprint covers a patch: "Your piece now covers rows 16-17 cols 6-7, one of the two patches MOUSE(row=36, col=51) exchanges." (req 61, and every turn 69–91; caveat: resembles the dead untried-action facts). U's moved line stayed silent after its first magenta click. Noise on Ucore's 25 games: non-arrow only → 3 firings total (dc22 119 and 134, one wa30 SPACE); with arrows 97 (65 in m0r0 mirror moves) → non-arrow only.
B. Fix the stale-state latch: `ARC3_STALE_STATE_BLOCK=1` refuses every later `action()` in a snippet after one no-op; the message says "StaleStateActionError can be caught if a snippet has good reason to keep acting through no-ops" but catching does not help (`_stale_state` clears only when gameplay changes). Fix: don't arm the latch when the no-op was a MOUSE click and the next action is a click at another cell, or let a caught error disarm it for the next call. Effect: the req-91 magenta click (≈95 min, action 169) executes; clear by ~100–105 min likely vs never. With A, ~45 min earlier, leaving time for L4 (U cleared it in 14 min).
No run won dc22 (best stopped at L5): score, not one of the 10 target wins.
Files: S/lab2/extract/{Ucore,U,uc2,ms1}/dc22.requests.jsonl; detectors S/dc22w/exch.py, exch2.py; latch code solver.py (~533, 1160–1230), python_tool_sandbox.py (~550–680).
