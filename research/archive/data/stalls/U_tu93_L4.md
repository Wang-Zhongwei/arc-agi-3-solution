# tu93 L4 stall in run U (mem-unified-10x)

Cause: the 2nd L4 death was the action budget running out (bar empty in the death frame); the model read it as a ranged kill by the orange and spent 24 requests / ~72k tokens / ~39 min building ranged-kill models. Run C had the identical stall (82 min).

## 1. Level and winning route
- 9x9 maze. Player (8,0), goal (2,0). Red turret at (0,2) facing down: its front cell (2,2) is the only way into the goal, so it must be eaten from (0,4) first. Orange patroller on column 6: one cell per move, rows 2→4→6→8→6→4→2→0 (period 8), kills on contact; column 6 is the only way up. Budget bar on row 63: 64 px, about -3.2 px per action → ~20 actions per attempt.
- Winning route (17 moves from a fresh reset): R L R L R R R R U U L U U L L D L.
- Fast runs: pf req 32 "the ORANGE enemy moves EVERY turn … after … capturing, its facing flipped to UP … it bounces/patrols up and down column 6"; planner at req 34 found the 17-move route. A died the same way on action 20 of an attempt, then at req 37–38 printed the bar per frame: "the bar is None (gone entirely) → the budget ran out … DEATH BY BUDGET"; req 39: "17 moves from the start — within the 20-move budget". A cleared L4 in 9 requests.

## 2. U's stall (L4: 1900→5205 s, requests 24–64, 33.6k→158.4k tokens)
1. Req 24–26 (1900–2101 s, ~11k): read the orange as a 2nd static turret, probed R R R; death 1 at action 75 was real contact (bar 54/64).
2. Req 27–32 (to 2621 s, ~33k): decided the orange is a pathfinding chaser; req 30 (12k tokens) concluded unwinnable; a worst-case belief-set solver found no plan; ignored the notch flipping D→U at the bottom (visible in the death frame); probe R R L L at req 32 refuted the chaser.
3. Req 33–37 (to ~2790 s, ~7k): found the bounce patrol (period 6, then 8); planned 18 more moves at step 83 with ~13 actions of bar left. Req 37: "Fine — the budget is generous (64)." Death 2 at action 96 = move 20 of the attempt, bar 3→0.
4. Req 38–61 (2798–5141 s, ~72k): misdiagnosis. Req 38: "So the orange killed the player from a distance?? … a sweeper that kills anything in front of it!"; req 41: "kills … along the column, through white blocks". Req 42, 44, 47, 48, 53, 57, 58: every planner under each lethality variant returned no plan; refined the premise instead of rejecting it (req 47: "R = 2 exactly is the only consistent model"; 57–59 parity arguments). Tooling: retention/NameErrors at 29, 48, 54, 55; hard-coded start cell at 56; req 58 cut at 12,288. Starvation: req 49 (cache 0 after compaction) took 20 min wall (3500→4693 s) for 2.9k tokens.
5. Req 61–64: req 61 "what if the player died because of a TIME LIMIT? The bar!"; req 62 "HUGE finding: the bar … shrank from 64 … to 3 by frame 95, and frame 96 (the death) has NO bar … TIMEOUT"; 16-move plan; cleared at action 115, ~5205 s.

## 3. Decisive gap
All evidence was in context: the death message's BAR RULE told it to check the bar in the death frame; the bar was gone in the death-frame image; positions at death (player (2,6), orange (6,6)) equalled the attempt-3 state at action 110, which it survived. It judged the bar by eye ("≈84%" at req 30; req 56 "the bar looked nearly full in the image"). At req 45 a no-op LEFT shrank only the bar ("changed only the edge") and it still didn't connect it. The confirmed-rules ledger amplified the wrong belief: kept quoting "Confirmed chase, speed 1, kills on contact" after refutation; at req 45 quoted "there's an additional lethal rule". Turning point: req 62, the bar measured numerically in code.

## 4. Classification (~125k tokens)
(a) misperception of the budget bar ~55% (death 2 and all of step 4); (b) wrong mechanic held ~25%; (c) tooling ~10%; (e) starvation ~20 of 55 wall-min, ~3% tokens; (d) loop minor.

## 5. Intervention: computed bar fact
- Detect: a single-colour run along a frame edge whose length only shrinks within an attempt and refills on RESET/level-up; fit px per action from the attempt's history.
- Every turn (`<this_level>`): "Budget bar (bottom edge): 42/64 px, −3.2 px per action this attempt → about 13 actions left before it empties." Would have fired at req 35–37, before the 18-move plan.
- On every GAME OVER, replacing the BAR RULE advice with a verdict: "Bar in the death frame: 0/64 (empty). It emptied on action 20 of this attempt, so this game over is the action budget running out. At death the player (2,6) and the orange (6,6) were on different cells." On a contact death: "Bar at death 54/64 (about 17 actions left): not a budget death." Would have fired at req 38 (C: req 44).
- Saving: after the insight it cleared in 2 requests → ~72k tokens / ~39 min; at req 37 it would likely have prevented death 2.
- Not covered by existing facts (own-rules quoting amplified the wrong belief). Not the dead "action-count/clock" fact: a verdict on the cause of death that refutes the hypothesis being built.
- Lab state: `tu93_L4_budgetdeath` at U req 38; success = reasoning names bar/budget as cause, no ranged-kill claim, or next plan from a fresh reset <= 20 moves. Replicate C req 44; positive control A req 37.

## Level 6 (brief)
Started ~6842 s (~7 min before the deadline): 8 requests, 24.8k tokens, req 119 cut at 12,288. Death at action 154 (4th move) was contact: at req 121 it predicted the patroller would come down to (2,6) and moved LEFT into it ("whatever, it works out"); an earlier UP no-op from misreading (0,8) as passable. Mostly (e) time: U 5/9 vs Ucore 9/9 due to the 55-min L4 stall; Ucore cleared L4 in 7 requests (its 2nd death was real contact, no budget trap).
Files: Ucore tu93 extract at S/lab2/extract/Ucore/tu93.*; bar/position tracks from kaggle-outputs/v4-mem-unified-10x/v1/artifacts/tu93-0768757b_p0_events.jsonl.
