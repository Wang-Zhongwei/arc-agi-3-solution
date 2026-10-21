# r11l L2 stall in Ucore (mem-unified-core-10x)

## 1. Level and winning route
- Mechanics (engine `r11l.py`, read statically): the orange blob sits at floor(mean) of its 3 diamond legs, the purple blob at the mean of its 2 legs. Clicking a leg selects it (white); clicking elsewhere teleports the selected leg. Gray 'g' walls refuse a leg silently (no move, no animation). Win: each blob overlaps its own-colour ring (orange ring (51,40), purple (18,57)).
- Hidden rule: if a move would leave a blob on sky blue, the blob blinks, the move is undone, and a hidden counter rises; the 5th undone move in one attempt is GAME OVER. Nothing on screen shows the counter.
- The rope is drawn over black and sky-blue cells (why sky blue appeared to "shrink").
- U's route: 13 clicks, one undone (*): (25,45) (9,49) (45,40)* (21,8) (57,40) (25,45) (51,40) (9,49) (45,40) (48,54) (11,57) (35,45) (26,57). None of the 12 other runs died on L2 (uc2, uc3, comb took 45-70 min).
- U's turning point: req 16, after its one undone click (on a black cell): "When I made the click at (45,40), the chain was {(9,49),(21,8),(25,45)} … So the resulting square would have been (30,31) which is in the sky-blue blob."; req 18 "the square's 5x5 footprint at (30,31) would overlap S"; wrote a checker; cleared at req 25 (13 min, 27.8k tokens).

## 2. Stall timeline (Ucore L2 6.0→101.0 min; requests 13-79; 134.7k tokens; 22 actions)
- P1 req 13-26 (6-16 min, 25k): mechanics right by req 16. Its probe click (30,30) landed on sky blue and was undone; it blamed the leg ("water blocks movement (the white diamond tried to move there and bounced back)"). Its planner checked leg cells and ropes against sky blue, never the blob's landing spot. Req 26 ran 9 clicks as a python loop of single `action()` calls stopping only on level-complete/game-over; clicks 10, 12, 14 undone (trace said NO-OP); click 16 = 5th strike → GAME OVER.
- P2 req 27-46 (16-57 min, 57k; incl. 14.4 and 12.2 min parked after trims at req 29, 41): hunting a contradiction. The harness death text steered it: "this action itself was fatal". Req 28 "The water (S) regions SHRANK when the rope crossed them! … ERODES the water" (rendering artifact). Req 30 had the right rule, then "Click 7 … was ALLOWED but we DIED. Hmm, that's still inconsistent". Reachability searches, L-shaped paths; req 45 "a14 fully submerged bounced, a16 20/21 executed. So it's not a simple threshold"; req 46 "a16 SPLITS the water region → allowed but FATAL?!"; req 40 hit 12,288 with no tool call.
- P3 req 47-58 (57-69 min, 29k): gave up on the death; assumed blob, ropes and legs must all stay off sky blue; beam-search planners (req 53 "It plateaus at h=7"); after the trim re-derived select semantics it had at req 16 (req 58 "KEY OBSERVATION: The centers are FIXED…").
- P4 req 59-71 (69-98 min, 13k; incl. 25.6 min parked after the trim at req 59): TypeErrors, broken simulator; an 8-click batch stopped by the no-op guard at (61,13), a gray wall (at req 42 it had decided "noise is NOT an obstacle").
- P5 req 72-79 (98-101 min, 10k): req 73 "the gray 'g' regions are walls"; req 74 "The only anomaly remains a16 … Whatever — avoid blocked cells entirely"; re-planned, cleared at action 29.

## 3. Decisive gap
Same mechanics, different first undone click: U's hit black (forcing the "blob on sky blue" explanation); Ucore's hit sky blue (fits both). The unchecked loop then stacked three more → hidden 5-strike death. Evidence on screen: the animation digest at req 16 showed the blink at the blob's would-be spot ("S3 (18,27)+(28,28)"; black 20 / sky blue 5 cells); every undone click's blob footprint overlapped sky blue (5, 5, 9, 25 cells). Only the strike count was invisible; it saw the contradiction but never counted. Cleared by ignoring the anomaly and finding the walls. Moved-objects lines were not the difference (Ucore found the centroid rule as fast; U's L2 reasoning never cites a moved line; undone moves move nothing; C and keep had no moved line and cleared in 6 and 18 min).

## 4. Classification (134.7k tokens)
| Class | Share | What |
|---|---|---|
| (a) misperception | ~45% | cause of undone moves and the hidden strike count (all of P2, part of P1) |
| (c) tooling | ~35% | unchecked action loop causing the death; planner bugs; a 30 s timeout; the 12k cut-off |
| (b) wrong hypothesis | ~10% | "rope melts water"; "gray noise is not a wall" |
| (d) loop | ~10% | re-deriving after trims |
(e) starvation: 52 of 95 min parked after three trims (req 29, 41, 59), at 1.6-4.4 tok/s vs a median of 57. Outside r11l too: across Ucore's 4 and U's 7 extracted games, 27 of 33 turns longer than 5 min came right after a trim (where the priority scheduler re-queues the game).

## 5. Intervention: "undone action" fact plus a stop
- Fact in the turn message after any action whose final board equals the pre-action board (ignoring the HUD) while its animation changed cells: "Action 10 MOUSE(8,54) was undone by the game (2nd undone action in this attempt). During its animation rows 26-30 cols 52-56 blinked; before the action those cells were black 20, sky blue 5." (undone: post == pre; blink cells toggle >= 2 times across animation frames; colours from the pre-action board.)
- At GAME OVER: "This attempt had 4 undone actions (8, 10, 12, 14) before the fatal action 16." When > 0, drop "this action itself was fatal".
- Behaviour: after an undone or no-op action, skip the remaining `action()` calls in the same snippet (the current guard only works within one batch, and did not stop U's batch on its animated undone click).
- Fires: req 16 (after action 8); inside req 26's snippet (stops after action 10, 2 strikes, no death); req 27 with the count.
- Effect: the rule it had at req 30 ("blob must not overlap water") is no longer contradicted; skips most of P2-P3 (~85k tokens, two trims, ~27 min).
- Not covered: moved-objects silent on undone actions; touching/chaser n/a; own-rules quoted trivia; `<this_level>` "animated and changed nothing" counts per clicked colour lack order within the attempt, the blink location and any link to the death.
