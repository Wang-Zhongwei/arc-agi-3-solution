# The levels that decide wins 8-10: transcript read (2026-10-13)

The target is 10 of 25 games won in the 2-hour run. Over the 11 recent full runs (uc1-3, pf, ms1-2, keep, A, B, C,
comb) wins 8-10 depend on five games' last levels. Five readers (one per game) compared the runs that cleared each
level with the runs that stopped there, using compact per-turn extracts of the request logs, the per-action boards
and, statically, the exact engine source. This file collects their reports; the lead checked every right move and
every fact line used in the state lab against the rebuilt engine states and the logged boards.

| game, level | runs that stopped there | main cause | lab state(s), round 2 |
|---|---|---|---|
| m0r0 L6 (and B's L1) | 8 of 11 at L6, never cleared; B scored 0 (stuck on L1) | two touching same-colour squares form one segmentation node and are read as "merged" (B L1, uc1 L6: one action from a win); hazards treated as walls in BFS; trimmed selected-dot rule | m0r0_L1_touch (B 19), m0r0_L6_touch (uc1 116), m0r0_L6_selected (A 165) |
| tu93 L9 | 7 | mostly time (clean L9 ~700-800 s); walking back into the chaser's cell (A); entering the goal from a live guard's front (uc1) | tu93_L9_trailback (A 141), tu93_L9_finishplan (B 154) |
| sc25 L6 | 4 | identical belief in A, comb, keep: pad A is a 2x2 spot; the corner frame that marks the next big-teleport pad moved unnoticed | sc25_L6_ringA (A 100) |
| r11l L6 | 4 | A: own L4/L5 rules trimmed, 54 calls before the first action; others out of time | r11l_L6_ledger (A 152), r11l_L6_noaction (A 172) |
| vc33 L4 (and L7) | 4 at L4/L5, 2 at L7 | the orange gate (open only at one bar-height state) never clicked or only seen mid-batch | vc33_L4_reprobe (A 56) |

States, criteria and variants: `analysis/state_lab/states/round2_*.json` on `tooling/state-lab`.


## m0r0 transcript read (reader agent, 2026-10-13; saved by the lead). Scripts: lab2/m0r0/ (l6sim.py rules port of
engine 492f87ba read statically; checked against logged boards of uc1, ms1, keep, B; dist.py/dist.pkl distance-to-win).
## L6
Two mirrored sky-blue squares P (3,4), Q (9,4); doors open only while a square stands on a matching plate; a blue dot
plugs the middle wall hole: click it (px 43,31) and arrows move only the dot; checkered walls reset every object;
win: both squares on one cell, or two side-by-side squares swap. Route 38 actions (U U, click dot, L U U L U U U U L,
click off, D D D, click dot, D D, click off, D L D D D R x6 D R D U U U U). Final frames show two squares stacked in
adjacent cells. Engine: P moves (dx,dy), Q (-dx,dy); players don't block each other; dot blocks players; hazard ->
8-frame flash then all sys_click sprites to level-start; doors removed while a player is on a matching plate; merge =
same cell or horizontally adjacent pair that swaps; 150 actions/level cap; L6 has no ordinary walls.
## Stuck runs (best dist to win; 38 at start)
A L6 at 6961 s, 16 req/5 acts, best 38 (i=165: selected the dot, tried only UP (blocked), "clicking the dot
deactivated the blocks - that's bad", deselected; its L3 rule (i=63) that a selected dot moves had been trimmed).
comb 4980 s, best 35 (second BFS dropped click toggles -> "No solution"). keep 6421 / ms1 6482 / ms2 3840 / pf 5640:
best 18-20, BFS treated checkered hazards as blocking though each had written "checkered = fatal"; pressed into the
wall, level reset. uc1 4140 s: best dist 1 at action 236 (~5040 s); its plan stopped one UP short because its merge
check fired on two touching squares (one 4x8 node); i=114 "merged-blob case (h=8) means both squares are in the same
cell -> win"; i=116 "So merging at (8,3) did NOT win" -> changed goals for ~2200 s; its L5 lesson (i=99 "adjacent,
not merged ... detection artifact ... one step short") trimmed by i=116. uc3: passed cell coords as pixels.
B (leader) scored 0: never left L1; one RIGHT from clearing L1 at action 18 (~600 s), and at 42 and 44; i=19 "merged
into ONE 10x5 block (segmentation shows a single S node at (9,29)) ... So the goal is something else"; long stalls.
Patterns: touching objects read as merged (B L1, uc1 L6, uc1 L5; each one action from a win); hazards as walls in BFS;
dot selection forgotten after a trim; pixel/cell coordinate confusion.
## Proposed states (agent's)
- m0r0_L6_touch | uc1 | req 116 | right: UP (wins L6 and the game) | success: actions_prefix [UP]; level_up; no_reset |
  change: new_fact "Same-colour objects that touch are listed separately: sky blue S x2 - 4x4 at rows 34-37, cols
  14-17 and 4x4 at rows 38-41, cols 14-17 (touching; segmentation shows one 4x8 node). Sky-blue pixel total: 32,
  unchanged since the level start." (split a component equal to k earlier same-colour objects when the colour total is
  unchanged); own_rule_quote: uc1 i=99 line.
- m0r0_L1_touch | B | req 19 | right: RIGHT (clears L1) | success: actions_prefix [RIGHT]; level_up; no_reset |
  change: "sky blue S x2 - 5x5 at rows 9-13, cols 29-33 and 5x5 at rows 9-13, cols 34-38 (touching; segmentation
  shows one 10x5 node). Sky-blue pixel total: 50, unchanged since the level start."
- m0r0_L6_selected_A | A | req 165 | right: LEFT (first of a 40-action win; RIGHT also moves the dot) |
  success: any_action [LEFT, RIGHT] within 2; first action not MOUSE; no_reset | change: own_rule_quote (L3 before
  action 72: "The yellow dot MOVED UP ... So the selected (yellow) dot is a movable object, moving in 4-cell steps,
  while the blocks are frozen."; L4 before action 143: "when a dot is selected, only the dot moves (blocks frozen).");
  untried_line: "Since MOUSE(row=43, col=31) at action 215 recoloured the board (b->Y, S->w): UP pressed once, no
  change. Not pressed since then: LEFT, RIGHT, DOWN."
Engine checks asked: uc1 history to 236 then UP wins; B history to 18 then RIGHT clears L1; A to 216 then LEFT moves dot.

## sc25 L6 transcript read (reader agent, 2026-10-13; saved by the lead because the agent could not write files)
Scripts and dumps: lab2/sc25/ (trace.py, parks.py, dump_<run>_L6.txt).

## Level
Start board identical in the 9 runs that reached L6 (md5 3591657e). Three regions joined only by teleports:
R1 (start: 4x4 block, red gate, big pad C at rows 37-40 cols 53-56), R2 (exit corridor behind an orange gate,
big pad A at rows 33-36 cols 29-32), R3 (pocket: small pad B, red target). Spells on the 3x3 keypad: plus/diamond
toggles 4x4<->2x2; yellow-L (50,25)(50,30)(55,30) teleports (2x2 -> B; 4x4 cycles C->A->C, reset at RESET/game over;
a 6x6 ring of yellow corner dots marks the next big destination); magenta column fires a ray (red target removes the
red gate, O target the exit gate). Budget 60 steps. Shortest route ~37 actions. Winners B (44 actions, 10 min), pf (47).
Engine read statically: experiments/arc-agi-3/data/environment_files/sc25/635fd71a/sc25.py (big-pad list [C, A],
index advances per big teleport, reset in on_set_level; ring redrawn around the next big pad).

## Deciding mistake (A, comb, keep: identical)
Both gates cleared, 4x4 block in R1, ring on A, win 11 steps away with 26-36 steps left. Each cast plus + yellow-L
(-> dead-end pad B) instead of yellow-L alone. Belief: A is a 2x2 spot "rows 34-35 cols 30-31" (row 33 reads YggY)
and big teleports only go to C (comb i=74 "TWO 2x2 pads (A and B) and ONE 4x4 pad (C)"; keep i=86 "The 4x4 teleport
goes to comp C - useless"; A i=100 "as a 2x2 at bracket #1"). From L5 (only big pad had the ring: "8 dots = 4x4").
Nobody noticed the ring move C->A at the first big teleport (A action 241, comb 193, keep 197); the tool result said
only "43 cells changed and changed back". keep's context at i=86 still held "Or maybe the teleport cycles through the
spots!" (dropped). All five winners called A a 4x4 pad; ms1 (i=105) chose the big teleport at this exact state.

## Timing (t of 7260 s)
A: L6 at 3877, mistake i=100 (t~4396), parked 4714->7076. comb: L6 at 2483, mistake i=80 (t~2970), parked
3389->6424; after RESET hit A on the run's last action (259), 7 moves short. keep: L6 at 3751, mistake i=86 (t~6270),
its first request after a 32-min park. uc3: never found the free 4x4 spot after clearing red (needed UP x3 into the
cleared target cell rows 33-36 cols 17-20; i=88-90 "nothing fits! ... unsolvable"); bar to 0; also believed A is 2x2
(i=110); parked 40 min. C stopped at L5 ("diamond only shrinks", i=93; 43 min parked). ms2: 96 min on L1.
Classes: A (a); comb (a)+(b); keep (a); uc3 (c)+(a)+(b). "Parked before the insight / reached L6 late" does not hold.

## Proposed states (agent's)
- sc25_L6_ringA | run A | request 100 | right move: MOUSE(50,25), MOUSE(50,30), MOUSE(55,30) (panel clicks at col<=9
  allowed first; block lands rows 33-36 cols 29-32) then UP, RIGHT, UP x5 (= B actions 241-248).
  success: any_action; no_reset; expr: level_up or the block is still 4x4 (16 b/S cells) inside rows 13-36 cols 29-36.
  change: new_fact, moved-marker registry: "Moved since level start (not by arrow keys): yellow 4-dot frame (6x6),
  corners (36,52),(36,57),(41,52),(41,57) around rows 37-40 cols 53-56 -> now corners (32,28),(32,33),(37,28),(37,33)
  around rows 33-36 cols 29-32. It moved at action 241 MOUSE(row=55, col=30), when the 16-cell b/S object moved from
  rows 33-36 cols 17-20 to rows 37-40 cols 53-56. Corner-marked boxes now: Y rows 33-36 cols 29-32 (inside 12 g: open
  4x4); Y rows 37-40 cols 53-56 (open 4x4); Y rows 40-43 cols 12-15 (inside p 2x2 at rows 41-42 cols 13-14)."
- sc25_L6_ringA_comb | run comb | request 80 | same (moved at action 193).
- sc25_L6_ringA_keep | run keep | request 86 | keypad cells MOUSE(49,24),(49,29),(54,29) (moved at action 197).
- optional uc3 i=88: UP x3 then plus; fact "cells opened since level start: rows 33-36 cols 17-20 (was the M box)".
Scheduler note: a final-level hold would likely have converted comb (and maybe keep) after the mistake.

## tu93 L9 transcript read (reader agent, 2026-10-13; saved by the lead)
Scripts: lab2/tu93/ (prefix.py, deaths.py, ledger.py, follow2.py, times.py, L9_<run>.json).
Maze grid 7x8; cell (i,j) top-left pixel (10+6i, 9+6j).

## Level
Start board identical in all 10 runs. Player (4,4), goal (5,4). Red guards (5,3) facing right (goal is its front
cell) and (6,3). Three orange patrollers (1 cell/move, bounce). Dark-red chaser wakes when the player enters its line,
then stands on the player's cell from 2 moves earlier (14/15 moves in A). Moving onto a red guard from the side removes
it; ending a move on a red guard's front cell lets it step onto you (game over).
Winning route (uc3, ms2, keep; 29 moves): L L U U R D D L U U R U U R R R D | D R D L D L D D L L U R
(eat (6,3) from the right, then (5,3) from below, then step right onto the goal).
Deaths on L9 (9): orange at move 6 (uc3, ms2, uc1 att 2); red guard onto goal cell at move 25 (uc1, keep); chaser
(A twice, comb once: stepping back into the cell left 2 moves earlier; pf once after the chaser woke).
"Walked into a known guard ray after the rule was trimmed" holds for uc1 and keep's first attempt, not for A.

## Stuck runs (t of 7260 s)
| run | L9 start | left | reqs | moves | tokens | outcome |
| uc1 | 5173 | 2087 | 42 | 31 | 114k | 24/29 right, died entering goal |
| pf | 2842 | 4418 | 108 | 11 | 208k | stalled building chaser models |
| A | 5952 | 1308 | 32 | 29 | 64k | 18/29 right, 2 deaths |
| ms1 | 6116 | 1144 | 31 | 4 | 63k | planner found no plan |
| B | 6527 | 733 | 26 | 20 | 41k | 20 moves on route, 9 known left |
| comb | 6497 | 763 | 15 | 5 | 45k | died at move 5 |
| uc2 | 6767 | 493 | 11 | 2 | 24k | out of time |
Winners needed ~1250-1900 s on L9 (plus stalls), B shows a clean run ~700-800 s.
Classes: uc1 (a: never read guard facing; stated the guard rule on L5 req 41, only a fragment left by L9);
A (a+b: derived "exactly 2 behind" on L8 req 110, trimmed; re-derived lag 3 (124), then "CURRENT position" (129);
req 141 ran D L U D into the chaser; solver planned entering the goal from the guard's front);
B (b, partly a: req 154, 236 s left, 8 calls asking whether the chaser was destroyed - it was hidden under an orange);
pf (a, partly c); ms1 (a+b); comb (b); uc2 (b).

## Proposed states (agent's)
- tu93_L9_goalfront | uc1 | req 137 | right: DOWN RIGHT DOWN LEFT DOWN LEFT DOWN DOWN LEFT LEFT UP RIGHT (wins) |
  success: won OR (actions_prefix [DOWN,RIGHT,DOWN,LEFT,DOWN,LEFT,DOWN,DOWN] AND not game_over) |
  change: contact ledger new_fact ("you moved onto an R sprite 14 times - each time the R vanished and you survived ...
  3 GAME OVERs came from an R moving one cell toward its marker pixel onto the cell you had just entered. Live R now:
  ... (marker right) ... (marker top)"); second arm own_rule_quote (req-41 sentence). Replicate: keep req 121.
- tu93_L9_trailback | A | req 141 | right: DOWN RIGHT DOWN LEFT ... | success: actions_prefix [DOWN,RIGHT] AND not
  game_over; regex reasoning t-2|2 (moves|steps|turns) (ago|earlier|behind)|lag.?2 |
  change: follower new_fact ("the dark-red r sprite was on the cell you occupied 2 moves earlier in 14 of your last 15
  moves (1 earlier: 0/16; 3 earlier: 0/14). Your cells 2 and 1 moves ago: ..."); computed by comparing each sprite's
  position with the player's k=1..3 moves earlier, reported at rate >=0.8 over >=4 moves. Earlier variant: A req 128.
- tu93_L9_finishplan | B | req 154 | right: LEFT DOWN LEFT DOWN DOWN LEFT LEFT UP RIGHT (wins) |
  success: won OR (actions_prefix [LEFT,DOWN,LEFT,DOWN] AND not game_over) | change: budget_time ("Run clock: 7024 of
  7260 s used, 236 s left; your last 17 calls on this game averaged 29 s.").

## r11l L6 transcript read (reader agent, 2026-10-13; saved by the lead). Tools: lab2/r11l/ (grade.py, ledger.py,
S1_ledger_notes.txt, trace.py, times.py, boards.json).
Level (engine read statically, r11l/495a7899/r11l.py): two white bodies at floor(mean of leg centres)-2; clicking a
leg selects it, any other click teleports the selected leg (only wall-overlap check). A body absorbs any colour piece
it overlaps. Goal met when a body on it has exactly the goal's colour SET. b2 collects b,p,N -> {p,N,b} goal (12,51);
b1 collects M,Y,S -> {Y,M,S} goal (55,11). R, r, O decoys; pink ring decoy goal. 60 actions/level. Start board
identical in 9 runs. Winners 17-24 clicks. pf route: M(61,8) M(19,4) M(55,18) M(7,18) M(61,8) M(25,56) M(19,22)
M(5,55) M(7,18) M(6,42) | M(43,49) M(3,40) M(31,14) M(49,2) M(57,50) M(61,48) M(61,18).
Stuck: A (L6 at 5173, 1786 s / 54 requests to first action, 59 req / 3 actions: class a); keep (6414, b); ms2 (6263,
b; at i154 one click from the win with ~95 s left it rechecked a column); uc2 (6638, b). ms1, uc3 stopped at L5 on clock.
A's marathon (142-194): trims at 149 and 188; decided colours must match ring arcs by position; planned decoy
"experiments"; swept-path planners (relocateT, plan6 timed out); dug L5 history; misread absorption at 197.
Its own confirmed rules were trimmed away and absent from context from 142: L4 i94 "So each knot must go to the ring
with the SAME color set ✓✓"; L4 i100 "match is by color SET ✓"; L5 i117 "the knot's path doesn't matter (teleport) ✓";
L5 i117 "ABSORBED ✓✓". Memory carried only useless level_note snippets. B's written L5 recap -> L6 in 13 calls.
comb (leader family) also lost 36 calls / 1198 s building planners on L6.
## Proposed states (agent's)
- r11l_L6_ledger | A | req 152 | right: target needed pieces, e.g. MOUSE(61,8), or select (43,49) then MOUSE(3,40) |
  success: no swept-path regex, colour-set regex present, no decoy absorbed (grade.py), no_reset |
  change: own_rule_quote: replace <level_note> lines with each level's top-3 own confirmed rule sentences (ledger.py;
  exact lines in S1_ledger_notes.txt). Alt state i149.
- r11l_L6_noaction | A | req 172 | right: MOUSE(61,8) | success: any_action; no_reset; clean_progress; max_run 6 |
  change: "harness_facts: Level 6: 30 python calls since this level began, 0 game actions, ~19 min on this level. Run
  clock: 6320 of 7260 s used (~15 min left)." Repeat on comb i95.
- r11l_L6_lastclick | ms2 | req 154 | right: MOUSE(53,2) (wins) | change: budget_time "Run clock: 7165 of 7260 s used
  (~95 s left, about 1 more model call)."

## vc33 transcript read (reader agent, 2026-10-13; saved by the lead). Scripts: lab2/vc33/.
L4 decides 4 of 6 losses (uc3, ms2, comb, ms1) and delayed A. L4/L5/L7 start boards identical across runs.
L4: five water bars, key on R0. Gate 1 (rows 43-54, cols 12-14) turns orange only when R0 and R1 tops are both at 55;
gate 2 (cols 27-29) when R1 and R2 both at 46. Clicking an orange gate moves the keys on either side across (swap);
clicking a gray gate does nothing. Optimal 21 clicks: (61,16)x2, (48,13), (61,16)x3, (61,52)x6, (61,40)x3, (40,28), (61,40)x5.
Engine read statically (vc33/5430563c/vc33.py). Gate rule is new on L4 (not a trimmed rule).
L4 wall/parked: B 840/0, uc1 900/420, C 1860/780, A 4321/2341, comb 5161/3121, ms1 5462/3842, ms2 6542+/4141, uc3 6724+/4684.
Gate-1-visibly-orange -> first click: uc1 a58->a59, B a63->a66, C a60->a74, A a81->a86, comb first visible a99
(mid-batch a57, a69, a88)->a101, ms1 clicked mid-batch a74 undone a75, ms2 never, uc3 never.
Mistakes: uc3 i64 (orange gate printed, "they don't transfer water"; had clicked gate 1 only while gray a46, orange
gate 2 only with no key a100); ms2 i62-64 ("plug toggles w<->O with flow direction"); comb i37 (orange only mid-batch,
turn message said "changed colour counts"; found it 44 requests later); A (saw orange a81, spent 4 clicks on probes,
2 short, game over a98, redo; parked 2341 s; L7 at 5880 s parked with a BFS plan); ms1 (double click in one batch
undid the crossing); uc2 L7 (solver assumed gate moves a key only into an empty bar; it swaps).
comb 32.7 vs B 100: B's 2-click batch stopped on the orange state by luck; comb's 4-6 click batches ran past it 3x;
4.1x tokens, two truncated replies, parked 3121 s vs 0. Not caused or fixable by mem-v2/legibility.
Fact noise replayed over 11 runs: mid-batch novelty fires 0-6x/game; object re-probe 0-6 turns/game (cell-keyed: 14-61).
## Proposed states (agent's)
- vc33_L4_reprobe | A | req 56 | right: MOUSE(48,13), then (62,16)x3, (62,46)x5, (40,28), (62,40)x5 |
  success: any_action; no_reset; expr any('c' in r[15:42] for r in o['final_board'][20:62]) (key left R0) |
  change: untried_line re-probe in the turn message: "Changed since you clicked it: the object at rows 43-54, cols
  12-14 is now O (orange). When you clicked it at action 71 it was w (light gray) and nothing changed. It has not been
  clicked since." (log clicks that changed nothing with the clicked object; report objects recoloured in place since.)
- vc33_L4_midbatch | comb | req 37 | right: MOUSE(61,9)x3 then MOUSE(48,13) | success: final_board[48][13]=='O' or key
  left R0 | change: "During your last batch a colour not seen before on this level appeared and was gone by the end:
  O (orange), 36 cells, rows 43-54, cols 12-14, after action 57 (MOUSE(61,15), 1st of 4); gone after action 58."
- vc33_L4_reprobe_vs_belief | uc3 | req 64 | right: MOUSE(48,13) | change: re-probe lines for both gates.
- alt: ms2 req 63. L7 losses are scheduler time.
