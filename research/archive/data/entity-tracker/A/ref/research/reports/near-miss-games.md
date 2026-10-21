# Near-miss games: re86, su15, ka59, dc22

*2026-10-10. Follow-up to `win-reliability.md` for the 12-win target. Each game was studied across the nine v4 full runs (user-combo-10x v1–v3, hats, 13x, 16x, accept05, curriculum, control), from transcripts, request logs and scheduler decisions. Win probabilities are per 25-game run; "p_now" is observed, "p_with" is the study's estimate with the lever.*

**Common findings.** The games are not lost to deaths, sandbox errors or action efficiency. They are lost to (1) the clock and the scheduler (late admission, long parks once a level gets expensive), (2) verified rules lost at trims and level changes (the `<memory>` block holds thinking fragments, not the deformation/recolor/selection rules), (3) the model losing track of which object is selected or acting in long batches that hide when a mechanic fired, and (4) novel objects whose behaviour is never probed. The levers that recur are a verified-mechanics card that survives trims, an active-object fact with a stop-on-surprise batch guard, and novel-object facts. Scheduler levers are small: the final-level hold measured +0.04–0.10 full clears per run in replay (`final-level-hold.md`, PR #155), and the park cap was rejected (−2.3 mean at the hidden ratio).

## re86 (8 levels; human baseline actions per level 26/42/86/108/189/139/424/241)

### Best levels
The best run cleared 7 of 8 levels (B, v4-user-combo-10x/v2-snowiers, score 77.78). re86 was never won and never fully cleared, so it is 0 of 9 on both counts.

Levels cleared per run, and where each stopped:
- C: 5. Stopped on L6, which it reached only at minute 97 after a 48.5-minute park on L4.
- A: 5. Stopped on L6 after 38.5 active minutes, then parked for the last 15 minutes.
- B: 7. Stopped on L8 after 20.8 active minutes, out of clock.
- D: 6. Stopped on L7: 24 active minutes, then a 25-minute park at minute 92.
- U: 6. Reached L7 at minute 120.
- T: 6. Stopped on L7: 27 active minutes, out of clock while close to done.
- K: 5. Reached L6 at minute 112 after about 54 parked minutes on L3/L4.
- X: 5. Stopped on L6. It had placed the ring, then was parked 47 minutes and never worked out the plus.
- H: 6. Stopped on L7 after 35 active minutes.

The mean is 5.78 levels. L1 to L5 were cleared 9 of 9 times. L6 was cleared 5 of 9 times, L7 1 of 5, L8 0 of 1.

Efficiency is not the problem. B used fewer actions than the human baseline on every level it cleared (23/39/49/73/83/68/151), so a full clear scores about 100.

### Decisive level
The decisive stretch is the L6, L7, L8 chain, played against the clock.

- **L6 is the first wall.** It introduces deformation. A box stretches like a rubber band, keeping height plus width (h+w) constant. When an edge is blocked, that edge stays put and the perpendicular side grows. A plus is really two bars; when one bar is blocked, the other still moves. Blockers are the gray 8x8 diamond-hole block and the board edges.
  - Four of the 9 runs stopped here: A and X because the model failed, C and K because they arrived late.
  - When cleared, L6 took 16 to 34 active minutes (B 15.7, H 22.5, D 24.5, T 29.9, U 34.0).
- **L7 is the second wall.** It combines deformation, recoloring from five palette pads, three objects, and a SPACE key that cycles which object is selected. B cleared it in 21.8 active minutes. D, T and H failed after 24 to 35 minutes.
- **L8 was reached only by B.** It adds two boxes, two obstacles, a row of recolor buttons to cross, and an off-board limit (the box's center has to stay on the board). B ran out of time with the yellow box still being routed and the sky-to-magenta box not started; it needed about 20 more minutes.

A win needs about 23 active minutes for L1 to L5, plus about 25 for L6, 30 for L7 and 30 or more for L8, about 105 to 115 in total. re86 is admitted in the second wave, 13 to 22 minutes late in 8 of 9 runs, which leaves at most about 100 minutes even if it is never parked.

### What blocks
**1. Time and the scheduler (the main blocker).**
- re86 started 13.5 to 22.4 minutes into the run in 8 of 9 runs. It was outside the first 14 admitted games; K's curriculum policy started it at minute 0.
- Parks, counted as gaps longer than 4 minutes between turns: C 48.5 min on L4; K about 54 on L3/L4, at prices that decayed 204 to 48 in sched_decisions; X 47.4 on L6, right after it had placed the ring; A about 49 (L4, L6, and the last 15 minutes); D about 40 (L6 11, L7 29); U about 40; T 28; H 12; B 6.5.
- The price formula multiplies by a "chance" term that falls as tokens and actions on the level grow. re86's L6 to L8 cost 100k+ tokens and 70 to 210 actions each, so its price collapses and it gets handed over.
- B, the best run, was barely parked and still ran out on L8, because L1 to L5 took it 39 active minutes.

**2. Knowledge lost at trims and level changes.**
- The context is trimmed every 5 to 10 turns (in B, 115 to 47 messages, 88 to 37, and so on).
- The `<memory>` block at L7/L8 holds only 200-character thinking fragments. In B, H, T and D none of them holds a deformation, recolor or SPACE-order rule. H's L6 note even reads "maybe the rule isn't 'blocked parts stay'". B's one world_model line covers only the L4 recolor.
- The consequences on L7:
  - H spent about 10 minutes re-deriving the ring's elasticity from L6 frames, then ran 10 turns in a row without an action (about 12 minutes) mining L3 frames.
  - T forgot the L5 rule that colours must match. It covered all 9 dots in the wrong colours, then spent about 10 minutes on recolor tie-breaks.
  - D re-derived the recolor rule over about 15 minutes.

**3. Wrong object moved after SPACE.**
- The model loses track of which object is selected, causing wrong-object episodes in 7 of 9 runs.
- They were decisive in two runs. T sent a 33-move plan to the orange box instead of the pink plus, and the box was recolored several times. D sent a 12-move batch to the pink plus instead of the ring, the ring "disappeared", and then came the 25-minute park.
- X wasted enough actions this way on L6 to need a RESET. B lost a couple of minutes this way on L7 and L8.

**4. Multi-action probes that hide when a deformation happened.**
- A and X sent batches of 6 to 10 moves on L6 and then could not tell which press changed the shape.
- A settled on a wrong "pinned tips" model (410 mentions of "pinned"), a "dots get collected and regenerate" model, and "the gray block does not block the plus". Its L6 thinking ran to 444k characters, against B's 135k.
- B, by contrast, measured one press at a time, wrote down an explicit rule, built a simulator, found the route by breadth-first search (BFS), and verified each step. That is how it cleared L6 in 16 minutes and L7 in 22.

**Ruled out:**
- Deaths: 0 game overs in all 9 runs, and the step bar was never close to running out.
- Sandbox errors: 0 to 4 per level and never decisive. The 30-second Python timeout fired twice on long verified batches (B L8, T L7), and B lost some saved variables and plans at times.
- Goal misread: not at the start, since everyone got "cover every dot with an object of the same colour, all at once" from L1 to L3. It was only forgotten later, as in T's L7.
- Clicks: re86 uses arrow keys and SPACE only.
- Perception: only minor. The white cursor splits an object into separate pieces, and dots are drawn on top of bars.

### Levers

| Lever | p now | p with | Evidence |
|---|---|---|---|
| Memory card with the verified rules and goal, plus a trim-time digest (report lever 1). The rules: box keeps h+w constant, a blocked edge stays and the perpendicular side grows; plus = 2 bars, a blocked bar stays and the other moves; recolor = the last palette fill touched; goal = every dot under an object of its colour, all at once; SPACE order X->Y->Z, white cursor marks the selected object's center; box center must stay on the board. | 0.00 | 0.06 | At L7/L8 the memory block holds no rule beyond B's one L4 recolor line; the rest is 200-character fragments (checked in B, H, T, D), and trims come every 5-10 turns. L7 time lost re-deriving rules: H about 10 minutes (ring elasticity) plus about 12 minutes mining old frames; T covered all dots in the wrong colours, then about 10 minutes of recolor tests; D about 15 minutes on the recolor rule. B wrote its own 'Level 6 recap' at the start of L7 and cleared it in 22 minutes. Flips if L7 were 10-15 minutes faster: D (L7 done about minute 82, before its park), H (about 22 minutes left for L8). Alone it does not fix the clock. |
| Active-object fact plus a stop-on-surprise batch guard (new; extends lever 5 and the controllable-entity tracker). Every turn: 'selected = orange 13x13 box at (51,21); SPACE order seen S->O->P'. Stop a batch when the moved object's shape, size or colour changes, or when a different object moves than the batch's first action moved, the same way the existing no-op guard stops a batch. | 0.00 | 0.04 | Wrong-object episodes in 7 of 9 runs. Decisive in T L7 (33-move plan moved the orange box instead of the pink plus, which was then recolored several times) and D L7 (12-move batch moved the pink plus, the ring 'disappeared', 25-minute park followed). X L6 needed a RESET after a wrong-object batch. A and X on L6 sent 6-10-move batches and could not tell when the box deformed. A then built a wrong 'pinned tips' model with 444k thinking characters, against B's 135k from one-press measurements. Saves about 5-10 minutes per level on L6-L8. |
| Cap early parks: max wait about 10 minutes, plus a fair-share floor while the game is still clearing levels (report lever 4) | 0.00 | 0.04 | Parks: C 48.5 minutes on L4 (then reached L6 only at minute 97), K about 54 on L3/L4 (prices 204->48, reached L6 at minute 112), X 47.4 on L6 right after placing the ring, A about 49, U about 40, T 28. A 10-minute cap gives back 9-38 minutes in 6 runs, but each still needs about 85 active minutes for L6-L8. Alone it flips about 0.3 of a run in 9 (C, T). Zero-sum against other games. |
| Final-level hold: B floor about 3 on the last level and the second-to-last level in the second half of the run (report lever 3) | 0.00 | 0.02 | Only B reached L8, and it was parked 6.5 minutes there (minute 107.5); it needed about 20 more. D was parked about 29 minutes on L7 (minutes 75 and 91.6-116); H 11.8 minutes on L7 at minute 74. Even with that time back, D and H would still have to clear L8 in about 0-20 minutes. It matters as part of a stack, not alone. |
| Admit long (8-level) games in the first wave (new; zero-sum; the report rated 'earlier start' 0 for the 12 games already won at least once) | 0.00 | 0.03 | First action at 13.5-22.4 minutes in 8 of 9 runs, because re86 was outside the first 14 admitted games; K's curriculum admitted it at minute 0. re86 is clock-bound: B's total active time was 97.5 minutes and it still fell about 20 short on L8. 15 more minutes makes B's L8 a coin flip. |
| Simulate before committing (report lever 2), as a harness check: a long plan must come with a predicted end state, and the batch stops at the first step that differs from the prediction | 0.00 | 0.02 | Every cleared L6/L7 in B used a coded state model plus BFS plus per-step checks (L6 17-move plan, L7 18/38-move plans), and B said it would verify its model before committing to a long plan. A ran BFS on a wrong model it never verified. 'Fill is not use': failing runs did not build the simulator, so only a harness check would help. Overlaps with the stop-on-surprise guard. |
| Forced action after a run of turns that end without an action (report lever 6) | 0.00 | 0.01 | H on L7: 5 and then 10 such turns in a row (20:35-21:01, about 18 minutes), mostly mining L3/L6 frames. B on L5: 7 in a row (8 minutes), but B went on to succeed, so this is a risk. |
| Sandbox state: keep plain-data variables and plans between calls (report lever 8) | 0.00 | 0.01 | B on L7: globals() blocked and a function lost because it depended on a variable, so plans had to be printed and pasted back, plus one stale-variable mix-up (a few minutes). The 30-second Python timeout cut off verified batches in B L8 and T L7. Sandbox errors are only 0-4 per level in all runs. |
| Novel-object facts (report lever 5) | 0.00 | 0.01 | On L8, B first took the white-plus pads for obstacles. On L6, A concluded the gray block does not block the plus, which is wrong. Successful runs tested the gray block within about 4 minutes anyway. |
| Bar-aware RESET (report lever 9) | 0.00 | 0.00 | 0 game overs in 9 runs; the step bar shrinks about 1 cell per 3-6 actions and was never close to empty. RESET was used once (X L6) to undo wrong-object moves. |
| Stack: card + active-object fact and guard + park cap + final hold + first-wave admission (+ simulate check and sandbox state) | 0.00 | 0.25 in a 25-game run (range 0.15-0.30); in the no-wait reliability panel, about 0.20 now and 0.35-0.40 with the card and object fact | Replaying each run on its own active per-level times with no queue and no parks: B +22 minutes on L8 (about 0.45), T +42 (about 0.3), D +56 (about 0.3), U +63, C +67 and others about 0.1-0.25. That puts re86 at about 0.2 now when its turns never wait, and gives each lever's share of the clock. Adding 10-20 minutes saved on L7/L8 by the card and object fact gives about 0.25 in a full run. The levers are super-additive for re86: it needs about 105-115 active minutes and every source of time matters. These are judgement weights from single runs, so read them as a ranking, not forecasts. |

## su15 (su15-1944f8ab): 9 levels, base actions [22,42,26,115,36,31,8,40,41]. The game is a merge ladder. Clicking pulls every token within about 9 cells to the click; two same-size, same-colour tokens merge (2 S 1x1 -> M 2x2 -> p 3x3 -> Y 4x4); the built piece is then delivered into the blue ball(s) shown in the header. L4 adds a pink 'invader' sprite.

### Best levels
Levels cleared, of 9: C 3, A 3, B 8, D 3, U 3, T 3, K 3, X 3, H 3. No run cleared all 9 (P(full clear) observed 0/9) and none scored 100.

**Eight of 9 runs stop on L4.** They reach it at 24-60 min and then spend 46-96 min of wall clock there until the 121-min cutoff:
- 15-37 active minutes (from score_timeline).
- 105-213k tokens.
- 41-141 actions.
- 5-12 RESET attempts each (C used 20 UNDOs instead).

**B (v2-snowiers) is the only run past L4.** It cleared L4 in 9 min, 29k tokens and 39 actions, then L5-L8 in 49 min. It stopped on L9 at the clock: 17 actions in, with a written plan of about 10 more actions and a 25-action bar left. B had also lost about 37 min before then: a 17-min queue before its first action and a 21-min park on L2 (tokens flat from 29 to 50 min).

**Score 100 is out of reach even for a full clear.** L7's base is 8 actions and B used 109, which caps a full clear at about 92. Score 100 would need L7 in 18 actions or fewer.

### Decisive level
**L4 (the 4th level).** The header asks for one Y 4x4 to be delivered into one ball (rows 53-61, cols 1-9). The board has 8 S dots, enough for exactly one Y, and a new pink invader at (19,52).

**How the invader behaves** (consistent in every run's logged frames):
- It is deterministic. Each action it moves about 4 cells toward the nearest token.
- A click within Chebyshev distance 9 of its centre pulls it about 12 cells toward the click instead. Of roughly 100 such clicks across all runs, about 85% produced a jump of 8-13 cells.
- When it overlaps a token it eats it. The token is replaced by a lone S dot, and the ball and header both turn gray. Gray means the level is unwinnable, since fewer than 8 S-equivalents remain.
- UNDO restores the eaten token.
- Its starting path runs straight into the dot pair at (27,31)/(29,36). So the naive plan of merging all four dot pairs first loses a token within about 5 actions.

**B, the best run.** B's first attempt was eaten at act 70; it used UNDO, then RESET at act 77.
- At step 32 B asked whether the invader responds to clicks like the tokens do. It made an isolated probe: click (20,44), within 7.5 of the invader and more than 9 from every token. The invader jumped (-12,+8).
- B checked this against act 78, (19,46), which gave a jump of about 12.7. It stated the rule: "if a click is within ~9 it moves toward the click by ~13, otherwise toward the nearest token by ~5". It also fixed the merge reach to Chebyshev 8 or less from a token's top-left corner.
- It then interleaved lure clicks (acts 78, 79, 83, 90, 96) with merges. The Y formed at act 95 and was delivered at act 101: 23 actions in the second attempt, one eat recovered by UNDO.
- B reused the same lure rule on L6 (lure an invader into a ball), L7 (invaders downgrade tokens) and L9 (invaders merge).

**A typical failing run** follows the same arc, about 57 attempts across 8 runs:
- It merges dot pairs first, a token is eaten at attempt action 4-7 (median 5), it RESETs and repeats.
- No failing attempt ever held two p tokens at once, and no Y was ever formed.
- Six failing runs made 10-25 clicks near the invader, and almost all produced the 12-cell jump. Only B turned that into a rule.

How individual runs went wrong:
- **C and T** probed by clicking the invader's exact centre, (21,54). At distance 0 it does not move, and both concluded it was not interactive. C first read it as a decorative 'holder' or 'sparkle', later as a 'leftward sweeper'.
- **X** decided the gray came from the number of clicks, not from eating.
- **K** found a partial rule ("click ON it throws it 8-12 away") but still lost a token in every attempt.
- **A** ('chaser') logged invader motions but could not explain its mixed jump sizes ("speed 4 ... then Δ(-8,-12)?!"). It ran 7 RESET attempts.
- **D** ('hunter') built a simulator in which the hunter only chases the nearest piece and only the nearest token moves. The simulator matched 4 logged transitions, none with a click near the hunter. D then ran beam searches (beam to beam11), concluded the decoy plan needs about 44 actions against a budget of about 40, and spent 58 analysis events, 19 tracebacks and 8 timeouts on it.

**How long each run spent on L4:** active minutes were 15-37 in failing runs against 9 in B. Wall clock and parks:

| Run | L4 wall clock | Parks on L4 |
|---|---|---|
| C | 58-104 min | not measured (no score timeline) |
| A | 56-108 min | 12 min |
| D | 41-120 min | 23 min |
| U | 60-106 min | 8 and 14 min |
| T | 24-120 min | 18 and 17 min |
| K | 33-121 min | 13 and 13 min |
| X | 31-110 min | 29 and 17 min |
| H | 43-91 min | 16 and 30 min |

Time did not decide L4; the missing invader rule did.

### What blocks
**1. Missing mechanic (primary, all 8 losses).** The runs lacked the invader's conditional policy: a click within 9 pulls it about 12, otherwise it chases the nearest token by about 4, and it eats on contact. Without it they could not protect the tokens. The goal was read correctly in every run (build a Y, deliver it to the ball).

**2. Perception (contributing).**
- The invader is a diagonal outline. 4-connected segmentation splits it into 6 one-pixel P nodes plus a 4-pixel B 'hole' node. Runs gave it 7 different names, and A tracked the hole as its 'core'.
- The eat event is easy to misattribute. The token vanishes, a lone S appears elsewhere, and the ball and header recolour b->g and Y->g in the same step. X never linked the gray to eating.
- The centre-click probe in C and T (distance 0, no motion) produced a false negative.

**3. Planning on a wrong model (D).** D's simulator was validated only on transitions without a near-invader click, so its search declared the level infeasible.

**4. Time and scheduler.** This did not decide L4, but it is the second gate for a full clear.
- su15 is queued for 15-17 min before its first action, and parks of 12-30 min hit its long levels.
- Clearing L5-L9 took B about 56 active minutes even with good understanding.
- B, the only run past L4, missed the win by about 10 actions on L9 after its 21-min L2 park.
- Five of the 8 failing runs (T, K, X, D, H) reached L4 early enough to have had about 70-88 min left at B's L4 pace. They would still have needed no parks.

**5. Not causes.** Deaths (2 game-overs on L4 in 9 runs), bar exhaustion, sandbox errors (outside D) and RESET-versus-UNDO choice. RESET refills the bar and costs about 5 re-merge actions.

### Levers

| Lever | p now | p with | Evidence |
|---|---|---|---|
| Novel-object facts (report lever 5), extended into an autonomous-mover and eat-cause fact line. Group the diagonal pink sprite into one 8-connected object with a stable ID. After each action, attribute its move, e.g. 'P moved (-12,+8), toward your click (Chebyshev 7.5 away); in 9 earlier actions with no click within 9 it moved about 4 toward the nearest token'. On a loss, say what caused it, e.g. 'M@(27,32) vanished while P overlapped it; same step: ball b->g, header Y->g; S-equivalents 8->7, fewer than the 8 the Y needs; UNDO restores'. Warn when a click lands 2 or fewer cells from a novel object's centre that a centre click may show no motion. | 0.04 (P of clearing L4 = 1/9; P of clearing L5-L9 in time given L4 ≈ 0.4) | 0.15 (P of clearing L4 ≈ 0.45) | About 100 near-invader clicks across the 9 runs; about 85% produced an 8-13-cell jump, and the rest of the time it moved about 4 toward the nearest token. So the rule can be recovered from the logs, but only B stated it (step 32/38, after its isolated probe at act 67). T made 25 near clicks, K 16, H 15 and U 11, and none of them formed the rule. C and T concluded 'not interactive' from a centre click at (21,54) with zero motion. X blamed the gray on click count. 57 failing attempts never held two p tokens. CPU check: replay l4state frames and confirm the line fires at the first eat in at least 7 of 8 losing runs. |
| Park cap (report lever 4): a maximum wait of about 10 min after an early-level trim, plus a floor for games that are still clearing levels. | 0.04 | 0.06 alone; 0.15 -> 0.22 when stacked with the novel-object line | B: tokens flat from 29 to 50 min on L2 (a 21-min park). B then ended on 8/9 with a written L9 plan of about 10 actions and a 25-action bar left at the 121-min cutoff. Its L9 pace was about 2.4 actions/min, so a 10-min cap probably flips B (flip chance about 0.6). L4 parks in the failing runs were 12-30 min (D 23, X 29+17, H 16+30, T 18+17, K 13+13). Once L4 is solved, L5-L9 needs about 56 active minutes, so these parks decide the full clear. Zero-sum against other games. |
| Memory card (report lever 1): carry the verified invader rule (lure radius 9, jump about 12, chase about 4, eats on contact, gray = unwinnable, UNDO restores, merge reach 8 or less from top-left) and the merge ladder across trims and across L4->L9. | 0.04 | 0.05 alone; +0.03 to +0.05 on top of the novel-object line (P of clearing L5-L9 given L4 0.5 -> 0.6) | L4 attempts ran 40-140 actions and 100-213k tokens, so trims happen mid-level. Partial rules found mid-level did not stick: D wrote 'the diamond chases my clicks!' and later reverted to chase-the-nearest-piece only; K's 'throw' rule was found at act 52-60 but eats continued. The same invader mechanic is reused on L6 (lure an invader into a ball), L7 (invaders downgrade tokens), L8 (invaders eat delivered pieces) and L9 (invaders merge into a dark-red invader). B reused it explicitly on each of those levels. |
| Simulate before committing, and suspect the model before the goal (report lever 2): when a model-built simulator says a plan is infeasible, the harness replays logged transitions and points out unverified branches. In D's case: 'your sim never saw a click within 9 of P; logged acts 66/89/90/92/93 had one and P moved 12; your sim predicts 4'. | 0.04 | 0.05 | D's sim5_step matched 4 logged transitions, none of them near-hunter clicks. Its beam searches (beam_search to beam11) concluded 'decoy needs ~44 actions > budget ~40' and gave up on L4. Only runs that build simulators benefit (mainly D; A partly). |
| Final-level hold (report lever 3): give the game a minimum budget floor on L8/L9 instead of handing it over at a trim. | 0.04 | 0.045 alone; about +0.02 once L4 is solved | su15 reaches L8/L9 only at about 100-120 min (B: L8 at 101 min, L9 at 115 min). Any handover there ends the run. B happened not to be parked on L8/L9, so this protects rather than flips. |
| Earlier start (not one of the report's levers): admit su15 before the 15-17-min queue. | 0.04 | 0.05 | First su15 action came at 15-17 min in every 10x run (31 min in C). B needed about 5 more minutes on L9. Zero-sum with other games. |
| Forced action after a yield streak (lever 6), sandbox state (lever 8), bar-aware RESET (lever 9) | 0.04 | 0.04 | No decisive role. Forcing actions without the invader rule just feeds tokens to the invader (D had 10+ yields at step 45, but its blocker was the wrong model). Sandbox errors were minor outside D's beam-search timeouts. There were 2 bar deaths in 9 runs, and RESET refills the bar. |
| All of the above stacked (5 + 4 + 1 + 2 + 3), counting a full clear. Score 100 additionally needs L7 in 18 actions or fewer against B's 109. | 0.04 full clear; about 0.0 score 100 | about 0.25 full clear (P of clearing L4 ≈ 0.5, P of finishing in time ≈ 0.75, P of clearing L5-L9 ≈ 0.65); about 0.03 score 100 | Scratch data is under /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/winrel2/su15/<run>/ (l4state.txt has per-action ball colour, bar, tokens and invader position; L4_think.txt and L4_all.txt have the reasoning and code). Scripts are in winrel2/scripts/ (timeline.py, l4state.py, idle.py, think.py). |

## ka59 (7 levels, base actions per level [28,109,51,51,33,132,326]). It is a push/launch puzzle. MOUSE selects a green block. A pushed block slides and can cross purple bands, which blocks cannot rest on. On L5 to L7, 6x6 red/orange pistons fire every 6 actions and launch blocks 9 to 15 cells across purple. The goal is to fill every charcoal frame with the block of matching shape. It is never won: 0/9 full clears, 0/3 on the 10x stack.

### Best levels
The best run is B (user-combo-10x v2-snowiers): 6/7, score 75. It reached L7 at 77 min, played it for 28 active min plus a 17 min park, and made 61 actions without clearing it. Five runs reached 5/7 and stopped on L6: C 37.4, U 51.8, T 24.0, K 43.4, X 46.3. Three runs reached 3/7 and stopped on L4: A 21.1, D 21.4, H 13.6. On the 10x stack the results are 3, 6 and 3 levels (A/B/D). Time on the stall level was B L7 45 min wall (28 active). L6: C 25 min, U 34, T 31, X 28, K 5 (arrived at 114 min). L4: A 47 min (42 active), D 14 (arrived at 107 min), H 29. Every run had one long early park of 18 to 55 min at its first trim on L1 to L3: B 27, C 18 (plus a 15 min late start), U 31+7, T 27, K 38 (the scheduler log confirms handover at 15.7 min at L3 with price 173 and readmission at 49 min), X 24, A 31 (plus an 11 min late start), D 49+6, H 29. The mean is 36 min parked per run, against 62 to 91 min of active play.

### Decisive level
L6 is decisive. 5 of the 6 runs that reached it failed there. B was the only solver, in 14 min and 82 actions, and it is the only run that formed a piston concept: 'piston' appears 536 times in its transcript against 1 to 16 in the others. B learned the concept on L5 ('the piston LAUNCHED upward'). The other runs cleared L5 without it, for example U 'Level 5 solved by getting the player into the hole'. L7 is the next wall. It has 3 frames (3x6, 6x3 and a plus), 2 green blocks, 2 pistons and a yellow plus that must cross a purple band with no piston under it. Its base is 326 actions, and it has never been cleared. L4 is decisive only for the three 3/7 runs, and in two of them (A, D) the real cause is time lost on L1 plus the park.

### What blocks
The losses come in this order of weight:
(1) Missing mechanic on L6. C, U and X read the 6-step red to orange piston cycle as decorative. U: 'the pulse is a decorative sweep animation'. X: 'decorative pulse' until its last minutes ('What if this is not decorative but a real piston'). C: 'a global clock'. The piston launches were then filed as 'teleports': U ('DOWN moved the player UP?!'), C ('pressing DOWN while a block is below ... the player moves up 15'). T found 'eruption every 6 actions' but treated it as a hazard that 'keeps knocking the player down' and ran out of time. B decoded the timing ('fires every 6 actions', used alternating moves as waits) and solved L6 in 14 min.
(2) Time and the scheduler. Each run parks for 18 to 55 min (mean 36) early. A full clear at B's pace needs about 50 active min for L1 to L6 plus an estimated 50 to 60 min on L7, which is close to the whole 121 min. B reached L7 at 77 min and was then parked another 17 min on its final level.
(3) L7 is hard and long. In 28 active min B placed both greens, then had to pull them back out to use as pushers, and was still testing push versus launch rules at the end ('push = 3 cells?', 'push into an invalid landing untested'). The plan it was converging on (launch a green into the top-left pocket and push the plus down into its frame) was not yet tested.
(4) L4 in A. A carried a wrong L3 rule ('slides until blocked + pit stops') and used a coarse 3-px grid with an off-by-2 offset. It spent 42 active min, mostly in one 20-request turn rewriting a solver: 8 sandbox errors and 11 unretained functions in 61 tool calls. It got the plus to (5,12) against the frame and did not finish.
(5) Slow L1 exploration in D and A. D clicked only empty floor and purple, concluded 'MOUSE is a no-op', and did not click the black-centred block for about 10 turns (20 active min). A took 16 active min to find that a pushed block crosses the purple wall.
(6) Perception of overlapping objects, a smaller factor: U's position helper was off by one column because of the white-centre rendering ('PD() unreliable'), A saw 'two blocks merged', B on L4 saw the green under the plus, and on L7 B's block bounding boxes broke where they overlapped a frame.
Not causes: deaths (0 game_over in any run); the step bar (about 64 cells, falling 1 per 2 to 3 actions, so roughly 190 actions per attempt, refilled by RESET; never exhausted); yield streaks (at most 1 zero-action turn per level). Sandbox errors appear but are minor outside A's L4.
Extracted transcripts are in /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/winrel2/ka59/{A,B,C,D,H,K,T,U,X}/transcript.txt, with excerpts in B_L4.txt, B_L6s.txt, B_L7b.txt, A_L4s.txt and U_L6t.txt. The scripts are in .../winrel2/scripts/.
Bottom line: even with every lever stacked, P(win) for ka59 stays around 0.10 to 0.15 per run, not 0.3. It needs about 105 active minutes at the best observed pace, so it is a weak candidate for the 12-win target. su15, re86 and dc22 should be preferred.

### Levers

| Lever | p now | p with | Evidence |
|---|---|---|---|
| Cap early parks (report lever 4): a maximum wait of about 10 min, or aging, after the first L2/L3 trim | 0/9 (about 0.02) | 0.05 alone; it is a prerequisite for every stack below | Every run lost 18 to 55 min (mean 36) to one early park: B 27 min on L2, U 31 (L2), T 27 (L2), X 24 (L2), K 38 (L3; the scheduler log shows handover at 15.7 min with curriculum price 173 and readmission at 49 min), A 31 (L1), D 49+6 (L1), H 29 (L3), C 18. With a 10 min cap, the L6 runs get 42 to 55 min on L6 instead of 25 to 34. U's accidental piston crossing came about 25 min into L6, and T understood the 6-action period by the end, so extra minutes convert. B would reach L7 at about 60 min instead of 77. A and D would reach L4 at about 54 and 68 min instead of 75 and 107. A cap alone does not fix the L6 misread or L7's length. |
| Novel-object facts (report lever 5), extended with a periodic-actuator fact: an object whose colours cycle with a fixed period (here 6 actions) plus, when the controlled block moves against or beyond the key pressed, a line naming the coincident transient extension; plus the 'never clicked on this level' registry | 0.02 | 0.05 alone; 0.10 stacked with the park cap | On L6, C, U and X called the piston cycle 'decorative pulse' or 'clock' and the launches 'teleport' (U: 'DOWN moved the player UP?!'). T saw 'eruption every 6' only as a hazard. Only B formed the piston model ('piston' 536 mentions against 1 to 16 in the other runs, learned on L5) and solved L6 in 14 min against 25 to 34 min unsolved elsewhere. The harness already has the evidence: U and X read the animation digests ('orange ghost moving up', 'O 15x6 seen at S4') and dismissed them. The click registry would have helped D's L1 ('MOUSE is a no-op' after clicking only floor and purple; about 10 turns before clicking the black-centred block, 20 active min). The fact should fire on L5 too, where every non-B run cleared the level without learning the piston. |
| Verified-mechanics card that survives trims and level changes (report lever 1) | 0.02 | 0.03 alone; it adds about 0.02 in a stack with the two levers above | A's L4 used a wrong carried rule ('slides until blocked + pit stops') and an off-by-2 coarse grid, spending 42 active min unsolved. B carried a correct L3 model (launch distance up to 15, crossing purple, yellow plus not selectable, MOUSE selects green) and cleared L4 in 10 min. For L7 the card must hold the L5/L6 piston rules (fires every 6 actions, launch 9 to 12, purple cannot be rested on, block-on-block push is 3 cells, frame outlines are passable when walking). B relearned some of these on L7 after its 17-min park. Limit: a card carries only rules that were learned, so it does not fix the L6 misread in C, U, T and X. |
| Final-level hold (report lever 3): B floor of about 3 on the last level, and no handover when the price is near the best waiter's | 0.02 | 0.03 (about 0.04 stacked) | Only B reached the final level, at 77 min, and was then parked 17 min (steps 59 to 60) at about 90 to 107 min, leaving 28 active min on L7. L7's base is 326 actions and B used 61 actions without finishing. The hold matters only once the earlier levers make L7 reachable at about 60 min. |
| Simulate before committing (report lever 2), including goal ordering | 0.02 | 0.025 | U on L6 pushed the plus 3 columns off its frame, where it locked, and needed a RESET in the final minutes. B on L7 seated both greens in their frames (about 40 actions) and then had to pull them out to use as pushers for the plus. Checking the push landing in code and ordering the goals (plus first) would have saved roughly 10 to 20 actions and several turns. Not decisive, because the model already builds BFS planners on L7. |
| Sandbox state (report lever 8) | 0.02 | 0.025 | A's L4: 8 tool errors (TypeError, ImportError, a truncated snippet) and 11 'was not retained' functions in 61 tool calls, inside one roughly 20-request turn that rewrote a solver several times ('I can't easily modify the retained function without redefining'). U's L6 had 3 NameErrors ('Oops, variables reset', 'Variables reset again'). T's L6 had 7 errors. Elsewhere errors are at most 3 per level. |
| Occlusion-aware object positions (variant of report lever 10): track a known block's full shape when it is partly hidden or overlaps a frame | 0.02 | 0.025 | U on L6: 'PD() is unreliable — the player's left column rendered WHITE', which made a pushing plan miss by one column. A on L4: 'the two green blocks have MERGED'. B on L4: 'the green block is UNDER the yellow plus'. B on L7: a 6-tall block reported as a 5-row 13-px box while it overlapped frame outline cells. Each cost 1 to 3 turns. |
| Forced action after a yield streak (6) and bar-aware RESET (9) | 0.02 | 0.02 (no effect) | No run had more than 1 zero-action turn per level, and there were 0 deaths. The step bar starts at 64 cells and falls 1 per 2 to 3 actions (about 190 actions per attempt); RESET refills it and no run exhausted it. |
| Stack: park cap, actuator and novelty facts, card, final hold, sandbox | 0.02 | 0.10 to 0.15 | Estimated chain: P(L1 to L5 done by about 55 to 60 min) about 0.8 with the cap, the card and the click registry. P(L6 cleared within about 20 min) about 0.45 with the actuator fact, against about 0.2 without it. P(L7 cleared with 45 to 60 min) about 0.35, judged from B's 28-min partial progress and a base of 326 actions. A full clear needs about 105 active min at B's best pace, so ka59 cannot reach p of 0.3 with these levers. |

## dc22 (6 levels; L5 has a 324-action human baseline and L6 a 578-action one; step bar 512 on L5)

### Best levels
Score-100 wins: 0 of 9. dc22 has never cleared L5 and no run has seen L6. Seven runs stop on L5 with L1-L4 cleared (4/6): C, B, D, U, K, X and H. In those runs L5 starts 41-104 min into the game and gets 16-45 active min plus 0-33 parked min. A (10x v1) reaches only 3/6: it was parked for 57 min on L1 and the run ended on L4. T (13x) reaches only 2/6: it found the L3 teleport at action 225, about 11 min before the run ended, after 44 min of parks. The best L5 run is B (10x v2). It is the only run that found the grab, at L5 action 79. At L5 action 106 it had the orange plank in the bridging spot, 9 arrow moves from milestone M4. No run ever reached M4 (the pad at row 20, col 26). A typical L5 run is C (control): the crane only ever visited the bottom rail and the first rail step up, and the plank was never grabbed. In the 58 L5 actions the token moved between 4 cells.

### Decisive level
L5 decides the game, and L6 sits behind it.

**Method.** The game source in competitors/polyphony-arc-3/vendor/arc/environment_files/dc22/4c9bff3e/dc22.py renders our fdcac232 L5 start board exactly (0 cells differ). Replaying each run's logged L5 actions through it reproduces the logged boards exactly in 6 of 7 runs. In B, 15 boards differ by 6 cells while the plank is carried, then match again.

**Solution, checked in the simulator: 108 useful actions (109 executed, one wasted UP).**
1. Click the blue hat. This opens the checker at rows 34-37, cols 26-29.
2. Drive the crane (the red 8x8 object at rows 24-31, cols 2-9). Its four buttons sit at row 29: col 44 = left, 49 = up, 54 = right, 59 = down. It runs on a ⊏-shaped track: vertical moves only on the left column, horizontal moves only on the top and bottom rails.
3. Take it to the top-right corner and click the red bar at row 35, col 52. That grabs the orange plank. Grab works only there, and there is no release.
4. Carry the plank to the bottom-right. That bridges the pad at row 30, col 26 to the pad at row 20, col 26.
5. Carry it back to the top-right. That bridges up to the teleporter at row 6, col 24.
6. Click the magenta hat to teleport to row 34, col 10.
7. Carry the plank to the bottom-left. That bridges up to the purple pickup at row 18, col 10.
8. Collect the pickup. A new control appears at rows 12-16, cols 46-58. It swaps the purple bar and the purple checker.
9. Click the blue hat again, click gray 4 times (fills the black bar), then use the new control, click yellow-L 4 times (slides the strip), use the new control again, and walk to the goal at row 50, col 10.

**Milestones.** Every run reaches M1 (through the blue checker) and most move the crane (M2). Grab (M3) happens once in 7 runs. The first bridge use (M4) happens 0 times in 7.

### What blocks
**Missing mechanic, caused by a perception and attribution error. Time is secondary.**

**1. Buttons look dead because whether they work depends on where the crane is.**
- In 7 of 7 L5 runs the first left-button click came with the crane at (0,0). The first right-button click came at (0,1) or (0,2). Both are rejected by the track, so they just blink.
- Every run concluded the sky-blue blocks are "locked/decoration" and that only the blue ones work, as "up/down of the red star/rotor/assembly". The crane was misread as a hazard, a rotor or an S-connector, and the white rails as walls.
- Grab was tried 2-17 times per run at the wrong crane cell. The harness reports these as NO-OP or blink.
- Crane cells visited: X only (0,0). U (0,0) and (0,1). D 4 cells, C 5, H 6, K 8 (never (2,3) or (3,3)). Only B re-tested the right button on the top rail, reached (3,3) and grabbed.
- The models do hypothesize the right subgoal: "if the orange bar could move down 4-6 rows it would bridge the gap" (H, U, C). They never find the control that moves it.

**2. B's false negative.**
- B placed the plank at the bridge spot (L5 action 106). It then moved it one rail step left and tested LEFT from row 32, col 26 into what was now void. It wrote "ORANGE IS NOT WALKABLE" and re-cast the plank as a "wire" between sockets.
- Counterfactual in the simulator: the arrow sequence U,U,L,U×5,R from action 106 reaches the pad at row 20, col 26.

**3. The goal gets abandoned.** All 7 runs voice goal doubt (5-12 sentences each), a symptom of being stuck.

**4. Time.**
- Parks on L5: C 24 min plus about 10 at the end, B 33, H 16, K 4 plus 20 at the end.
- Early parks: A 57 min on L1, D 33 min on L2, X 25 min on L2, T 44 min on L2/L3.
- Time alone did not help: U and K had 44-45 active L5 min without a grab.
- A win also needs L6 (578-action baseline). L6 adds a path-guided crane that carries bridges with collision checks, crane buttons hidden until pickups reveal them, and colour-cycling teleporters. That plausibly needs 60+ min after L5, which no run's clock allows.

**Not decisive:**
- Step bar: 274-492 of 512 steps left at the end.
- Deaths: 1-9 per run. Each one is a toggle that removes the floor under the token, costs 20 steps and is auto-reverted.
- Sandbox: 0-2 NameErrors per run. The no-op halt guard (StaleStateActionError) fires 2-20 times per run.
- Yields: 6-13 turn-budget yields per run, each followed by an action.
- Trims: 0-1 per run. L3/L4 rules (teleport needs the token on the pad, a pickup adds a button, checker means closed, the strip slides right, gray cycles the bar) carried into L5 in all runs.

**Verdict.** dc22 is not a viable p ≥ 0.3 win candidate. Clearing L5 would be worth about +23.8 game points, which is +0.95 mean points per run in which it happens.

### Levers

| Lever | p now | p with | Evidence |
|---|---|---|---|
| Novel-object / re-probe facts (report lever 5), extended to context-dependent controls. Keep a per-control effect table keyed by the position of the object the control moved or animated, e.g. '(29,54): moved R +4 cols when R was at row 24; rejected at R rows 20, 16'. When a control was rejected (blink or NO-OP) and the object its blink animated has since moved, mark the control untested again. List contacts never tried, e.g. 'GRAB (35,52) never clicked while R touches O'. Let control sweeps continue through blink rejections. | P(win) 0.00 (0/9); P(L5 clear) 0.00 (0/7); P(grab) 1/7 | P(win) about 0.005; P(L5 clear) about 0.10; P(grab) about 0.55 | In 7 of 7 L5 runs the first left click came at crane (0,0) and the first right click at (0,1) or (0,2). Both were rejected by the track, and every run wrote the sky blocks off. Grab failed 2-17 times per run at the wrong cell; it works only at (3,3). B is the only run that re-tested right on the top rail; it reached (3,3) at L5 action 78 and grabbed at action 79. Models already name the right subgoal: H says 'if the orange bar could move DOWN by 4-6 rows it would bridge the gap'; U says 'some control moves the orange bar down to bridge rows 24-29 cols 26-29'. Checkable on CPU now with the game simulator and replay scripts in winrel2/scripts_dc22 (crane.py, replay.py). |
| Simulate before committing and suspect the model before the goal (report lever 2), plus a small new fact on blocked moves. For a blocked arrow move, name the colours or objects under the target cells, e.g. 'LEFT blocked: target rows 32-33, cols 24-25 = c (void), not O'. Re-test the walkability assumption before re-casting an object. | P(L5 clear) 0.00; P(M4, the first bridge use) 0/7 | P(L5 clear) +0.03 alone, about 0.13 stacked with the re-probe facts; P(win) about 0.005 | At L5 action 106, B's plank sat at row 22, col 20 (pixels at cols 22-25, rows 22-33), the bridging spot. At actions 108-109 the model clicked down (rejected) and then left, which moved the plank to col 16. At action 113 LEFT from row 32, col 26 hit void, and the model wrote 'ORANGE IS NOT WALKABLE' and then 'the orange column is a WALL... wire connecting sockets'. Counterfactual in the simulator: U,U,L,U×5,R from action 106 reaches the pad at row 20, col 26 (M4), which no run reached. All 7 runs voiced goal doubt (5-12 sentences each). |
| Final-level hold extended to the second-to-last level (report lever 3), plus the early-park cap (report lever 4). | P(reach L5) 7/9 = 0.78; P(L5 clear) 0.00 | P(reach L5) about 0.9; P(L5 clear) about 0.00 alone, about 0.20 stacked with the two levers above; P(win) about 0.01 | Parks on L5: C 24 min plus about 10 at the end, B 33 min (12 + 21; it parked 6 active minutes after its near-M4 moment), H 16 min, K 4 min plus 20 at the end (price 1 in sched_decisions). Early parks cost A 57 min on L1, so it never reached L5. D lost 33 min at L2 and started L5 at 93 min; X lost 25 min at L2 and started L5 at 104 min; T lost 44 min across L2-L3. Time alone does nothing: U and K had 44-45 active L5 min and never grabbed. After the insight, the rest of the L5 chain is about 80 actions (2 more plank moves, teleport, pickup, the new control twice, gray, yellow-L), so the extra time matters only once the first two levers fire. L6 (578-action baseline, new path-guided crane with collision, buttons hidden until pickups, colour-cycling teleporters) would need an estimated 60+ min that no run has. |
| Memory card with a trim-time digest (report lever 1). | P(L5 clear) 0.00 | +0.01 to +0.02 on L5 (stacked only); P(win) about 0 | The cross-level rules needed for L5 were already carried in every run. C: 'a PICKUP (like the level-3 purple hat pickup)'. K: 'In L3, collecting the purple mini-trophy added a purple button'. All runs knew that teleport needs the token on the pad. The crane is new in L5, so the card cannot hold it in advance. Within L5, B's clamp facts survived its 12-min park. The value is mainly in keeping 'grab only at top-right' and 'the plank is floor' across L5 parks once learned. Trims on L5: 0-1 per run. |
| Sandbox state (report lever 8), plus letting control sweeps continue through blink rejections under the no-op halt guard. | P(L5 clear) 0.00 | about +0.005 | NameErrors ('variables reset', e.g. i4 not defined) 0-2 per run on L5. StaleStateActionError 2-20 per run: the guard halts a snippet after any no-op, which cuts control sweeps at the first rejected crane button. That compounds the context-dependent misread; it is better handled inside the re-probe lever. |
| Forced action after a no-action streak (report lever 6). | 0.00 | about 0 | 6-13 turn-budget yields per L5 run. Each was followed by an action in the next turn, and none sits at a decisive moment. |
| Bar-aware RESET (report lever 9). | 0.00 | 0 | The L5 step bar (512) never bound: 274-492 steps were left at the end. Deaths (C 0, B 1, D 2, U 5, K 1, X 1, H 9) are clicks that remove the floor under the token; each costs 20 steps and is auto-reverted, and none mattered. RESET was used 1-3 times only to re-inspect the start board. |
