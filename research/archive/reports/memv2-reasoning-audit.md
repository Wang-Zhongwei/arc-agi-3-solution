# Memory-v2 run audit: reasoning, output, actions and memory

2026-10-10. Run `mellamoa/v4-mem-v2-10x`: user-combo-10x plus memory-v2 (`ARC3_MEM_SOLVED`, `ARC3_MEM_CARD`, `ARC3_MEM_EFFECTS`, `ARC3_MEM_SEEN`, placement both, reply-wins). W4A16 Qwen3.8-Flash-Next with thinking on and one `python` tool. **Score 43.81** with 106 levels. Six games cleared every level (ar25, cd82, ft09, lp85, sb26, tr87), four of them at 100. The three best-config runs (`v4-user-combo-10x` v1, v2-snowiers, v3-snowiers) scored 56.94, 52.70 and 49.09.

The user asked for two things. First: "analyze very carefully what each ai on each game responds with, and find gaps (especially in its reasoning, because from my testing from a long time ago i remember that it always second guessed on the right intuition)". Then: "check especially reasoning vs model output vs actions taken and what memory is specifically being saved".

## The answer in brief

- **The second-guessing is real, and it happens in every game.** The model stated a correct idea and then dropped it **143 times across all 25 games**, 4 to 9 times per game. I re-checked every episode against the engine version the run actually used. In 137 the dropped idea was right, in 6 it was right once corrected, and in none was it wrong. In **18 of the 19 games that did not finish**, a correct idea was dropped on the level where the game stopped. Four of those levels had a short finish from the model's own state, confirmed by replaying the engine: sp80 needed 3 actions, cn04 7, m0r0 39 (which wins the whole game) and lf52 54.
- **It is mostly not pure doubt.** About 22% of the reversals are untested hedging. The rest have other causes:
  - about 36%: the model trusted a wrong reading of its own crop, filter, frame index or tool output;
  - about 14%: the rule was lost when the context was trimmed;
  - about 10%: a search or simulator the model wrote said "impossible", and it believed the search instead of fixing it;
  - about 10%: a confounded test.
  
  Real counter-evidence, read correctly, almost never caused a reversal.
- **Reasoning, output and actions.** The reply text is almost always empty, so the model's output is the python code: 2,740 of 2,771 responses ended in a tool call and 31 hit the 12,288-token cap with no call. I logged 279 mismatches (about 11 per game). The most common were misreading a tool result (27%), code that does not match the reasoning (21%) and code that kept acting after the plan had failed (18%). The python mostly does what the code says. The losses come earlier and later: the code drifts from the plan, and the model ignores results that contradict the plan, even when the harness printed the contradicting fact.
- **Memory-v2 saves paths and statistics, not knowledge.**
  - It wrote 141 full `<memory>` blocks (shown only after trims) and 280 `<memory_update>` deltas in 2,780 requests.
  - They contain cut-down routes, per-key and per-colour effect counts, game-over lines, "Seen before / Untried" lines and a "how:" sentence copied from the model's reasoning. None of them holds a goal, a rule, a step size, a budget or a cause of death.
  - The model wrote only 10 notes of its own, in 4 games. Roughly 2 of them helped and 3 hurt.
  - Several lines written by the harness actively misled the model (wa30, re86, r11l and others).
- **Memory-v2's prompt wording appears to suppress the model's own notes.** I measured this for this report. The best-config runs wrote **58, 39 and 34** notes (in 13, 11 and 6 games); memory-v2 wrote **10** (in 4). Memory-v2 replaced "Memory that survives history trimming … so findings from dropped turns are not lost" with "Paths, action effects and deaths are already recorded by the harness". This is one run per config, so it is a strong hint, not proof.
- **The fixes that would help are facts the harness computes and places in tool results, plus guards on actions.** Section 5 ranks them. The cheapest strong ones are:
  - a cross-level rule ledger built by the harness, replacing the "how:" sentence and the colour tallies;
  - restoring the best config's memory wording;
  - stop rules for runaway loops;
  - a confirmation step before RESET.

## How this was done

- One analyst per game read every turn: the opener, the thinking, the code, the tool result and the memory lines. Each analyst also read the engine source to learn the true rules, and compared against the best-config runs. A synthesis pooled the 25 analyses.
- Two verifiers then re-checked the synthesis.
  - **Quotes and replays:** 12 headline claims were checked against the transcripts and request logs, and 7 of them by replaying the run's own action history in the engine.
  - **Engine source:** all 25 games were replayed frame by frame on the exact engine versions the run used, with 0 frame mismatches. All 143 episodes and the 25 memory critiques were re-checked against those engines.
- **Engine versions:** the vendor copy under `competitors/polyphony-arc-3` is a different version for 14 of the 25 games (cn04, dc22, ka59, m0r0, r11l, re86, s5i5, sc25, sk48, sp80, su15, tn36, tu93, vc33). The versions this run used are in `~/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/<game>/<version>/`. Use those for any future ground-truth work.
- **Terms.** "Thinking" is the `reasoning_content`. "Output" is the tool call's code plus any reply text. "Actions" are what `action()` actually executed. A "call" is one model request. A "step" is one analysis turn.
- **Data:** [`data/memv2-reasoning-audit/`](../data/memv2-reasoning-audit/) (described at the end).

## 1. Does the model second-guess the right intuition? Yes

### How often

| | Count |
|---|---|
| Episodes where a correct idea was stated, then dropped | **143** (every game; 4–9 per game) |
| Analysts marked the idea right by the engine source | 140 (cn04's 3 were left open over the version mismatch) |
| Engine-source verifier, on the exact run versions | **137 right, 6 right with a correction, 0 wrong** |
| Unfinished games where an episode hit the level the game ended on | **18 of 19** (all but tu93, which ran out of clock with the optimal plan computed) |

Episodes per game: ar25 6, bp35 4, cd82 9, cn04 6, dc22 8, ft09 5, g50t 7, ka59 6, lf52 9, lp85 4, ls20 5, m0r0 5, r11l 5, re86 7, s5i5 5, sb26 4, sc25 6, sk48 5, sp80 5, su15 5, tn36 5, tr87 5, tu93 5, vc33 6, wa30 6.

The 6 partial episodes, with their corrections:
- **ar25 L4:** the model did not misread its search. It wrote "rb=42 → rows 30-47 ✓" and then chose the other solution without noticing piece B was already in place.
- **cn04 L2 (two episodes):** see below.
- **ka59:** a push is a kick. The pushed ring slides 3 cells per frame for 5 frames, and the pusher stays where it was.
- **m0r0 L6:** the blue block is a movable stopper, but it cannot hold a button down.
- **vc33 L4:** a gate opens only when both water surfaces sit exactly at the gate's edge. Being equal to each other is not enough.

**cn04, settled on its real engine (2fe56bfb).** There are two hidden connector types, pixel values 8 and 13, and both are drawn red. Any two tips that overlap turn gray. The win check counts only pairs of the same type, and it never checks whether bodies overlap. At action 97 all 12 tips were gray, but two pairs were 8-with-13. The model's own fallback, "if it fails to win, try rot 0" (calls 56–58), was right and never run: click p, then SPACE, SPACE, LEFT, UP, UP, UP wins L2 in 7 actions. At calls 69 and 74 it dropped "run one of the 792 clean, fully matched placements". Every one of those wins in the engine, even though its stated reason ("no body overlap") is not the engine's rule.

**Caveat on the 96% "right" rate.** Analysts logged an episode only when a dropped idea later looked correct, so a high rate is partly built into how episodes were chosen. What the data does show is how often this happens, what triggers it, and what it costs.

### What triggers the reversal

Each episode was assigned one main trigger by hand, so the counts are approximate.

| Trigger | ≈ Episodes | Examples |
|---|---|---|
| **Misread its own output or perception** | 51 (36%) | Hand-read ASCII crops off by one: cd82 read `BpppppppppOB` as cols 0–7; tu93 L6 misquoted its own printed row `BBBBRgRgWBBBB`. Filters that hid the evidence: sc25 L6 printed only the cursor while the yellow next-pad ring moved to pad C; dc22 L3's diff dropped 1-pixel changes; bp35 L2's run-length print dropped Y/W, so the spike's tip looked like a wall. A parser counting new objects as terrain: vc33 L5 `surf()` read 52/52 as 42/42 (engine-confirmed). Reading `history[k-1]` (the board before the winning move) as the win board: cn04, g50t, s5i5, su15, vc33, lf52. |
| **Untested hedge, or a new idea adopted without a test** | 32 (22%) | ar25 L3 "probably blocked" (the dropped plan was the final solution and cost 39 actions); sb26 L2 "OK let me just try H", then picked C; vc33 L4 "that's a stretch"; cn04 L1 "tips don't go in". |
| **Rule lost at a context trim** | 20 (14%) as the main cause, contributing in ~35 more | m0r0 L6 lost click-to-select from L3/L4; s5i5 L4 lost "a child arm follows its parent"; ka59 L2 went back from 3-cell moves to 1-cell moves; g50t L5 lost yellow=toggle and the 2-echo cap; tu93 L6 lost the guard-facing rule; lf52 L5 lost the purple-peg rules; wa30 L5 lost SPACE-release. |
| **"Impossible" from its own search, simulator or proof** | 15 (10%) | ar25 L4/L7/L8 over-constrained searches; re86 L6 "odd dims only"; vc33 L5 false conservation and mod-6 invariants; sk48 L2 simulator (missing 2 rules) called a state 37 moves from the win a dead end; ls20 BFS treated walkable black cells as walls. |
| **Confounded test, over-generalised** | 14 (10%) | A clamp no-op generalised to the whole level (cd82 ×3); ka59 push blocked by a wall read as "no push"; sc25 L6 n=2 teleports (the cycle restarts on RESET); su15 probes clicked out of range. |
| **Wrong budget belief** | 4 | sc25 L4 "32 vs 35" (the real budget is 35, plus a +10 pack on its route). |
| **Misleading harness or memory line** | 5 | tn36 "same signature" digest; wa30 "spent 164, 0 deaths"; lf52 stale level_note; r11l frame_diff "swap"; sp80 "UP: no change". |
| Other | ~2 | |

### What it cost

**Levels where it was decisive.** Starred finishes were replayed on the run's engine version. The rest are the analysts' estimates.

| Game, level | Finish from the model's own state | What it did instead |
|---|---|---|
| sp80 L4 | **DOWN, DOWN, SPACE\*** | Lost the second-spout and no-spill rules at trims; time ran out |
| cn04 L2 | **7 actions\*** (purple at rotation 0) | Never ran its own "try rot 0" fallback; spent ~70 min on invented rules |
| m0r0 L6 | **39 actions\***, which wins the game | Selected the blue block at action 276, pressed only UP and DOWN (both blocked), then RESET at 280 |
| lf52 L5 | **54 actions\*** (LEFT×6, DOWN×2, then jumps) | RESET at action 327 although the opener said "Untried: LEFT, UNDO"; it called the state a "DEADLOCK" |
| vc33 L5 | Gate B was already active at 52/52\* | `surf()` reported 42/42; it wrote "the rule is NOT 'clear of the gate'" |
| tn36 L2 | 9 actions (4×{bar1, stem3}) | A contaminated test (`BBwwwB`, really {bar1, stem1, stem3}) gave "= DOWN 4"; it dismissed re-checking as "unambiguous" |
| sc25 L6 | One 4×4 cast plus ~7 moves | Read two first-teleports-after-reset as "always pad B"; a 47-min truncated call ended the run |
| sk48 L2 | 37 moves | Its simulator, missing "LEFT releases a jammed block", called the state a dead end |
| g50t L5 | 51 actions | Lost yellow=toggle and the 2-echo cap at trims |
| r11l L3 | ~12 actions | A hedge ("to be safe, require the swept shape to be free") hardened into a rule |
| Also | | bp35 L2, dc22 L5, ka59 L2, ls20 L4, re86 L6, s5i5 L4, su15 L7, wa30 L5 |

**Games cleared anyway paid in actions and tokens:**
- ar25: about 100 extra actions, and 208k tokens against 125–178k in the best-config runs.
- cd82 L2: 69 actions against a 6-action plan the model stated at its first L2 turn.
- sb26 L2: 35 actions, where the optimum is 15.
- ft09 L3: 27 actions against 14.
- lp85 L3 and L7: about 10 extra actions each.

A typical episode costs 5–40 actions and 3–20 calls. The worst ones happened inside single calls of more than 12k tokens that took 20–60 minutes.

### Verdict on the user's hypothesis

The hypothesis is **confirmed in frequency and in cost**. It is only **partly confirmed as an "intuition" problem**. Pure doubt (an untested hedge) is about a fifth of cases. Most reversals happen because the model trusts something wrong over a correct idea it already stated:
- a wrong reading of its own crop, filter, frame index or tool output;
- its own broken search or simulator;
- or nothing at all, because the trim deleted the idea.

The fixes therefore aim at what the model reads and keeps, not at its confidence.

## 2. Reasoning vs model output vs actions

The model's output is almost entirely its python code. Of 2,771 responses, 2,740 ended in a tool call and 31 hit the 12,288-token cap with none: s5i5 8, tu93 4, ka59 3, sk48 3, sp80 3, and others. Those 31 include sc25's 47-minute call that ended the run, and the calls where several decisive reversals happened (sk48 step 18, wa30 call 127).

The analysts logged 279 mismatch rows, about 11 per game:

| Kind | Rows | What it means |
|---|---|---|
| Misread result | 75 (27%) | The tool printed the answer and the model read it wrong |
| Reasoning → code | 59 (21%) | The code does something other than what the thinking decided |
| Code → action | 49 (18%) | The code ran actions the plan did not intend (loops, repeats, errors) |
| Perception | 44 (16%) | The board itself was misread (crops, segmentation, scale) |
| Plan drift | 31 (11%) | It stated a cheap decisive test, then ran something else |
| Other | 21 (8%) | Sandbox errors, truncations, budget models |

**Reasoning → code.**
- **Memory writes announced but never made.** About 29 times in about 17 games, the thinking said it would record something and no tag followed in that message. Examples:
  - bp35 calls 20, 24, 30 and 38: "Let me record the world model in memory first, then act".
  - dc22 calls 19, 57, 73 and 74; call 73 was a bare `<memory>` string.
  - g50t calls 61, 72 and 95; call 72 was "Level 3 recap for memory: red = hold-only…".
  - cd82, sb26, tr87 and wa30 twice each, and one each in ft09, lf52, lp85, ls20, r11l, re86, sc25, sp80, tn36 and tu93.
  - In ft09 call 38 it wrote correct rules as plain `#` comments.
- **Code that silently uses a different convention from the plan:**
  - lp85 L3 checked a push model with pull code, printed "pred_mismatch 0", and sent 15 clicks to the wrong arrows.
  - ar25 L1 pressed RIGHT×4 the wrong way.
  - m0r0 L2 and L4 coded hazards as walls.
- **Off-by-N:** s5i5 used `N[3]` for `N[4]`; g50t counted 7 cells and sent 6 UPs.
- **Plans missing a step:**
  - ar25 L3 left out the SPACE that switches piece, costing about 22 actions.
  - tn36 L2 ran its decisive test on a tablet that still held stale columns.

**Code → action.**
- **Loops with no stop condition:**
  - tr87 L5, call 81: `while i!=tgt_i: action(["LEFT"])` pressed LEFT 98 times (97 of them in a row, actions 148–244) and ran out the budget, which ended the attempt in GAME OVER. The cursor moves by tile, not by slot, so the loop never reached its target.
  - s5i5: about 29 clicks that blinked in loops which printed "BLK" but stopped only on a level change.
  - ar25 L7 and lp85 L7 kept clicking after printing MISMATCH.
- **Repeats:** r11l retried the same reverted click 3 times and reached 4 of the 5 hazard hits allowed.
- **The no-op guard fires wrongly in both directions:**
  - It stops correct batches when changes near the edge are classed as "edge only" (vc33 actions 75 and 126, wa30 L1).
  - It stays silent when helpers move every turn (wa30 L3–L5).
- **Sandbox friction costs about 5–17 calls per game:**
  - lp85 had 12 tool errors, r11l 14 tracebacks, and 17 of ls20's 138 results were exceptions.
  - Typical errors: a NameError on a variable that did not persist, `import time` or `pickle` not allowed, and the 30-second timeout.

**Misread result.**
- **Reading the end of a batch instead of each step:** re86 L6 "squashed" the ring; ka59 built snap and teleport theories.
- **Reading the board before the win as the win board (6 games):** s5i5 wrote "history[69] … = the winning board ✓".
- **Explaining away correct evidence:** re86 "maybe RIGHT4/5 changed the HUD bar only"; dc22 "8 = HUD only".

**Plan drift.**
- sb26 L2: "OK let me just try H", then ran C.
- dc22 L5: "test the red bar at (12,14)", then spent 4 calls mining history.
- cn04 L2: planned the rot-0 fallback and never ran it.
- lf52 L5: planned LEFT, then pressed RESET.
- m0r0 L6: asked itself "LEFT/RIGHT only?" and never tried it.

**Output vs action, in one line:** the actions match the code, but the code often does not match the thinking, and the thinking often does not read the result. In several games the harness result already held the contradicting fact, and the model ignored or overrode it:
- tn36: the frame bounding boxes.
- vc33: the node list.
- lf52: "Untried: LEFT".
- su15: "not tried yet this game: UNDO", shown all game.

## 3. Memory: exactly what is being saved

### What the harness writes

Counts over the run: **141 full `<memory>` blocks** and **280 `<memory_update>` deltas** in 2,780 requests. A full block (about 0.4–3.7 KB) appears only as the start of the oldest remaining message after a context trim, 2–9 times per game. Deltas ride on turn messages.

A real block (wa30, first shown at request 70):

```
<memory>
  <solved>L1: 33 acts [U2 S U L U S D L3 U S D R4 U S D R3 U2 S D2 L2 U S], spent 72, 0 deaths</solved>
  <solved>L2: 50 acts [D3 R6 S D L7 D4 R U S D R7 S L8 U S L6], spent 120, 1 death | how: The player is at (9,2) facing up (last successful move was UP).</solved>
  <solved>L3: 90 acts [U3 L U R S R3 D2 S U2 L7 U R S R6 S D7 L6 R S R5 U2 S D3 ...+32], spent 135, 0 deaths | how: Each DOWN/UP costs ~0.67.</solved>
  <actions level="4">
  UP: moved 4 | game death 1 (1 board), moved 49, changed 22, edge only 1
  DOWN: not tried this level | game clear 1, moved 22, changed 27, edge only 4
  ...
  </actions>
</memory>
```

Line by line:
1. **`<solved>`**, one per cleared level. It holds the winning attempt's path with loops cut out, truncated with `...+N` after about 25 items, then "spent N, D deaths". Sometimes it adds `| how: …`, a sentence copied from the reasoning of the winning call.
2. **`<actions level="N">`** counts what each key, and each click on a cell of a given colour, did this level: moved, changed, edge only, anim only or no change. The game-wide counts follow `|`. Click rows look like `MOUSE@c 5 targets: moved 5, no change 1; median 2 cells`. A final line reads "colours on the board not clicked this level: W w N".
3. **`<hazard>`** lines: "L2: attempt 1 ended in GAME OVER after 8 actions; last action M(33,39) from the board first reached at action 7". These cover only the current level and are dropped when the level changes.
4. **Deltas** report firsts and solved levels, e.g. `+ MOUSE@R: first moved at M(18,40)` and `+ solved L1: 7 acts [L D R4 S], spent 11, 0 deaths`.
5. **"Seen before"** lines in turn messages: "board interior matches … after action #7 … From it you tried: M(10,8) -> no board change … Untried: UP, DOWN, LEFT, RIGHT, UNDO."

None of these lines holds a goal, a rule, an object's identity, a step size or unit, a budget, or what caused a death.

### What the model writes

| | memory-v2 (this run) | best config v1 | v2-snowiers | v3-snowiers |
|---|---|---|---|---|
| Model-written tags (`world_model`, `hypothesis`, `level_note`) | **10** | 58 | 39 | 34 |
| Games with any tag | **4** | 13 | 11 | 6 |

The counts above cover assistant messages only (reasoning, reply text and tool-call code), with each message counted once and the prompt's example tags excluded. They were measured for this report with `tags_compare.py`. The mem-v2 count matches the verifier's.

The 10 tags in this run, all of which reached later memory blocks:
- **tr87, 5 tags.** Three are inside python strings (request 26) and two are in the reply (request 27, restating request 26). They hold class labels from L1 that mean nothing from L2 onward. Neutral.
- **lf52, 2 tags,** in reasoning (request 72):
  - A `world_model` that is correct but thin.
  - A `level_note`, "goal = 1 peg total", which is true for L3, where no purple pegs exist. The harness relabelled it "level 4: L3: …". In L5 it argued three times against the true goal of greens only. **Hurt.**
- **cd82, 1 tag,** in reasoning (request 78): "7 poses … small stamp at the U pose only". It is wrong: there are 8 slots, and the small stamp works at U, R, D and L. It sat in memory exactly when L4 needed the L slot, and was never revised after being disproved. **Hurt.** L4 took 35 actions against 21, the main score loss on cd82.
- **tu93, 2 tags,** in python strings:
  - The L6 guard rule (request 110) is correct and was used in L7. **Helped.**
  - The L7 note "the chaser … cannot be eaten" (request 123) is wrong in 4 ways and was cited in L8. **Hurt.**

Net: about 2 helped and about 3 hurt.

**Why so few: the prompt wording changed.**
- The best config's section is titled "Memory that survives history trimming". It says the `<memory>` block holds "what you recorded, so findings from dropped turns are not lost", and adds "A refuted hypothesis is as valuable as a confirmed one".
- Memory-v2 retitles it "Memory kept by the harness". It explains the harness lines first, then calls tags "the other memory lines", and says "Paths, action effects and deaths are already recorded by the harness, so a note holds what the frames do not show".

The rate fell from 34–58 tags per run to 10. Even the lowest best-config run wrote 3.4× as many. This is one run per config, so the cause is likely but not proven. Example: in su15, v2-snowiers stored the lure, shrink and gray rules as tags and cleared L7; this run had none and failed L7.

### Harness lines that misled the model (verified)

- **wa30 "spent 164, 0 deaths".** The 164 actions were two attempts split by a RESET at action 421 (93 + 71), which the line does not show. At step 93 the model wrote "Confirmed: P is non-fatal (score only)". It hit GAME OVER at action 697 when the step bar reached 0. The engine ends the game when the steps run out.
- **re86, the L2 "how:" line.** "The orange X remains at (18,27), needs to go to (48,18): DOWN 10, LEFT 3" was cited in L6 calls 69, 71, 72 and 80. Call 69 says "so L2's goal was to MOVE a collector to a target position! Not dots!", which is wrong.
- **su15, the L2 "how:" line.** "The 4th p2 click must be > 6 from p1" encodes a wrong rule; the real rule is that blocks of different tiers touching is invalid. It was cited in call 56 and set aside, so it is a weak case.
- **Click tallies keyed by colour.** They label a selection as "moved" (r11l, ka59, lf52, sp80, m0r0) and count clicks on the HUD as board clicks. In su15, "MOUSE@P no change" was read in L7 as "clicking a sprite did nothing".
- **Units.** cn04's "median 135 cells" is in pixels, and it led to a wasted probe.
- **Hazards.** ls20 and tr87 blame the arrow key for deaths that were really the budget running out.
- **"colours not clicked".** It prompted predictable no-op clicks in ka59, cn04, sk48 and tn36. In m0r0 L3 it did point at the right object.
- **Truncated paths** hid the winning tail: bp35 L1 (`...+4` hides `M M L L`), dc22 L3 and L4, and s5i5.

### What the model actually used

- **The `<solved>` path, mostly to rebuild a past level:** sk48, sc25's click patterns, sp80 L3's click-then-move, and g50t's SPACE count (correct, then dropped).
- **"Seen before / Untried", used well a few times:** cd82 L2's breakthrough came from "Untried: RIGHT", though 6 steps late; also bp35, lf52 L2 and g50t.
- **"Seen before / Untried", ignored at the decisive moments:** lf52 L5 ("Untried: LEFT" right before the RESET), sc25 L6, su15 (UNDO all game) and m0r0 L6.
- **The real working memory was the harness's `history` and `transitions` frames.** These are facts the harness computed. ft09 fixed its colour rule from them, ls20 recovered "black is walkable", and sk48 recovered the goal order. This matches the standing lesson that facts in tool results get used.

### What was missing at the decisive level

In every game where the analyst checked, the decisive rule had been learned and then trimmed:

| Game | Rule that was lost |
|---|---|
| ka59 | Moves go 3 cells; frames don't block |
| m0r0 | A click selects the b block, then arrows move it |
| s5i5 | Child arms follow the parent when it lengthens and when it shortens |
| g50t | Plate types; the 2-echo cap |
| lf52 | Purples cannot be selected or captured |
| wa30 | SPACE releases on target; the bar reaching 0 = death |
| tu93 | Guard facing; entering a room eats its guard |
| vc33 | Gate activation; clicking a gate moves the key |
| su15 | Push and lure; gray = no longer reachable |
| re86 | Covered dots are only hidden |
| sp80 | The second spout; a spill means no win |

Also missing everywhere: the budget and step size per level, the cause and position of each death, and click meanings attached to objects that would carry across levels.

### Verdict on memory-v2

As built, it stores **paths and statistics, not knowledge**. Its prompt wording appears to have cut the model's own notes by more than two thirds. Its scraped "how:" text, its colour tallies and its death counts misled the model in several games. The parts that earned their place are the full solved path (if it were never truncated) and "Seen before / Untried".

The 43.81 against 49–57 cannot be pinned on memory from one run. The per-game swings go both ways:
- mem-v2 was the best of the four runs on lf52, wa30 and sp80, and tied best on dc22, sk48 and s5i5.
- It was the worst on r11l (14.29 against 47.6–100), ka59 (0.50 against 21–75) and cn04 (3.09 against 4.8–71.4).

## 4. Other patterns

- **Wall-clock stalls (not a memory-v2 cost).** In about 15 games the first request after a trim came back after 12–47 minutes, having generated only a few hundred tokens. Examples: ka59 at 20:47→21:22 and 21:32→22:04, sk48 at 984→3,730 s, g50t 24.7 min, ls20 2,106 s, vc33 36 min and sb26 36 min. These stalls ate the clock that ls20, g50t, ka59, sk48, vc33 and tu93 needed for their last level. The verifier found three things:
  - Later trims in the same games came back in seconds to about 3 minutes.
  - The stalls cluster between 20:47 and 22:04.
  - The best-config runs have as many gaps of 20 minutes or more: 15, 24 and 21, against 22 here.
  
  So the stalls are probably server contention at 10× concurrency. They are neither caused by trimming nor specific to memory-v2.
- **The board after a win is shown, but badly labelled.** A winning action returns 2–27 engine frames. The harness records them as that action's animation (`last_animation_frames`, `animation_for(n)`), and the next message says "animated over 2 frames … Animation digest (novel): action 36". But that board is not in `history` or `transitions`, and the line describes it as "cells changed and changed back". So in 6 games the model read the board before the win as the winning board.
- **Perception at the wrong scale.** The model hand-reads 64×64 ASCII while the game logic runs on a coarser grid:
  - bp35 has 6-pixel cells and produced invented "erosion", "slits" and "pitch 5 or 6" rules.
  - ka59 confused 1-cell and 3-cell moves; m0r0 read a bordered block as a 2×2 marker.
  - lf52 and ls20 are scrolling worlds with no world coordinates.
  
  Every recovery came after the model printed run-lengths or indices from code.
- **Its own searches and simulators.** It read every "impossible" result as the mechanic being wrong rather than the search being wrong. When the simulator was right, play was optimal:
  - tu93: every plan it made once it had the rules matched the engine optimum (L2 10, L3 19, L4 17, L5 29, L6 28, L7 14, and L8 21, which it computed but never got to run).
  - Also ar25 L6, lp85 L8, g50t L3, tr87 L6, and sb26 L3–L8.
- **Wrong budget models in most games.** It read the bar as a timer (ka59, lp85, ls20) or as a 64-action limit (lf52, ar25). In dc22 L5 it believed about 34 actions were left when about 460 were. It missed refunds (sc25). The results were needless RESETs (sc25 ×4, lf52 ×2, ka59) and risky idling (wa30 L5).
- **Exploration order.** Distinct, unexplained objects were clicked late or never:
  - tn36's selector boxes at L2 action ~70; v1 clicked them at actions 2–3 and won the game.
  - dc22's purple marker and m0r0's blue block.
  - UNDO was almost never tried, even in su15, where it was free and would have un-grayed the board.
- **Unbounded thinking.** 31 calls hit the cap. sc25 call 91 took 47 minutes. re86 L6 used 96k tokens and dc22 L5 149k tokens for 42 actions. The model never saw how much time remained; tu93 found out by accident.
- **Hidden rules no analysis caught, found by the source verifier:**
  - cd82's win check ignores both main diagonals.
  - g50t's SPACE rewind re-toggles plates it passes back over.
  - sc25's teleport cycle restarts on RESET.
  - su15's gray means the target can no longer be reached with the blocks left.
  - In wa30 a carried box can enter hatched cells, though the player cannot.
- **Analysis-tooling note.** The compact `games/*.md` openers show `<memory_update>` blocks as empty, but the raw requests contain the deltas. Judge memory content from `memory/*.md` or the raw requests.

## 5. Interventions, ranked

Ranked by expected score per unit of compute and by how directly each one hits the final level of the unfinished games. All of them are harness-side facts or guards, so none depends on the model following optional structure.

| # | Intervention | What it fixes | Cost | Overlap with existing arms |
|---|---|---|---|---|
| 1 | **Rule ledger built by the harness** (replaces "how:" and colour tallies) | Rules lost at trims: the main cause in 20 reversals and a contributing one in ~35; the final level in ~8 games; three verified misleads | Memory-builder work; block stays ~1–2 KB | Extends memory-v2 (PR #148); `mem-layout-10x` already pushes a per-level block every call |
| 2 | **Restore the best config's memory wording** | Model notes fell from 34–58 per run to 10 | Prompt edit, ~0 tokens | Reverts part of PR #148 |
| 3 | **Per-action object-delta digest** | Misread-result and perception reversals (36% of reversals; ~119 of 279 mismatches) | CPU; ~100–300 tokens per action | Entity tracker failed Stage-0 (keep this a dumb component diff with no roles); attention-prompt |
| 4 | **Batch stop rules and loop guards** in `action()` | tr87's 98 LEFTs, s5i5's ~29 blinks, ar25/lp85 clicks after MISMATCH, r11l's repeats, both no-op-guard misfires | Small | Extends the existing batch no-op and stale-state guards |
| 5 | **Confirm before RESET**, with an untried-actions list | lf52 L5, m0r0 L6, ka59 L2, sc25 ×4, su15 L7 (UNDO) | One extra call per RESET | none |
| 6 | **Steps left and minutes left in every result**, plus a thinking cap | Wrong budget beliefs in most games; 31 truncated calls; clock-blind deliberation | Trivial | budget-alert (PR #60), part of fable-facts-10x |
| 7 | **Label the board after a win, and add death facts** | Win misreads in 6 games; deaths blamed on the wrong cause (bp35, tu93, ls20, tr87, wa30) | Small | none |
| 8 | **Persistent sandbox globals and perception helpers** | ~5–17 wasted calls per game; off-by-one crop reads | Moderate | sandbox-v3 design |
| 9 | **Investigate the 10× stalls** | 12–47 min stalls in ~15 games, as common in best-config runs | Infrastructure | throughput work |

### Details

**1. Rule ledger built by the harness.**
- Key effect tables by object type, not pixel colour, and keep them across levels. Examples:
  - "click on the blue-bordered 4×4 block: it turns yellow and the players turn gray; arrows then move only it (L3, L4)".
  - "arrow: the selected object moves 3 cells".
- Record how far things move, with units.
- Keep every hazard, with its cause, for the whole game.
- Report RESETs separately from deaths.
- Drop the "how:" scrape. Drop "colours not clicked" once MOUSE is known to be a no-op, and leave HUD and selection colours out of it.
- Never truncate the `<solved>` path.
- Optionally, quote the model's own "✓ confirmed" and "KEY DISCOVERY" sentences word for word, labelled with their step number.
- Put the ledger in the tool-result area at every level start, not only at trims.

Evidence:
- m0r0 L6 lost the L3/L4 click-to-select rule.
- s5i5 L4 lost "the parent pulls the child", which was the solution.
- ka59's 3-cell step reverted to 1 cell and cost about 55 actions.
- g50t L5 lost the echo cap and yellow=toggle.
- lf52 L5 lost the purple rules, which led to the RESET.
- wa30 L5 lost SPACE-release.
- tu93 L6 spent 47 minutes and 6 deaths re-learning guard facing.

**2. Restore the best config's memory wording.** Keep the harness lines, but put back "Memory that survives history trimming … so findings from dropped turns are not lost" and the refuted-hypothesis sentence. Drop "Paths, action effects and deaths are already recorded by the harness". It costs nothing and can be checked on a panel by counting tags.

**3. Per-action object-delta digest.** After every action, and after every step inside a batch, print each changed play-area component, with the HUD rows excluded:
- its colour and size;
- old → new bounding box;
- displacement in pixels and in detected grid cells;
- whether it appeared, vanished or changed colour.

Make animation signatures position-aware: tn36's "same signature" was true by its own definition, but it described two different animations as a repeat. Cap the list at the top 8 changes. Do not assign roles; the behavioural entity tracker made wrong claims about 16% of the time and failed its gate.

Evidence:
- tn36 L1: the false "same signature" line.
- r11l: frame_diff called a recolour a "swap".
- sc25 L6: the ring moving to C went unseen.
- vc33 L5: the handle strips were counted as water.
- re86 L6: it read batch endpoints instead of each step.
- ka59: the 3-cell step went unnoticed.
- dc22 L3: the 1-pixel swap was filtered out.

**4. Batch stop rules and loop guards.**
1. Treat "anim only" and reverted moves as no-ops.
2. Stop a python call after about 8 consecutive actions that change only a cursor or nothing, or that revisit a board already seen in that call.
3. Refuse an exact repeat of an action that just reverted on an unchanged board.
4. Count only HUD rows as "edge".
5. For the stale-state guard, compare the controlled object, not the whole board.
6. Accept an optional `expect=` and halt on the first difference.

Rough effect: about 150 actions saved across the run. tr87 would go from 93.1 to about 100.

**5. Confirm before RESET.** The first RESET in a level returns "not executed", together with facts the harness computed:
- the untried actions from this exact state, including UNDO and arrows on the selected object;
- distinct objects not yet clicked;
- the steps left.

Calling RESET again executes it. In the same spirit: when UNDO is available and the last action produced a loss signal, the result says that UNDO restores the previous state at no step cost.

**6. Steps left and minutes left.** Estimate steps left from the bar's per-action decrement, and flag increases as refunds. Print "steps left ≈ N" and "game time left: M min" in every result. Cap thinking per call at roughly 6–8k tokens. budget-alert (PR #60) already covers the steps line, and fable-facts-10x is queued.

**7. The board after a win, and death facts.**
- On `level_completed`, give the winning action and name the board after the win explicitly as "this is the finished board", not as an animation. Add a computed relation list for that board: which objects overlap which targets, and where the player sits relative to special cells.
- On GAME OVER, give the cause class (bar at 0, or contact), the player's last cell, and what moved into it.
- Keep a `<level_facts>` line per level that is never trimmed.

**8. Persistent sandbox globals and helpers.** Keep globals, or a `mem` dict, between calls, and allow `import time`. Add `grid()` (the cell lattice), `runs(row)` (indexed colour runs) and `objects()` (play area only).

**9. The 10× stalls.** Measure queue and prefill time against generation time on the stalled requests, in this run and in the best-config runs. Since the stalls are equally common in the best-config runs, this is a lever for every config, not a memory-v2 fix.

## 6. Corrections to the draft synthesis

The verifiers changed these points, and the report above uses the corrected versions.

1. **Episode count.** It is 143, not 142: cd82 has 9 episodes, not 8. The analyst outputs were saved after all, as `games.json`. The source verifier re-checked all 143 on the exact engine versions and found 137 right, 6 right with a correction and 0 wrong. The draft's 139/142 rested on the analysts' own flags.
2. **cn04's rule** is "same hidden type" (8 or 13), not "no body overlap". See §1.
3. **Model tags.** All 10 strings are distinct (the draft said 8). They were written in reasoning (3), python strings (5) and reply text (2).
4. **Announced-but-not-written memory notes:** about 29 in about 17 games, not 28 in 14.
5. **The board after a win is shown** as the action's animation, poorly labelled. The draft said it is never shown.
6. **The stalls are not memory-v2 specific**, and trimming does not cause them by itself.
7. **Citations.**
   - re86 cites the L2 "how:" line in calls 69, 71, 72 and 80, not 84.
   - su15 cites it in call 56, not 55, and set it aside, so it is a weak case.
   - bp35's announcement is call 24, not 25.
   - ar25 L4 call 51 was a selection slip, not a misread.
8. **wa30's "0 deaths" is literally true.** The misleading part is "spent 164", which hides two attempts split by a RESET.
9. **Engine versions.** The vendor copy is a different version for 14 games. The analyst's sp80 simulator ports 0ee2d095, not the run's 589a99af; the real engine gives the same 3-action finish.
10. **New for this report:**
    - mismatch rows counted exactly (279);
    - truncations counted exactly (31);
    - the tag comparison against the best config, which revises the draft's view that the model simply ignores optional tags. It wrote 34–58 per run under the old wording.

## 7. Per-game appendix

Score columns: memory-v2, then best-config v1 / v2 / v3. Ep = second-guessing episodes; Mm = mismatch rows.

| Game | Levels | Score (mem-v2; v1 / v2 / v3) | Ep | Mm | What decided it |
|---|---|---|---|---|---|
| ar25 | 8/8 | 100; 100 / 100 / 100 | 6 | 15 | Won, but ~100 extra actions; the L3 call-31 plan equalled the final solution and was dropped for "probably blocked" (+39 actions) |
| bp35 | 1/9 | 2.22; 6.67 / 2.02 / 1.85 | 4 | 9 | L2: a run-length print dropped Y/W, so the spike tip read as a wall and it died twice; v1 cleared L2 with LEFT×2 at the same decision |
| cd82 | 6/6 | 89.13; 100 / 71.36 / 100 | 9 | 13 | L2 took 69 actions against a 6-action plan stated at call 30 (miscounted crop); L4 took 35 against 21 with the wrong "7 poses" note in memory |
| cn04 | 1/6 | 3.09; 47.62 / 4.76 / 71.43 | 6 | 9 | L2 fully paired at action 97 but two pairs were 8-with-13; the planned "try rot 0" was never run (7-action finish\*) |
| dc22 | 4/6 | 47.62; 26.79 / 47.62 / 47.62 | 8 | 9 | L5 (no run has cleared it): 149k tokens for 42 actions; the red-bar test was planned and not run; budget misread (~34 left against ~460) |
| ft09 | 6/6 | 100; 100 / 100 / 100 | 5 | 10 | Won; L3 took 27 against 14 because L2 taught "white → red" instead of "white → centre colour" |
| g50t | 4/7 | 35.71; 48.90 / 0 / 35.71 | 7 | 13 | Reached L5 at minute 111 (a 24.7-min stall at L1); had lost yellow=toggle and the 2-echo cap; 51-action L5 solution |
| ka59 | 1/7 | 0.50; 21.12 / 75.00 / 21.43 | 6 | 10 | The 3-cell step was lost at the first trim, so it used a 1-cell model for ~100 actions; two ~35-min stalls |
| lf52 | 4/10 | 16.76; 8.28 / 1.82 / 4.53 | 9 | 12 | Best of 4; L5 RESET at action 327 from a state 54 actions from the win\* while the opener said "Untried: LEFT" |
| lp85 | 8/8 | 100; 100 / 100 / 100 | 4 | 11 | Won; +10 actions on L3 (push/pull convention), +10 on L7 (no break on MISMATCH) |
| ls20 | 3/7 | 12.39; 16.01 / 30.62 / 52.30 | 5 | 12 | Its BFS treated black cells as walls; a 2,106-s stall; L4 cut off with 2.5 s left |
| m0r0 | 5/6 | 71.43; 71.43 / 47.62 / 71.43 | 5 | 14 | L6: selected the blue block at action 276, tried only UP/DOWN, RESET; 39 actions from there would win the game\* |
| r11l | 2/6 | 14.29; 100 / 71.43 / 47.62 | 5 | 12 | frame_diff called a selection recolour a "swap"; a hedge hardened into a rule; 4 of 5 hazard hits used in L3 |
| re86 | 5/8 | 41.67; 39.62 / 77.78 / 58.33 | 7 | 12 | L6: the 28×10 ring target stated at call 63 was dropped over "even dims impossible"; the L2 "how:" line misled it 4 times |
| s5i5 | 3/8 | 16.67; 16.67 / 16.67 / 7.71 | 5 | 13 | "The parent pulls the child" was found twice, never saved and trimmed before L4; 8 truncated calls (37% of tokens) |
| sb26 | 8/8 | 100; 100 / 94.24 / 91.95 | 4 | 8 | Won; L2 took 35 against 15: "OK let me just try H", then ran C |
| sc25 | 5/6 | 71.43; 86.31 / 79.89 / 71.43 | 6 | 11 | L6: the teleport cycle restarts on RESET, so both of its tests hit pad B; its filter hid the ring moving to C; a 47-min truncated call |
| sk48 | 1/8 | 2.78; 0.47 / 1.30 / 2.78 | 5 | 11 | "LEFT releases a jammed block" reversed from a false memory; the L2 simulator called a state 37 moves from the win a dead end; 46 + 12-min stalls |
| sp80 | 3/6 | 28.57; 27.05 / 14.29 / 21.49 | 5 | 12 | Best of 4; ended 3 actions from clearing L4\*; the L4 rules were lost at trims |
| su15 | 6/9 | 46.67; 13.33 / 68.98 / 11.81 | 5 | 10 | L7: lure, shrink and gray rules lost at trims; UNDO "not tried" all game; v2-snowiers had them as tags and cleared L7 |
| tn36 | 1/7 | 3.57; 100 / 75 / 3.57 | 5 | 10 | False "same signature" digest in L1; a contaminated test killed the 9-action L2 solution; selector boxes first clicked at L2 action ~70 |
| tr87 | 6/6 | 93.10; 100 / 100 / 71.43 | 5 | 9 | Won; one uncapped loop pressed LEFT 98 times, causing GAME OVER in L5 attempt 1 |
| tu93 | 7/9 | 52.78; 80 / 59.94 / 100 | 5 | 11 | The guard-facing rule was trimmed, and L6 cost 6 deaths and 47 min; the optimal 21-action L8 plan was computed 70 s before the cutoff |
| vc33 | 4/7 | 27.68; 100 / 75 / 21.43 | 6 | 11 | L5: gate B active at 52/52 but `surf()` read 42/42, so it dropped the confirmed rule\*; a 36-min stall in L4 |
| wa30 | 4/9 | 17.30; 13.33 / 2.22 / 11.53 | 6 | 12 | Best of 4; "spent 164, 0 deaths" became "the bar is non-fatal", then GAME OVER at 697 when the bar hit 0 |

\* replayed on the run's engine version by the verifier.

## Data and scripts

Everything is in [`data/memv2-reasoning-audit/`](../data/memv2-reasoning-audit/):

- `games.json`: the 25 per-game analyses. Each has these fields:
  - `outcome` and `true_rules`;
  - `levels` (the first correct idea per level, whether it was abandoned, and the blocker);
  - `second_guessing` (143 episodes, each with step, right idea, reversal, trigger, cost and the analyst's source flag);
  - `mismatches` (279 rows: kind, evidence, effect);
  - `memory` (what the harness saved, what the model saved, what was wrong, used and missing);
  - `other_gaps` and `takeaways`.
  
  This is the analysts' text before verification; §6 lists what changed.
- `synthesis.json`: the pooled draft. `verification.json`: both verifier reports. The source verifier gives 168 verdicts, one per idea and memory critique.
- `tags_compare.py` and `tags_compare.out`: the tag counts for memory-v2 against the three best-config runs (assistant messages only).
- `scripts/`: the extractors that built the compact transcripts and memory views (`extract.py`, `memextract.py`), the verifier checks (`verify/`, `verify/cn04/`) and the engine replay helper (`replay/rp.py`). They hard-code the scratch paths of this session's copy of the run. The run itself is `mellamoa/v4-mem-v2-10x` and the engines are in `experiments/arc-agi-3/data/environment_files/`.
