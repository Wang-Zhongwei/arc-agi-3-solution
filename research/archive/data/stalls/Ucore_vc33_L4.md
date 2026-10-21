# Ucore · vc33 · L4 stall read

Lead question: yes, Ucore repeated it. It clicked gate 1 once while gray (a77, nothing happened) and in the next 57 actions never clicked a gate while orange. Gates were orange 9 times (a98 and a105 mid-batch; a113, a121, a123, a128, a130-134); 3 still orange at turn end (i51, i60, i62). The run ended at a134 with gate 1 orange on the board.

## 1. Level and winning route (fast run U)
- Five water pools P1-P5, a key (plug) floating on P1. Blue pistons move 3 rows of water between neighbouring pools. Gate 1 (rows 43-54, cols 12-14) turns orange only when P1 and P2 both sit at 55; gate 2 (rows 34-45, cols 27-29) when P2 and P3 both sit at 46. Clicking an orange gate carries the key across. Goal: key on P3 with P3's top at 31 (band lines up with the yellow slot).
- U's route (a51-a73, 23 clicks, ~6 min, i26-i36, ~19.5k tokens): (61,9),(61,15)x3 opens gate 1; click gate 1 (53,13); (61,15)x3,(61,39)x2,(61,51),(61,39) opens gate 2; click gate 2 (44,28); (61,51)x5,(61,39)x5.
- U's turning point, i30: the turn message said "Colours on the board not clicked on this level: white, light gray, charcoal, black, yellow, orange." U: "Something new appeared! … The light-gray segment of tower 12-14 has turned ORANGE!" … "the orange is new/unfamiliar → per the game guidance, test it." i32: "clicking the orange gate moved the boat through the tower into pool B."

## 2. Ucore timeline (L4: t=1089 s (i23, a57) to the budget end 7261 s, uncleared; 41 requests, 78 actions, 121.9k tokens)
| Phase | Requests | Level-min | Tokens | What happened |
|---|---|---|---|---|
| P1 | i23-29 | 0-6 | 19.4k | 3-row transfer, key floats. Correct. |
| P2 | i30-38 | 6-27 | 34.9k | Clicked gray gate 1 (a77). i31 "Clicking the light gray = no-op … So the light gray isn't clickable." i37 "So the ONLY interactive elements are the 6 blue pistons." Goal already right at i32: "the plug must TRAVEL from R1 to R3!" |
| P3 | i39-46 | 27-35 | 30.3k | Tested flow/suction; gate 1 orange mid-batch a98, gone by a99; saw only white "jets"; own `read4` corrupted by them (i42 "+10 units created!"); bar low → reset (a103). |
| Starved | i47 | 35-75 | 1.1k | One request took 39 min. |
| P4 | i48-58 | 75-82 | 28.8k | i49 "the light gray window turned ORANGE"; filed a rule ("GATE that opens (turns orange) when the … levels match"); tested flow-through, merged pumping, both gates open — always as a valve. i55 "the gate seems to do nothing except open at equal levels." |
| Starved | i59 | 82-100 | 1.9k | |
| P5 | i60-62 | 100-103 | 5.5k | Gate 1 open, key level with it: pumped (a129) instead of clicking the gate. i62 "So the gates are pure red herrings?!" |
Starvation ~68 of 103 min for ~3k tokens.

## 3. Decisive gap
- Root cause: one click on the gray gate became a lasting "gate is inert" belief.
- Mid-batch misses (a98, a105): orange appeared and vanished inside a batch; missed for ~20 requests.
- Truncated colours line (i51, i60, i62): `<this_level>` `effect_rows` has a 700-char budget and the colours line renders last. At i51 the click rows used 651 chars, so "Colours on the board not clicked on this level: white, orange." (64) was dropped. At i60/i62 the light-gray click row overflowed first and the loop's `break` dropped the 57-char "…: orange." line, which would have fit (`arm/mem-unified:ARC3-Inference/inference/agent/memory_layout.py:373-386`).
- The model uses that line as a to-do list (same episode): i47 "That's a notable gap! … notes say 'Colours on the board not clicked on this level: white, yellow'" → clicked yellow; i58 "What about clicking the WHITE? Never tried! … notes say '…: white'" → clicked white. At i60/i62 the only untried colour was orange.
- The orange was visible (diff image, Ucore's own `oranges()` prints): noticed, never treated as clickable. No turning point.

## 4. Classification (~122k tokens)
(b) wrong mechanic held ~45%; (a) misperception a fact could fix ~20% (mid-batch orange, truncated line); (d) loop ~12% (key-carry test x4, flow-through x5); (c) tooling ~7% (cells within 4 of the edge treated as edge-only so real refills at rows 61-63 counted as no change → false batch halts at i27, i31-33, i47; own `read4` misread jets); correct exploration (P1) ~16%; (e) starvation ~66% of wall-clock, almost no tokens; (f) not needed.

## 5. Intervention
A. Never truncate the untried-colours line; put new colours first with location. Render before the click rows, `break` → `continue` in the budget loop, mark colours absent from the level's first board: `Colours on the board not clicked on this level: orange (new on this level; rows 43-54, cols 12-14), white.` Signal: a colour on the current board, inside the edge crop, never clicked on this level. Would fire at i51 (gate 2), i60, i62 (gate 1); it had fired at i23-i47 and was dropped at i48-53.
B. Stop a batch when a new colour appears (mirrors the no-change guard): stop after the first action whose board shows a colour on no earlier board of this level. Note: `Note: the sequence was stopped after action 98 because a colour not seen before on this level appeared: orange, 36 cells, rows 43-54, cols 12-14. The remaining 2 action(s) were never executed.` Would fire at a98 in i41's batch (i42 sees the open gate at ~level-minute 29), then a105. Saving: the model had the goal and pump plan at i32; one gate click → clear within ~25 actions, ~70 min saved. Noise: earlier replay 0-6 firings per game.
Coverage: none of the existing facts (moved/touching: the gate changes colour, doesn't move; own quoted rules present from i50, no click; chaser n/a). Re-probe text failed in the lab (A 1→2/20, uc3 1→0/30), so B plus A's short to-do line is the bet.
Cheap check: freeze Ucore at i60 (board after a128); control vs restored "Colours on the board not clicked on this level: orange."; success = a click inside rows 43-54, cols 12-14.
Files: S/lab2/extract/Ucore/vc33.requests.jsonl, S/lab2/extract/U/vc33.requests.jsonl, S/lab2/stats_r2_A.json, S/lab2/stats_r3_uc3.json.
