# Ucore · tn36 · L4 stall report

## 1. Level and fast route
- Right: an 8x8 yellow piece with a 4x2 notch at the bottom; below it a magenta wall with a 1-cell gap at cols 41-44; below that a small U-socket whose free space is a 4x4 notched shape. Program: 6 columns of toggles (3 bars + 3 stems each, rows 33-48). The ball at (58,57) runs the program; a failed run reverts.
- Legend: the 4 boxes bottom-left (centres (58,5), (58,15), (58,25), (58,35)). Clicking a box writes its code into the left demo panel and animates a demo piece doing that move. Codes here: left={S1,S3}, big={S2}, down={B1,S1}, small={B1,S2}.
- Fast route: U cleared it in 17 actions / 8 requests / 12.9k tokens: click the 4 boxes, program small, left, down x4, run. All 7 fast runs open L4 with these box clicks; all had clicked the boxes on L2 or L3.
- U's turning point: i68 "the left machine's program = the encoding of the selected box … click each box in turn and read the left panel's program + the piece's motion"; i71 "So {bar1 + stem2} = scale DOWN one step!"

## 2. The stall (L4 from 50.6 min (i32, action 91), open at the 121-min deadline; 41 requests, 128.6k tokens, 103 actions, 12 attempts, no deaths)
| Phase | Requests | Wall (min) | Tokens | What happened |
|---|---|---|---|---|
| P1 | i32-35 | 50.6-57.3 | 19.4k | Roles reversed (cup as plug, block as socket). i34 hit the 12,288 cap with no tool call fitting a +3-column offset. Ran [U1x4, GROW]. |
| P2 | i36-40 | 57.3-59.5 | 7.0k | Read the animation, fixed roles. i40 "So the cup is stationary = the socket ✓. So the block is the plug." GROW took the block 8→12. |
| P3 | i41-58 | 59.5-86.4 | 61.2k | Blind search for a SHRINK code: 9 runs, ~18 combos. Never tried {B1,S2}: its L2 table said "{top bar,mid stem}=noop" (i51 "{S2,B1} = noop"). |
| P4 | i59-61 | 86.4-101.7 | 7.2k | Starved: 15 min for 3 requests. |
| P5 | i62-72 | 101.7-111.6, idle to 121 | 33.8k | Clicked the boxes but read them as "4 sub-puzzles/previews"; detour into "clip the plug at the panel edge"; i69 another 12,288-token reply with no tool call; clicked box4 (action 192) but its helper read the wrong columns; final reply cut at the deadline. |
P3 confusions: program not cleared between runs (a move credited to {B2}); frame-to-column mapping mixed up; misaligned box crops (i48-50: "box3: empty … decorative robot faces / progress indicator"); a false rule at i46 "{S3}=ROTATE", which the rules ledger then quoted ("So the block ROTATES ✓") every turn until i71 found it false. At i41 it nearly explored the boxes ("maybe the 4 bottom boxes correspond to the 4 'row2' glyph variants … Let me just test") but tested glyphs, not boxes.

## 3. Decisive gap
Missing: the boxes are clickable, a click writes its code into the demo panel, and one of them is shrink. Always on screen; at i32 it even guessed box4 means "scaled down (tiny)". Ucore never clicked the boxes on any level: it solved L2 by brute-forcing codes (64 actions, 42 min), and its table recorded shrink ({B1,S2}) as "noop" because the L2 piece was already at minimum size. That stale noop sat in `<previous_levels>` and shut off the right experiment; never retested. Turning point at i61 (83 actions, 101.7 min): the colour-level fact landed: "the <this_level> notes say 'Colours on the board not clicked …' — so I have NOT clicked any of the 4 boxes … That's a big unexplored area!" (U's own L2 box exploration started the same way at i23: "Colours not clicked … Clicking yellow/gray will tell"). Too late; then a reader bug (`leftglyphs` read cols 9/14/19 instead of 11/16/21) misread the demos.

## 4. Classification (~129k tokens)
(b)+(f) ~55% (wrong goal hypotheses; code search held to a stale L2 noop; the needed insight is gettable from a click); (c) ~30% (two capped 12k replies with no action = 19% of all tokens; stale program state; frame mapping; misaligned crops/readers); (d) ~5% (repeated no-op runs; actions 173 and 186 repeat 139's animation); (e) ~10% of tokens but ~30 of 70 min (gaps 70.6-78.6, 86-102, 111.6-121); (a) is the fix lever.

## 5. Intervention: object-level, game-wide "never clicked" line
- Compute: segmentation components of the current frame; drop background-sized and repeated grid cells; a component is clicked if any click in this game (any level) landed in its bbox or enclosing frame; list never-clicked ones with centres; flag ones whose pixels differ from the previous level's start frame.
- Text at L4 start (i32): "Never clicked in this game: 4 framed boxes at (58,5) [blue frame], (58,15), (58,25), (58,35). New since level 3: (58,15) now shows a 5x5 yellow square, (58,35) a 1-cell yellow dot."
- Fires every turn while such objects exist (first at L2 start, i12); moves to the top of the turn message after >= 3 "animated and changed nothing" run clicks on the level with never-clicked objects remaining (L4: i42, 61 min, ~40 min before the real box click).
- Coverage: moved-objects no help (runs revert); touching n/a; own rules hurt (quoted the false ROTATE rule; the L2 noop came from the recap); chaser n/a; the colour "not clicked" line partly covers it and was the real trigger in both Ucore (i61) and U (L2 i23). Caveat: the lab found "untried-action facts: no effect" → needs a resample check at a frozen stuck state (e.g. i45 or i53).
- Secondary: cap reasoning at ~6k tokens and force a tool call; two runaway replies (i34, i69) burned 24.6k tokens with no action.
Data: S/lab2/extract/{Ucore,U,uc1,uc2}/tn36.requests.jsonl; Ucore events kaggle-outputs/v4-mem-unified-core-10x/v1/artifacts/tn36-ef4dde99_p0_events.jsonl. Also extracted U, uc2, ms1, C, ms2 into S/lab2/extract/.
