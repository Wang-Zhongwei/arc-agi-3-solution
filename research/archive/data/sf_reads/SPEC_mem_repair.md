# SPEC: mem-repair (2026-10-14)

## Why
Goal (the user's): 9-10 of 25 games fully won in the 121-min Kaggle run. The user: "optimizing the prompts and
improving upon memory is making it better". The current arm `stall-fix-10x` (SF, 51.85, 8 wins) = `mem-unified-10x`
+ six stall-fix flags. Eight reads of SF's stalled levels (tn36 L2, vc33 L4, sc25 L3/L6, tr87 L2, m0r0 L3/L5, su15 L4,
cn04 L2/L4) plus a run-wide memory audit found that the harness-written memory itself often carries WRONG, STALE or
NOISY content, or drops a correct fact exactly when it matters. This arm REPAIRS existing memory lines (prefer
correcting/removing over adding). Each change sits behind its own default-off env flag. The final test is ONE
121-min run: arm `mem-repair-10x` = `stall-fix-10x` + all flags below.

## Model facts (don't re-derive)
Qwen3.8-Flash-Next W4A16 (small, local). Harness-computed FACTS in tool results / turn messages can work; ADVICE text
is ignored or hurts; optional structure is ignored. State lab (k/20 right next moves at frozen stuck states):
"moved objects since level start" 0 -> 16 (sc25 L6); "touching same-colour objects listed separately" 5 -> 10;
"your own confirmed rules quoted" 3 -> 9; untried-action / action-count / clock facts: no effect. The model reads the
rules line as harness-verified (tr87 i52: "the harness note says 'Confirmed: cycles are glyph-determined'").

## Shared rules (same as the stall-fix build)
- S = /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad.
  Base = `arm/stall-fix` 575d2197. Harness code: `<worktree>/ARC3-Inference/inference/` (agent loop
  `agent/tool_agent.py`, sandbox `agent/python_tool_sandbox.py`, memory `agent/memory_layout.py`,
  `agent/confirmed_rules.py`, `agent/board_facts.py`, `agent/memory_facts.py`, segmentation `utils/segmentation.py`).
  Option reference `ARC3-Inference/CONFIGURATION.md` (see the "Stall-fix arm" section for house style).
- Python: `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/.venv/bin/python`
  with cwd = `<worktree>/ARC3-Inference` (check once: `python -c "import inference; print(inference.__file__)"`).
  Always `nice -n 15`. Targeted tests freely; the FULL suite only once at the end, under the shared lock:
  `until mkdir $S/suite.lock 2>/dev/null; do sleep 30; done; nice -n 15 <python> -m pytest -q -x tests; rmdir $S/suite.lock`
  (always rmdir the lock, even on failure).
- Flags off => byte-identical requests to `arm/stall-fix` (golden tests `tests/test_mem_unified*.py`,
  `tests/mem_unified_golden.py`, `tests/fixtures/pf_keep_v2_golden.json` must pass). Read flags at call time like the
  neighbouring code. Flags that must reach the sandbox subprocess go the way existing ones do.
- Real data: SF raw logs `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/kaggle-outputs/v4-stall-fix-10x/v1`
  (`<gid>_p0_requests.jsonl` = every request's full messages; `artifacts/<gid>_p0_events.jsonl` = every action with
  its 64x64 `board`, level, game_over, board_changed, action_display, animation frames where logged). U (mem-unified)
  logs: `.../kaggle-outputs/v4-mem-unified-10x/v1`. Compact extracts: `S/lab2/extract/<run>/<game>.requests.jsonl`
  (0-based `i`; fields i, action, level, last_text, reasoning, code, usage) and `.actions.jsonl`. Read-only data.
  Small fixtures you cut go under `ARC3-Inference/tests/fixtures/` (gzip, < 100 KB each).
- VERIFY CONTENT, NOT PLACEMENT: replay your change over all 25 SF games (and U where useful) and report per-game
  firing counts, typical length, and the exact text at the moments named below. A line that fires everywhere is
  noise; one that fires at the wrong moment misleads. Tune on that evidence and say what you tuned.
- Commits: small, one logical change each, imperative subjects; end every message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01AULSqB7ZpbNcZV9DbF3pfR`.
  Commit only to your branch; do NOT push, do NOT open PRs, do NOT edit `kaggle/arms.json` or `kaggle/runs.json`.
  Add your flags' rows to `CONFIGURATION.md` (a "Mem-repair arm" section after "Stall-fix arm").
- Prompt strings are behaviour: change only the lines this spec names; keep text factual, no advice/imperatives.
- Never run game engine code. Never edit run directories or other worktrees.
- Return (final message, <= 800 words): commits (hash + subject), what each flag does exactly (one real example
  line from replay), replay firing stats, the texts at the named moments, test results (targeted + full suite
  counts), deviations from the spec with reasons.

## Builder RULES (`arm/mem-repair-rules`, worktree `S/wt/mr-rules`)
Evidence (run-wide audit, 512 distinct SF rule entries hand-labelled in `S/lab2/audit/SF_rules_labeled.json`,
stats script `S/lab2/audit/rstats.py`): real mechanic 36%, trivia (state/coordinates/arithmetic) 24%, plan 17%,
hypothesis stated as fact 15% (30% of impressions on stalled levels), about the model's own code 4%, echo of
memory/prompt 3%. 23 entries later contradicted; in 7 cases both sides shown at once (sp80 i34-43). Refuted rules
shown while stuck: re86 L7 ("Level 6: the dots were absorbed when covered…", 11 turns), dc22 L3 ("So the purple
region can NEVER be entered"), wa30 L3 ("the 3 left boxes can never be delivered", later cleared by delivering them),
sc25 L6 ("Either way, hitting any dark red destroys all dark red ✓." shown i87-i101 after refutation at i91; i95
"So the outer frame may just be a 'big marker' that's decorative…" promoted though hedged), m0r0 L3 ("dots are
impassable walls, and clicking a dot is a penalty" shown after the model refuted it), tr87 L2 ("The order list: I'll
maintain as a list…"; "Confirmed: cycles are glyph-determined" wrong). Current filter (`confirmed_rules.py`): ✓ /
confirmed / verified or a "So/Thus" sentence after a marked one; needs a rule word; 40-260 chars; no leading plan
words; hedge list lacks "may"; at most one coordinate pair (row/col RANGES not counted); <= 8 digits; newline
splitting makes fragments; top 3 per level by strong words; frozen into `<previous_levels>` at the clear.
1. `ARC3_RULES_FILTER_V2`: reject candidates that (a) contain any coordinate pair or row/col range ("rows 42-53",
   "cols 3-5", "(12, 40)"), (b) are fragments (end in ":" / "…" / "...", start with ")" / a lowercase word /
   "and|but|or", unbalanced brackets or quotes), (c) are echoes (contain quoted text or name a level: "level 3",
   "L3", "Level 6"), (d) are plans or about the model's own code ("let me", "let's", "I'll", "I will", "I need",
   "we need", "first,", "actually", "next", "my/the code|solver|search|BFS|helper|function|simulator|model"),
   (e) are hedged ("may", "might", "maybe", "perhaps", "probably", "likely", "seems", "appears", "could",
   "possibly", "guess", "assume", "?"), (f) are UNMARKED follow-on conclusions ("So/Thus …" without their own ✓)
   containing never/cannot/can't/impossible/only/must/always. Apply to both the current-level line and the rules
   frozen into `<previous_levels>` (same extraction). Measure on the labelled set: mechanics share, share of
   mechanics kept, hypotheses kept, contradicted kept — before vs after (target: mechanics >= 45% of kept entries,
   >= 85% of mechanics kept). Then replay over all 25 SF games: what each game's rules line would show at its stall
   moments (tr87 i16-24, sc25 i87-101, m0r0 i58-80 and i125+, su15 i90+, vc33 i72+, re86 L7, dc22 L3, wa30 L3).
2. `ARC3_RULES_ACTION_TAG`: suffix each quoted rule with the game action number at which it was written, e.g.
   `"…" (action 63)`, and order them oldest -> newest, so contradictory pairs are ordered. Keep the existing label
   text ("Rules you confirmed on this level") unchanged (lab-tested wording).
Also, cheaply (one more flag, `ARC3_RETAINED_DELTA`, in `tool_agent.py`): the "Your retained functions from earlier
Python calls in this game: …" list is 37% of a turn message's text (median 117 tokens, p90 464, max 2,048), identical
to the previous turn's list 69% of the time; ~30% of it is default-argument values (m0r0 alone 127k tokens). When the
list is unchanged since the last list still present in the context, send "Your retained functions are unchanged
(N functions)." instead; send the full list again after any context trim/rewrite that removed the last full list.
Shorten default values longer than 30 chars to a type summary (`FREE=<tuple, 74 items>`). Replay: new tokens saved
per game; check the full list is re-sent right after every trim in SF's logs.

## Builder FACTS (`arm/mem-repair-facts`, worktree `S/wt/mr-facts`)
1. `ARC3_MOVED_PERSIST` — sc25 L6 (SF, coin-flip game: won in 5 of 13 runs). At i89-91 the moved-objects line and
   the "Boxes marked…" line (corner-pixel box: "rows 33-36 cols 29-32 (4x4: 12 g cells and the 4 Y corners…)") were
   correct and decisive (i90: "this is the NEW 4x4 destination marker (it moved…)"). At action 229 a shrink hid the
   yellow 6x6 corner frame and a purple ring appeared around pad B. `moved_lines` only reports markers on the board
   NOW, and the boxes line only prints beside a moved line, so both vanished from i92 on — exactly when the
   ring-size link mattered (2x2 teleports, a RESET discarding progress, run ended unsolved; fast run B cleared L6 in
   10 min). Change: (a) print the boxes line whenever a corner-pixel box is on the board (not only beside a moved
   line); (b) keep reporting a marker that was on the board earlier on this level and is now gone, and report new
   markers, e.g. "Hidden since action 229 MOUSE(60,30) (the click at which the 16-cell b/S block became 4 cells): the
   yellow 6x6 corner frame (last at rows 33-36 cols 29-32). Appeared at action 229: a purple 6x6 corner frame around
   rows 40-43 cols 12-15; the 4-cell block landed inside it at action 233." Trigger: a marker present earlier on
   this level is gone, or a new one appears, at an action. Replay on SF sc25 (must fire from i92 and be present at
   i93, i95, i99) and all 25 games (firing counts; no spam on games where markers flicker — tune).
2. `ARC3_CLEARED_BOARDS` — vc33 L4 (SF, 28 near-idle requests i44-i71, ~75k tokens): the model read each earlier
   level's LAST `history` frame, which is the board BEFORE the clearing click, so every "winning board" looked one
   move short; it concluded its correct recap was "wrong/hallucinated" (i48). A code grep finds models reading
   earlier levels' last frames in 18 game-runs across 9 runs (cn04 U i90, Ucore i133, A i102; r11l uc1/A; tu93, tn36,
   m0r0, sc25). m0r0 L3 (SF i63) reused a misleading level-2 recap about the winning geometry. Change: expose in the
   sandbox `cleared_boards[N]` = the board the clearing action produced (its last animation frame before the next
   level loads; if no such frame exists, None), and add one fact line to the level-cleared message and to that
   level's `<previous_levels>` entry: "Level 3 was cleared by action 50, click (56, 46). history's last level-3 frame
   (step 49) is the board BEFORE that action; the board it produced is cleared_boards[3]." (or "…the board it
   produced was never shown; history's last level-3 frame is one action short of the clear."). Check in the event
   logs which frames exist for clearing actions (animation frames) across SF's games and say how often a produced
   board is available.
3. `ARC3_THIS_LEVEL_AFTER_TRIM` — vc33 L4: from i58 to i71 the context held no `<this_level>` at all: the trims
   dropped every L4 turn message and no-action replies don't render one; i68: "re-derive the whole level-4 board
   layout from scratch, since my memory of it may be imperfect". Change: when a context trim/rewrite leaves no
   `<this_level>` block for the current level in the kept messages, include the current `<this_level>` block in the
   next message the harness sends (including "executed NO actions" messages). Replay: how often SF's trims left no
   `<this_level>` (per game), and the message at vc33 i58.

## Builder OBJECTS (`arm/mem-repair-objects`, worktree `S/wt/mr-objects`)
Evidence: the SF colours row ("Colours on the board not clicked on this level: …", flag `ARC3_MEM_COLOURS_FIRST`) is
click-framed and colour-level.
- m0r0 L5 (SF i98-150, 130k tokens, never cleared; fast runs 9-10 min): coloured 4x4 pressure plates open same-colour
  bars while a square stands on them. SF never stepped onto the purple plate until i140 and held "the coloured blocks
  are walls" (i108, i125) without testing it; the colours row led it to CLICK gates 7 times (i131: "I have NOT
  clicked on a gate! Let me click the O gate"). Fast runs pressed into the purple block in their first 5 actions.
- tn36 L2 (SF, never cleared; fast run uc1 4 min): the two legend boxes at (58,11) and (58,21) are a demo; SF found
  them only at i42 (73 min). The colours row dropped "yellow" after the model clicked the yellow block (i26: "colors
  not clicked: … yellow. Let me click the yellow block" -> a no-op), although the yellow in the boxes was never
  clicked. An object-level row would list the boxes as untouched objects with locations.
- su15 L4: pink invader (the real novel object) was never tagged "new" because it was on the level's first board;
  34 of 35 "new on this level" tags were the model's own merge products, shown every turn (noise). cn04: all 51
  "new" tags were recolourings caused by the model's own selection clicks (noise).
- cn04 L2 (SF i20): action 19 clicked (55,54), the background gap inside the blue U-shape; `<this_level>` said
  "Clicks on orange cells … (55, 54) did nothing" (correct) but the model doubted it ("maybe it mislabels") and lost
  ~11 requests before discovering that clicks select pieces. ms2 also clicked into piece O's hole on cn04 L4.
- Audit: 152 of 1,237 `<this_level>` blocks drop >= 1 clicked-colour row to the 700-char budget (462 rows; lf52 70
  blocks, m0r0 25, dc22 24, sc25 14); that colour then appears in neither the click rows nor the not-clicked list.
  44% of arrow presses are reported as "changed cell colours" even when an object moved (lf52 "RIGHT (24 presses):
  moved cells once, changed cell colours 19 times" while the box was moving; m0r0 L5 same).
1. `ARC3_MEM_CONTACT_ROW` (replaces the colours row when on): an object-level row first in `<this_level>`, e.g.
   "Not touched yet on this level (never clicked, entered or bumped): purple 4x4 at rows 50-53, cols 14-17; light
   green 4x4 at rows 26-29, cols 34-37; orange 4x4 at rows 26-29, cols 58-61; 2 yellow boxes at rows 56-60, cols
   9-13 and 19-23." Objects: segmentation components (>= 4 cells, not background, not the HUD/budget bar), grouped
   when identical shape+colour. Touched = clicked inside its bbox, overlapped by a moved object, or adjacent (in the
   press direction) to an object whose arrow press was blocked. Tag "(new in this game)" only when its colour+shape
   was not on any earlier level's board, and never tag cells that the model's own action just recoloured (cells
   that held another non-background colour one frame earlier). Cap the row (e.g. 6 items, ~300 chars) and choose
   items deterministically (new-in-game first, then larger/rarer). Works for keyboard games too. Replay on all 25 SF
   games and report the text at m0r0 i100 and i108, tn36 i26 and i42, su15 i41, cn04 i20 and a few cn04 turns
   (no selection-recolouring noise), vc33 i28 and i72, sc25 i93 (the "light green" pocket). Tune for noise.
2. `ARC3_CLICK_MISS`: after a MOUSE action that changed nothing, if the clicked cell is background and lies inside,
   or within 3 cells of, another object's bounding box, add a turn-message line: "Action 19 MOUSE(55,54) hit
   background (orange) inside the outline of the blue object (rows 53-58, cols 50-58); it did not touch it. Nearest
   blue cells: (56,54), (55,52)." Replay: per-game firing counts and the text at cn04 i20.
3. `ARC3_MEM_CLICKED_ALSO`: when the 700-char budget skips click rows, add a short row "Also clicked on this level:
   yellow x1, purple x2." so no clicked colour disappears silently.
4. `ARC3_EFFECT_MOVED_FIX` (only if the cause is clear and the fix small): find why arrow presses that moved an object
   are counted as "changed cell colours" and make the effect classification report "moved cells" when a component
   translated. Show before/after counts on lf52 and m0r0 L5.
