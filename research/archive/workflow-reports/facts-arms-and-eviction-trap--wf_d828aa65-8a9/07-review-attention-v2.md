# facts-arms-and-eviction-trap: review:attention-v2

Workflow run `wf_d828aa65-8a9`, phase Review. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

Review of `arm/attention-prompt` (9c3a40b..f43ca80 on top of `arm/stack-v2`). I found 6 defects, none blocking. Two of them put a false fact in front of the model: an "appeared" fact on a real board, and the description of positions in the system prompt. The rest are low or nits. No edits or commits were made.

## Defects, most severe first

**1. Medium-low: objects that move in from the ignored edge band are reported as "appeared".**
- **Where:** `inference/agent/salient_facts.py:249-260` (`_is_new`). It only checks the previous board's interior cells under the object. An object that was in the band and then moves into the interior passes both tests.
- **Real case:** ar25 level 6, step 32: `appeared W 1x1 at (34,4) (34,7) +17`. I checked the board: the S/W dotted bar at rows 0-2 (in the band) moved to rows 33-35. The white dots did not appear; they moved.
- **Scale:** in my niced replay of the 5 held-out games, 19 of 54 detected "new" objects were these band entries. They make up one emitted fact out of about 20 "appeared" facts.
- **Synthetic repro:** a 3x3 at rows 0-2 slides to rows 5-7 and gives `Salient: appeared O 3x3 at (5,20).`
- **Fix:** in `update()`, count the keys of small objects touching the band on a full-board segmentation (`segment_interior(grid, 0)`, about 2 ms). Store that count with `_prev`. In `_is_new`, return False when the previous band had more of `obj.key` than the current band.
  - On held-out this check also suppresses two ar25 reflection pieces whose earlier copies left the band at the same time: L4 `c 3x3 (24,18)` and L7 `c 12x6 (48,45)`. Look at those before accepting the fix.
  - Add a unit test for the band-to-interior slide.

**2. Low-medium: the system prompt describes positions wrongly.**
- **Where:** `prompts.py:150` says "Positions are the (row,col) of an object's top-left cell". The code reports the top-left corner of the bounding box (`salient_facts.py:74-75`).
- **Real case:** for non-rectangular objects that corner is not part of the object. re86 level 6, step 24 reports `appeared b 25x12 at (18,36)`; cell (18,36) is black, and the blue cross starts at (18,45). A model that indexes `current_frame[18][36]` finds the wrong colour.
- **Cut objects:** for objects cut by the band, the corner is clipped at the interior edge. su15 reports `Y first seen ... at (4,30)` for a block that starts at row 3.
- `CONFIGURATION.md:81` already has it right.
- **Fix:** change the wording to "the (row,col) of the top-left corner of an object's bounding box". The text is static and about 3 tokens longer; the `SECTION` constant in the test follows automatically.

**3. Low: facts that are dropped are recorded as shown, so two claims are overstated.**
- **Where:**
  - `salient_facts.py:372-374` marks every pending fact as emitted, including facts cut by `MAX_FACTS` or `MAX_CHARS`.
  - `render()` at `:201` and `:207` drops a colour-only fact whenever a pending NEW or ODD fact names that colour, even if that fact was itself cut.
- **Effect:** such facts are never shown on that level. That contradicts two statements:
  - "a fact is listed once per level" (`prompts.py:150-152`)
  - "the first line after the hold lists everything pending" (`CONFIGURATION.md:54`). With `ARC3_ATTENTION_AFTER_TOKENS` set, everything past 4 facts is dropped for the rest of the level.
- **Fix:**
  - Build `about_objects` from the facts actually rendered (two passes).
  - Then either keep cut facts pending, or reword both texts, e.g. "lists up to 4 pending facts; the rest are not shown on this level".

**4. Low: the "off the rows" fact fires on regular lattices.**
- **Where:** `_off_line` (`salient_facts.py:180-192`) only requires exactly one member with a unique `r0` or `c0`.
- **Real case:** I checked the tu93 case the implementer left unchecked. tu93 level 5, step 15 gives `W 3x3 at (20,9) off the rows of its 58 same-shape peers`. The board is a maze lattice with a 3-cell pitch. Row 20's only other node is an O square with a p cell, so it has a different shape. The W node sits on the lattice.
- The statement is literally true but marks a normal cell as an anomaly. There were 0 such facts on the held-out set, so the impact is small.
- **Fix:** require the lone coordinate to be off the peers' pitch (not congruent modulo the GCD of peer row gaps), or drop this fact type.

**5. Low: the tests don't guard against drift from stack-v2.**
- **Where:** `test_salient_turn_line.py:81-88` and `test_attention_guidance.py:35-41` compare "flag unset" with "flag = 0" in the same tree. Both paths stop at `_attention_guidance_enabled()`, so the tests cannot catch a change from `arm/stack-v2`.
- **What actually holds:** byte identity is shown only by the out-of-tree dumps. I re-ran `cmp` on `salient/id_*`, `idc_*` and `idf_*` for both trees: all identical. The code also confirms it:
  - `tool_agent.py:2759` is gated on the flag.
  - `:5231` extends the message with an empty list when the flag is off.
- **Also untested:** a death, RESET or UNDO putting the board back to the level's start produces no "appeared" fact.
- **Fix:** commit a sha256 of the flags-off system prompt plus three turn messages (from `identity_dump.py`) as a pinned constant, and add the reset-to-start unit test.

**6. Nits.**
- **Tracker can silently never start:** `tool_agent.py:4797` creates the tracker only when `_session_runtime_dir` differs from the saved value. If both are None, `self._salient` is never set, every call raises `AttributeError`, and the line silently never appears. This can't happen in `analyze()`, because `_ensure_session` runs first. Use a sentinel default.
- **Mirror flips:** `_canonical_shape` (`salient_facts.py:78-86`) covers rotations only. A sprite that mirrors while moving its own width is reported as appeared. Synthetic repro: `appeared R 3x2 at (10,14)`. There were 0 cases in the held-out replay.
- **Border value in the prompt:** `tool_agent.py:2762` formats the raw border value ("1 cells", negative values not clamped), while `SalientFacts` uses `max(0, …)`.

## Checks that passed
- **Boards spot-checked:**
  - cn04 L2 steps 8-9: the red 6x3 is two touching 3x3 markers, and the grey "unlike" facts are markers that changed colour when a piece docked. Both true; the second is useful.
  - ar25 L6 step 32: the c reflection "appeared" is genuine; the W dots are defect 1.
  - re86 L2 and L6: "first seen" is true. The `+45` count is because diagonal lines split into 1x1 objects under 4-connectivity. The 25x12 coordinate is defect 2.
  - su15 L2 and L4: "first seen" facts for the HUD indicator colour are true.
  - tu93 L5: defect 4.
- **Facts, not advice:** every string the model sees is declarative: the two system sentences (`prompts.py:144-153`) and every fact template (`salient_facts.py:272-328`). The "Salient:" label steers attention, as the user asked.
- **Cache safety:** the line goes only into the new opener, between "Valid actions" and "Only tool" (test `:104`). The system text is static and set once per agent. History is never rewritten.
  - If a carrying message is lost (rolled-back turn, pruned resume opener, history replaced by a rolling summary, trim at request time), the facts are re-sent one turn later.
- **Caps:** 100 characters and 4 facts are enforced (`render` `:205`, `:213`). The longest single fact is about 70 characters, so the first fact always fits.
- **Hold:** `_tokens_at_level_start` is updated during tool dispatch at a level change, before the next opener (`tool_agent.py:3934`), so the hold restarts correctly.
- **Short resumptions:** they neither show nor use up facts (`:4792`).
- **Exceptions:** the whole path is wrapped in `try/except` and adds nothing on failure (`:4821`); a test covers it.
- **Cost:** 1.9 ms of CPU per turn on 53 re86 boards. The history substring check is about 0.02 ms. Nothing runs per action.
- **Docs and arms:** both flags are in the `CONFIGURATION.md` table plus a subsection. `attention-prompt` and `stack-v2-attention` are in `kaggle/arms.json`. The notebook patch is `git diff bundle-base..HEAD`, so it includes the new module. Other arms are documented only in `CONFIGURATION.md`, so not touching the READMEs is consistent with them.
- **Tests:** 28 targeted tests pass (niced; imports confirmed to load from the worktree).
- **Git:** the tree is clean, all 4 commits have imperative subjects and the exact trailers, nothing was pushed.
- **Eviction/handover item:** not applicable to this branch.

My diagnostic scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/review-attention/`: `new_diag.py` and `new_diag2.py`.
