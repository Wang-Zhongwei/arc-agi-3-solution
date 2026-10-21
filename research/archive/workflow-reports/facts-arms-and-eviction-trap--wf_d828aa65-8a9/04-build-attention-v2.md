# facts-arms-and-eviction-trap: build:attention-v2

Workflow run `wf_d828aa65-8a9`, phase Build. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

Branch `arm/attention-prompt` no longer gives the model advice. It now gets two declarative sentences in the system prompt and, on some turns, one `Salient:` line of facts the harness computes from the board. With the flags off, every request is byte-identical to `arm/stack-v2`. Nothing has been run with a model yet: all numbers below come from replaying recorded boards.

**Commits** (on top of 5c54594; nothing pushed, no PR edits):
- `9c3a40b` Add a tracker of salient board facts for the turn message
- `1f2a1a5` Add the Salient facts line to each turn message
- `8bacd3d` Replace the 'Where to look' advice with a description of the Salient line
- `f43ca80` Document the Salient facts line and ARC3_ATTENTION_AFTER_TOKENS

**How it works**
- The flag is still `ARC3_ATTENTION_GUIDANCE`, so the Kaggle arms `attention-prompt` and `stack-v2-attention` work unchanged. `ATTENTION_GUIDANCE_ADDENDUM` is now `SALIENT_FACTS_ADDENDUM`.
- New sub-flag `ARC3_ATTENTION_AFTER_TOKENS` (default 0) holds the line until the model has generated that many tokens on the current level.
- The line goes only into the new user message, after "Valid actions" (and after the death ledger when that is on), so the cached history is untouched.
- The edge band set by `ARC3_NOOP_GUARD_BORDER` is ignored, and objects it cuts in half take no part in the shape facts.
- Each fact is keyed without coordinates, so a moving object doesn't re-trigger it. A fact is shown once per level, or again once the message that carried it has been trimmed from history.
- The line is capped at 4 facts and 100 characters; everything resets on a level change, and any exception adds nothing.
- Rule 2 (prefer known controls) is left to memory-v2's action-effect table, and rule 3's far-from-activity colour changes to the indicator-notes arm. Neither is duplicated; the docs say so.

**What the model sees.** The system prompt gains this section (120 tokens, down from about 300 for the old advice), between the action meanings and the Python guidance:
```
Salient facts:
- A user turn can include a line starting `Salient:` that the harness computes from the current board, ignoring cells within 4 cells of the edge: objects that appeared during this level, colors first seen this game, objects unlike their same-color or same-shape peers or off the row or column those peers share, and colors that cover few cells.
- Positions are the (row,col) of an object's top-left cell and sizes are height x width; a fact is listed once per level unless the line that listed it has left the conversation.
```
Three example lines from recorded games:
```
Salient: appeared S 5x10 at (9,29).
Salient: g 6x6 at (54,52) unlike 5 g 6x6 in shape; rare colors S (5,5).
Salient: O first seen this game at (24,30) (40,30); rare colors g (22,30) (26,30) +3.
```

**Validation** (replayed over recorded `fz-ledger-on-trim-r2` boards, one niced process, about 1 s of CPU per 5-game replay)

| Set | Games | Turns | Lines per 100 turns | Tokens per line (median / p90 / max) |
|---|---|---|---|---|
| Held out (never used for tuning) | ar25, cn04, lp85, re86, su15 | 281 | 20.3 | 31 / 54 / 59 |
| Tuning, r2 | ft09, tu93, m0r0, dc22, sp80 | 349 | 14.0 | 26 / 50 / 53 |
| Tuning, v4-stack-v1 | ls20, vc33, sc25 | 299 | 11.4 | 36.5 / 58 / 62 |

- On the held-out set the line fired on 17/58, 6/38, 8/44, 14/53 and 12/88 turns (ar25, cn04, lp85, re86, su15), mean 32.5 tokens. "rare colors" appeared in 24 lines, "appeared" in 20, "unlike" in 17, "first seen" in 7, and "off the rows/columns" in none.
- With `ARC3_ATTENTION_AFTER_TOKENS=15000` the held-out rate drops to 11.4 per 100 turns.
- Cost is about 2.5 ms of CPU per turn and roughly 7 uncached tail tokens per turn on average.
- My first version fired 18 times per 100 turns at up to 125 tokens per line. Tuning removed most of that: requiring three agreeing peers, a 4x size-ratio limit, alignment computed across colours, excluding objects cut by the edge band, and not counting in-place recolours as "appeared".

**Five real lines, checked against board crops** (turn = opener number):
1. m0r0 L1, turn 9: "appeared S 5x10 at (9,29)". A sky-blue block appeared in a gap that was black the turn before.
2. sp80 L4, turn 29: "appeared b 3x9 at (44,32) (44,44)". Two blue bars appeared under the red bar, with a charcoal 3x3 between them.
3. re86 L4, turn 17: "g 6x6 at (54,52) unlike 5 g 6x6 in shape". Of six gray frames, this is the one the green diagonal line enters, which breaks its border.
4. su15 L2, turn 9: "appeared M 2x2 at (36,17)". A magenta 2x2 now sits where a single sky-blue cell was.
5. lp85 L1, turn 1: "Y 4x4 at (19,42) unlike 4 Y 2x2; rare colors R (29,4), N (29,56)". Only a partial hit: the four yellow 2x2 are corner markers around the selected tile, but the line names the yellow tile as the odd one. R and N are the pieces at the left and right edges.

**Tests** (targeted only, niced, imports confirmed to load from the worktree): 34 passed. That covers `tests/test_salient_facts.py` (14), `tests/test_salient_turn_line.py` (8), `tests/test_attention_guidance.py` (6) and `tests/test_ledger_on_trim.py` (6). The turn-line tests check that flag unset and flag `0` give identical openers and messages (images included), that with the flag on the only change is one `Salient:` line in the new message, plus the hold, re-emission after a trim, short resumptions and the failure path. Separately, I dumped the system prompt and three turn messages (including a level change) from this worktree and from the stack-v2 worktree; with flags off they match byte for byte under both the default and competition flag sets. `kaggle/build_notebook.py --arm attention-prompt` builds (local, gitignored), and its patch includes `salient_facts.py`. I did not run the full suite, as instructed.

**Limitations**
- There's no score evidence. The arms still need Kaggle runs against `control` and `stack-v2`, compared with `make significance`.
- The replay doesn't simulate history trims, so re-sends after a trim aren't counted (at most one extra line per trim per level).
- Tile and lattice games produce state-driven noise: lp85's odd-one-out names the tile instead of the cursor markers, ft09's odd colours flip as tiles are clicked, ar25 had three "appeared" facts in one line after a big layout change, and tu93's "off the rows of its 58 same-shape peers" was not checked.
- "rare colors" at each level start is often the player sprite, repeated every level by design.
- Only the board at each turn's start is seen, so objects that appear and vanish within a batch are missed, and a second copy of an object that already "appeared" isn't reported again.
- Playfields that reach into the 4-cell edge band lose those cells.
- No Fable code was ported, so no attribution was needed.
- I didn't edit `arms.md`. The backlog entry "Harness-flagged new elements" is now implemented by this arm, so it should move, and the PR #57 description is stale.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad`:
- Worktree: `wt/attention-prompt/ARC3-Inference/`
  - new: `inference/agent/salient_facts.py`, `tests/test_salient_facts.py`, `tests/test_salient_turn_line.py`
  - changed: `inference/agent/tool_agent.py`, `inference/agent/prompts.py`, `tests/test_attention_guidance.py`, `CONFIGURATION.md`
- Validation (`salient/`): `replay_salient.py`, `heldout_final.json`, `tune_r2_final.json`, `tune_v1_final.json`, `heldout_hold15k.json`, `identity_dump.py`, `crop.py`
