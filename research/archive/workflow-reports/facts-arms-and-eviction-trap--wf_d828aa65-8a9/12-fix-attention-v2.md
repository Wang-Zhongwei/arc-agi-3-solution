# facts-arms-and-eviction-trap: fix:attention-v2

Workflow run `wf_d828aa65-8a9`, phase Fix. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

All six review findings are dealt with: 8 fixed and 1 rejected, in 11 new commits on `arm/attention-prompt`. With the flags off, requests are still byte-identical to `arm/stack-v2`, and a new test now pins that. The 47 targeted tests pass and nothing was pushed. No model has run on this yet: the numbers below come from replaying recorded boards.

**Per finding**
1. **Objects sliding in from the edge band reported as "appeared": fixed** (`5321354`, sped up in `46917be`).
   - A colour+shape now counts as appeared only if more copies arrived in the interior than left the band since the last board. I used this count instead of the reviewer's "band count went down" check, so a genuinely new copy that arrives at the same moment is still reported.
   - The edge band is only segmented on turns where something arrives, so CPU stays at about 2.2–3.3 ms per turn.
   - I looked at the two ar25 cases the reviewer flagged. In both, the band copies moved in, so hiding them is right:
     - L4 `c 3x3 (24,18)`: two c 3x3 blocks left rows 0–2 in the same step the mirror bar moved.
     - L7 `c 12x6 (48,45)`: the identical shape at (48,0), inside the band, disappeared in the same step and on the same row.
   - It also removed 4 lines on the v4-stack-v1 games, all checked against the boards: ls20's key-glyph change in the bottom HUD, and vc33 pieces sliding in from column 3.
2. **System prompt described positions wrongly: fixed** (`b7f5d33`). It now says positions are "the top-left corner of an object's bounding box". `CONFIGURATION.md` also says that corner may not be a cell of the object.
3. **Facts cut by the cap still counted as shown: fixed** (`3bc252c`, `d12cff9`).
   - `render()` no longer hides a colour fact because of an object fact the cap then dropped. A first-seen colour that an odd-one fact also names is now considered just after the odd-one facts.
   - I kept the policy that a fact cut by the cap counts as shown, so a full line doesn't spill into the next turns. The wording now matches it: the prompt says "at most once per level", and the `CONFIGURATION.md` sentence about the first line after the hold was corrected.
4. **"Off the rows" firing on regular lattices: fixed** (`8f8cb90`). The lone object must now be off its peers' spacing (the gcd of the gaps between their rows or columns).
   - This removed 4 tuning lines (3 in tu93, 1 in ft09); each one was an ordinary lattice node.
   - The sp80 case, where all peers share one row, still fires.
5. **Tests couldn't catch drift from stack-v2: fixed** (`d1f952a`, `7312838`).
   - New `tests/test_flags_off_identity.py` hashes the system prompt and three turn messages (one a level change). It checks them under the default environment and under the full Kaggle notebook environment plus the stack-v2 flag.
   - The pinned hashes came from running the same code on the stack-v2 tree (d55df75) without writing any files; that tree is still clean. Turning the flag on changes the hash, so the test does catch differences.
   - Image data is replaced by a marker so the hash doesn't depend on the PNG encoder.
   - A reset-to-start unit test was added.
6. **Small issues:**
   - The tracker that never starts when no session folder is set: fixed (`c2a13fd`).
   - The edge-band size in the prompt now uses the same clamped value as the tracker (`343f171`). With the default of 4 the text is unchanged, and a band of 1 now reads "1 cell".
   - Mirror-image sprites: **rejected**. Over all 13 replayed games, none of the 83 "appeared" detections was a mirror flip. Treating mirror images as the same shape would merge mirrored peers in the odd-shape facts and hide new reflections in mirror games like ar25.
   - The review said the eviction item doesn't apply to this branch, so I left the GitHub issue alone.

**Final git log arm/stack-v2..HEAD**
```
5e5d2df Point the Salient docs at the flags-off identity test
46917be Segment the whole board for band copies only when an object arrives
7312838 Test that a reset to the level's start adds no appeared fact
d1f952a Pin the flags-off requests to arm/stack-v2's
343f171 State the same clamped edge band in the prompt and the tracker
c2a13fd Create the Salient tracker even when no runtime dir is set
8f8cb90 Skip a lone object that sits on its peers' lattice pitch
d12cff9 State that a fact the cap drops is not shown on that level
3bc252c Only drop a color fact for a fact that made the line
b7f5d33 Describe Salient positions as the bounding box's top-left corner
5321354 Net out objects that slide in from the edge band
f43ca80 Document the Salient facts line and ARC3_ATTENTION_AFTER_TOKENS
8bacd3d Replace the 'Where to look' advice with a description of the Salient line
1f2a1a5 Add the Salient facts line to each turn message
9c3a40b Add a tracker of salient board facts for the turn message
5c54594 Add attention-prompt and stack-v2-attention Kaggle arms
ed399dc Document ARC3_ATTENTION_GUIDANCE in the configuration guide
ed6695e Test that the attention section is opt-in and changes nothing else
1406ec5 Add an opt-in 'Where to look' section to the system prompt
```

**Targeted tests** (niced; imports confirmed to load from the worktree): `47 passed in 0.22s`. That covers `test_salient_facts.py`, `test_salient_turn_line.py`, `test_attention_guidance.py`, `test_ledger_on_trim.py` and `test_flags_off_identity.py`. I did not run the full suite. `kaggle/build_notebook.py --arm attention-prompt` still builds (local only), and its patch contains the fixes.

**PR summary**
- **Hypothesis:** a short, harness-computed line of board facts steers the model's attention without the cost that advice text showed in rellik13's runs. Rules 2 and 3b are left to the memory-v2 and indicator-notes arms.
- **Change:** `ARC3_ATTENTION_GUIDANCE` adds two static sentences to the system prompt and, on some turns, one `Salient:` line in the new user message only. The line lists objects that appeared this level, colours first seen this game, odd-one-out objects, objects off their peers' row or column, and rare colours.
- **Limits:** at most 4 facts and 100 characters, keys without coordinates, any exception adds nothing, and an optional hold via `ARC3_ATTENTION_AFTER_TOKENS`.
- **Review fixes:** objects sliding in from the edge band, the position wording, the cap and dedupe interaction, lattice nodes, the tracker start, and the band value in the prompt.
- **Tests:** 47 targeted tests, including pinned hashes of the flags-off requests against arm/stack-v2 under the default and Kaggle environments.
- **Replay results:**

| Set | Lines per 100 turns | Tokens per line (median / p90 / max) |
|---|---|---|
| Held out (281 turns) | 20.3 | 31 / 52 / 59 |
| Tuning, r2 | 13.5 (was 14.0) | |
| Tuning, v4-stack-v1 | 10.0 (was 11.4) | |
| Held out, 15k-token hold | 11.4 | |

  - On the held-out set, 3 ar25 lines changed. The wrongly reported band entries are gone, and two genuine facts the cap used to cut now show: `c 9x3 (48,42)` and `c 3x3 (57,21)`.
- **Cost:** about 2–3 ms of CPU per turn and about 7 uncached tail tokens per turn on average.
- **Still needed:** Kaggle runs against `control` and `stack-v2`, compared with `make significance`.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad`:
- Worktree: `wt/attention-prompt/ARC3-Inference/`
  - new: `tests/test_flags_off_identity.py`
  - changed: `inference/agent/salient_facts.py`, `inference/agent/tool_agent.py`, `inference/agent/prompts.py`, `CONFIGURATION.md`, `tests/test_salient_facts.py`, `tests/test_salient_turn_line.py`, `tests/test_attention_guidance.py`
- Replays and scripts (`fix-attention/`): `heldout_fix.json`, `tune_r2_fix.json`, `tune_v1_fix.json`, `heldout_hold15k_fix.json`, `golden.py`, `show.py`, `offline.py`
