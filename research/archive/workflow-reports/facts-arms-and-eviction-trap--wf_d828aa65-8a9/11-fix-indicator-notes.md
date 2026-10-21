# facts-arms-and-eviction-trap: fix:indicator-notes

Workflow run `wf_d828aa65-8a9`, phase Fix. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

All 7 review findings were real and reproduced with the review's probes. I fixed 6 fully. Of defect 3 I fixed the bar half and documented the counter half, because replacing the move test does not work. Nine commits sit on `arm/indicator-notes` (worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/indicator-notes`). Nothing is pushed and no PR is open.

**Per finding**
1. **False "first seen" baseline: fixed** (`4ed8979`, `ef51c63`).
   - Every reference now points to content that was recorded:
     - "back to its level-start content", meaning the level's first frame, or the last RESET frame;
     - "back to its content before step N" (N is the step that found the patch);
     - "the same change as after step K", when the before and after both match;
     - "back to its content after step K".
   - dc22 step 203 now reads "back to their content before step 202".
   - probe2 now reads correctly: step 4 says "back to its level-start content" and step 5 says "back to its content before step 4".
   - New tests cover the dc22-shaped case, probe2, and a RESET that changes the level start.
2. **Labels that claim a meaning: fixed** (`4ed8979`).
   - Lines now start with `Far change:`. "Indicator", "LOST" and "AGAIN" are gone, and so is "main change".
   - An UNDO revert is now described as a change, not judged as a loss (test added).
   - Docs updated.
3. **HUD counters: partly fixed.**
   - Bar half fixed (`fde719c`): `_bar_like` no longer counts runs of the background colour, so the pip flush with col 63 is now reported.
   - This also removed a false positive: on dc22 step 196 an orange bar moved 10 cells left, and its far half had been reported twice.
   - The background colour is sampled from every 4th cell, so the cost stays about 55 µs per action.
   - Counter half rejected, with the limit documented instead (`7271880`). The suggested shift test also misreads a counter: shifting the "2" down two rows reproduces the 3's changed cells (checked in code). The 1→2 probe was also caused by defect 4, not by the move test. "counter" is removed from the docs.
4. **"Main change" and patch swapping roles: fixed by the wording** (`4ed8979`). Lines now say "the largest change in that step", which is how the cluster is actually picked. I kept size-only selection and documented the role swap in the docstring and `CONFIGURATION.md`. A probe-based test was added.
5. **Weak byte-identity test: fixed** (`3c6805c`).
   - A new test builds the full `_build_user_message`, including the image part, under the template flags. With the flag on and the gate closed it is byte-identical to the flag off. With the gate open, only the text part differs, by the one line.
   - I also re-ran `bytes_check.py` against `arm/stack-v2` (d55df75): with the flag off, the output is identical.
6. **Rollback could bring back gate-dropped events: fixed** (`383c216`). The taken list is now cleared on the early return (test added).
7. **`MAX_LINES=0` gave one line: fixed** (`e097e91`). 0 or less now shows nothing, and the docs say the count line comes on top of the cap (test added).

I found one more problem with the reference checker I wrote: 5 of 52 events had cells change to two colours, but the line named only one. They now name both pairs ("charcoal->gray and charcoal->light gray"). "It had changed to X" only names X when that change produced a single colour (`1420240`).

The review had no eviction item, so I left no GitHub issue comment.

**git log arm/stack-v2..HEAD**
```
9c6dbf5 Refresh the far-change replay numbers in CONFIGURATION.md
3c6805c Test the full far-change opener, image parts included
7271880 Document what the far-change detector cannot report
fde719c Ignore background runs when looking for timer-bar ticks
1420240 Name both colour pairs of a mixed far change
ef51c63 Tie far-change returns to the level-start board
4ed8979 Word indicator lines as far changes with recorded references
e097e91 Treat ARC3_INDICATOR_MAX_LINES=0 as no indicator lines
383c216 Keep gate-dropped indicator events out of a rollback
59cfc12 Add the indicator-notes and stack-v2-indicator Kaggle arms
9460aab Document the indicator-notes flags
f032b9c Show indicator notes in the turn message behind ARC3_INDICATOR_NOTES
848430f Test the indicator detector on synthetic frames
03a5b0a Add an indicator detector ported from the Fable layer
```

**Targeted pytest** (niced; imports resolve to the worktree): `35 passed in 0.10s` for `tests/test_indicator_notes.py`. I did not run the full suite.

**PR summary**
- **Hypothesis:** a short fact line about a small patch changing far from an action's main effect (often a HUD mark, like sk48's tray) gives the model evidence it would otherwise miss. It is ported from leoprovorov's Fable layer (Apache 2.0), which was never scored.
- **Change:** `ARC3_INDICATOR_NOTES` (off by default) adds at most 3 lines plus a count line, starting `Far change:`, right after the valid-actions line of the new turn message. They only appear once the level has used 15k generated tokens.
- **Wording:** facts only, with no advice and no judgement such as "indicator" or "lost". Every reference (level start, before step N, same change as step K) is computed from recorded frames. Any error shows nothing.
- **Detector fixes over Fable:** 9 from the port, plus 3 from review (level-start baseline, background-run bars, mixed colours). Limits are documented: a digit glyph change reads as a move, "largest" is chosen by size only, and a change on its own is not reported.
- **Tests:** 35 targeted tests, including the off/on comparison of the system prompt and the full opener with its image. With the flag off the output is byte-identical to `arm/stack-v2`, checked separately with `bytes_check.py`.
- **Validation:** replay of 5 games of `fz-ledger-on-trim-r2` (1447 actions, 297 turns):
  - 30 lines on 6.7% of turns (before the review fixes: 32 on 7.4%), about 230 characters per such turn, about 55 µs of CPU per action.
  - All 50 events the tracker emitted were checked against the frames (colours, plus 8 level-start, 7 same-change, 3 before-step, 10 "had changed to" and 3 also-changed claims) and all hold.
- **Caveat:** only sk48's hits are HUD marks; the others are true far state changes, such as dc22's button-toggled bar caps and wa30's block borders. Whether they help needs the Kaggle arms `indicator-notes` and `stack-v2-indicator`.
- **For the integrate step:** the fork-README row should now read: `indicator-notes` | "`Far change:` fact lines when a small patch far from the action's largest change flips, tied to level-start / earlier recorded content; `stack-v2-indicator` stacks it on stack-v2" | "Replay: 6.7% of turns, sk48 tray caught, 50/50 events verified against frames".

Validation scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/indicator-val/`:
- `verify_refs.py`
- `bench.py`
- `list_events.py`
- `replay_after_review_fix.json`
- `replay_before_review_fix.json`
- `bytes_off_v2.json`
- `bytes_stackv2_v2.json`
- `bytes_on_v2.json`
