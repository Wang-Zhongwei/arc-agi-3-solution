# facts-arms-and-eviction-trap: integrate

Workflow run `wf_d828aa65-8a9`, phase Integrate. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

The four branches passed their full suites, built their notebooks and were pushed. They now have PRs: #57 revised, #59, #60 and #61 new. The three fact-line arms merged into `arm/facts-stack` with only trivial conflicts, and that has a draft PR (#62). No arm has run on Kaggle: we're out of compute, so all evidence is from replays of recorded runs.

**One rule broken:** I launched the first three full suites in a single batch, so they probably ran at the same time (about 20 s each, niced) instead of one after another. Everything after that ran one at a time.

**Full suites** (niced, imports confirmed to load from each worktree)

| Branch | Result |
|---|---|
| `arm/attention-prompt` | 73 passed |
| `arm/indicator-notes` | 67 passed |
| `arm/budget-alert` | 72 passed |
| `arm/eviction-guard` | 45 passed, 3 skipped (memory-v2 placement configs, which run once that arm lands) |
| `arm/facts-stack` | 151 passed |

**Notebook builds** (local only, never pushed to Kaggle)
- Built: `attention-prompt`, `stack-v2-attention`, `indicator-notes`, `stack-v2-indicator`, `budget-alert`, `stack-v2-budget`, `stack-v2` on eviction-guard (it has no new arms), and `facts-stack` (415 KB).
- In every notebook the embedded patch matches the branch's committed changes, holds the new code, and the env cell sets the arm's flags.
- Checker: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/integrate/check_nb.py`

**PRs and issue** (all against `arm/stack-v2`)
- #57 (attention-prompt), title and body rewritten to describe the change from advice to the `Salient:` facts line: https://github.com/adsdemaybe/daniel-franzen-sol/pull/57
- #59 indicator-notes: https://github.com/adsdemaybe/daniel-franzen-sol/pull/59
- #60 budget-alert: https://github.com/adsdemaybe/daniel-franzen-sol/pull/60
- #61 eviction guard, body says "Fixes #58": https://github.com/adsdemaybe/daniel-franzen-sol/pull/61
  - The memory-v2 trap issue already existed as #58, so I didn't open a new one: https://github.com/adsdemaybe/daniel-franzen-sol/issues/58
  - #58 won't close on its own when #61 merges, because the base isn't `main`.
  - #61 also carries d01f845, the one commit on `arm/memory-v2`, which has never been pushed.
- #62 facts-stack, draft: https://github.com/adsdemaybe/daniel-franzen-sol/pull/62

**README** (`docs/fork-readme`, commits 0db5211 and 35993e2, pushed)
- The `attention-prompt` row now describes the facts line instead of the advice section.
- New rows for `indicator-notes` (#59), `budget-alert` (#60) and `facts-stack` (#62, draft).
- The fixes table has a row for #58 pointing to #61.

**arms.md** (`/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md`), under "In progress"
- I revised the attention-prompt entry and kept a note of the replaced advice version.
- New entries for indicator-notes, budget-alert, facts-stack and the eviction guard, with hypothesis, flags, replay numbers, PR number and "Not run".
- The "Harness-flagged new elements" backlog idea now says the three arms and facts-stack cover it.
- Nothing was deleted. A backup is at `scratchpad/integrate/arms.md.bak`.

**Merge check**
- The three arm merges hit conflicts in `tool_agent.py`, `CONFIGURATION.md` and `kaggle/arms.json`. Every one was the two branches adding lines at the same spot, so I kept both sides and changed no arm code.
- I then added three commits on `arm/facts-stack`:
  - the `facts-stack` Kaggle arm: `ARC3_LEDGER_ON_TRIM=1`, `ARC3_ATTENTION_GUIDANCE=1`, `ARC3_INDICATOR_NOTES=1`, `ARC3_BUDGET_ALERT=both`;
  - `tests/test_fact_lines_together.py`, which checks that the lines appear in the order `Salient:`, `Far change:`, `Step bar`, that turning all three off leaves the message unchanged, and that the real trackers together add only fact lines;
  - a "Fact lines together" section in `CONFIGURATION.md`.
- Worktree: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/facts-stack`

Every commit has the required trailer. I didn't touch the main repo's working tree (still on `arm/animation-digest`).
