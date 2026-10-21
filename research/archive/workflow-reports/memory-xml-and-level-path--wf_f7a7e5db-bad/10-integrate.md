# memory-xml-and-level-path: integrate

Workflow run `wf_f7a7e5db-bad`, phase Integrate. Research, design, build, review and integrate two arms on memory-v2: memory writes in the Qwen3 XML tool-call format with a memory addendum and revision log, and a shortest-known-path tracker shown after every action

---

All three arms are built, tested, pushed and have PRs: memory-xml and level-path are ready to review, and memory-v3 (both features together) is a draft. One test needed a looser setting to pass: the #58 guard with both flags on, which you should check (point 1 below). Nothing has run on Kaggle (out of compute).

**Three things to decide**
1. **The #58 guard with both flags on only passes at a larger context window.** With both flags on, the system prompt plus tool schema are 991 sim tokens longer than with memory-v2 alone. In the test's 14000-token window, a trim then leaves less than one turn of history, so the next request trims again. That fails the "no two consecutive full re-prefills" check at requests 7 and 35. The other three checks pass at 14000. I set the combined config (`mem-v3`) to 15000, which gives it the same history room as plain memory-v2. On Kaggle (128k window, 58k drain) about 1k extra tokens can't cause this. Say if you'd rather handle it another way; the reasoning is in the commit, the config comment and the PR.
2. **The memory-v2 base has one unpushed commit.** cf40ac1 (the `panel-memory-v2-both` smoke arm) is on local `arm/memory-v2` but not in PR #64. Until it's pushed, all three new PRs also show that one `kaggle/arms.json` hunk. I didn't push it because that branch was read-only for me; each PR body says so.
3. **A throughput commit is still unpushed in the fork README worktree.** Another agent's commit 4682b91 ("README: add the throughput-stack arms…", marked "PR pending") was sitting there unpushed. I pushed only my two README commits and left it unpushed, rebased on top of mine.

**Full test suites** (one at a time, niced; imports load from each worktree)
| Branch | Tests | Time | Notebook built locally |
|---|---|---|---|
| `arm/memory-xml` | 227 passed | 33 s | `v4-memory-xml`, 451 KB |
| `arm/level-path` | 159 passed | 38 s | `v4-level-path`, 439 KB |
| `arm/memory-v3` | 289 passed, 0 skipped | 41 s | `v4-memory-v3`, 492 KB |

For each notebook, the embedded patch equals the branch's `git diff bundle-base..HEAD`, it contains the arm's code, and the env cell sets every flag. Nothing was pushed to Kaggle.

**Conflicts in the memory-v3 merge** (memory-xml merged cleanly first; level-path conflicted). Each kept both sides:
- **`tool_agent.py`:**
  - The prompt builder passes both features' settings.
  - The tool description gets the `level_path` sentence, then the `memory_items` sentence. This is now built in one function, a separate refactor commit with the same bytes for every flag combination.
  - The "Only tool:" line lists `level_path`, `level_path_len`, `memory_items`, `memory_log`.
- **`prompts.py`:** `memory_addendum` takes both sets of arguments.
- **`python_tool_sandbox.py`:** the sandbox reads both `memory` and `level_path`.
- **`memory_sim.py`:** the two branches hashed the tool schema differently. I kept one form: one hash per distinct schema, in send order. The new-flags list now has all four new flags.
- **`memory_v2_on_golden.json`** (added on both branches): regenerated from d252fa3 with the merged `memory_sim.py`. Every hash except the tool-schema one equals both branches' versions.
- **Tests, eviction configs, CONFIGURATION.md, README.md, kaggle/README.md:** both sides kept.
- **memory_ledger.py** merged without conflict and was checked by hand.

**Added on memory-v3**
- A golden recorded on each arm's own branch (fedada6, a438802). With one flag on, the merge must match it byte for byte in both placements; it does.
- With both flags on, the output must differ from each single flag, and the tool schema must stay the same on every request.
- The `mem-v3` #58 eviction config.
- Advice-word checks on the combined text.
- The `memory-v3` Kaggle arm (`memory-v2-both` + `ARC3_MEM_XML=1` + `ARC3_MEM_PATH=1`).
- **Cost over `memory-v2-both`:** about 840 cached tokens (2,229 chars of system prompt, 723 of tool schema), plus the per-turn route line, route note on each action and memory field.

**PRs**
- memory-xml: https://github.com/adsdemaybe/daniel-franzen-sol/pull/65 (base `arm/memory-v2`)
- level-path: https://github.com/adsdemaybe/daniel-franzen-sol/pull/66 (base `arm/memory-v2`)
- memory-v3 (draft, lists #65 and #66): https://github.com/adsdemaybe/daniel-franzen-sol/pull/67 (base `arm/memory-v2`)

Each PR body has the quoted request, the hypothesis, examples of what the model sees, the tests, the validation numbers, "Not run: out of Kaggle compute" and a keep/drop rule. The memory-v3 rule: keep if the two-run mean is at least `memory-v2-both` + 3 and not below the better single arm; drop if it is at most `memory-v2-both` + 1, or 2 or more below the better single arm.

**Docs**
- **memory-v3 branch:** CONFIGURATION.md section 10 has a new memory-v3 part; README.md and kaggle/README.md list the arms, with PR numbers in the README rows.
- **`docs/fork-readme`:** pushed a50dbc1 (rows for memory-xml #65, level-path #66, memory-v3 #67 draft). It broke the table with `|` characters inside one cell; 77c4b1a fixes that.
- **`arms.md`:**
  - Backed up first to `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/arms.md.bak-memv3`.
  - The memory-xml and level-path entries were already under "In progress". I added their PR numbers and an integration line each.
  - Added a new memory-v3 entry. Nothing was deleted.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/`:
- `wt/memory-v3/`
- `integrate/memv3/suite_*.txt`
- `integrate/memv3/pr/*.md`
- `integrate/memv3/gen_golden.py`
