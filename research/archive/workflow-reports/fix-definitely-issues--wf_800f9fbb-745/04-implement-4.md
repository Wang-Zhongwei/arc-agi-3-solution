# fix-definitely-issues: implement:#4

Workflow run `wf_800f9fbb-745`, phase Implement. Fix DEFINITELY issues #5, #4, #3, #6 one at a time: implement in a worktree, Sonnet review, fix

---

Issue #4 is fixed on branch `arm/fix-4-tool-deadline`. The 30 s budget now counts only the snippet's own time, and a killed or crashed call keeps its printed output and reports how many actions already ran. The full test suite passes. I did not push or open a PR: the workflow rules say never push, so the PR still has to be opened from this branch.

**Worktree:** /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-4 (base `arm/kaggle-builder` @ 4d63980)

**git log --oneline 4d63980..HEAD**
```
4923b06 Test the timed-out python result ToolAgent shows the model (#4)
00d4cf7 Document the python tool deadline and partial output on timeout (#4)
3c70f93 Read the python tool's wall cap factor from ARC3_TOOL_WALL_CAP_FACTOR (#4)
3aa61ff Test that host action time does not use up the tool deadline (#4)
14108cc Charge only the snippet's own time against the python tool deadline (#4)
da60ad6 Test sandbox output that survives a timeout or crash (#4)
dcdae01 Keep sandbox print output and the action count when a snippet is killed (#4)
```

**pytest:** `10 passed in 12.45s` (imports resolve to the worktree). Before the fix, the three tests for partial output after a timeout or crash failed. The two tests for normal and error results passed both before and after the fix, so those results are unchanged.

**What changed (python_tool_sandbox.py, tool_agent.py, CONFIGURATION.md):**
1. **Deadline:** time the host spends serving `action()` and animation requests (from the request arriving to the reply being written) now pushes the 30 s deadline back. A separate total cap of 3 × the timeout (90 s by default) still ends a runaway action loop. `ARC3_TOOL_WALL_CAP_FACTOR` changes the multiple; setting it to 1 brings back the old 30 s wall-clock limit.
2. **Output survives a kill:** the child sends its new print output to the host as `{"type":"stdout"}` JSON lines. It sends them at every `action()` and animation request, and a small background thread sends the rest every 0.25 s.
   - If the child never sends its final message (timeout, crash, bad or unknown reply), the host returns the output received so far as `stdout`.
   - The result keeps the same shape and still has no `keepable_functions`, so it stays compatible with fix-5's retention check.
   - A finished call still returns the child's full output buffer, exactly as before.
3. **Small robustness change:** a reply that is valid JSON but not an object is now reported as "invalid response". Before, it raised an exception in the host. The action and animation serving code moved into two local helpers so both are timed in one place; what they do is unchanged.

**What changes for the model:**
- **Timeout or crash results** now include the `stdout` printed before the stop. When actions had run, the error gets a count added, for example:
  - `Tool timed out after 30s (37 actions already executed)`
  - `Sandbox process exited unexpectedly (1 action already executed).`
  - With no actions executed, the error text is unchanged (fix-5's test still expects that).
- **New error when the 90 s cap stops a call:** `Tool timed out after 90s in total, Ns of it executing actions`.
- **Snippets that used to die on host time** now finish, so the model sees their normal output instead of a timeout.
- **System prompt:** unchanged. It still says "hard time limit of 30 seconds", which is now the conservative reading.

**Still needs a Kaggle run to check:**
- Whether the timeout count falls from about 7 per 25-game run, by replaying g50t, tr87 or sk48, or a synthetic 60-action loop at N≈300.
- Whether long scripted sequences now use enough stream time to matter. A single call can now take up to 90 s instead of 30 s, which also means a runaway loop can spend up to about 3× more actions before it is stopped.
- How much of the budget the child still uses decoding the full state it receives after every action at large N. That decode still counts against the 30 s; I only checked it with tiny states, not real N=300 games.
- That the timed-out results in the logs now show output and action counts.
