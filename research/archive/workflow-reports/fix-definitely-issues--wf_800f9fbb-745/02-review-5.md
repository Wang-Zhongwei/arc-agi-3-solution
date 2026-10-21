# fix-definitely-issues: review:#5

Workflow run `wf_800f9fbb-745`, phase Review. Fix DEFINITELY issues #5, #4, #3, #6 one at a time: implement in a worktree, Sonnet review, fix

---

No defects. The fix addresses issue #5's root cause, and I found nothing in the diff that breaks existing behaviour or needs changing before merge.

**Fix**
- `ARC3-Inference/inference/agent/tool_agent.py:4617-4623` returns early when `keepable_functions` is absent from the result. Only results the host synthesizes lack that key: timeout, EOF/crash, invalid JSON, unknown message type, failed `Popen`.
- The `final`/`error` branch at `python_tool_sandbox.py:964` always adds the key, even as `[]`. So reported results, including the child's own "retention failed, cleared" result, are unchanged.
- `_run_python_tool` calls `_record_retained_functions` at `tool_agent.py:5920`, before the level-transition clear. With `levelup` scope, the early return followed by the clear is the same as before.
- Persistence-off still returns first, so nothing changes there.
- The new check sits outside the `try`, but `in` on a dict cannot raise.

**Tests**
- All 8 pass, and they also pass with `REPAIR_HINTS`, `IMPORTS` and `ANIMATION` set.
- Three more runs were stable.
- I reverted the fix in a scratch copy and exactly 5 tests failed (timeout, 3 never-reporting children, failed start). The 3 behaviour-preserving tests still passed.
- The tests use the real `run_sandboxed_python` and a real 1s timeout, so they are meaningful.

**End-to-end probe** (real sandbox, `SCOPE=game`)
- The model defines `shift`, then a call times out, then `shift(1)` runs from the restored pool and returns 2.
- The retained-functions context still lists `shift(x)`, and no "no longer retained" notes appear.

**Other checks**
- Process-group kill, queue handling and Popen failure paths are unchanged.
- No other code consumes `function_retention` or the removed note text.
- The new test file only adds a few KB to the arm-builder patch, far below the Kaggle size limit.
- `PERSISTENT_FUNCTIONS_LINE` in `prompts.py` says tool results report which functions were retained. That stays true for reported results, and a timed-out call reports nothing, as specified.

**Low severity, optional**
1. **Redefinitions are reverted silently.** I ran a snippet that redefined an existing helper and then timed out. The next call used the old helper, with no signal to the model. Before the fix this was loud, because everything was wiped with notes.
   - **Where:** `CONFIGURATION.md` section 7, the new sentence ("only the definitions that call introduced are lost") understates this.
   - **Fix:** change the wording to "definitions or redefinitions made by that call are lost". This is a documentation nit only, because the spec asks for no notes.
2. **Pre-existing and not a regression: a slow retained helper now survives a timeout.** The model must redefine it, since the pool is no longer wiped. This is the trade-off the issue accepts, and the implementer already flagged it for the Kaggle run.

Worktree: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-5`. It is clean; I made no edits there.
