# fix-definitely-issues: implement:#5

Workflow run `wf_800f9fbb-745`, phase Implement. Fix DEFINITELY issues #5, #4, #3, #6 one at a time: implement in a worktree, Sonnet review, fix

---

Issue #5 is fixed on branch `arm/fix-5-retained-functions`: a sandbox call that never answers no longer wipes the retained-function library. All 8 tests pass. Nothing was pushed, and the fix still needs a Kaggle run.

**Branch:** `arm/fix-5-retained-functions`, cut from `arm/kaggle-builder`
**Worktree:** `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-5`

**`git log --oneline arm/kaggle-builder..HEAD`:**
```
f277419 Document that unanswered sandbox calls keep the pool (#5)
97dbe8a Test that unreported sandbox results keep retained functions (#5)
cf1789e Keep retained functions when the sandbox never reports (#5)
```

**pytest:** `8 passed in 1.19s`. Imports load from the worktree (`.../wt/fix-5/ARC3-Inference/inference/__init__.py`). With the fix reverted, 5 of the new tests fail; the 3 tests covering unchanged behaviour still pass.

**The fix** (`ARC3-Inference/inference/agent/tool_agent.py::_record_retained_functions`): if the result has no `keepable_functions` key, the function now returns before touching anything:
```python
if "keepable_functions" not in sandbox_result:
    return
```
That case covers a timeout kill, a crash or early exit, invalid JSON, an unknown message type, and a failed start. The previous pool is kept and no retention notes are added. Normal final and error results always carry the key, even as an empty list, so they work exactly as before.

**Tests** (`ARC3-Inference/tests/test_retained_functions_survive_failures.py`, a new file because this base branch had no `tests/` directory). They go through `_run_python_tool` end to end:
- **Cases that now keep the pool:**
  - a real 1 s timeout;
  - stand-in sandbox children that crash, send invalid JSON, or send an unknown message type;
  - a failed process start.

  Each checks that the pool is unchanged, the tool result has no `function_retention`, and the next prompt still lists `shift(x)` and `double(x)`.
- **Cases that behave as before:**
  - a reported error still adds the new helper;
  - a reported empty list still clears the pool with "no longer retained" notes;
  - a snippet that deletes helpers still clears them with "was not retained" notes.

**Docs:** one sentence added to section 7 of `ARC3-Inference/CONFIGURATION.md`.

**What changes for the model:** this only applies with `ARC3_PERSISTENT_FUNCTIONS=1` and only after a call the sandbox never answers.
- The tool result no longer includes the "Your previous function X is no longer retained; redefine it if needed." lines.
- The next prompt keeps listing the earlier retained functions instead of an empty or shrunken list.
- Functions defined inside the failed snippet itself are still lost, and nothing tells the model so.
- The error text is unchanged. Every other path is unchanged.

**Not done:** having the sandbox child report its retained functions before the kill. The host kills the whole process group on timeout, so this would need a child-side soft deadline. That isn't simple, so I left it out.

**Still needs a Kaggle run to confirm:** run with `ARC3_PERSISTENT_FUNCTIONS=1` and `SCOPE=game`, and check that:
- after each "Tool timed out after 30s" the next prompt's retained-function list doesn't shrink;
- transcripts no longer show the model remarking that its functions were dropped, re-emitting helpers, or hitting `NameError`;
- token use per run drops slightly (the issue estimates about 20–30k).

Also confirm that no game gets stuck in repeated timeouts because a kept helper now survives them. Restoring the pool only re-runs `def` statements with literal defaults, so I expect this to be unlikely.
