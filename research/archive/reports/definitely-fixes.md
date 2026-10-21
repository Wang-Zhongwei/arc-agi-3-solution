## #5 (arm/fix-5-retained-functions)
### Implementation
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
### Review
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
### Fixes
Branch: arm/fix-5-retained-functions
Worktree: /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-5 (clean, not pushed)

**Review findings**

1. Redefinitions are reverted silently, and the documentation understated this. **Fixed** in 5406c50. I confirmed it in the code. The pool (`self._kept_functions`, a mapping from name to source) changes only when the child reports a result. If a call redefines a helper and then times out, the redefinition is never recorded, and the next call restores the old version with no note. That matches the issue's "emit no notes" spec, so the behaviour stays. `CONFIGURATION.md` section 7 now says: "any definitions or redefinitions made by that call are lost, so a function it redefined silently reverts to its earlier retained version". The comment on the early return in `ARC3-Inference/inference/agent/tool_agent.py` says the same. Only text changed, no code.
2. A slow retained helper now survives a timeout. **Rejected, no change.** This is the trade-off the issue accepts by asking to keep the earlier pool unchanged. Clearing the pool on timeout would bring the bug back. The model still sees the timeout error and can redefine the helper. It stays on the list of things to check in a Kaggle run.

**git log arm/kaggle-builder..HEAD --oneline**
```
5406c50 Note that unanswered calls also lose redefinitions (#5)
f277419 Document that unanswered sandbox calls keep the pool (#5)
97dbe8a Test that unreported sandbox results keep retained functions (#5)
cf1789e Keep retained functions when the sandbox never reports (#5)
```

**pytest:** `8 passed in 1.19s`. Imports resolve to the worktree: `.../wt/fix-5/ARC3-Inference/inference/__init__.py`.

**PR summary**
- Fixes #5: a sandbox timeout, crash, invalid or unknown reply, or failed start wiped the whole retained-function pool and added a "no longer retained" note for every function.
- `_record_retained_functions` now returns early when the result has no `keepable_functions`. Only results the host builds itself lack that key; every result the child reports includes it, even as an empty list.
- What the model sees: after a call the sandbox never answers, the previous functions stay available and listed, with no wipe notes. Definitions or redefinitions from that call are lost, and a redefined function silently goes back to its earlier version. All other paths are unchanged.
- Eight tests use the real sandbox with a real 1s timeout. They cover timeout, three kinds of child that never reports, failed start, and three paths whose behaviour should not change. `CONFIGURATION.md` describes the new behaviour.
- Still needs a Kaggle run (competition config, `SCOPE=game`): check that the retained list stays the same size after each timeout (about 7 per run), that no "functions were dropped" re-emits appear, and that slow retained helpers don't cause repeated timeouts.

## #4 (arm/fix-4-tool-deadline)
### Implementation
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
### Review
Note: claude-sonnet-5[1m] (the safety classifier) was unavailable (timed out) when reviewing this subagent's work. Please carefully verify the subagent's actions and output before acting on them.

**Verdict: the root cause is fixed and the logic is correct, but there is one medium defect (unbounded host memory on a print flood) and one low-medium gap (output lost when the child holds the GIL). Ranked below.**

**What I ran**
- The suite gives `10 passed` on the branch. The 8-test deadline file passed 8 times in parallel, so I found no timing flakiness.
- Against base (`arm/kaggle-builder`), 6 of the 8 new deadline tests fail. The two "result unchanged" tests pass on both, so they do guard unchanged behaviour.
- `git merge-tree` with `arm/fix-5-retained-functions` merges clean, and the merged tree gives 18 passed. That means fix-5's check on the missing `keepable_functions` key still holds.
- I traced the deadline arithmetic: extending by served time, the 3x wall cap, and the timeout-versus-cap message choice. The branch on `now >= deadline` picks the right message in every ordering I could construct.
- Forwarded stdout stays a contiguous, in-order prefix of the real output. I checked this with 3.1M lines under a kill.
- The action and animation serving code was moved into helpers with no behaviour change. A non-dict JSON line is now a clean "invalid response".
- The normal and error result shapes are unchanged.

**Defects, most severe first**

1. **MEDIUM: a print flood now blows up host memory** (`python_tool_sandbox.py:1117-1119`, plus `"".join` in `_unanswered_result` at 866-888).
   - `forwarded_stdout.append(text)` has no cap, and the child has no memory limit and no cap on what it forwards (`_ForwardedStdout`, 168-190).
   - A snippet like `while True: print("state dump " + "x"*80)` that gets killed was harmless on base.
   - On this branch it makes the host process grow by about 1.6 GB in 3 s through `ToolAgent._run_python_tool`, and 777 MB of stdout in 5 s through `run_sandboxed_python`. On base the same run grew the host by 0.1 MB and returned `stdout=''`.
   - That extrapolates to many GB at 30 s, in the process that also holds every game. The model only ever sees about 4k chars after `_trim_tool_text`.
   - Fix: bound the host side, for example keep the first and last ~256 KB of forwarded text and drop the middle. Also bound what the child queues for forwarding (stop forwarding past a few MB, keep the full buffer for the final message). A finished call's `final` payload stays as it is.

2. **LOW-MEDIUM: output is lost entirely when the child holds the GIL in one C call** (pump design at `python_tool_sandbox.py:158-216`; doc claim at `CONFIGURATION.md:163`).
   - The 0.25 s pump thread cannot run during a catastrophic `re.match`, a huge bigint `7**(10**9)`, a giant `sum(range(...))`, or a `list(itertools...)` consumed in C.
   - Repro: `print("trying regex"); re.match(r"(a+)+$", "a"*40+"b")` gives `stdout=''`, same as base. Printing after an `action()` and then hitting such a call also loses the post-action print.
   - Pure-Python loops are fine. The docs only admit to losing "the last ~0.25 s", which understates this.
   - Fix: have `_ForwardedStdout.write` forward synchronously from the snippet thread when at least ~50 ms have passed since the last forward (or under a burst limit), and keep the pump for the trailing text. Otherwise correct the doc sentence. Write overhead is acceptable either way: 1M prints cost +0.12 s.

3. **LOW (repo policy): the change is default-on with no env flag or arms.json entry.**
   - `kaggle/README.md:23-29` says every arm is flag-gated and off by default, with a flagless exception only for "same behaviour, only speed" changes (history-perf). This one changes timeout behaviour and what the model sees.
   - So a control built at this commit already differs from base. `ARC3_TOOL_WALL_CAP_FACTOR=1` is not a full off switch: stdout forwarding and the "(N actions already executed)" suffix stay on.
   - Fix: either add a flag and an arms.json arm, or add a sentence to the README listing this as a deliberate behaviour-changing exception.

4. **LOW (stacking note, not a bug in this branch): it conflicts with `arm/history-state-perf`.** There are two conflict hunks in `python_tool_sandbox.py` where history-perf rewrote the inline action/animation serving, and one in the `tool_agent.py` import line (69) plus CONFIGURATION.md. Whoever resolves it must port history-perf's change into `_serve_action`, not the old inline block.

**Not defects, noted**
- Printing a lone surrogate crashes the child on base and on this branch alike, so no regression.
- The system prompt (`prompts.py:326`, "hard time limit of 30 seconds") is unchanged by design and is now only approximately true. The new error text explains the 90 s case.
- The tests have no case for partial stdout on an invalid or unknown reply, and none for flood handling.
### Fixes
**Per-finding results** (branch `arm/fix-4-tool-deadline`, worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-4`, not pushed)

1. **MEDIUM print flood blows up host memory: FIXED** (17b1239)
   - Confirmed before the fix: a killed `while True: print(...)` with a 2 s timeout returned 256 MB of stdout and grew the host by 970 MB.
   - The child now forwards the first 64K characters. After that, each forward sends only the latest 64K it has, plus a `skipped` count.
   - The host (`_ForwardedOutput`) keeps the first 64K and the latest 64K, joined by `... [N chars omitted] ...`.
   - After the fix the same run returns 131K of stdout, the host grows by 0.4 MB, and the marker carries the real omitted count (about 520M characters).
   - A finished call's final message is unchanged.
   - `take_unsent` now takes only what is queued at that moment. This is a defensive bound, not a fix for an observed bug. I first suspected a livelock there, but that came from scripts that imported the main checkout's code instead of the worktree's.
2. **LOW-MEDIUM output lost when the child holds the GIL: FIXED** (6271c70), docs corrected in a25aa3a
   - Confirmed: `print("trying regex"); re.match(r"(a+)+$", "a"*40+"b")` returned `stdout=''`, and so did a print right after an `action()`.
   - A write that completes a line now forwards from the snippet's own thread, up to 5 lines per 0.25 s pump interval. This stops once the 64K head is used up.
   - Both repros now return the line. The new tests for them fail on the previous commit.
   - Cost: 500k prints take about +0.02 s.
   - Still lost: a partial line with no newline, or a burst of more than 5 lines per 0.25 s, just before a GIL-holding call. CONFIGURATION.md now says exactly this.
3. **LOW no flag: ADDRESSED with a README note, no flag added** (38df48d)
   - Reason: #4 is a confirmed bug, and the task rules say to prefer fixing over flags for clear bugs. The risky part, the longer wall time, already has the `ARC3_TOOL_WALL_CAP_FACTOR` setting.
   - `kaggle/README.md` now calls this a deliberate exception to the flag rule. It says `=1` restores only the old deadline, and to compare with the previous batch's control using `--new-batch`.
   - I added no arms.json entry: with no env to set, such an arm would be identical to the control at its commit, so the builder would refuse it as a duplicate.
4. **LOW stacking conflicts: FIXED**
   - ddc8eb8: the action and animation serving blocks are back inline, with host time measured from the loop top.
   - a25aa3a: the #4 docs moved to their own subsection at the end of section 6.
   - e837c57: `WALL_CAP_FACTOR` is now imported at its call site, so the shared top-level import line is unchanged.
   - `git merge-tree` is now clean against history-state-perf, fix-5-retained-functions, nvfp4, stack-v1 and serving-bench.
   - The merged tree with history-perf gives 54 passed, and history-perf's delta-encoding change lands correctly in the inline action block.

**`git log --oneline arm/kaggle-builder..HEAD`**
```
38df48d Note in the arm README that the #4 fix has no flag (#4)
e837c57 Import the wall cap factor where the python tool uses it (#4)
a25aa3a Document what a killed python call keeps, in its own section (#4)
d213cbf Test flood bounds, interpreter-holding calls and bad replies (#4)
6271c70 Forward a printed line at once, not only from the pump (#4)
17b1239 Bound the print output a killed python call forwards and keeps (#4)
ddc8eb8 Serve action and animation requests inline again (#4)
4923b06 Test the timed-out python result ToolAgent shows the model (#4)
00d4cf7 Document the python tool deadline and partial output on timeout (#4)
3c70f93 Read the python tool's wall cap factor from ARC3_TOOL_WALL_CAP_FACTOR (#4)
3aa61ff Test that host action time does not use up the tool deadline (#4)
14108cc Charge only the snippet's own time against the python tool deadline (#4)
da60ad6 Test sandbox output that survives a timeout or crash (#4)
dcdae01 Keep sandbox print output and the action count when a snippet is killed (#4)
```

**pytest:** `18 passed in 19.55s`. Imports resolve to the worktree.

**What changes for the model, compared with the reviewed version of this branch:**
- A killed call whose forwarded output passes 128K now shows head + `... [N chars omitted] ...` + tail. The `[truncated N chars]` count from tool-output truncation then covers only the kept text.
- A line printed just before a GIL-holding call now survives the kill.
- Finished calls are unchanged.

**Still needs a Kaggle run:**
- How often timeouts happen per run, compared with the previous batch's control (about 7 per 25-game run before).
- Stream idle time when the 90 s cap is hit.
- How often partial output or the omitted-chars marker reaches the model.
- Score and actions spent on games with long scripted loops (g50t, sk48, tr87, sc25).

**PR summary:**
1. The python tool's 30 s deadline now counts only the snippet's own time: time the harness spends serving `action()` and animation requests pushes it back, under a cap of `ARC3_TOOL_WALL_CAP_FACTOR` (default 3) times the timeout.
2. A killed or crashed call returns the print output forwarded so far and says how many actions had already executed.
3. Printed lines are forwarded at once (up to 5 per 0.25 s) and the rest every 0.25 s or at `action()`, so output before a GIL-holding call survives.
4. A killed print flood is bounded to the first and latest 64K characters, in both the child and the host.
5. The change is unflagged by design (the README notes it), and it merges cleanly with history-perf, fix-5, nvfp4, stack-v1 and serving-bench.

## #3 (arm/fix-3-prefix-reuse)
### Implementation
Issue #3 is fixed on the harness side, behind a default-off flag (`ARC3_VERBATIM_HISTORY`). The pytest suite passes (30 tests). Nothing is pushed and no PR is open, because the task rules say never to push. Whether this actually cuts the re-prefilled tokens is not measured yet: that needs the two new Kaggle arms.

**Branch:** `arm/fix-3-prefix-reuse`, worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-3`

**`git log --oneline arm/kaggle-builder..HEAD`**
```
84dce5b Store the rolling summary reply verbatim under the flag too (#3)
dd0ecf2 Add the prefix-reuse and prefix-reuse-60 arms (#3)
2de03b6 Document ARC3_VERBATIM_HISTORY and ARC3_PREFIX_REUSE_DIAG (#3)
edc38c8 Test the prefix-reuse probe against the served chat template (#3)
19c191b Log prefix reuse per request behind ARC3_PREFIX_REUSE_DIAG (#3)
dd17c6e Test that tool-call arguments in history equal what the model sent (#3)
76a87e5 Test that reasoning and content round-trip verbatim into history (#3)
429647a Keep reply text verbatim in history behind ARC3_VERBATIM_HISTORY (#3)
```

**pytest:** `30 passed in 0.16s`. Imports resolve to the worktree's `ARC3-Inference/inference/__init__.py`.

**What was done**
1. **Verbatim history (`ARC3_VERBATIM_HISTORY`, off by default).** With it on, history stores reasoning and content exactly as the server returned them, apart from `.strip()`. I checked the real `chat_template.jinja` from Qwen/Qwen3.8-Flash-Next: it applies `|trim` to both fields, so that strip changes nothing rendered. It covers every place history stores a reply:
   - tool-call turns;
   - replies with no tool call;
   - the rolling-summary reply.

   The transcript, logs, solver reasoning, knowledge parsing and all branch decisions still use the old cleaned-up text. The resume/yield branch only builds the user message that reopens a turn, so it needed no change. It is a flag rather than a plain fix because it changes what the model sees, and the control and the "fix only" arm need it off and on at the same commit.
2. **Tool-call arguments.** I checked them and found no mismatch, so there was nothing to fix. `_run_python_tool` rstrips a parsed copy of the code it runs, never the stored string, so history keeps the arguments exactly as the server returned them. The server's parser removes one newline on each side and the template adds them back, so they re-render byte-identically. A test now pins this, with and without the flag. One known exception, not changed: tool calls recovered from markup inside the text can never re-render identically.
3. **Diagnostic (`ARC3_PREFIX_REUSE_DIAG`, off by default, logging only).** The new module is `ARC3-Inference/inference/agent/prefix_reuse.py`. Each request logs one `PREFIXDIAG` line showing:
   - whether the previous reply's stored message matches what was generated, and if not, where they first differ;
   - whether the earlier messages are unchanged;
   - where the cache hit ended (`reply_end`, `prompt_end`, `partial`, …);
   - how many reply tokens were prefilled again.

   Every 100 requests a `PREFIXDIAG totals` line adds them up, with lost tokens split by match versus mismatch. That split is what separates cause (a), the harness, from cause (b), cache-slot pressure on the server. If the diagnostic itself fails, it logs and never ends a turn.
4. **Arms in `kaggle/arms.json`** (both use the defaults: all 25 games, 121 min):
   - `prefix-reuse`: both flags plus serve override `MAMBA_CACHE: 84`.
   - `prefix-reuse-60`: both flags, server unchanged.

   The notebooks build at about 319 KB, and I confirmed the embedded patch applies cleanly to `bundle-base`.

**What changes for the model** (with `ARC3_VERBATIM_HISTORY=1` only; with it off nothing changes)
- Stored reasoning, content and summary replies keep their blank and whitespace-only lines, any literal `<think>`/`</think>` tags, and line separators other than `\n`. Before, all of these were removed or converted.
- Leading and trailing whitespace is still stripped; the template trims it anyway.
- Stored text is slightly longer, so the context estimate grows a little and trims may come marginally earlier.
- Tool-call arguments, stubs and the transcript are unchanged.
- All of this is written up in `ARC3-Inference/CONFIGURATION.md`, sections 12 and 14.

**What still needs a Kaggle run**
- **Run both arms against a control at this commit.** The commit has new code, so the control has to be rebuilt with `--new-batch` first.
- **Judge by re-prefilled tokens and mean running requests, not score.** Re-run chain.py and `serve_report.py` on the arms' `serve.log`. Targets from the issue: the rate of hits that stop at the previous prompt's end well below 20%, and total uncached input down from 10.6M toward 8-9M.
- **Read the `PREFIXDIAG totals` lines.**
  - On `prefix-reuse-60`, `history_match` should be near 100%. Any tokens still lost with a match point at the server (slot pressure, or re-tokenization the harness cannot see).
  - Comparing `prefix-reuse` against `prefix-reuse-60` isolates the effect of 84 slots.
- **Check memory with 84 slots.** Confirm the server still starts and that the KV pool loss (about 1.3 GB) does not raise queueing.
- **Optional:** a control-plus-diagnostic arm (`ARC3_PREFIX_REUSE_DIAG` only) would measure the harness-side mismatch rate directly. I did not add it.
### Review
No blocking defects. The diff does what the spec asks, behaviour with both flags off is unchanged, and all 30 tests pass (`30 passed in 0.18s`). The worktree is clean; I built both arms into the gitignored `kaggle/out/` and removed it afterwards.

**What I checked**
- **Stored text:** every assistant message built in `tool_agent.py` goes through the verbatim path. That covers tool-call turns, no-tool-call replies and the rolling-summary reply. The resume/yield branch only builds a user message. Transcript, knowledge parsing, solver reasoning, stubs and degenerate-text checks still read the normalised text.
- **Tool-call arguments:** `tool_calls` is a JSON round-trip of the server's own objects, and `_run_python_tool` rstrips a separately parsed copy of the code. I confirmed no mismatch exists. I also simulated the server's tool-call parse (one newline stripped each side) and the real chat template on six cases. Rendering was byte-identical to the generated text in all six.
- **Test strength:** I reverted the history store, the no-tool-call content store, the summary store, and rstripped `code` into the stored arguments, one at a time. Each revert made at least one test fail.
- **Kaggle arms:** `prefix-reuse` and `prefix-reuse-60` both build (about 320 KB). The built `prefix-reuse` notebook has `MAMBA_CACHE` 84 applied before `--max-mamba-cache-size` is read. The notebook's own env sets `ARC3_REASONING_HISTORY_KEY=reasoning_content`, which the probe's renderer assumes.
- **Probe edge cases:** a markup-recovered tool call, a malformed-arguments reply and a flag-on turn all ran without error.

**Defects, ranked**
1. Low: `ARC3-Inference/inference/agent/prefix_reuse.py:262-263` (`check`) clears the stored reply before the request. After a context-overflow retry or a timeout, the next `check` reports `history=none cache=first`, so a non-first request is counted as first in the totals. The affected request had a trimmed prefix anyway, so it only skews the `history_none`/`none_first` counters. Fix: keep `_reply` until `report` runs, or restore it when `_chat_completion` raises (in `tool_agent.py` around L6853-6863).
2. Optional: there is no diagnostic-only arm (`{"env": {"ARC3_PREFIX_REUSE_DIAG": "1"}}`). Both new arms turn the fix on, so the harness-side mismatch rate on unmodified code is never logged. The control-versus-`prefix-reuse-60` comparison on `serve.log` still shows the fix's effect.

**Caveats, not defects**
- The probe compares text, not tokens. It cannot see a server-side parse or re-tokenization difference, and the docs say so. A trailing `\n\n` before `</think>` or a single `\n` before `<tool_call>` can never match because the template fixes those.
- Hits that stop at the previous prompt's end with `history=match` still point at the server. Short gaps (the issue measured 92% misses under 0.5 s) are not obviously explained by blank-line removal, so the Kaggle run has to settle that.
- With the default `ARC3_REASONING_HISTORY_KEY=reasoning`, the template ignores the stored reasoning, so the flag does not help. This only matters outside the Kaggle config, which already sets `reasoning_content`.
- Replies cut off at the length limit still can't match: the malformed tool call is dropped from history, but the generated text contained it.
- No PR exists: the branch is not pushed, so the user's "real PRs" request is still open for whoever has push rights.

Relevant paths:
- `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-3/ARC3-Inference/inference/agent/tool_agent.py`
- `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-3/ARC3-Inference/inference/agent/prefix_reuse.py`
- `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-3/kaggle/arms.json`
### Fixes
Both review findings are fixed on branch `arm/fix-3-prefix-reuse`. The worktree is clean, nothing was pushed, so there is still no PR, and Kaggle was not run.

**Finding 1: fixed.** The bug was real. I reproduced it by driving `ToolAgent.analyze` through three turns, where the third turn's request fails with a context-overflow error and is retried after the oldest exchange is dropped.
- **Before:** the retried request logged `history=none prefix=- cache=first`, so it was counted in the `history_none` and `none_first` totals.
- **After:** it logs `history=match prefix=changed cache=below_prompt prev_prompt=1300 prev_output=100`.
- **Fix:** in `ARC3-Inference/inference/agent/prefix_reuse.py`, `check()` no longer clears the stored reply; `report()` clears it, and `report()` only runs once a request completes. A request retried after a context overflow or a timeout is therefore checked against the same previous reply, and `gap_s` includes the failed attempt. Nothing changed in `tool_agent.py`.
- **Tests and docs:** I added a unit test (`test_probe_checks_a_retried_request_against_the_same_reply`) and an end-to-end test through the real overflow-retry path (`test_tool_agent_checks_a_context_overflow_retry_against_the_previous_reply`). Both fail with the fix removed and pass with it. `ARC3-Inference/CONFIGURATION.md` now says how retries are reported.

**Finding 2: fixed.** I added a `prefix-diag` arm to `kaggle/arms.json` with only `ARC3_PREFIX_REUSE_DIAG=1` and the control settings otherwise. Without it, the mismatch rate and lost tokens on unmodified history would never be measured, and that is the evidence that this cause is real. It is logging only and is built only when named. I built it locally: the env is just `{'ARC3_PREFIX_REUSE_DIAG': '1'}`, `ARC3_VERBATIM_HISTORY` is not set, and the notebook keeps `ARC3_REASONING_HISTORY_KEY=reasoning_content`. I deleted the build output afterwards.

**Caveats:** I agree with all of the reviewer's caveats and none needs a code change.

**What changes for the model:** these three commits change nothing the model sees; they only change diagnostic logging and add an arm. The branch as a whole changes what the model sees only with `ARC3_VERBATIM_HISTORY=1`, which is off by default: reasoning and content in history keep their blank lines and are only trimmed at the ends.

**Still needs a Kaggle run:**
- `prefix-diag` against control: how much of the miss rate comes from harness-side mismatches.
- `prefix-reuse` and `prefix-reuse-60` against control: whether the `prompt_end` misses and `lost_tokens` in `serve.log` drop, and whether any remaining losses with `history=match` point at the server's checkpoint capacity (`MAMBA_CACHE` 84 vs 60).

**Tests:** `32 passed in 0.22s`, with imports resolving to the worktree.

`git log --oneline arm/kaggle-builder..HEAD`:
```
e45e15c Add a diagnostic-only prefix-diag arm (#3)
0135a14 Test the probe across a context-overflow retry (#3)
894faab Keep the previous reply in the probe until a request completes (#3)
84dce5b Store the rolling summary reply verbatim under the flag too (#3)
dd0ecf2 Add the prefix-reuse and prefix-reuse-60 arms (#3)
2de03b6 Document ARC3_VERBATIM_HISTORY and ARC3_PREFIX_REUSE_DIAG (#3)
edc38c8 Test the prefix-reuse probe against the served chat template (#3)
19c191b Log prefix reuse per request behind ARC3_PREFIX_REUSE_DIAG (#3)
dd17c6e Test that tool-call arguments in history equal what the model sent (#3)
76a87e5 Test that reasoning and content round-trip verbatim into history (#3)
429647a Keep reply text verbatim in history behind ARC3_VERBATIM_HISTORY (#3)
```

**PR summary (#3):**
1. With `ARC3_VERBATIM_HISTORY` on (off by default), history keeps the reply's reasoning and content as the server returned them, trimmed only at the ends as the chat template trims them. A re-rendered reply then matches the generated tokens, and the server can reuse the cached reply instead of prefilling it again.
2. Off, nothing changes. Tool-call arguments, stubs, the transcript, knowledge extraction and every decision still use the normalised text.
3. `ARC3_PREFIX_REUSE_DIAG` logs one `PREFIXDIAG` line per request (whether history matches the reply, where the cache hit ended, tokens lost), plus running totals. A request retried after a failure is checked against the same reply.
4. Tests check the probe's rendering against the served Qwen chat template, the verbatim round-trip, tool-call arguments, and the overflow-retry path.
5. New Kaggle arms: `prefix-diag` (logging only), `prefix-reuse` (`MAMBA_CACHE` 84) and `prefix-reuse-60`. They separate harness-side mismatches from checkpoint-capacity pressure on the server.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-3/`:
- `ARC3-Inference/inference/agent/prefix_reuse.py`
- `ARC3-Inference/tests/test_prefix_reuse_probe.py`
- `ARC3-Inference/CONFIGURATION.md`
- `kaggle/arms.json`

## #6 (arm/fix-6-cold-start)
### Implementation
Issue #6 is fixed on branch `arm/fix-6-cold-start`; nothing has run on Kaggle yet. The control notebook still behaves the same, all 54 tests pass, and nothing was pushed. Whether the restored cache actually removes the ~145 s compile is the main thing still unproven.

**Worktree:** /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-6

**git log --oneline arm/custom-serving..HEAD**
```
259feb9 Test the launcher's JIT cache blocks with SGLang stubbed (#6)
f580018 Tie the JIT cache to the target model's config (#6)
59577cf Document the cold-start flags and how to publish the JIT cache (#6)
9d867c9 Add coldstart-save, coldstart-overlap and autotune-off arms (#6)
a16d8ed Let an arm mount extra Kaggle datasets (#6)
3efc744 Save and restore the JIT caches behind CFG flags (#6)
a3c89d8 Add CFG STARTUP_WEIGHT_LOAD_MODE to the launcher (#6)
081cb6f Embed the JIT cache helpers in the serving cell (#6)
440fb63 Add JIT cache snapshot and restore helpers for the launcher (#6)
```

**pytest:** `54 passed in 0.19s`, run from ARC3-Inference as `-m pytest -q ../kaggle/tests`. The import check resolves to the worktree (`.../wt/fix-6/ARC3-Inference/inference/__init__.py`).
- The tests are in `kaggle/tests/`, not `ARC3-Inference/tests`. Anything under ARC3-Inference goes into the harness patch every notebook carries, so it would have changed the control and forced a new batch. The `arm/serving-bench` branch already keeps its tests in `kaggle/tests/`.
- This branch has no `ARC3-Inference/tests`, so the command as written fails with "no such directory".
- The flags are documented in `serving/CUSTOM.md`, not `ARC3-Inference/CONFIGURATION.md`, which explicitly excludes serving settings.

**What I added**
- **The cache module** is `serving/kaggle/jit_cache.py`, which the builder embeds in the serving cell.
  - It tars the triton, flashinfer, sglang/jit, torchinductor and cuda caches plus `~/.tilelang`, keeping file times and skipping lock files and Triton's half-written entries.
  - The manifest records the GPU, CUDA, Python, torch, SGLang and kernel-package versions. It also hashes the serving config, the kernel-related env vars and the target model's `config.json`.
  - The cache is restored only if the manifest matches. Otherwise the log says why and the launch continues cold.
  - Kaggle may unpack a `.tar` on upload, so the restore also accepts the unpacked folder.
- **The arms**:
  - `coldstart-save`: saves the cache, 5-minute panel.
  - `coldstart-overlap`: overlap weight loading, 20-minute panel.
  - `autotune-off`: no FlashInfer autotune, 20-minute panel.
- **`"datasets"`**: an arm can now list extra datasets to mount. Arms without the key get the same metadata as before.
- **A consumer arm that loads the cache is not added**, because the dataset doesn't exist yet. `serving/CUSTOM.md` has the arm snippet and the publish steps.

**What changes for the model:** nothing with the defaults.
- I dry-built `control` and `panel-control`. Only the serving cell differs, and every difference is an added line:
  - three new settings, off by default;
  - the embedded helper module;
  - one check;
  - blocks that only run when a flag is set.
- Launch arguments, server env, kernel metadata and the harness patch are unchanged. The code hash is unchanged because nothing under ARC3-Inference changed.
- Every arm's fingerprint changes because the serving cell text changed.
- New `runs.json` records will carry `"datasets": []`.

**The flagged arms**
- `coldstart-overlap` passes `--startup-weight-load-mode overlap`. It hasn't been tested with AutoRound, the NEXTN draft model or `--ple-offload-embedding`, and Franzen's prefetch patch edits the same loader code.
- `autotune-off` passes `--disable-flashinfer-autotune`, which may lower decode speed.
- `coldstart-save` adds a probe of the runtime versions (a few seconds before launch). It then writes the tar in the foreground once the server is ready, which delays the games by however long that takes.

**Still needs a Kaggle run**
1. **Does a restored cache work in a fresh container?** Run `coldstart-save`, publish the cache, then run a consumer arm. Success is `cuda_graph target_verify` in serve.log falling from ~145 s to a few seconds.
   - FlashInfer's and SGLang's compiled modules may be rebuilt anyway. Their build tool compares file times with the freshly installed package sources.
   - Triton may rerun its autotune benchmarks even on cached kernels.
2. **Is the cache complete?** Check the save log's `not in the snapshot:` lines for other cache folders that hold compiled kernels.
3. **Is overlap loading correct and faster?** Check `coldstart-overlap`'s serve.log for errors, plus its accept length and tok/s against `panel-control`.
4. **How much of the silent ~125 s is autotune, and what does turning it off cost?** Compare `autotune-off`'s startup timings and tok/s with `panel-control`.
### Review
VERDICT: 2 defects need fixing (the `coldstart-overlap` arm cannot work, and `autotune-off` is probably a no-op), plus 2 minor ones. The cache save/restore code and the defaults-unchanged claim hold up.

SGLang source references are to the pinned commit `d00d88ef` in `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-rtxpro6000`. Its `server_args.py:3363` is the same line the issue cites.

**Verified, no defect**
- **Tests:** `python -m pytest -q ../kaggle/tests` gives 54 passed. The spec's `tests` path fails because `ARC3-Inference/tests` does not exist. The implementer's reason holds: files under ARC3-Inference go into the harness patch and would change the code hash.
- **Control and panel-control:** I built both at `arm/custom-serving` and at HEAD and diffed every cell. Only the serving cell differs, with 445 added lines and 0 removed. Metadata and the other notebook keys are identical, and the harness patch hash is the same. The notebook grows from 301,878 to 322,519 bytes, under the 950,000 limit.
- **Arms:** all five arms build and the serving cells compile. `coldstart-save` overrides `JIT_CACHE_SAVE`, `coldstart-overlap` overrides `STARTUP_WEIGHT_LOAD_MODE`, and `autotune-off` overrides `AUTOTUNE`. No extra dataset sources appear for any arm.
- **`/health` timing:** the launcher's claim is correct. `/health` returns 503 while the server status is `Starting`, and the status becomes `Up` only after the warmup request (`http_server.py:674`, `2319`). Saving at `/health` is therefore saving after warmup.
- **sglang/jit restore:** SGLang's JIT cache is content-addressed (`kernels/jit/utils/compile/cache.py`), so restoring it does not depend on file times.

**Defects, most severe first**

1. **`coldstart-overlap` cannot work, and the server will die at model load.**
   - Where: `kaggle/arms.json:19-24`, `serving/CUSTOM.md:29` and `:85`, `serving/kaggle/launch_sglang.py:370`.
   - Why: `StartupWeightLoadManager._get_unsupported_reason` (`startup_weight_load.py`) rejects this launch config on several independent rules, so it raises `ValueError: --startup-weight-load-mode=overlap is not supported: ...`.
     - **Speculative decoding** (line 360): NEXTN is on.
     - **Page-cache dropping** (line 394): `DROP_CACHE_AFTER_LOAD` defaults to True.
     - **Quantization** (line 411): auto-round.
     - **Multimodal** (line 416): the model takes images.
     - **Architecture allowlist** (lines 38-43): Llama, Qwen2 and Qwen3 only, so Qwen3-Next is not allowed.
     - **Draft worker:** draft workers are also rejected.
   - The call site (`load_model_utils.py:318-325`) does not catch the error and there is no fallback to serial. The builder turns the server exit into a RuntimeError, so the arm burns a GPU session and produces no games.
   - The docs say "Unverified with AutoRound, the NEXTN draft...". The source shows a hard reject, and it also kills issue #6 proposal item 2.
   - Fix:
     - Delete the arm from `arms.json`.
     - Rewrite `CUSTOM.md:29` and `:85` to say overlap is rejected on this stack and list the reasons above.
     - Optionally replace the assert at `launch_sglang.py:169` with one that rejects `"overlap"` with that reason list, so it fails in seconds instead of mid-launch.
     - Note on issue #6 that hiding the capture behind the weight read must come from the cache restore instead.

2. **`autotune-off` is very likely a no-op, so the arm and its docs are misleading.**
   - Where: `kaggle/arms.json:25-30`, `serving/CUSTOM.md:24` and `:84`.
   - Why: FlashInfer autotune only runs if `should_run_flashinfer_autotune` (`runner/flashinfer_autotune.py:53-126`) returns True. That needs the MoE runner to be one of the `flashinfer_*` backends (line 88), or a modelopt fp4/fp8 model (the checks before line 122). Otherwise it returns False at lines 122-123.
   - This launch uses `--moe-runner-backend auto` with `--quantization auto-round`. `arg_groups/overrides.py:2775-2834` leaves `auto` unchanged for that quantization. So startup should be identical with `AUTOTUNE` on or off.
   - The ~125 s silent period is then not autotune, and the "autotune picks the kernels, compare tok/s" note in `CUSTOM.md:84` is wrong.
   - I found no serve.log locally to confirm, so this rests on the source. The cheap check is to grep an existing serve.log for "Running FlashInfer autotune".
   - Fix: drop the arm unless that grep shows autotune running, and correct `CUSTOM.md:24` and `:84`. If autotune does run, its cache is written to `cache/sglang/flashinfer/autotune/` (default `SGLANG_FLASHINFER_AUTOTUNE_CACHE` is True). That path is not in `CACHE_SUBDIRS`, so it would also need adding.

3. **A failed save leaves a partial tar in the kernel output (minor).**
   - Where: `serving/kaggle/jit_cache.py:200-208`.
   - Why: any exception inside `tarfile.open(tmp, "w")` leaves `sgl-jit-cache.tar.partial` in `/kaggle/working`. I reproduced this by making `tar.add` raise on the second file, which left a `.partial` file behind.
   - One cache file vanishing between the file listing and the tar (for example a build finishing) also aborts the whole snapshot, and there is no retry. `.staging-*` build directories from SGLang's JIT cache are not skipped.
   - Fix: wrap the write in try/except and unlink `tmp` on failure. Skip files that raise `FileNotFoundError`. Add `.staging-` to `_skipped`.

4. **The publish steps would upload the kernel log too (minor).**
   - Where: `serving/CUSTOM.md:112`.
   - Why: `fetch_version_output.py` also writes `<slug>.log` into the output directory, so `kaggle datasets create -p /tmp/sgl-jit-cache` publishes `v4-coldstart-save.log` into the cache dataset.
   - Fix: add a step to delete that log, or fetch into a subfolder and publish only the tar and manifest.

**Not defects, still unproven on Kaggle (the implementer already lists these)**
- Whether FlashInfer's ninja build treats the restored objects as up to date.
- Whether Triton reruns its benchmarks on cached kernels.
- Whether the snapshot is complete. The `not in the snapshot:` log lines cover only directories under `PREFIX/cache`.

Worktree untouched: `git status` is clean at `259feb9`.
### Fixes
All four findings were real and are fixed in four new commits on `arm/fix-6-cold-start`. The kaggle tests pass: **51 passed in 0.19s**. Nothing was pushed, and I did not comment on issue #6.

Worktree: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/fix-6`

**Per finding**

1. **`coldstart-overlap` cannot work: fixed (703f17f).**
   - **Check:** at `d00d88ef`, `StartupWeightLoadManager._get_unsupported_reason` rejects our launch on speculative decoding, draft workers, page-cache dropping, quantization, multimodal and the Llama/Qwen2/Qwen3 allowlist. `load_model_utils.py:318-330` has no fallback, and nothing else checks the mode first. Franzen's prefetch-lookahead patch does not touch these rules.
   - **Change:** I removed the knob entirely rather than keeping it behind a reject-assert, because no value but `serial` can ever work for this model. That removes the arm, the CFG key `STARTUP_WEIGHT_LOAD_MODE`, its launch argument and its entry in `CFG_IGNORED`. An arm that still sets the key fails in seconds on the existing `set(_SERVE_OVERRIDES) <= set(CFG)` assert.
   - **Docs:** `serving/CUSTOM.md` now lists the rejection reasons and says the JIT cache is the only way to shorten the capture. The tests were updated to match.
2. **`autotune-off` is a no-op: fixed (152042c).**
   - **Check:** `should_run_flashinfer_autotune` needs a `flashinfer_*` MoE runner, or ModelOpt FP4/FP8 or MXFP8 GEMMs. For AutoRound, nothing in `overrides.py` moves `--moe-runner-backend auto` or `--speculative-moe-runner-backend auto` off `auto`; the Qwen3-MoE override only fires for fp8, modelopt_fp4 or no quantization. Franzen's patches don't touch this code.
   - **Change:** I dropped the arm and corrected the `AUTOTUNE` row and the cold-start text.
   - **Not checked on a log:** no serve.log exists locally, so this rests on the source. The docs say so and point to the `Running FlashInfer autotune` grep. The autotune cache path was not added to `CACHE_SUBDIRS`, since nothing is written there.
3. **A failed save leaves a partial tar: fixed (6bb2bc2).** In `jit_cache.py`:
   - `save()` now removes the `.partial` tar on any error, and `_write_atomic` does the same for the manifest.
   - A file that disappears after the listing is skipped instead of aborting the snapshot.
   - `.staging-*` directories are skipped.
   - The manifest is now written last in the tar, so its file and byte counts match what was stored. Restore finds it by name, so the position doesn't matter.
   - Two new tests cover a file that disappears and a failure on the second file; both fail on the old code.
4. **The publish steps upload the kernel log: fixed (5dbd3dd).** The fetch now goes into a scratch folder with the pattern `'sgl-jit-cache\.(tar|manifest\.json)$'`, which also excludes `.partial` files. Only those two files are moved into the dataset folder.

I also rebuilt `control` and `panel-control` from `arm/custom-serving` and from HEAD. Only the serving cell differs, with 464 lines added and 0 removed; metadata and every other cell are identical.

**git log --oneline arm/custom-serving..HEAD**
```
5dbd3dd Keep the kernel log out of the published JIT cache dataset (#6)
6bb2bc2 Make the JIT cache save survive vanishing files and clean up on failure (#6)
152042c Drop the autotune-off arm: autotune never runs on this stack (#6)
703f17f Remove the overlap startup mode: SGLang rejects it for this model (#6)
259feb9 Test the launcher's JIT cache blocks with SGLang stubbed (#6)
f580018 Tie the JIT cache to the target model's config (#6)
59577cf Document the cold-start flags and how to publish the JIT cache (#6)
9d867c9 Add coldstart-save, coldstart-overlap and autotune-off arms (#6)
a16d8ed Let an arm mount extra Kaggle datasets (#6)
3efc744 Save and restore the JIT caches behind CFG flags (#6)
a3c89d8 Add CFG STARTUP_WEIGHT_LOAD_MODE to the launcher (#6)
081cb6f Embed the JIT cache helpers in the serving cell (#6)
440fb63 Add JIT cache snapshot and restore helpers for the launcher (#6)
```
a3c89d8 and 9d867c9 add the knob and arms that 703f17f and 152042c remove. Squash them before the PR if you want a clean history.

**pytest:** `51 passed in 0.19s`, run as `nice -n 10 env -u ARC3_PRIORITY_POLICY <venv python> -m pytest -q ../kaggle/tests -p no:cacheprovider` from the worktree's `ARC3-Inference`. `ARC3-Inference/tests` does not exist. `import inference` resolves to the worktree.

**What changes for the model:** nothing. All defaults are unchanged; the serving cell only gains code that runs when `JIT_CACHE_SAVE` or `JIT_CACHE_DIR` is set.

**Still needs a Kaggle run:**
- `coldstart-save` should log `JIT cache saved to ...`, and any `not in the snapshot:` lines would show cache directories it misses.
- A consumer arm with the published dataset should log `JIT cache restored`, and `cuda_graph target_verify` should fall from about 145 s.
- Whether ninja (FlashInfer, SGLang JIT) rebuilds the restored objects anyway, and whether Triton re-runs its autotuning on cached kernels.
- Any serve.log should have no `Running FlashInfer autotune` line, which would confirm finding 2.

**Suggested comment for issue #6:** proposal 2 (overlap loading) is impossible on this stack for the reasons in finding 1. Proposal 3 (autotune on/off) is answered from the source: autotune does not run here. Only proposal 1, the cache restore, can recover the ~145 s.

**PR summary**
1. Cold start (#6): save the Triton, FlashInfer, SGLang-JIT, TorchInductor, CUDA and TileLang caches once `/health` is up (`JIT_CACHE_SAVE`), and restore a matching snapshot from a mounted dataset before launch (`JIT_CACHE_DIR`); both are off by default.
2. A snapshot is used only if the GPU, CUDA, torch and kernel package versions match, along with a hash of the kernel-relevant CFG and env and the target's `config.json`. A mismatch, a truncated or unsafe tar, or a failed save leaves the launch cold, exactly as today.
3. Arms can mount extra datasets (`"datasets"`), and the new `coldstart-save` arm produces the cache. `serving/CUSTOM.md` covers publishing it and adding a consumer arm.
4. Overlap loading and `--disable-flashinfer-autotune` were checked against the SGLang source and dropped. SGLang rejects overlap for this model (NEXTN, AutoRound, multimodal, architecture allowlist), and autotune never runs for `--moe-runner-backend auto` with AutoRound.
5. Control and panel-control notebooks are unchanged apart from the added serving-cell code, and the 51 kaggle tests pass without a GPU. Validation still needs the `coldstart-save` run and a consumer run.