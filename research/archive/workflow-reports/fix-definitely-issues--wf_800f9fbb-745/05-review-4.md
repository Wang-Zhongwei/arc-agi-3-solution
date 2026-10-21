# fix-definitely-issues: review:#4

Workflow run `wf_800f9fbb-745`, phase Review. Fix DEFINITELY issues #5, #4, #3, #6 one at a time: implement in a worktree, Sonnet review, fix

---

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
