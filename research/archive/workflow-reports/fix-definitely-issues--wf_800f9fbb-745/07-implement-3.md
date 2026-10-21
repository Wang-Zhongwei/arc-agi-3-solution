# fix-definitely-issues: implement:#3

Workflow run `wf_800f9fbb-745`, phase Implement. Fix DEFINITELY issues #5, #4, #3, #6 one at a time: implement in a worktree, Sonnet review, fix

---

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
