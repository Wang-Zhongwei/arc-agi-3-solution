# fix-definitely-issues: review:#3

Workflow run `wf_800f9fbb-745`, phase Review. Fix DEFINITELY issues #5, #4, #3, #6 one at a time: implement in a worktree, Sonnet review, fix

---

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
