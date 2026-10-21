# GB10 overnight arm tests (2026-09-26): index and summary

Every backlog item in `test.md` that can be switched on in the harness got its own branch
`gb10/<arm>`, a build, an independent review, one or two live GB10 runs, and an overview at
`notes/gb10-arms/<arm>.md` on that branch. This file is the index (branch `gb10/runner`).
Read the per-arm files for the detail and evidence.

## How it was tested

- **Model:** the exact Kaggle checkpoint, `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@4c67bf6`,
  downloaded to the GB10. It is served with the Kaggle bundle's launch profile flag for flag:
  MTP 3, `qwen3_coder` tool parser, `qwen3` reasoning parser, 49152 context, prefix caching,
  bf16 KV, 12 sequences, 4096 batched tokens, async scheduling, `preserve_thinking`. See
  `ARC3-Inference/scripts/gb10/serve_intel_w4a16.sh`.
  - The one forced difference is the runtime. Kaggle's vLLM build puts the 95 GiB PLE n-gram
    table in host RAM, and on the GB10 host RAM is GPU memory. So the GB10 uses the b12x build,
    which reads the table from NVMe, with the plain safetensors loader, because b12x's own loader
    can't read AutoRound scales.
- **Agent:** the Kaggle notebook's `RUN_CONFIG["env"]` exported verbatim
  (`scripts/gb10/kaggle_run_env.json`), plus only the arm's overrides
  (`scripts/gb10/arms/<arm>.env`), through the same TAAF entrypoint
  (`scripts/gb10/run_arm.sh` → `make interactive`).
- **Games and budget:** ar25 and sk48 for 45 min each, 8 concurrent streams per batch (Kaggle
  also runs 8). Every batch had its own control. Batch 6 had two controls, to measure
  control-to-control spread inside one batch.
- **Scale caveat:** the GB10 generates about 5-6 tok/s per stream under this load, roughly 7x
  slower than Kaggle. A game therefore gets only ~9-15 model turns, about 6-7 Kaggle minutes.
  - ar25 level 1 falls at turn 3-7 in every run, so level 2 is the only event that can
    discriminate between arms. Nobody cleared sk48.
  - Across 8 controls, ar25 reached levels **2,1,2,1,1,1,1,1**. A one-level difference from a
    single run is noise, so the arms are judged mainly on process evidence from the transcripts.
- **Reporting:** `scripts/gb10/arm_report.py` gives levels, level-up turn and action, turns,
  requests, tokens, the server's `cached_tokens` hit rate, and transcript pattern counts.

## Verdicts

| arm (branch `gb10/…`) | verdict | the one-line reason |
|---|---|---|
| **ledger-fix** (bug fix) | **ship** (+ the v2 note fix) | The memory ledger was starved on the current code: `_absorb_memory` ran only `if content:`, and 95% of this model's replies have no visible text. Fixed, memory blocks appear (9/20 vs 0/24 requests), but the level note kept the turn-1 guess; that is fixed in code, not yet run live. |
| **numpy-perception** (v1+v2) | **promising → Kaggle** | Cleared ar25 level 2 in both runs. The mirror-stripe move reached the model only through the `<perception>` block. v2 restored the cache hit rate. |
| **frame-diff** (v1+v2) | **promising → Kaggle** | Cleared ar25 level 2 in both runs, by the same mechanism: an exact "the wall moved +3" line that the model's own python filtered out. v2 removed the fake 1x1 moves (21/31 → 0/33). |
| **sandbox** | **promising → Kaggle** | 0 python errors in 47 calls, helpers used in 34 of them. sk48 fitted 15 turns vs 8-11. Cleared ar25 level 2 (1 run). |
| **alloc-policies** (offline sim) | **Kaggle smoke run** | `knapsack@0.375` beats `ucb@0.375` in all 4 simulated world models (+0.09 to +1.16). Also found that the a1/a2 Kaggle arms silently ran `uniform`, and fixed a thread race. |
| **state-manager** (v1+v2) | inconclusive → optional Kaggle | 60 blocks, 0 wrong facts, ~50-100 tokens, but the model cites it in only ~6% of requests. Its purpose (surviving compaction) needs long games. |
| **hypothesis-loop** (v1+v2) | inconclusive | Compliance via a tool parameter reached 93%. The useful part is the harness's "actually …" diff; the model never reads the ledgered refutations. Superseded by frame-diff. |
| **reasoning-summary** (v1+v2) | inconclusive | Holds ~2x more history before compaction and v2 keeps its findings, but no level change, and it costs cache hits. Next: condense only at compaction time. |
| **token-efficiency** (+ `reasoning_effort=medium`) | not a speed-up | Prompts were 33% smaller, but GB10 wall time is ~99% decode, so it bought no turns. `medium` does not shorten thinking (only `low` asks for brevity). |
| **prompt-cache** | fold into defaults, no A/B | Cache hit 0.79 vs 0.71 as predicted, but prefill is ~6% of wall time. The real wins it found: compaction triggers on an estimate that is ~1.7-2x too high, and there is no `max_tokens` cap. |
| **xml-format** | no effect | Zero cost and zero detectable change. The model never refers to the structure. |
| **scoring-guide** (math policy) | no effect here | The prompt fired but batches got shorter, not longer, and the model used the guidance 1 time in 36 responses. Judge it on the Kaggle b4 run. |
| **thinking-guide** | drop | 0/28 `Predict:` lines. Every reply is reasoning + a tool call with no visible text, so "end your reply with…" has nowhere to go. |
| **ascii-upscale** | drop as is | 9x the image tokens (576 vs 64 per board). The model barely read the letters, got 2 of 4 coordinate reads wrong, and was never told the overlay exists. |
| **action-tools** | drop | The `act` tool was called 0 times. The failure modes it guards against occur 0 times in 142 control responses on this model. |
| spec-decode (MTP scan) | **keep MTP 3** | Aggregate tok/s at 8 real requests, by MTP depth: 0: 34.8, 2: 39.2, **3: 42.0**, 4: 41.8. The current Kaggle depth is already optimal (+21% over no speculation). |
| not run | see `not-run.md` | Custom vLLM build, hidden-activation feedback, layer looping and the LAYA checker, with what it would take and the tested nearest proxy for each. |

## What the night says overall

1. **Exact change information is the one lever that showed up across arms.**
   - numpy-perception (x2), frame-diff (x2) and sandbox (whose `delta()` helper does the same)
     each hand the model an exact, harness-computed "what moved" statement. All 5 of those runs
     cleared ar25 level 2, against 2 of 8 controls.
   - In each case the transcript shows the key fact reaching the model through that text, while
     its own python filtered it out.
   - This is still small-sample: pooled Fisher p is ~0.03, and the pooling was chosen after
     seeing the data. It is the best-supported idea of the night, not a proven one.
2. **This model does not follow prompt-only output formats, but it does fill tool parameters.**
   Every reply is reasoning plus a tool call with empty visible text: 5% of 281 control replies
   had any text. The results line up:
   - `Predict:` lines: 0/28.
   - Model-written memory tags: 0 in 511 replies.
   - `prediction` tool parameter: 93%.
   - A second `act` tool: 0 uses.

   Anything that must come from the model should be a tool parameter or computed by the harness.
3. **On the GB10, only generated tokens cost time.**
   - Measured over 337 request blocks: 0.145 s per generated token vs 0.002 s per uncached prompt
     token.
   - Prompt trimming and caching barely move turns here.
   - Check the ratio on Kaggle before investing further in prompt size there.
4. **Control-side bugs found along the way** (all present on `my-main`):
   - The ledger feed was gated on visible text, and the level note kept the oldest guess (fixed
     on `gb10/ledger-fix`).
   - The context estimate counts reasoning twice at chars/3, so compaction fires at ~17-19k real
     tokens.
   - There is no `max_tokens`: at least 3 of 58 game streams lost 6-16 min to a reply that hung
     until the 900 s timeout.
   - A `formatted_code` echo plus F821 lint on preloaded sandbox names when ruff is available to
     the sandbox. Found in code review; not observed in these GB10 runs.
   - The a1/a2 allocation arms ran `uniform`.
   - The system prompt says past turns are text-only, but past board images stay in history.

## Recommended next Kaggle runs (132 min, 25 games, same-day control)

1. `gb10/ledger-fix` merged into the control itself. It is a bug fix; rebase every other arm
   onto it.
2. `PERCEPTION_SUMMARY=1 PERCEPTION_NUMPY=1` (numpy-perception) and
   `MULTIMODAL_FRAME_DIFF=1 MULTIMODAL_ANIMATION=` (frame-diff): the change-information family.
   They overlap, so run them as separate arms, not stacked.
3. `SANDBOX_V2=1`.
4. `ALLOCATION_POLICY=knapsack EXPLORATION_FRACTION=0.375`.
5. Cheap defaults with no A/B needed: a finite `max_tokens`, and compaction triggered on the
   server's real `prompt_tokens`.

## Results table (every arm-run; generated by `combine.py` from each batch's `report.json`)

| batch | arm | ar25 levels | ar25 level-ups (turn/action) | turns ar25 / sk48 | requests ar25 / sk48 | sk48 actions | gen tok per request ar25 / sk48 | cache hit ar25 / sk48 |
|---|---|---|---|---|---|---|---|---|
| b1 | control | 2 | t4/a18,t9/a35 | 9 / 10 | 20 / 16 | 27 | 869 / 1051 | 0.717 / 0.603 |
| b1 | hypothesis-loop | 1 | t4/a19 | 7 / 5 | 14 / 13 | 8 | 939 / 1358 | 0.663 / 0.614 |
| b1 | numpy-perception | 2 | t4/a19,t9/a35 | 9 / 5 | 13 / 9 | 11 | 1263 / 1680 | 0.478 / 0.467 |
| b1 | state-manager | 1 | t4/a20 | 9 / 9 | 18 / 15 | 40 | 966 / 1111 | 0.711 / 0.606 |
| b2 | control | 1 | t5/a23 | 10 / 11 | 16 / 18 | 30 | 927 / 924 | 0.593 / 0.687 |
| b2 | ascii-upscale | 1 | t6/a20 | 10 / 9 | 21 / 15 | 11 | 780 / 1059 | 0.667 / 0.563 |
| b2 | scoring-guide | 1 | t7/a24 | 13 / 7 | 23 / 13 | 12 | 663 / 1021 | 0.645 / 0.546 |
| b2 | thinking-guide | 1 | t4/a18 | 9 / 8 | 13 / 15 | 10 | 1021 / 1007 | 0.703 / 0.604 |
| b3 | control | 2 | t5/a17,t12/a39 | 13 / 11 | 24 / 15 | 22 | 790 / 1239 | 0.711 / 0.655 |
| b3 | prompt-cache | 1 | t4/a22 | 12 / 7 | 21 / 11 | 10 | 859 / 1046 | 0.79 / 0.657 |
| b3 | reasoning-summary | 1 | t2/a19 | 7 / 13 | 17 / 23 | 26 | 1012 / 687 | 0.615 / 0.721 |
| b3 | token-efficiency | 1 | t4/a19 | 7 / 9 | 14 / 15 | 26 | 998 / 1268 | 0.662 / 0.651 |
| b4 | control | 1 | t5/a18 | 12 / 8 | 20 / 13 | 13 | 811 / 1304 | 0.705 / 0.573 |
| b4 | action-tools | 1 | t3/a15 | 8 / 9 | 18 / 17 | 30 | 940 / 907 | 0.71 / 0.696 |
| b4 | frame-diff | 2 | t4/a17,t8/a33 | 9 / 13 | 18 / 19 | 31 | 1008 / 606 | 0.669 / 0.795 |
| b4 | sandbox | 2 | t5/a17,t13/a35 | 14 / 15 | 21 / 26 | 36 | 871 / 666 | 0.745 / 0.755 |
| b5 | control | 1 | t3/a23 | 10 / 10 | 15 / 18 | 19 | 1036 / 938 | 0.709 / 0.641 |
| b5 | hypothesis-loop | 1 | t3/a19 | 4 / 11 | 12 / 15 | 19 | 1376 / 1141 | 0.532 / 0.578 |
| b5 | state-manager | 1 | t6/a21 | 11 / 6 | 17 / 16 | 11 | 995 / 1045 | 0.631 / 0.628 |
| b5 | xml-format | 1 | t4/a17 | 8 / 11 | 16 / 19 | 22 | 1056 / 857 | 0.562 / 0.691 |
| b6 | control | 1 | t6/a30 | 11 / 8 | 18 / 16 | 27 | 900 / 997 | 0.655 / 0.675 |
| b6 | control2 | 1 | t3/a15 | 6 / 15 | 15 / 19 | 43 | 991 / 892 | 0.627 / 0.571 |
| b6 | numpy-perception | 2 | t7/a19,t15/a42 | 15 / 11 | 17 / 14 | 44 | 928 / 1042 | 0.627 / 0.63 |
| b6 | token-efficiency-medium | 1 | t4/a27 | 11 / 13 | 18 / 18 | 68 | 925 / 908 | 0.502 / 0.64 |
| b7 | control | 1 | t5/a21 | 12 / 11 | 23 / 15 | 42 | 689 / 828 | 0.74 / 0.688 |
| b7 | frame-diff | 2 | t4/a17,t10/a37 | 10 / 11 | 21 / 15 | 13 | 782 / 1020 | 0.614 / 0.605 |
| b7 | ledger-fix | 1 | t6/a18 | 13 / 9 | 19 / 14 | 29 | 788 / 996 | 0.66 / 0.646 |
| b7 | reasoning-summary | 1 | t4/a17 | 14 / 14 | 19 / 24 | 48 | 838 / 607 | 0.553 / 0.692 |

Controls: n=8; ar25 levels [2, 1, 2, 1, 1, 1, 1, 1] (level 2 reached 2/8); turns per game 6-15; sk48 levels all 0.

## Where things are

- Branches `gb10/<arm>` in `wang-duck-harness` (local only, not pushed). Worktrees are at
  `arc-agi-3-v3/gb10-wt/<arm>`.
- Raw runs are on the GB10 under `~/gb10-arms/<arm>/ARC3-Inference/runs/local/*gb10-<arm>-b<N>/`,
  with batch logs in `~/gb10-arms/logs/`.
- The GB10 is still serving the Kaggle model on :1234. The old 27B container `arc3-gb10` was
  stopped, not removed; `docker start arc3-gb10` restores it after stopping ours with
  `ACTION=stop bash scripts/gb10/serve_intel_w4a16.sh`.
