# Two bugs behind the 6 h adaptive-reasoning-ucb v5 submission

2026-09-17. Submission `56293413` (kernel `jonathanwang2022/adaptive-reasoning-ucb` v5, public
score 6.18) stopped after about 6 h of its 9 h slot. Kaggle does not show rerun output, so the
evidence comes from the public v5 run of the same notebook:
`ARC3-Inference/runs/kaggle/adaptive-reasoning-ucb/v05/output/` (local only; `runs/` is gitignored).

Line numbers are for HEAD `c173b40`. The v5 bundle (`5f09405`, dirty) has the same code.

## Bug 1: `/kaggle/working` fills up and the run ends early

1. `/kaggle/working` is a 20 GB loop device. The SPN serving setup unpacks the vLLM runtime into it
   (`flashnext-gcp-runtime`, plus `flashnext-model`), leaving **2.9 GiB free** for the game.
   - Kernel log at 81.6 s: `/dev/loop1 20G 17G 2.9G 86% /kaggle/working`.
   - Teardown frees 16.7 GiB (`flashnext-teardown.json`).
   - The vendor script builds these paths from `TAAF_KAGGLE_WORKING_DIR`
     (`kaggle_flashnext_setup.py:43-48` in the `matijaludvig/spn-rrw-bundle` dataset).
2. `HarnessSolver` writes per-game files to the same volume in submission mode too. It never reads
   `minimal_diagnostics`. Public v5 wrote these in 2.2 h:

   | Files | Size |
   | --- | --- |
   | `artifacts/*_events.jsonl` | 317 MiB |
   | `solver_analysis/*.html` | 148 MiB |
   | `transcripts/*.txt` | 144 MiB |
   | `vllm.log` | 5.7 MiB |
   | **Total** | **0.60 GiB** |

   The rate had reached 0.35 GiB/h by the last hour, which uses up 2.9 GiB about 8.7 h into
   gameplay. The rerun plays 110 games instead of 25. It only has to write about 1.5× faster to
   run out at about 6 h. Most of these bytes come from Bug 2.
3. Out of disk space, `HarnessSolver._allocation_event` (`inference/framework/solver.py:1317`)
   raises, because its append has no error handling. The exception leaves
   `_run_allocated_games`, and `Benchmark.run` catches it
   (`tufa-arc-agi-framework/src/taaf/benchmark.py:177`). It marks the remaining games crashed
   and closes the scorecard. The submission is scored on what had been played, so it gets a
   valid score and ends early.

**Control run.** `adaptive-reasoning/v04` used rolling_extension and our own vLLM with weights
from `/kaggle/input`. It ran the full 9 h as a submission while hitting Bug 2. Its setup snapshot
(`v04/output/vllm-setup-provenance.json`) shows **19.5 GiB free** on the same device.

**Not verified.**
- Disk usage during the rerun itself.
- A second possible cause: Kaggle's gateway auto-closing the scorecard. Its settings are not
  published.

**Fix**
- Stage the SPN runtime and model outside `/kaggle/working`. In v04, `/` (overlay) had 1.1 TB free.
- Skip or cap solver file writes when `minimal_diagnostics` is set.
- Catch `OSError` in `_allocation_event` so a logging failure cannot end the run.
- To check the fix, do a public run with `serving_budget_s=31440` that logs `df /kaggle/working`
  every few minutes.

## Bug 2: a 400 → 500 retry loop freezes a game

One cycle takes about 1 s plus request time and plays no action:

1. `ToolAgent.analyze` builds the system prompt, `_history_messages`, and a new user message
   (grid plus animation images). The harness estimate says this fits its 31,744-token budget.
   vLLM rejects it as too long with **400**. The response body is not logged.
2. `_is_context_length_error` matches. The harness drops the oldest history, logs
   `context_overflow_recovered`, and resends (`inference/agent/tool_agent.py:1862-1878`).
3. vLLM answers the trimmed request with **500**:
   `Multi-modal cache miss ... P0/P1 cache drift; returning a retryable response so the items are resent with data.`
   A 500 is not a length error, so it is raised and logged as `request_error: 500 ...`, and
   `analyze` returns `retryable_failure=True` (`tool_agent.py:2006-2022`).
4. The failed turn restores the untrimmed history:
   `self._history_messages = previous_history_messages` (`tool_agent.py:2045`).
5. `_HarnessGameSession.play` sleeps 1 s (`solver.py:479-483`) and calls `analyze` again with
   the same board, which is step 1 again.

Each cycle logs the ~16 KB system prompt again (`tool_agent.py:1771`), along with its events and
HTML.

**Evidence**
- Public v5 `vllm.log`: 2,790 × 200, 6,218 × 400, 6,165 × 500.
- In log order, 4,834 of the 500s come right after a 400.
- The 6,165 cache misses contain only three image-hash sets, one per stuck game, and they repeat
  every cycle:

| Game | Loop started | At action | Cycles | Left the loop |
| --- | --- | --- | --- | --- |
| tn36 | 01:39:49 | 30 | 4,187 | no (run ended 03:28:41) |
| wa30 | 02:38:58 | 237 | 1,278 | no |
| lp85 | 02:57:55 | 122 | 700 | no |

`adaptive-reasoning/v04` has the same loop on a different vLLM build and model (NVFP4): 2,286
cache misses, su15 2,139 cycles, r11l 147.

**Why the trimmed retry always misses.** This is inferred; the patched vLLM source was not read.
vLLM keeps one copy of its image cache in the API server and one in the engine, and assumes both
see every request. The 400 request passes through the API server but never reaches the engine. On
the next request, the API server sends only hashes for those images, which the engine never
received.

This is not a logging problem. Each cache miss is a failed request with no output, and `vllm.log`
is under 1% of the disk writes.

**Fix**
- After a 400, keep the trimmed messages even when the retry fails, so the next attempt does not
  send the same over-long prompt.
- Find why the harness estimate is below vLLM's count. Log the 400 response body first.
- Stop retrying a request that keeps failing the same way.
- On the server, try disabling the multimodal processor cache (`--mm-processor-cache-gb 0`).
  First confirm the SPN runtime's vLLM supports that flag.
- To check the fix, rerun the public v5 config. `Multi-modal cache miss` in `vllm.log` and
  `context_overflow_recovered` in the transcripts should both be close to 0.
