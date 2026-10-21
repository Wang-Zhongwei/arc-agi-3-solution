# gsq-coder-gguf-runner: review:runner-gguf

Workflow run `wf_0ae4547f-80a`, phase Review. Serve the ISTA GSQ-RCO Coder GGUF the way its Hugging Face repo documents (llama.cpp or Strata, both shards, 256 experts, mmproj), validate the harness contract locally, build staging/engine/test notebooks and the 2-hour final-run notebook

---

**Verdict:** I found nothing in `arm/runner-gguf` (HEAD `d64ab57`) that would stop the smoke or final run from working, and the server is set up the way the model card says. There are two things to change before the final run: the host prompt cache is undersized for 25 games, and two go/no-go checks in `gguf_report.py` can fail on a healthy smoke run. Two smaller items follow them.

## Defects, ranked, with fixes

**1. The final run's host prompt cache is too small, and idle slots fill it with copies (score per compute).**
- **Idle-slot copies:** llama-server's `cache_idle_slots` setting is on by default in b11476 (`common.h:634`).
  - Every time a new request starts, it copies the full state of each idle slot whose prompt has changed into the host cache (`server-context.cpp:2641-2656`). With 10 separate slots this duplicates state that is still on the GPU.
  - It isn't needed: a slot handed to a different game is already saved before reuse (`server-context.cpp:1768-1786`).
- **Size:** one game's state at about 100K tokens is roughly 3.5–4 GB (f16 KV at 24–33 KiB per token, about 113 MB of recurrent state, and up to 8 checkpoints).
  - `-cram 32768` holds about 8 of them, and the cache drops the oldest entry when full (`server-task.cpp:1761-1769`, `update()`).
  - So games that lose their slot also lose their cached state. Each time one comes back, it has to reprocess 60–116K tokens. At an estimated 1.5–4K tok/s that is about 15–75 s, and it slows every other stream meanwhile.
- **Local evidence:** with only 2 slots, 6 minutes and prompts under 14.5K tokens, `gguf-local/serve-run1.log` shows 33 idle-slot saves and 66 state saves (5.1 GB in total).
- **The smoke doesn't test this:** it runs with `-cram 8192`, so the final run's memory setup is never exercised.
- **Fix,** in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gguf/serving/kaggle/gguf_serve.py`:
  - Add `--no-cache-idle-slots` to `launch_args` (line 374ff).
  - Set `CACHE_RAM_MIB` (line 68) to about 80 GiB (`81920`) for both smoke and final. The host has 176.9 GiB (from the `[sys]` lines of `v4-panel-control`), so this still leaves room for shard 2 in the page cache and stays under the 160 GiB RAM gate.
  - Update both golden files (`kaggle/tests/golden/gguf_launch_{smoke,final}.json`) and CUSTOM.md §8.
  - Have `gguf_report.py` count the "cache size limit reached" and "removing oldest entry" warnings in `serve.log`.

**2. Two of `gguf_report.py`'s go/no-go checks can fail on a healthy run (`/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gguf/kaggle/gguf_report.py`).**
- **(a) Timeout threshold:** `FULL_TIMEOUT_S = 600` (line 40), but `build_fable_gguf.py` sets `analyzer_timeout = 1800`.
  - Each request's read timeout is the smaller of 1800 s and the game's remaining time (`solver.py:536-550`).
  - So any timeout at a game's end that falls between 600 and 1800 s is counted as a full analyzer timeout.
  - In the 15-minute smoke games a real 1800 s timeout can't happen at all, but a slow request started in the first 5 minutes would flip `no_full_analyzer_timeouts` to FAIL.
  - Fix: use 1800 and count `abs(t - 1800) < 1`.
- **(b) Reasoning check:** `reasoning_95pct` counts the Fable layer's acknowledgement turns. These are assistant messages with no reasoning by design (`ACK_TEXT`, "Noted: these are harness-computed facts…").
  - On F001's healthy SGLang output (`scratchpad/f001-v2-out`), all 8 assistant turns without reasoning were these acknowledgements. The share was 262 of 270, which is 0.970, only 2 points above the 0.95 floor.
  - A slower 15-minute smoke has fewer turns per game, so the same acknowledgements can push it below 0.95.
  - Fix: in `snapshot_turns`, skip `[ASSISTANT]` blocks that start with `ACK_TEXT`.
- Add a test for each fix.

**3. If llama-server crashes, the final run is lost (a robustness gap, not a spec defect).**
- `MetricsPoller` only records. If llama-server exits during the 2-hour run, every game stops after `ARC3_MAX_ANALYZER_FAILURES` (10) consecutive failures and about 2.4 GPU hours are wasted. The 10-slot code in b11476 is only days old, so this is a real risk.
- Fix: in the poller (`gguf_serve.py:578-610`), when `proc.poll()` is not None, relaunch with `TIMINGS["argv"]`, wait for `wait_healthy`, and log the restart. The files will still be in the page cache, so a restart should take about 2–5 minutes.
- Optionally raise `ARC3_MAX_ANALYZER_FAILURES` (for example to 40) in the GGUF notebooks only, so games survive that window.

**4. Minor: one edit in cell 4 does nothing.**
- `SERVER_STARTUP_TIMEOUT = 45 * 60` (`kaggle/build_fable_gguf.py` `ENV_EDITS`) is never read by the new cell 14. `bring_up` uses its own per-stage budgets, which add up to 51 minutes.
- The docs describe it as if it limited startup. Either drop the edit and the doc line, or cap the total time of `bring_up` with it.

## Checked and correct
- **Model card match:**
  - Both shards are placed under their original HF names, so llama.cpp derives shard 2 from shard 1. The clothespin fallback is symlinked under the same name.
  - `check_headers` requires `expert_count` 256, `split` 0/2 and 1/2, and `per_layer_token_embd.weight` (IQ4_NL, the only tensor in shard 2). There are no masks and no Intel n-gram table.
  - The mmproj is passed with `--mmproj`, and `/props` confirms vision is on.
  - The embedded template's sha `12827f24…` equals the real shard-1 header (9,993 chars). Its default effort is xhigh, and the harness sends no effort, so every request runs at xhigh.
- **Engine flags:**
  - Every flag in the argv exists in the b11476 source.
  - `--lazy-mode auto` applies to qwen4exp's n-gram table: it is created lazy (`TENSOR_READ_LAZY` in `qwen4exp.cpp:214`), it is over the 4 GiB threshold, and CUDA on this discrete GPU supports mmap. So the "enabling prefetch" line `check_serve_log` waits for will be logged.
  - `GGML_CUDA_CCCL_VERSION` is a real CMake option.
  - The amd64 driver stub package's sha256 matches the pinned value.
- **Engine build:** a full sm_120a build passed in a container (aarch64, `-j4`, 661 s, CUB top-k 300403, GLIBC 2.34 and GLIBCXX 3.4.30). That run used the script before its last edit, so the new manifest-checks step has still not run in a container.
- **Notebooks:**
  - All 7 push folders rebuild byte-for-byte from HEAD.
  - All pass `nbcheck`.
  - The 145 runner-gguf tests (incl. the staging suites) pass, run niced.
  - The staged shard-1 manifests pass `--check-manifests`.
- **GPU notebooks:**
  - Internet is off, the competition is attached, the image is pinned and the machine is the RTX Pro 6000.
  - `kernel_sources` lists all 8 notebooks the run needs.
  - Only cells 4, 14 and 18 differ from F001; cells 2 and 6 (harness patch and Fable layer) are identical.
  - The context-overflow patch's anchor occurs once after the harness patch, and nothing imports the harness before cell 16.
- **Final run:** 25 games (`demo_excluded_games = []`), concurrency 25, 121 minutes per game (control used `int(121*60)`), 10 streams to match `-np 10`. Each slot gets 139,264 tokens, the same as SGLang.
- **Smoke:** the balanced 10-game panel, 15 minutes per game, all 10 slots busy.
- **Server contract:** every row of `server-contract.md` is covered.

Files reviewed:
- `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gguf/serving/kaggle/gguf_serve.py`
- `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gguf/kaggle/gguf_report.py`
- `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gguf/kaggle/build_fable_gguf.py`
- `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gguf/serving/gguf/build_llama_server.sh`

Scratch cell extracts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/rev-gguf/`. I made no commits or pushes.
