# gsq-coder-gguf-runner: fix:runner-gguf

Workflow run `wf_0ae4547f-80a`, phase Fix. Serve the ISTA GSQ-RCO Coder GGUF the way its Hugging Face repo documents (llama.cpp or Strata, both shards, 256 experts, mmproj), validate the harness contract locally, build staging/engine/test notebooks and the 2-hour final-run notebook

---

All four review findings are fixed and every push folder is rebuilt. The branch is on GitHub, and the PR is https://github.com/adsdemaybe/daniel-franzen-sol/pull/71 (`arm/runner-gguf` into `arm/runner-gsq`). Nothing has run on Kaggle yet: there is no score, and the smoke and final runs are still to push.

**Status of each finding**

| # | Finding | Status | Commit |
|---|---|---|---|
| 1 | Host prompt cache too small, filled with idle-slot copies | Fixed: `--no-cache-idle-slots` added and the cache raised to 80 GiB (`-cram 81920`) in both smoke and final. I confirmed in the b11476 source (`server-context.cpp:2649` and `:1768-1786`) that the copies are real and not needed. The report now counts cache evictions, restores, oversized states and idle-slot copies. Docs updated. | `4779d26`, `70522fb` |
| 2a | Timeout check could fail a healthy run | Fixed: only a read timeout of 1800 s (±1) counts as a full analyzer timeout. A test ties that number to the value the notebook sets. | `7e6f0f5` |
| 2b | Reasoning check counted the Fable layer's acknowledgement turns | Fixed: those turns are counted separately. On F001's healthy run the share goes from 262/270 to 262/262. The real-source test checks the text matches the Fable layer's. | `af47c78` |
| 3 | A llama-server crash would lose the final run | Fixed: the metrics poller checks the server every 5 s and relaunches it with the same command, at most 3 times, allowing 15 min each to come back healthy. Cell 14 sets `ARC3_MAX_ANALYZER_FAILURES=40` (about 16 min of failures, up from about 4), so games outlast a relaunch. A test kills a fake server and checks it comes back. The report gets a `no_server_relaunch` check, and its token totals stay correct when a relaunched server starts its counters at 0. | `e3c778c`, `e5ca24d` |
| 4 | The `SERVER_STARTUP_TIMEOUT` edit in cell 4 did nothing | Fixed: edit dropped. Only F001's old serving cell read it; the new cell 14's stage budgets limit startup (51 min in all). | `a625ec3` |

**Tests and notebooks**
- **Tests**, run niced: the GGUF and staging suites give 153 passed and 2 skipped (the opt-in HF API tests), and the launch golden tests pass 29. I did not rerun the whole `kaggle/tests` suite.
- **Notebooks:** all seven push folders were rebuilt from HEAD `a625ec3` and copied into the scratchpad push folders. The engine and staging notebooks differ from the previous build only in the embedded git sha; I decoded the payloads to check. The smoke and final notebooks contain all the fixes, and all seven pass `nbcheck`. Sizes: smoke 744,474 bytes, final 744,356, the rest about 14–20 KB.
- **GitHub:** `arm/runner-gsq` was pushed first (a fast-forward of the 3 local commits), then `arm/runner-gguf`.

**Commits on top of the earlier 15**

```
a625ec3 build_fable_gguf: drop the startup-timeout edit nothing reads
e5ca24d gguf_report: gate on llama-server relaunches
e3c778c Relaunch llama-server if it exits during the games
70522fb gguf_report: count the host prompt cache's evictions and restores
af47c78 gguf_report: leave the Fable layer's acknowledgements out of the reasoning share
7e6f0f5 gguf_report: count only 1800 s read timeouts as full analyzer timeouts
4779d26 Keep idle slots out of an 80 GiB llama-server prompt cache
```

**Push order** (each Kaggle push needs approval; folders are in the scratchpad)

| # | Notebook (folder) | Machine | Expected minutes |
|---|---|---|---|
| 1 | `mellamoa/gguf-engine-build` (`push-gguf-engine-build`) | CPU, internet on, pinned F001 image, no GPU quota | 20–35 (75-min cap). If it exits with code 3 within seconds, Kaggle used another image: rebuild with `--gpu` or `--image latest --bundle-glibc`. |
| 2–4 | `mellamoa/gsq-coder-shard2-p1`, `-p2`, `-p3` (`push-gguf-stage-shard2-p1..p3`) | CPU, internet on | 5–10 each |
| 5 | `mellamoa/gsq-coder-mmproj-p1` (`push-gguf-stage-mmproj-p1`) | CPU, internet on | 2–4 |
| – | Do not re-push `gsq-coder-stage-p1..p3`; shard 1 is already staged. | | |
| 6 | Fetch only the manifests, then run `build_gguf_engine.py --check-manifest`, `build_gsq_stage.py --check-manifests --file shard2`, and the same with `--file mmproj` | local | about 1 |
| 7 | `mellamoa/v4-gguf-smoke` (`push-gguf-smoke`), then `kaggle/gguf_report.py` on its outputs; every go/no-go check must pass | RTX Pro 6000, internet off | 12–18 startup + 15 play, about 35 GPU minutes |
| 8 | `mellamoa/v4-gguf-final` (`push-gguf-final`), with no other GPU session running. Score it and run `make significance` against control's 46.25. | RTX Pro 6000, internet off | about 15 startup + 121 play, about 2.4 GPU hours |

**New risks from the fixes**
- **Host RAM:** with the 80 GiB cache, RAM use should stay under the report's 160 GiB limit (F001 peaked at 113 of 176.9 GiB). But llama.cpp's own RAM use on Kaggle has never been measured, so the smoke's RAM check is the real test.
- **Startup time:** the server takes 12–18 min to come up before the first game action. That is fine for this offline run. An actual competition submission would need the warmup RESET to happen before the server starts.

The laptop is on its final battery warning, so I ran nothing heavier than these tests.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/`:
- `wt/runner-gguf/` (the worktree)
- `push-gguf-engine-build/`, `push-gguf-stage-shard2-p1/`, `-p2/`, `-p3/`, `push-gguf-stage-mmproj-p1/`, `push-gguf-smoke/`, `push-gguf-final/`
- `regen-gguf-a625ec3/new/` (the rebuilt copies)
- `pr_runner_gguf.md` (the PR description)
