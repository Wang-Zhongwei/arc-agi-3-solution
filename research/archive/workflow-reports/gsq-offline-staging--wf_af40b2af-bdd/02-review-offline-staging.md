# gsq-offline-staging: review:offline-staging

Workflow run `wf_af40b2af-bdd`, phase Review. Make the F001-on-GSQ test run offline: CPU staging notebooks that fetch the Coder GGUF shard 1 in parts as notebook outputs, and an offline test notebook that mounts them; review and fix

---

Two defects found, both in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gsq/serving/kaggle/stage_gsq.py`. Everything else on the checklist passes.

**1. (Medium, reproduced) The 15-minute reassembly limit does not hold if a read from `/kaggle/input` hangs.**
- `reassemble()` checks its deadline only after each block arrives. Its consumer waits in `_prefetch` with `q.get()`, which has no timeout.
- If the reader thread is stuck inside `f.read()`, the notebook waits forever. The generator's `finally` also loops `while thread.is_alive()` with no limit, so even a `TimeoutError` could not get out.
- I reproduced it with a FIFO as the part file and a 2 s deadline: the call was still blocked after 10 s.
- On Kaggle, one hung mount read would hold the RTX Pro 6000 until the session limit instead of failing at 15 minutes.
- Fix:
  - Pass the deadline into `_prefetch` and use `q.get(timeout=max(0.1, deadline - time.time()))`.
  - Turn `queue.Empty` into `TimeoutError("reassembly past its budget …")`.
  - In `finally`, set `stop`, drain the queue once, then `thread.join(timeout=5)` and leave the daemon thread behind.
  - Add a test that reassembles from a FIFO part with a 1 s deadline and expects `TimeoutError` within a few seconds.

**2. (Low–medium) The staging download gives up long before its 3-hour limit.**
- `stage()` calls `fetch_range` with the default `rounds=3` and there is no pause between rounds.
- Each chunk gets 6 attempts with about 62 s of backoff in total. So when only a few chunks are left, an HF outage or a run of 429s lasting more than about 3 minutes fails the run.
- On failure `stage()` deletes the partial file (`keep_partial=False`), and a re-push starts with an empty disk anyway. That means re-downloading all 9.87 GB.
- The report's claim that 429s are "handled by retries and extra rounds within the 3-hour budget" is therefore not true.
- Fix:
  - Keep starting new rounds until the deadline (for example `rounds=None` meaning until the deadline).
  - Pause between rounds, starting at 30 s, doubling, capped at 5 min.
  - Honour `Retry-After` on 429/503.
  - Add a test where the stand-in server fails for longer than 3 rounds and then recovers.

**Optional (not a defect).** The "mount read speed" risk is probably small. Our F001 v2 and panel logs show `/kaggle/input` model and dataset mounts reading at about 450–770 MB/s (185.4 GB in 242–411 s on 3 threads; 6.7 GB in about 9 s on 16 threads), against the 33 MB/s the limit needs. That is model and dataset mounts, not notebook outputs. If you want the whole-file sha256, the real mount layout and the read speed checked before any GPU time, a small CPU notebook could do it. It would have the 3 staging notebooks as `kernel_sources`, no GPU, no competition and internet off, and run `locate()` plus a hash-only pass. `reassemble()` would need a mode that skips writing for that.

**What checks out:**
- **Byte ranges:** the 3 parts cover shard 1 exactly: (0, 9,869,197,312), (9,869,197,312, 9,869,197,312) and (19,738,394,624, 9,870,051,872). They sum to 29,608,446,496 and each is under 15 GB. Two parts would be 14.80 GB each.
- **Downloads:** any answer other than a 206 for exactly the requested range and total size is refused before the body is read. Downloads resume from the state file, and only the `.partK` file and `manifest.json` are left in `/kaggle/working`.
- **Manifests:** `check_manifest` checks the format, file, size, whole-file sha256, split, offset, size, name and the shape of the part's sha256.
- **Finding the parts:** `locate()` tries `/kaggle/input/notebooks/<owner>/<slug>`, then `/kaggle/input/<slug>`, then searches by name up to depth 5. It checks each manifest and the size on disk, and fails in seconds naming every missing part.
- **Plan check on part 1:** this is valid because the converter's `--plan-only` reads only the header, and the header sha256 is unchanged.
- **Embedded code:** the code inside the staging notebooks and the test notebook's cell 14 is byte-equal to HEAD `91a565f`.
- **Staging metadata:** no GPU, no machine shape, internet on, nothing attached.
- **Test metadata:** identical to F001's except the id, title, code file and `kernel_sources`, which lists exactly `mellamoa/gsq-coder-stage-p1..p3`. That keeps internet off, the competition, both datasets, both models, the RTX Pro 6000 and the pinned docker image.
- **Unchanged parts of the test:** only cells 4 and 14 differ from the source. Cells 2 (harness patch), 6 (Fable layer) and 18 (panel) are byte-identical, and so is the notebook-level metadata. The time limits are the same except download→assemble (15 min).
- **No internet needed:** nothing left in the test's code needs the network.
- **Notebook checks:** all four notebooks pass `nbcheck`, and the test notebook is 916,755 bytes, under the 950,000-byte limit.
- **Tests:** I re-ran them niced, one file at a time. `test_stage_gsq.py` gave 43 passed, 1 skipped; `test_build_gsq_stage.py` gave 6 passed.
- **Commits and CLI:** all six commits end with the required trailer lines, and `kaggle kernels output --file-pattern` exists in CLI 2.2.4.
