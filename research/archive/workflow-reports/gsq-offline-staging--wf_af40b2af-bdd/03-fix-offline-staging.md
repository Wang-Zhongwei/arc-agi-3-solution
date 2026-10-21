# gsq-offline-staging: fix:offline-staging

Workflow run `wf_af40b2af-bdd`, phase Fix. Make the F001-on-GSQ test run offline: CPU staging notebooks that fetch the Coder GGUF shard 1 in parts as notebook outputs, and an offline test notebook that mounts them; review and fix

---

Both review findings were real and are fixed. The branch is pushed to `origin/arm/runner-gsq` (`9562eb7..cb35752`), and nothing was pushed to Kaggle.

**Finding 1, the 15-minute reassembly limit failing on a hung read: fixed (`90715a9`).**
- **Reproduced first:** with a FIFO as part 1 and a 1 s deadline, `reassemble()` was still blocked after 8 s. After the fix it raises `TimeoutError("reassembly past its budget waiting on a read of part 1 (…)")` at 6 s: the 1 s budget plus a 5 s wait for the reader thread.
- **The change:**
  - `_prefetch` now takes the deadline and waits for each block with `q.get(timeout=…)`. If nothing arrives by the deadline it raises `TimeoutError` naming the part and its path.
  - The reader thread stops at its next queue write. On failure the queue is emptied once and the thread gets a 5 s join (`READER_JOIN_S`). A thread still stuck in a read is logged and left behind.
  - `reassemble()` now always closes the block reader, even when it fails.
- **New test:** `test_reassemble_stops_at_its_deadline_while_a_read_hangs` covers a FIFO that hangs on open and one that hangs after its first 1000 bytes. Both raise `TimeoutError` in under 4 s.

**Finding 2, the staging download giving up long before its 3-hour limit: fixed (`dbcb09d`).**
- **The change:**
  - When `stage()` has a deadline it now passes `fetch_range(rounds=None)`, which means "keep starting rounds until the deadline". Without a deadline it still stops after 3 rounds.
  - Rounds start 30 s apart. The pause doubles while rounds fetch nothing, up to 5 min, and never runs past the deadline.
  - Each retry waits at least the `Retry-After` of a 429 or 503 (seconds or an HTTP date), capped at 15 min.
- **One addition beyond the review:** retrying until the deadline would also make an unfixable error loop for 3 hours. So answers no retry can fix now stop the run at once with a new `PermanentError`. These are: Range ignored, a wrong range, a wrong file size, and HTTP 400, 401, 404, 410 or 416. 403 and 5xx are still retried.
- **New tests:**
  - A 429 outage with `Retry-After: 7` that lasts 6 rounds: `rounds=3` fails; `rounds=None` finishes with round pauses of `[30, 60, 120, 240, 300, 300]`.
  - The round pauses stop at the deadline.
  - `PermanentError` is raised after at most one request per thread, for a 404, an ignored Range and a wrong file size.
  - `Retry-After` is read correctly in seconds and as a date.
  - At the `stage()` level, a 120-request outage that 3 rounds could not survive now completes.

**Docs (`cb35752`):** `serving/CUSTOM.md` section 7 and `kaggle/README.md` now describe the retry schedule, the immediate stop, and the reassembly limit on a hung read.

**Optional suggestion not done:** I didn't build the CPU notebook that would hash the staged parts before any GPU time; the reviewer marked it optional.

```
cb35752 Document the staging retries and the reassembly guard on a hung read
dbcb09d stage_gsq: retry the staging download until its deadline
90715a9 stage_gsq.reassemble: stop at the deadline while a part's read hangs
91a565f Document the staged GGUF and the offline F001 x GSQ test
86b28f6 build_fable_gsq: reassemble shard 1 from the staging notebooks, internet off
fefde27 stage_gsq.locate: resolve the input root at call time
e041ed9 Add the staging notebook generator for GGUF shard 1
2208fcf Add stage_gsq.locate and reassemble for the offline test notebook
```
All 3 new commits end with the required trailer lines.

**Tests** (niced, one file at a time, using `scratchpad/gsqvenv`):
- `kaggle/tests/test_stage_gsq.py`: 52 passed, 1 skipped (was 43 passed).
- With `GSQ_HTTP_TESTS=1`, the real 2 MiB read from Hugging Face: 1 passed.
- `kaggle/tests/test_build_gsq_stage.py`: 6 passed.
- `kaggle/tests/test_build_fable_gsq.py` (with `FABLE_F001_NOTEBOOK` set): 12 passed.

**Regenerated notebooks** (from `cb35752`):
- `nbcheck.py` passes on all 4, and every `kernel-metadata.json` is valid JSON and unchanged from the previous build.
- The staging notebooks are about 17 KB each. The test notebook is 918,855 bytes, under the 950,000-byte limit.
- The code embedded in all 4 notebooks matches HEAD exactly.
- The test notebook still differs from F001 only in cells 4 and 14. Cells 2, 6 and 18 and the notebook metadata are byte-identical. Compared with the previous build, only cell 14's embedded code and commit hash changed.

**Push commands, in order** (`S=/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad`):
1. Push the three staging notebooks; they can run at the same time. Expect about 5–12 min each: 1–2 to start, 1–5 to download 9.87 GB, about 0.5 to hash, 2–5 to save the output. An HF outage now makes them wait, up to the 180-minute limit, instead of failing.
   - `kaggle kernels push -p $S/push-gsq-stage-p1`
   - `kaggle kernels push -p $S/push-gsq-stage-p2`
   - `kaggle kernels push -p $S/push-gsq-stage-p3`
2. Repeat `kaggle kernels status mellamoa/gsq-coder-stage-p1` (and p2, p3) until all three show COMPLETE.
3. Fetch only the manifests:
   - `kaggle kernels output mellamoa/gsq-coder-stage-p1 -p $S/gsq-stage-manifests/p1 --file-pattern 'manifest\.json$'`
   - the same for p2 and p3.
4. From `$S/wt/runner-gsq`, check them: `python kaggle/build_gsq_stage.py --check-manifests $S/gsq-stage-manifests --parts 3` (seconds).
5. Only after all three are COMPLETE: `kaggle kernels push -p $S/push-fable-f001-gsq`.
   - Expected total about 50–70 min. Locating the parts takes seconds and reassembly 1–4 min (limit 15), then conversion (12 max), launch (25 max), smoke test (10 max) and 25 min of games.
   - The time limits cap it at about 100 min. This total is my estimate; the conversion and launch times haven't been measured on Kaggle.
6. `kaggle kernels output mellamoa/v4-fable-f001-gsq -p runs/kaggle/v4-fable-f001-gsq/v1`

Changed files are in `$S/wt/runner-gsq`:
- `serving/kaggle/stage_gsq.py`
- `kaggle/tests/test_stage_gsq.py`
- `serving/CUSTOM.md`
- `kaggle/README.md`
