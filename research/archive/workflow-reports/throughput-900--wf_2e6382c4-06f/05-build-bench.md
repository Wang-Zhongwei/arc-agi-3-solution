# throughput-900: build-bench

Workflow run `wf_2e6382c4-06f`, phase Build bench. Find serving levers to reach 900+ tok/s peak with W4A16 Flash-Next, rank them, and build a Kaggle serving benchmark arm

---

The bench is built and tested on branch `arm/serving-bench`, but nothing has run on a GPU yet, so there are no speed numbers. All 33 unit tests pass in 8.5s, and dry builds of both benches and the control arm succeed. I did not push or run Kaggle.

For your 900+ goal, the most direct config is C7: Franzen's launcher with 14 slots instead of 10, and no lossy changes. Our earlier 14-stream run (fz-streams14) already logged a median of 849 tok/s, p90 1026 and max 1122 at 14 running requests. That's the slot count behind the 900+ peaks others report. C7 runs second so the time budget can't drop it. Every config also reports its peak window and the share of windows at 900 or above.

**Branch:** `arm/serving-bench` (from `arm/custom-serving`), worktree `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/serving-bench`.

**git log** (each commit ends with the two required trailer lines):
```
a89a949 Document serving benchmarks
241f044 Test the bench plan, requests, log parsing, client and build
d134ea5 Build serving benchmark notebooks with --bench
b4ea6fc Define the throughput plan's benchmark configs
bee456d Add bench patches that widen the SM120 low-M GEMM to M=64
537e6bc Add the serving benchmark driver
170471f Add a prompt corpus of real harness conversations for serving benchmarks
408db4b (base) Document every layer of the serving stack we can now modify
```

**Tests:** run from `<worktree>/ARC3-Inference` with `.../.venv/bin/python -m pytest -q ../kaggle/tests/test_bench.py`. The client and the bench driver run end to end against a small fake server. Normal arm notebooks are unchanged: control keeps fingerprint `1fa948b10cf43854`.

**How to launch** (from the worktree root):
- `python kaggle/build_notebook.py --bench smoke` builds `kaggle/out/v4-bench-smoke/`. It runs C0 with short phases, about 15 minutes, to check the machinery.
- `python kaggle/build_notebook.py --bench levers` builds the full session (about 582 KB). Add `--push` to push `mellamoa/v4-bench-<name>` and record it in `runs.json`. Benches are recorded as `"kind": "bench"` with `batch_open: false`, so they never join or block an arm batch. The duplicate-fingerprint check still applies.
- Results go to `/kaggle/working/bench_results.json` and `bench_results.md`, rewritten after every phase. Each config's serve.log is in `bench/<config>/`, and C0's profile trace is in `bench/C0/profile/`.
- I'd push `smoke` first, then `levers`.

**What each config does:** it launches SGLang through the same launcher text the arms use, with that config's overrides. Patches go onto a fresh copy of the installed SGLang. It waits until healthy, sends one test request, runs its timed load phases, then kills the server.

**The load:**
- Clients replay real harness conversations from `kaggle/bench_corpus.json.gz`: 16 games from fz-arms-v4-control, built by `kaggle/make_bench_corpus.py`. Speculative acceptance depends on what the model writes, so filler text would measure a different accept length.
- All streams share the 5k-token system prompt, and each turn extends the previous one, so the prefix cache is exercised.
- Requests match the harness: T 0.7, top_p 0.95, top_k 20, thinking on, the `python` tool. They are identical across configs.
- On the plan's temperature conflict: the harness code does send T 0.7. The T 1.0 in serve.log is only the server's default for requests that omit it.
- Phases:
  - `warmup10`: 60s at 10 streams, not reported.
  - `clean10`: 240s at 10 streams, outputs of 1,536–2,048 tokens.
  - `mixed10`: 180s, non-streamed like the harness, a quarter of requests cold.
  - Plus sweeps at 1, 4, 7 and 8 streams, `over12` (12 clients on 10 slots), `clean14`/`over16`, and a profile phase.

**Configs (`kaggle/bench.json`, bench `levers`, run in this order):**

| Config | Change | Note |
|---|---|---|
| C0 | None (Franzen's launcher) | Plus sweeps, 12-client phase and profile trace |
| C7 | 14 slots (`MAXREQ`/`CUDAGRAPH_MAXBS` 14, `MAMBA_CACHE` 84) | Lossless |
| C1 | Relaxed MTP acceptance 0.5/0.5 | Changes outputs, needs a score check |
| C2 | Relaxed MTP acceptance 0.3/0.3 | Changes outputs, needs a score check |
| C3 | GEMM patches `m64.py` + `nosplitk.py`, plus shape logging | Lets bs 9/10 use the faster kernel |
| C4 | FP8 port + `m64.py` | **Disabled**: the FP8 port isn't written yet |
| C5 | Fastest of C0/C3/C4 plus fastest of C0/C1/C2 | Picked during the run by clean bs10 median |
| C6 | C5 plus `OMP_NUM_THREADS=1`, overlap plan stream, CUDA graphs at every bs 1–10 | |

- **Budget:** 150 minutes from notebook start. The estimate is about 142 minutes plus setup. If it runs behind, C6 is dropped, then C2.
- **C4:** to enable it, add `serving/bench_patches/fp8-port.py` and set `enabled: true`.
- **`min_tokens`:** if this SGLang fork rejects it, the bench runs that config with natural output lengths and records a warning instead of failing.

**Main files:**
- `serving/kaggle/bench.py`
- `kaggle/bench.json`
- `kaggle/build_notebook.py` (adds `--bench`)
- `kaggle/make_bench_corpus.py`
- `kaggle/bench_corpus.json.gz`
- `serving/bench_patches/` (`m64.py`, `nosplitk.py`, `README.md`)
- `kaggle/tests/test_bench.py`
- `kaggle/README.md` (new "Serving benchmarks" section)
