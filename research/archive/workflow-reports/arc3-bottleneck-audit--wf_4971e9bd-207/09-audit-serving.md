# arc3-bottleneck-audit: audit:serving

Workflow run `wf_4971e9bd-207`, phase Audit. Audit the whole Franzen harness (inference, serving, TAAF, notebook, configs) for bottlenecks; verify each; classify DEFINITELY vs MAYBE for GitHub issues

---

```json
{
 "findings": [
  {
   "title": "Prefix cache drops the previous turn's whole output in ~45% of turns (Mamba checkpoint all-or-nothing)",
   "location": "notebook cell 12 launch (notebook_cells.py:10435-10436 MAMBA_RADIX=extra_buffer, GDN_MTP_CACHE_MODE=none; :10693 --mamba-track-interval 64; :10709-10716 NEXTN spec) + main:serving/patches/patch-sglang-mamba-final-prefill-v2.py:27 (sparse prefill: only cold-start branch + final checkpoint; also main:serving/build_bundle_pennyroyal.sh:322,452)",
   "resource": "prefix-cache",
   "problem": "Chaining consecutive same-stream requests in serve.log shows the radix hit is bimodal. In ~55% of pairs cached_input_len == floor64(prompt1+output1), so the previous output is reused. In ~45% it is exactly floor64(prompt1), the end of the previous PROMPT, so all of the previous generated output (mean 2.1k tokens) is re-prefilled even though it is present, with a similar token count, in the next prompt. There are essentially no partial hits (4 of 1358). That is the signature of Mamba states existing only at the prompt-end and decode-end checkpoints: if the decode-end state is missing or any token in the re-rendered assistant message differs from the generated ids, everything after prompt-end is lost. The miss rate rises with output length (24% for <300 tokens, 58% for >4k). It is 92% when the next request arrives <0.5s after the previous finished, 71% at 0.5-1s, and ~33% at >=3s, which hints at a commit or visibility lag or at message re-rendering. About 99% of replies end in a tool call (2274 of ~2300 in the transcripts), so these are ordinary tool-call turns.",
   "evidence": "fz-arms-v4-control/serve.log, ReqTimeStats chained by entry/end time (my script chain.py in scratchpad/serving). 1358 uniquely matched pairs: 742 full, 613 none; cached2-floor64(inp1)==0 in 612 of 613 none cases. Re-prefilled previous-output tokens in the matched subset: 1.30M, which is 12.3% of all 10.6M uncached input tokens; extrapolated to all requests about 2.2M. Replicated: ledger-on-trim 1.22M (46% none), streams14 1.37M (49%), streams12 1.24M (46%), audit60k 1.57M (52%). The none rate is similar with 60, 72 and 84 Mamba slots, so it is not slot pressure. Pure prefill is ~10.5k tok/s (WRITEUP.md:108), so this is ~125-210 s of GPU, 1.7-2.9% of the 7,334 s wall. Prefill also stalls decode of the other 9 streams. Competition-scale (8.8h): ~9-15 min of GPU.",
   "estimated_cost": "~2-3% of wall-clock/GPU (about 1.3-2.2M prefill tokens per 2h run, ~10-15 min over the 9h run), plus ~0.2s added latency per affected turn",
   "fix": "Discriminate engine vs round-trip cause: replay one stream's request pair against a live server twice, once with the assistant message echoed byte-exact (same reasoning_content and tool_calls text) and once as the harness sends it, and compare usage cached_tokens. If it is the engine: make cache_finished_req always materialize a page-aligned decode-end Mamba state, either by cutting the last speculative verify so seq_len lands on a 64 multiple (costs <1.5% accept length) or by a <=63-token final extend; or add intermediate Mamba checkpoints inside the decode region so a mismatch becomes a partial hit. If it is round-trip: fix the whitespace/tool-call re-render difference between generated text and the chat template output in the harness.",
   "proposed_class": "DEFINITELY"
  },
  {
   "title": "Prefill is ~15% of wall-clock and serialized with decode; 48% of uncached tokens are 55k-token trim re-prefills",
   "location": "notebook_cells.py:10688-10689 (--chunked-prefill-size 8192, --max-prefill-tokens 16384; enable_mixed_chunk False and prefill CUDA graph disabled in serve.log:'Breakable CUDA graph is incompatible with multimodal model'); trims driven by notebook_cells.py:10204-10205 (context 128k, drain 58k)",
   "resource": "gpu-throughput",
   "problem": "Uncached input is 10.6M tokens per 2h run (166M input, 93.6% hit). At ~10-11k tok/s prefill (47.4k tokens in 4.8s with 1 running) that is ~1,000 s, about 14-15% of wall (WRITEUP.md:110 says 15.25%). Decode lines whose interval contained a prefill batch are 34.6% of all decode lines and run at 526 tok/s versus 768 for clean intervals (-31%). So the 9 other streams stall while any stream prefills. 92 trim re-prefills (>20k uncached, input 48-64k) account for 5.09M tokens (48% of all uncached tokens, 458 s of initial_prefill plus 132 s of queue). Another 77 partial misses (8-13k uncached) account for 0.8M. This is the gap between decode-only throughput (741 median, 793 at bs=10) and the 611 harness tok/s.",
   "evidence": "fz-arms-v4-control/serve.log (misses.py, analyze2.py). Time-weighted GPU split reconstructed from ReqTimeStats and decode-log intervals. No run varied --chunked-prefill-size, --enable-mixed-chunk, --prefill-decode-interval or chunk=16384 (all server_args identical across runs, apart from the stream-count and spec/KV arms).",
   "estimated_cost": "~14-15% of wall-clock (~1,000 s per 2h run, ~75 min per 9h run) is prefill. Part is intrinsic to the trim design; a 1.3-1.5x faster or better overlapped prefill would be worth +4-5% tokens.",
   "fix": "A/B (a) --chunked-prefill-size 16384 (max-prefill already 16384), (b) --enable-mixed-chunk if it is compatible with NEXTN and hybrid Mamba, (c) a nonzero --prefill-decode-interval, (d) profile why eager prefill is only ~10k tok/s (Marlin W4A16 MoE at M=8192, FlashInfer GDN prefill, QSA) rather than leaving prefill graphs off. On the harness side, reducing trim frequency (larger drain/hysteresis) trades directly against this cost.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "10 streams in 10 slots: batch averages 8.95; only 44% of time at full batch",
   "location": "notebook_cells.py:10421 MAXREQ=10, :10248 ARC3_MAX_ACTIVE_STREAMS=10, :10691 cuda-graph bs list",
   "resource": "gpu-throughput",
   "problem": "The server never has a request waiting to fill a slot freed while a stream is in harness work (python tool, env step, rendering, HTTP). Time-weighted running-req: bs10 44.2%, bs9 29.4%, bs8 13.7%, bs7 7.1%, bs<=6 5.6%, mean 8.95. Each stream is on-server only 89.5% of the time (about 3.3 s of harness time per 28.3 s request). Clean decode throughput is 826 tok/s at bs=10 versus a time-weighted ~778, so ~6% of decode capacity is idle. The previous 12/14-stream arms confounded streams with context cuts (110k/96k) because the shared 1.011M-token KV pool is nearly full at 10 streams: peak 1,002,816 tokens (99%) in several 10-stream runs, with no retractions logged.",
   "evidence": "serve.log time-weighted distribution (tw.py): control 8.95 mean, streams14 12.9 mean. Clean tok/s by bs (control): 7: 689, 8: 746, 9: 792, 10: 826. Step time is smooth (27.7, 29.2, 30.7, 32.3 ms at bs 7-10), so the batch is compute/bandwidth saturated at the top and each extra request costs ~1.5 ms/step. Max full-token usage 0.99 in control, experts16-full and ledger-on-trim.",
   "estimated_cost": "up to ~6% decode throughput (about 36 tok/s) if the batch were always full; the score benefit is unproven given the 12/14-stream results",
   "fix": "Test oversubscription with context unchanged (128k/drain 58k): ARC3_MAX_ACTIVE_STREAMS=11 with --max-running-requests 11 and mamba cache 66, or let the gate admit the next game's request as soon as a stream starts a long python tool call. KV is the binding constraint, so only do this together with a KV-neutral change, and watch for retractions. Treat as low-priority upside.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "A server crash anywhere in the 9h run ends all games within ~100 s; GPU headroom is ~1.3-1.5 GB and an OOM was already seen",
   "location": "notebook_cells.py:10420 MEMFRAC=0.96, :10729 Popen(..., start_new_session=True) (never polled after startup, only :10741 during startup), :10701 --watchdog-timeout 1800; main:ARC3-Inference/inference/framework/solver.py:105 and :694 (ARC3_MAX_ANALYZER_FAILURES=10, retry backoff 10s)",
   "resource": "reliability",
   "problem": "Nothing supervises SGLang after startup. The harness is deliberately fail-fast: 10 consecutive analyzer failures at a 10 s backoff end every game in ~100 s, so a mid-run crash forfeits the rest of the run (hours). Memory margin in the competition config is thin: init avail 4.25 GB, nvidia-smi peaks 94.1-94.3 of 95.6 GB, i.e. ~1.3-1.5 GB free. In fz-spec-adaptive (3.22 GB avail after init, ~1 GB less) the scheduler OOM'd in an eager prefill: 'expandable_segments: memory mapping failed with OOM ... free: 22085632', in qwen4_exp.py PLE forward via model_runner._forward_raw -> eager_runner._execute_extend, 14 minutes in. That run then scored 16.08. If the server dies during startup the notebook loop just breaks (notebook_cells.py:10741-10743) and games run against nothing for the retry grace window (ARC3_HTTP_RETRY_INITIAL_SECONDS=900).",
   "evidence": "fz-spec-adaptive/serve.log:2043-2045 (OOM, 'Scheduler hit an exception'); max_total_num_tokens/available_gpu_mem lines in control (4.25 GB) vs spec-adaptive (3.22 GB); fz-arms-v4-control and fz-streams14 census lines (GPU0 94.1-94.3/95.6). No OOM in 8 competition-config runs (~16h) but each was 25 games, versus 110 games, 9h and more varied prompts and images in the real run.",
   "estimated_cost": "Low-probability but near-total loss (up to 100% of remaining run) if it triggers; each +0.01 mem-fraction is worth ~+0.95 GB / +76k KV tokens but eats the margin",
   "fix": "Add a supervisor thread in the notebook: poll /health every ~15 s and relaunch the same args on exit (weights come back from page cache/disk in ~5-10 min). Set ARC3_MAX_ANALYZER_FAILURES=-1 and a longer backoff so games wait rather than give up. Consider mem-fraction 0.955 (the WRITEUP's earlier value), -0.5 GB KV (-38k tokens), for ~2 GB headroom. Make the startup loop abort the notebook instead of breaking.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Startup is 8-14 min of the 9h budget with ~140 s of repeatable JIT and a BF16 PLE read",
   "location": "notebook_cells.py:10648 (SGLANG_JIT_CACHE_DIR/TRITON_CACHE_DIR under fresh /tmp/sgl-intel/cache), :10701 --warmups structured_output, :10697 --ple-offload-embedding, :10336/:10725 precache; serve.log 'Engine startup timings'",
   "resource": "startup-teardown",
   "problem": "READY ranges 472-820 s across the 13 runs (control 595 s). The first CUDA graph capture (target verify, bs=10) takes 139 s and the other 6 batch sizes plus the draft graphs only ~15 s (target_verify 144.6-151.9 s in every run). That is JIT compilation (TileLang/FlashInfer/Triton) re-done on every fresh Kaggle container because the cache dirs are empty. The first real prefill then takes ~11 s (5,504 tokens at 489 tok/s) because the warmup only runs a 10-token structured-output request. Weight load is 254-586 s, I/O-bound on ~185 GB (BF16 PLE table ~115 GB host-offloaded; RAM 106 GB used). The notebook also runs the arc install and harness patch (~70 s) strictly before launching SGLang. One run (spec-adaptive) took 820 s and passed the 12-min SERVER_STARTUP_TIMEOUT, so the games were released against a still-loading server.",
   "evidence": "serve.log of control: 'Capturing batches (bs=10) ... 139.03s/it', 'Engine startup timings: load_weight=386.86, scheduler_e2e=563.10, target_verify=144.80'; first prefill '#new-token: 5504 ... input throughput 489'; READY lines per run: 472-820 s; notebook precache logs 223-700 s for 185.4 GB.",
   "estimated_cost": "~2.5-4 min recoverable (JIT ~140 s, first-request JIT ~11 s, overlap of ~70 s of installs) = 0.5-0.8% of the 540-min budget, ~90-150k generated tokens",
   "fix": "Ship the Triton/FlashInfer/TileLang/SGLang JIT caches from a prior Kaggle run inside the wheelhouse dataset and copy them into /tmp/sgl-intel/cache before launch (same GPU arch). Launch SGLang before installing the arc runtime. Add a long-prompt (8k + image) warmup request. Optionally use Albucino's FP8 PLE table (halves the ~115 GB read, ~1-2 min) if quality holds, though Franzen chose BF16 deliberately.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "SM120 low-M BF16 GEMM covers only M<=32 and ships an empty tuning table; decode at bs 9-10 (74% of time) has M=36-40",
   "location": "main:serving/build_bundle_pennyroyal.sh:102 (embedded PATCH string: _MAX_M = 32, _SM120_TUNED_TACTICS = {}, use_sm120_lowm_bf16_gemm), env at notebook_cells.py:10653 SGLANG_ENABLE_SM120_LOWM_BF16_GEMM=1",
   "resource": "gpu-throughput",
   "problem": "Target-verify graphs run num_tokens_per_req=4, so M = 4 x bs: 32 at bs=8 but 36 at bs=9 and 40 at bs=10. The patch's own docstring says it was tuned for bs<=4 decode and C4 verify (m<=16), and that cuBLAS under CUDA graphs serves these projections with 1-warp split-K kernels at only 20-70% of DRAM bandwidth. With _MAX_M=32 the patch does nothing for bs 9-10, which is 73.6% of time-weighted decode, and the per-shape tuned table is empty so only the generic tactic is ever used.",
   "evidence": "Patch source (line 102) and serve.log 'Capture target verify CUDA graph ... num_tokens_per_req=4, bs=[1,2,4,7,8,9,10]'. Measured step time rises smoothly 27.7/29.2/30.7/32.3 ms at bs 7/8/9/10, so no cliff at M>32 is visible and the BF16 projections may be a small share. The upside is uncertain.",
   "estimated_cost": "unknown, plausibly 0-5% decode throughput at bs 9-10",
   "fix": "Run with SGLANG_BF16_GEMM_LOG_SHAPES=1 to see the (m,n,k) distribution and time share; raise _MAX_M to 48 and tune/populate _SM120_TUNED_TACTICS for M=36 and 40; A/B decode tok/s at bs=10.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Shared system-prompt Mamba branch is evicted: ~half of trim re-prefills and all ~20 new-game starts re-prefill the 4.7-5.5k shared prefix",
   "location": "notebook_cells.py:10423 MAMBA_CACHE=60 (60 slots, 4 held per running request -> ~20 left for the radix tree); main:serving/patches/patch-sglang-mamba-final-prefill-v2.py:27 (single cold-start branch checkpoint, MRU refresh)",
   "resource": "prefix-cache",
   "problem": "All games share one 4,672-token system prompt (and ~5,568 with the first user turn). In the 92 trim-like re-prefills, 47 hit cached=4672 but 45 had cached=0, because the one Mamba branch checkpoint at that prefix had been LRU-evicted from the ~20 free slots. 20 new-game starts of ~5.1k tokens also had cached=0. The loss is the shared prefix re-prefilled plus an extra chunk break to re-create the branch.",
   "evidence": "fz-arms-v4-control/serve.log: cached-length histogram of trim-like requests {0:45, 4672:47}; 20 short requests with cached==0 (93k tokens); mamba num 40/60 at 10 running. Extra tokens ~210k + 93k = 0.30M, 2.9% of uncached input.",
   "estimated_cost": "~0.3M prefill tokens (~30 s, ~0.4% of wall) per 2h run; larger in the 110-game run, with more new-game starts",
   "fix": "Pin (never evict) the shared-prefix Mamba node, or lower the LRU priority of nodes that have >1 child. Do not simply add slots: each slot is ~54 MB, so +12 slots costs ~52k KV tokens.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Runaway generations: 26 of 2,311 requests hit the 12,288-token cap = 7% of all generated tokens",
   "location": "notebook_cells.py:10204 LOCAL_ANALYZER_MAX_OUTPUT=12*1024; main:ARC3-Inference/inference/utils/openai_compat.py:38-97 build_chat_payload (non-streaming, no stop/penalty fields); main:ARC3-Inference/inference/agent/tool_agent.py _chat_completion (~5218-5340, max_tokens=self._max_output_tokens)",
   "resource": "tokens",
   "problem": "The request is non-streaming and carries no repetition guard (no stop sequences, presence/repetition penalty, or early abort), so a degenerate or looping reply is only detected after the full 12,288 tokens are generated, and the harness drops it afterwards. 26 requests (1.1%) reached >=12,000 output tokens: 319k tokens, 7.0% of 4.59M generated, ~4.3k stream-seconds (6.5% of in-flight time at 75 tok/s). 79 requests (3.4%) exceeded 8k tokens and hold 18% of generated tokens. The transcripts show 25 finish_reason=length versus 2274 tool_calls.",
   "evidence": "fz-arms-v4-control/serve.log ReqTimeStats output_len histogram (1k bins: ... 8:13, 9:11, 10:13, 11:8, >=12k:25); transcripts finish_reason counts. Per-stream decode averages 74.6 tok/s (median).",
   "estimated_cost": "up to ~5-7% of decode tokens/stream-time if most capped replies are non-productive (not verified, some long replies may be legitimate)",
   "fix": "Stream and abort the HTTP request client-side on detected repetition (compressibility check on the tail every ~1k tokens), or lower max_tokens for non-first turns (e.g. 6-8k) with the existing 'keep reasoning shorter' nudge, or add a modest presence_penalty. Validate that accept length does not drop.",
   "proposed_class": "MAYBE"
  }
 ]
}
```
