• Yes—there are worthwhile experiments, but increasing cache size would be low on my list. The strongest candidates are
  improving prefix reuse, tuning the prefill batch budget, and testing MTP separately.

  I reviewed the log, saved Prometheus metrics, current config, and SPN launcher. No files changed.

  The run’s measurements:

   Metric                                                                      Observed
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   KV pool                                  18.42 GiB; reported capacity 791,347 tokens
  ───────────────────────────────────────  ─────────────────────────────────────────────
   Reported KV usage                                           Median 57.7%; peak 73.9%
  ───────────────────────────────────────  ─────────────────────────────────────────────
   KV preemptions                                                                     0
  ───────────────────────────────────────  ─────────────────────────────────────────────
   Cumulative prefix-cache hit rate                                               35.1%
  ───────────────────────────────────────  ─────────────────────────────────────────────
   Multimodal processor-cache hit rate                                            91.2%
  ───────────────────────────────────────  ─────────────────────────────────────────────
   Average request                                   21,342 input / 1,770 output tokens
  ───────────────────────────────────────  ─────────────────────────────────────────────
   Average queue / prefill / decode time                         0.25s / 1.23s / 62.96s

  These come from the log (ARC3-Inference/runs/kaggle/_external/matijaludvig-spn-rrw/output/vllm.log:274) and cumulative
  metrics (ARC3-Inference/runs/kaggle/_external/matijaludvig-spn-rrw/output/vllm-metrics-final.prom:75). The final logged
  47.8% prefix-hit rate is a rolling statistic, not the whole-run rate. Request timing is dominated by decoding, although
  prefills can also slow other requests’ decoding.

  One important configuration trap: the Kaggle SPN launcher pins its own serving settings. This run used TP=1,
  max_num_batched_tokens=6144, GPU utilization 0.96, FP8 KV, and async scheduling. The JSON’s server section has different
  values for the Slurm deployment. Changing server.max_num_batched_tokens alone would not tune this Kaggle run; the adapter
  also explicitly rejects nonempty serving_profile_env. See launcher (ARC3-Inference/runs/kaggle/_external/matijaludvig-spn-
  rrw/bundle/kaggle_flashnext_setup.py:706) and adapter (ARC3-Inference/inference/framework/kaggle.py:587).

  My priorities would be:

  1. Improve prefix reuse before allocating more cache.
     The current agent drops old history as it fills the context and removes animation parts from persisted messages. Both
     can change previously cached prefixes. Investigate trimming several turns at once to leave room for subsequent append-
     only turns, rather than repeatedly shifting the history. That trades retained context for reuse, so compare game
     progress as well as throughput. Keep preserve_thinking=true for the baseline. History implementation (ARC3-Inference/
     inference/agent/tool_agent.py:1669).

     A smaller serving-only experiment is --prefix-match-unit 16. The saved cache config reports physical block_size=3136,
     mamba_cache_mode=align, and no explicit match unit. That does not prove matching currently happens only every 3,136
     tokens—the runtime derives it from cache groups. Verify the resolved granularity first; finer matching could be
     redundant. It also does not create additional recurrent-state checkpoints by itself. vLLM documentation.

  2. A/B test the prefill budget: 6,144 versus 4,096.
     Smaller prefill chunks may reduce interruptions to ongoing decoding. If that loses overall throughput, test 8,192 next.
     Measure generated tokens/sec, completed agent turns, and latency on the same workload; the log cannot establish the best
     value. This is the cheapest useful serving experiment. vLLM tuning guidance.

  3. Index sharing is an MTP experiment here—not an independent switch.
     The run has speculative_config=None and MTP disabled. The MTP_INDEX_SHARE_FOR_ITERATION control documented in your other
     serving bundle requires MTP; it cannot simply accelerate this existing configuration. Existing experiment notes (ARC3-
     Inference/configs/smoke/README.md:71).

     Test MTP=1 first, then MTP=3 with supported index reuse, after checking this runtime/checkpoint combination. Benchmark
     at 25 concurrent requests, plus a low-concurrency case: faster single-request decoding does not guarantee higher
     aggregate throughput. Upstream also reports configurations where MTP hurts, so I would not enable it by default.
     Main KV is already FP8; that does not establish the QSA indexer cache’s precision. Current upstream supports an FP8
     indexer path. Check whether the pinned build supports it before considering an upgrade/backport, and validate quality
     because it changes precision. Upstream implementation.

  Also, the historical run’s approximately 29.5-minute startup includes expensive checkpoint verification. Your current
  adapter already skips weight hashing by default, so that improvement is already implemented; later measurements show
  loading still dominates. Startup investigation (notes/serving/spn-startup-latency.md:1).

  My first two trials would be 4,096-token batching and verified finer prefix matching, each independently. Keep FP8 KV,
  prefix caching, and async scheduling enabled. Follow with MTP only after establishing those baselines; none of the proposed
  speedups is measured yet.