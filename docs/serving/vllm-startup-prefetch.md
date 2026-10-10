# vLLM startup: parallel shard reads

The saved Intel AutoRound runs start SGLang in 8–9 minutes and vLLM in 14–16 minutes. Target weight loading accounts for most of the difference. This change ports SGLang's parallel byte-range reads to the Lord vLLM launcher's existing bounded page-cache prefetcher.

## Saved startup measurements

These runs serve `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1` on Kaggle RTX PRO 6000 hosts. Times come from `serve.log` or `vllm-openai-server.log` and each notebook's timestamped output. Ready time is measured from notebook start.

| Exact run under `ARC3-Inference/runs/kaggle/` | Preparation | Target load | Draft load | Server ready |
|---|---:|---:|---:|---:|
| `jonathanwang2022/startup-read-threads/v02/` (SGLang) | wheel precache 7.34 s; launch at ~53 s | 224.39 s | 0.87 s | 486 s |
| `ticbudja/thinking-budget-priority-four-waves/v03/` (SGLang) | wheel precache 9.40 s; launch at ~52 s | 267.13 s | 0.83 s | 539 s |
| `matijaludvig/arc3-vcintel-25x1/v01/` (vLLM) | runtime extraction 119 s | 477.25 s | 64.86 s | 854 s |
| `ticbudja/priority-one-wave-lord-vllm/v01/` (vLLM) | runtime extraction 167 s | 567.99 s | 41.81 s | 960 s |

These are observations on different hosts, not a controlled engine A/B. SGLang uses the separate Albucino draft checkpoint; vLLM's MTP load rereads the target checkpoint. vLLM reports model loading totals of 577.98 s and 644.66 s, followed by 64.93 s and 57.62 s of engine initialization. Faster reads can reduce the target load but do not remove runtime extraction, draft loading or warmup.

## What differs in the readers

SGLang runs a 6-thread whole-model precache 60 s after server launch and an 8-thread, one-shard lookahead prefetcher inside its loader. Both split files into ranges, so a single ~5 GB shard can use several NFS read streams. The [startup experiments](startup-read-threads.md) found that an early whole-model precache outran the loader, exceeded the available page cache and caused rereads; its target load took 1,011 s.

The vLLM launcher already prefetches within an 8 GiB window, follows target-load progress and evicts passed shards. Before this change its four workers each read an entire file through one stream. A ~5 GiB shard often filled enough of the window to prevent another shard being submitted, leaving one worker active.

There was also a filename mismatch: the launcher excluded `ple-*`, but Intel uses `model-ple-*`. The priority-one-wave v01 prefetch report counted all 38 shards (168.75 GiB), including 22 PLE shards that the PLE loader already reads in parallel. Only two shards finished prefetch before the loader reached them. vcintel's report correctly excluded those PLE files: 16 GPU shards, 73.38 GiB, also only two staged before arrival.

## Change on `allocation-vc-one-wave`

- Use eight workers in one shared pool to read disjoint, block-aligned ranges of each GPU shard. `kaggle.vllm.prefetch_threads` controls the total; the default is 8.
- Retain the 8 GiB byte window and additionally admit only the current shard and its immediate successor in the full loader order. This prevents `model_extra_tensors.safetensors` from being warmed far ahead while the intervening PLE files load.
- Exclude both `ple-*` and `model-ple-*`; begin reads after the target loader's first progress line.
- Cancel every range of passed shards and wait for all their reads to end before eviction. Preserve the watchdog's separate prefetch process, bounded shutdown and per-restart log offsets.
- Keep the per-shard timing report, aggregating the first range start and last range finish. Add `reader_mode: parallel_ranges` and `lookahead_shards: 1` to identify the candidate.

The range reads warm page cache only. Model weights, quantization, draft selection, prompts, scheduler and concurrency are unchanged. This ports parallel reading into the bounded window; it does not add a second whole-model scan.

## Validation and remaining measurement

Offline tests check exact range coverage including short tails, cancellation, short reads, loader-order and byte bounds, both PLE names, eviction only after all readers finish, report aggregation, partial progress lines and restart offsets, and propagation of the configured thread count into the rendered watchdog. The existing INC-route and serving tests also run.

Validation: 134 tests passed across `test_lordhansolo_prefetch.py`, `test_lordhansolo_inc_ple.py`, `test_lordhansolo_serving.py`, `test_serving_config.py` and `test_pennyroyal_optional_patches.py`. Ruff and `git diff --check` passed. Tests used the existing project virtualenv with this worktree and its TAAF source on `PYTHONPATH`.

The candidate has not been launched on a GPU. Its speedup is unmeasured. A startup A/B should keep the Intel checkpoint, draft, runtime, engine and harness settings fixed, compare the baseline at `1d35483` against this candidate over repeated cold hosts, and record runtime extraction, target/draft loading, readiness and `vllm-shard-prefetch.json`. The saved cross-engine timings above do not establish a speedup for this patch.
