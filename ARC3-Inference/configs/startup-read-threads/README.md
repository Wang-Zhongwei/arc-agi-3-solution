## Startup read threads

Quick runs that measure model load time only. Each is [new-baseline v3](../new-baseline/v3.json) with 5-minute public games and a different model read.

The model is 185.4 GB. sglang's loader prefetches one shard (~5 GB) ahead with 4 threads; the launcher precache reads every file in order with 3 threads from ~113 s (60 s after server launch). In the baseline the precache trails the loader the whole load (new-baseline v01: loader 70–336 s, precache 113–353 s), so it only re-reads shards the loader already dropped. v2 shows the read limits even fast hosts: it loaded at 5.9 s/shard.

| Config | Launcher precache | sglang prefetch | Result |
|---|---|---|---|
| Daniel / new-baseline v3 | 3 threads, from ~113 s | 4 | 266–858 s loads |
| [v1.json](v1.json) (x4 precache) | 12 threads, from ~14 s | 4 | Kaggle v1: precache 185 GB in 335 s, load **1,011 s**. 185 GB plus ~100 GB pinned PLE exceeds 177 GB RAM, so the early read was evicted and re-read. |
| [v2.json](v2.json) (x2 both) | 6 threads, from ~113 s | 8 | Kaggle v2: load **224 s** (5.9 s/shard), precache 185 GB in 189 s (982 MB/s), server ready 486 s after notebook start. Its wheel read (913 MB/s) was median, so the host was typical; the 22 baseline runs loaded in 266–858 s. 14 threads gave ~1.3x the 7-thread rate (~60 vs ~90 MB/s per thread). |
| [v3.json](v3.json) (prefetch only) | off | 16 | Pushed as Kaggle v3, then dropped for v4. |
| [v4.json](v4.json) (x2 again) | 12 threads, from ~113 s | 16 | |

Read from each run: `Load weight end. elapsed=` and the per-shard progress in `serve.log`, the `Checkpoint prefetch: ... up to N range-reader threads` line, and the `Precaching: loaded 62 files (185.40 GB) in N seconds` line in the notebook log. Machine speed varies (266–858 s loads across 23 runs), so compare seconds per shard against the ~7 s floor, not one load time.
