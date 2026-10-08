# Startup read threads

**Decision (2026-10-08):** the Pennyroyal launcher reads the model with 6 precache threads and 8 sglang prefetch threads, starting the precache at its usual time (60 s after server launch). Daniel's cell used 3 + 4. These are now the defaults of `kaggle.sglang.MODEL_PRECACHE_THREADS` and `PREFETCH_THREADS` ([CONFIGURATION.md](../../ARC3-Inference/CONFIGURATION.md)), so every Pennyroyal config gets them unless it sets its own.

## Why the read matters

The target and draft checkpoints are 185.4 GB (62 files). The weight load ends when the read does: across 22 runs with 3 + 4 threads, loads took 266–858 s, tracking the read rate (795 MB/s on new-baseline v01, 306 MB/s on v02). A slow load can also run past `server_startup_timeout_s` and release the games before the server answers, as new-baseline v02 did.

Two readers fetch the same bytes:

- **Launcher precache** reads every file in order into page cache.
- **sglang prefetch** reads one shard (~5 GB) ahead of the loader, then drops each loaded shard (`--weight-loader-drop-cache-after-load`).

With 3 threads from ~113 s the precache trails the loader for the whole load (new-baseline v01: loader 70–336 s, precache 113–353 s).

## Runs

All are [startup-read-threads](../../ARC3-Inference/configs/startup-read-threads/README.md) notebook versions: new-baseline v3 with 5-minute games. The wheel read (6.7 GB, 16 threads, identical in every run) measures the host before any tuning.

| Kaggle version | Precache | Prefetch | Wheel read | Weight load | Server ready |
|---|---|---|---|---|---|
| Baseline (22 runs) | 3, from ~113 s | 4 | 219–1,175 MB/s | 266–858 s | |
| new-baseline v01 (fastest baseline) | 3, from ~113 s | 4 | 1,175 MB/s | 266 s | ~517 s |
| v1 (x4, early) | 12, from ~14 s | 4 | 858 MB/s | **1,011 s** | after the 1,200 s deadline |
| **v2 (x2)** | **6, from ~113 s** | **8** | 913 MB/s | **224 s** | **486 s** |
| v3 (prefetch only) | off | 16 | | cancelled after ~1,000 s without a ready server | |
| v5 (x2 again) | 12, from ~113 s | 16 | 704 MB/s | 236 s | 504 s |

## Findings

- **More readers help up to about 1 GB/s.** v2 loaded faster than all 22 baseline runs on a typical host; on hosts with similar wheel reads (~890–930 MB/s) the baseline took 269–334 s. 14 threads gave about 1.3x the rate of 7 (~60 vs ~90 MB/s per thread).
- **Doubling again does not pay.** v5's precache read 917 MB/s against v2's 982 MB/s on a slower host, and loaded in 236 s against 224 s.
- **Do not precache early or far ahead.** 185 GB plus ~100 GB of pinned host PLE exceeds the host's 177 GB of RAM, leaving roughly 60 GB of page cache. In v1 the early 12-thread read finished at 357 s while the loader was at shard ~7 of 38; the evicted shards were read again and the load took 1,011 s.
- **Prefetch alone was slower.** v3 had not served after ~1,000 s.

## Not tested

Each result is one run on a random host, so differences under ~30% are within host variance. Reading fewer bytes (the FP8-PLE checkpoint, about 50 GB less) would change model numerics and needs a quality comparison.
