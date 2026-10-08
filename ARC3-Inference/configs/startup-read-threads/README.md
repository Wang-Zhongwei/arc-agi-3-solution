## Startup read threads

Quick runs that measure model load time only. Each is [new-baseline v3](../new-baseline/v3.json) with 5-minute public games and a faster launcher precache. sglang's one-shard prefetch stays at 4 threads: once the precache is ahead of the loader its reads come from RAM, and changing one reader keeps the result attributable.

| Config | Launcher precache | sglang prefetch threads |
|---|---|---|
| Daniel / new-baseline v3 | 3 threads, starts 60 s after server launch (~113 s) | 4 |
| [v1.json](v1.json) (x4) | 12 threads, starts after the wheel precache (~14 s) | 4 |
| [v2.json](v2.json) (x2) | 6 threads, starts after the wheel precache | 4 |

The model is 185.4 GB and the load ends when the read does: 795 MB/s (266 s) on new-baseline v01, 306 MB/s (621 s) on v02. Run v1 first; push v2 only if v1 does not roughly double the read rate. If `Load weight end` still lands right at the precache end, the loader is catching the precache; then raise `PREFETCH_THREADS`.

Read from each run: the `Precaching: loaded 62 files (185.40 GB) in N seconds` line in the notebook log, `Load weight end. elapsed=` and `The server is fired up` in `serve.log`. Machine speed varies (266–858 s loads across 23 runs), so compare read rates, not one load time.
