# SPN startup latency investigation — 2026-09-16

The saved `runs/kaggle/_external/matijaludvig-spn-rrw/output/spn-rrw.log`
shows runtime import at 95s, checkpoint verification completed at 1015s,
server up at 1755s, and setup ready at 1774s. The first benchmark summary at
2380s follows the ordinary 600s reporting interval; the allocation stall gate
is unrelated.

The vendor's deterministic eight-shard sample is 128.03 GiB in the attached
`W4A16_FILE_MANIFEST.json`, including the 95.37 GiB BF16 PLE shard. The source
comment describing approximately 5 GB of sampled reads is incorrect. The
920s interval includes discovery, hashing, and header checks; no finer timing
proves precisely how much hashing consumed.

The remaining 740s server startup includes 550.91s loading weights, 563.38s
model loading overall, and 94.30s engine initialization (including 14.99s
compilation). At load start vLLM reports an NFS checkpoint of 168.75 GiB,
4.44 GiB available RAM, and skips automatic prefetch. Pre-launch resource
output reports roughly 170 GiB buff/cache and cgroup memory.current near
180 GB after verification. Hashing may influence the loader's memory/cache
state, but causation and the speedup from skipping it require a new run.

The comparison log downloaded via `jonathanwang2022/adaptive-reasoning-2`
(latest and /1) internally identifies itself as `adaptive-reasoning-ucb`.
It reports setup complete at 573.8s, server ready in 460.05s, weight loading
205.99s, and engine initialization 98.11s (17.23s compilation). Its NFS
checkpoint is 125.91 GiB with 145 GiB available RAM, and the GPU worker enables
8-thread prefetch. These artifacts establish a loading-path difference but
not the identity of the original adaptive-reasoning-2 execution.

Implemented weight-hash skipping and explicit stage timing in our SPN adapter.
See the [startup reference](../../ARC3-Inference/docs/spn-startup.md) for the
policy and opt-in vendor verification. Mounted historical source stays intact.
No prefetch is forced: loading the large BF16 host table while prefetching the
whole checkpoint needs memory measurements. No change to serving precision,
KV allocation, compilation, or model weights was made. Local tests exercise
validation behavior and generated commands; GPU speedup remains unmeasured.

## V3 result and simple next experiment

V3 measured verification at 0.706s, runtime installation at 90.086s, and
server startup at 859.46s; setup ready was 958.595s from notebook launch.
Weight loading took 651.63s. Prefetch remained disabled: 168.75 GiB checkpoint
versus 129.46 GiB available RAM. Removing hashing fixed verification but did
not establish a loading speedup.

The next smoke run changes only the default startup OpenMP thread count to one,
applying the vendor's existing post-load thread reduction earlier. No loader,
prefetch, checkpoint packaging, or compilation changes are part of this trial.

The one-thread path was abandoned at the user's request. The user observed the
first v4 summary near 1981s, suggesting benchmark start around 1381s versus
approximately 960s in v3 (exact stage completion not recovered here). The local
adapter's OMP_NUM_THREADS default was removed; weight-hash skipping remains.
This does not alter the already running v4 notebook.

V6 combined four loader threads with prefix caching disabled. The adaptive run
returns to the v3 serving behavior: default loader, prefix caching enabled, and
no OMP override. It retains metadata-only checkpoint verification.

V6 was intended to disable prefix caching, but its engine log confirms it stayed
enabled: removing the positive flag left the runtime default on. The adapter
now emits the explicit negative flag for value 0. The adaptive deployment uses
value 1, matching v3. Historical v6 is not a valid prefix-off comparison.

## Verified prefix-cache counters (v3–v6)

Fetched using `ApiListKernelSessionOutputRequest.version_label`; the installed
Kaggle CLI parses a version suffix but omits it from its output request.
The rates below are total cached tokens divided by total queried tokens,
including startup smoke requests.

| Version | Cached tokens | Queried tokens | Hit rate |
|---|---:|---:|---:|
| v3 | 5,350,016 | 11,815,804 | 45.28% |
| v4 | 5,475,456 | 11,944,582 | 45.84% |
| v5 | 4,685,184 | 10,964,862 | 42.73% |
| v6 | 5,180,672 | 11,789,225 | 43.94% |

All four engine logs confirm `enable_prefix_caching=True`.
