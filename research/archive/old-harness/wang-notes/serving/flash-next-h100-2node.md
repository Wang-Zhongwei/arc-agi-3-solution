# Flash-Next FP8 on two separate HPC3 H100 nodes

Status (2026-09-13): **upstream UVA PLE offload passed text, vision, and tool-call
smoke tests across two H100 nodes** (job `76407`). The older runtime remains
blocked as documented below. The matching 32k/C28 throughput benchmark also
passed (job `76417`): **205.12 output tokens/s**, versus 134.55 on four A100s.

## Takeover and repairs

- Original job `76336`: `g18` and `g19`, one NVIDIA H100 PCIe
  (81,559 MiB) each; Ray saw both GPUs but its head workers failed to start.
- The Ray head daemonized and its Apptainer invocation exited. The head raylet's
  container filesystem returned `Transport endpoint is not connected` when
  accessing Python's library directory. Its driver waited in
  `unix_stream_data_wait`; Ray reported worker registration timeouts.
- Keep **both** head and worker Ray containers alive with `ray start --block`.
  Limit Ray CPUs to the 12 allocated per node and its object store to 4 GiB/node.
  Install cleanup before launching Ray; use the Slurm working directory instead
  of the submitter's directory.
- Restart `76337` exited immediately due to the old submit-directory dependence;
  corrected job `76339` formed a healthy 24-CPU / 2-GPU Ray cluster.

## Confirmed result in job 76339

Both ranks loaded the real FP8 model, using **62.36 GiB per GPU** and native
FP8 / FlashAttention 3. TP0 loaded in 83.70 s, TP1 in 138.44 s. This establishes
weight fit and distributed initialization, not successful inference.

The PLE CPU worker ran on `g18`, binding:

```text
ipc:///tmp/flash-next-h100-76339/tmp/8d840e95-3764-4a0f-93b8-a6059ecd5c70
```

It waited for two GPU registrations but received only TP0's. The socket did not
exist on `g19`. TP1 logged that it had sent registration, but that is not an
acknowledgment: `connector.py::_register_with_offload_worker` logs after a
ZeroMQ PUSH send, which can queue without a connected peer. The registration
payload contains shared-memory tensors and CUDA IPC handles, not remotely
accessible buffers. Changing the socket address to TCP alone cannot fix this.

The pinned runtime's `gpu_worker.py::_validate_ple_offload_config` rejects
`nnodes != 1`, but this Ray launch reached initialization despite spanning two
physical hosts. Passing the configuration check is therefore not evidence of
physical multi-node PLE support. The node-local IPC design is the actual limit.

The stalled job was cancelled after collecting evidence. No smoke requests
completed in that attempt; the later upstream results are below.

## Resolution

The old worker/IPC implementation needs a different architecture for multi-node
use. Upstream already supplies that architecture: process-local pinned-memory
UVA lookups. We upgraded the isolated H100 runtime instead of patching the old
IPC protocol. Do not merely remove the old validation guard or point CUDA IPC
at a network socket.

The shared A100 image/checkpoint are unchanged. Launcher changes remain staged,
not committed. Existing four-A100 benchmark results remain valid.

Evidence under `ARC3-Inference/.cache/flash-next-h100/`:

- `logs/76336.{out,err}`: original Ray startup failure.
- `logs/76339.{out,err}`: repaired cluster formation.
- `runs/76339/server.log`: both model loads and head PLE registration.
- `runs/76339/g19-worker.out`: preserved remote worker log.

## Successful upstream UVA test: job 76407

### Runtime and launch

- Official upstream image: `vllm/vllm-openai:nightly`, resolved before download to
  amd64 digest `sha256:53f466b1524d3a9730a302007632d4d640d2ff956245f71a69b608ca35efa70e`.
- Build commit: `2671fedfc7ae604761990603fc736c0c4f21de57`;
  image creation timestamp `2026-09-13T05:21:41Z`.
- Installed vLLM: `0.29.1rc1.dev9+g2671fedfc`; PyTorch `2.13.0+cu130`.
- Includes upstream [UVA PLE / Engram sharding PR #54371](https://github.com/vllm-project/vllm/pull/54371),
  merged September 9. **No runtime patches or Ray overlay**.
- Isolated image: `ARC3-Inference/.cache/flash-next-h100-upstream/vllm-openai.sif`.
  The original A100 image and checkpoint are unchanged.
- Same FP8 checkpoint as the A100 test, revision
  `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- Native multi-node `mp`: `--nnodes 2`, TP2 + EP, rank 1 `--headless`.
- Explicit `--engram-config '{"cpu_offload":true,"embedding_across_dp":false}'`.
- BF16 activations/KV, context limit 32,768, max sequences 4, batch-token cap 1,024,
  GPU memory utilization 0.90, prefix caching off, MTP off.
- Both ranks confirmed `weight_dtype=torch.float8_e4m3fn, weight_device=cpu, pinned=True`.
  NCCL confirmed `NET/IB` for cross-node transport.

Reproduce from the repository root after staging the pinned image:

```bash
ssh coe-hpc3 'cd /home/018270239/arc-agi-3/duck-harness/ARC3-Inference &&
  sbatch --parsable --partition=gpuqs --nodelist=g18,g19 scripts/smoke_flash_next_h100_2node.slurm'
```

The launcher uses node-local caches, binds the API to loopback, waits up to
20 minutes for readiness, runs the smoke checker, and stops both ranks.
Each worker has a 40-minute process timeout inside a 45-minute allocation.
Create `.cache/flash-next-h100-upstream/logs/` before submission.

Image staging used `crane pull --platform linux/amd64` at the digest above,
then Apptainer `build` from the Docker archive with local `APPTAINER_TMPDIR`
and `--mksquashfs-args "-processors 4 -Xcompression-level 1 -mem 4G"`.
The CLI help command cannot infer a device on the CPU-only login node; the
actual two-GPU launch validated the serving arguments.

### Verified outcome

Slurm **COMPLETED**, exit **0:0**, elapsed **3m43s**, on `g18` and `g19`
(one H100 PCIe 81,559 MiB each, driver 590.44.01).

| Check | Verified output | End-to-end time |
| --- | --- | --- |
| Text | `4` | 2.768 s |
| Synthetic red image | `Red` | 0.323 s |
| Required tool | `report_answer({"value": 4})` | 1.034 s |

All three requests returned HTTP 200. Thinking was disabled. The first request
included additional kernel compilation; these tiny requests are **not throughput
measurements**, long-context validation, or an ARC-AGI score run.

- Model weights: **62.36 GiB/GPU**.
- Available KV memory: **6.79 GiB/rank**.
- Reported KV capacity: **448,669 token slots**, **13.69x at 32,768 tokens/request**.
  This is a memory-derived estimate, not measured concurrency; max sequences was 4.
- Post-request GPU usage: **72,627 MiB on each GPU** (about 70.93 GiB).
- Engine profiling/cache/warm-up: 52.95 s.
- FlashInfer's NVSwitch-specific allreduce fusion workspace was unavailable;
  initialization continued and all smoke checks passed.
- Both ranks were intentionally terminated after the pass marker. Slurm step
  cancellation/kill messages are expected teardown, not failed inference.
  No serving allocation was left running.

Evidence under `ARC3-Inference/.cache/flash-next-h100-upstream/`:
`logs/76407.{out,err}`, `runs/76407/server-rank-{0,1}.log`,
`runs/76407/responses.json`, and `runs/76407/metrics-after.txt`.

## Matched 32k / concurrency-28 benchmark: job 76417

Completed on `g18` and `g19`, exit **0:0**, allocation duration **6m44s**.
Append `benchmark` to the launcher and override Slurm time to `01:10:00`:

```bash
ssh coe-hpc3 'cd /home/018270239/arc-agi-3/duck-harness/ARC3-Inference &&
  sbatch --parsable --partition=gpuqs --nodelist=g18,g19 --time=01:10:00 --job-name=flash-next-h100-32k-c28 scripts/smoke_flash_next_h100_2node.slurm benchmark'
```

Benchmark mode sets **max sequences 28** and **GPU memory utilization 0.95**,
matching the A100 benchmark's settings. Server lifetime is capped at 65 minutes,
client at 45 minutes. Smoke defaults remain four sequences and 0.90 utilization.

Both tests use the same FP8 checkpoint, random seed 42, exact **31,744 input +
1,024 output tokens**, 28 measured requests at client concurrency 28, one excluded
warm-up, temperature zero, EOS ignored, prefix caching off, MTP off, and
max batched tokens 1,024. All H100 per-request token lengths were verified.

| Metric | 4x A100 40GB, one node (76328) | 2x H100 80GB, two nodes (76417) |
| --- | ---: | ---: |
| Successful / failed | 28 / 0 | 28 / 0 |
| Measured duration | 213.10 s | **139.78 s** |
| Aggregate output throughput | 134.55 tokens/s | **205.12 tokens/s** |
| Aggregate input + output throughput | 4,305.48 tokens/s | 6,563.94 tokens/s |
| Median time to first token | 100.71 s | **54.79 s** |
| p95 time to first token | 186.97 s | 113.67 s |
| Median end-to-end latency | 147.36 s | **124.15 s** |
| p95 end-to-end latency | 212.43 s | 139.15 s |
| Median per-request time per output token | **45.55 ms** | 67.80 ms |
| KV token capacity | 250,091 | 691,908 |
| Observed maximum running requests in periodic logs | 7 | 21 |
| Preemptions | 0 | 0 |

H100 aggregate throughput was **1.525x** the A100 result, and batch completion
time was **34.4% shorter**. However, median time per output token was worse:
more requests were served concurrently and queueing fell, but each active
request did not generate tokens faster under this load. Do not interpret the
aggregate gain as a per-game decode-speed gain.

The H100 server allocated **10.47 GiB KV memory/rank**, reporting **21.12x**
full-32k capacity. Logs reached 21 running requests with seven waiting and
99.6% KV usage. No contexts were preempted. Post-run GPU memory was
**76,907 MiB on each GPU**. The client produced 888,832 input and 28,672 output
tokens; server counters agree after including the one warm-up.

This compares **two complete serving configurations**, not GPUs alone:
A100 uses older vLLM `0.1.dev20073+g8e685d198`, Marlin weight-only FP8 and a
dedicated CPU PLE worker; H100 uses upstream `0.29.1rc1.dev9+g2671fedfc`, native
FP8, local UVA offload and cross-node InfiniBand. The benchmark clients also
come from their respective pinned runtimes. Both are finite first-burst tests,
not repeated steady-state measurements, and both incurred additional batch-shape
JIT compilation after the one warm-up.

All worker processes were stopped after results were collected. No serving
allocation remains. Artifacts under
`ARC3-Inference/.cache/flash-next-h100-upstream/runs/76417/`:
`benchmark-32k-c28.json`, `benchmark.log`, `server-rank-{0,1}.log`, and
`metrics-{before,after}.txt`; allocation logs are in `logs/76417.{out,err}`.

## Local animation game run (job 76469)

Submitted on 2026-09-13 via:

```bash
cd ARC3-Inference
make sbatch CONFIG_PATH=configs/inference.flash-next.json RUN_NAME=flash-next-h100-animation
```

Job **76469** uses source commit `bc6c6a4` on branch `a100-h100`,
two nodes `g18,g19`, one H100 and 220 GiB host memory per node, and 24
allocated CPUs total. One harness drives one native-MP TP2/EP server.
It retains 25 official games × ten passes, 60 minutes per game, uniform
allocation, and action animations; concurrency is 28 and Slurm wall time is
16 hours.

The production image processor caps images at 65,536 pixels, matching the
256×256 rendered boards, and allows 512 images per prompt so historical boards
and current animation panels remain eligible. Encoder profiling uses 16 images
within its 1,024-token budget. Initial startup reported 11.84 GiB KV cache per
rank and 781,390 token capacity (23.85 full 32k requests). This differs from the
one-image text benchmark's vision profiling reservation.

The server reached ready and accepted game chat requests with HTTP 200;
28 game slots initialized without connection errors. This is startup
validation, not a completed game-score or sustained-throughput result.

Run directory:
`ARC3-Inference/runs/local/20260913_150541_flash-next-h100-animation/`.
See `server.log`, `stdout.log`, `stderr.log`, and `artifacts/*_events.jsonl`.

The preceding job **76463** was canceled after detecting a stale `g16`
endpoint inherited from a previous remote-server node file. Its distributed
server was healthy, but game requests went to the wrong host. The solver now
pins owned-server connections to their configured loopback port; a regression
test covers this. The complete inference test suite has 135 passing tests and
one pre-existing `kaggle.backend` config-audit failure, also reproduced against
the pre-change commit `e60b7e9`.
