# Flash-Next on HPC3's four-A100 node

Experiment branch: `hpc3-flash-next-a100`. This is a bounded serving smoke test,
not an ARC-AGI score run. It does not change the Kaggle NVFP4 configuration.

## Hardware verified on 2026-09-13

Slurm job `76279` completed successfully on `cs001`:

- Four NVIDIA A100-PCIE-40GB GPUs, 40,960 MiB each, compute capability 8.0.
- NVIDIA driver 590.44.01; Apptainer 1.5.2.
- No NVLink: the GPU topology reports SYS/NODE PCIe paths.
- Two Intel Xeon Gold 6326 CPUs: 32 physical cores / 64 logical CPUs total.
- Approximately 250 GiB host memory.

The single-node requirement is specific to the pinned runtime's PLE CPU
offload. `gpu_worker.py::_validate_ple_offload_config` rejects `nnodes != 1`;
this is not a universal restriction on tensor-parallel inference.

## Reproducible runtime

- Image: `vllm/vllm-openai:qwen38-flash-next` pinned to amd64 manifest
  `sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`.
- Installed vLLM: `0.1.dev20073+g8e685d198`; PyTorch `2.13.0+cu130`.
- Model: `Qwen/Qwen3.8-Flash-Next-FP8`, revision
  `236dfdf285828023ca3bcd3f37366c58a3469b13`, approximately 173 GiB.
- Separate cache: `ARC3-Inference/.cache/flash-next-a100/`; existing serving
  images and model caches are untouched.

The [official Flash-Next recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
validates newer GPUs. This A100 experiment uses FP8 weights with BF16 activations
and the Marlin MoE backend, plus PLE CPU offload. It is not native FP8 compute.

Four tensor-parallel ranks also enable expert parallelism: each rank owns
whole experts with intermediate dimension 640. Plain TP4 splits that dimension
to 160; the pinned FP8 implementation then forces a Triton block-scale-refinement
path whose FP8 activation kernels do not support A100. EP avoids that refinement.
Whether the complete configuration runs successfully must be established by the
test, not inferred just from available VRAM.

## Run the checks

Stage the pinned image into the isolated cache with the existing
`container/scripts/01_pull_image.sh`, setting `VLLM_CONTAINER_STORE` and
`VLLM_IMAGE` to the paths/digest above. Convert its Docker archive to
`vllm-openai.sif` using Apptainer. Download the checkpoint revision above to
the cache's `model/` directory, including configuration/tokenizer assets and
all safetensors shards. Compute nodes run offline. All 131 indexed shards were
confirmed present after staging.

Use node-local temporary storage for the Apptainer build and
`--mksquashfs-args "-processors 4 -Xcompression-level 1 -mem 4G"` for packaging.
Shared-storage extraction and default gzip level 9 were very slow in this test;
the completed SIF is approximately 8 GiB.

The launcher also puts Triton, vLLM, FlashInfer, CUDA, and temporary compilation
caches in a unique node-local `/tmp/flash-next-a100-<job-id>.*` directory. In the
first runtime attempt, all four workers were observed in NFS directory-removal
waits (`do_rmdir`, `vfs_rmdir`, `rpc_wait_bit_killable`) after graph compilation.
Weights and run logs remain on shared storage.

From the repository root, the real-model serving check is:

```bash
ssh coe-hpc3 'sbatch --parsable --job-name=flash-next-a100-smoke \
  --account=default --qos=normal --partition=gpuqs --nodelist=cs001 \
  --nodes=1 --ntasks=1 --gres=gpu:a100:4 --cpus-per-task=32 --mem=230G \
  --time=00:30:00 \
  --chdir=/home/018270239/arc-agi-3/duck-harness/ARC3-Inference \
  --output=/home/018270239/arc-agi-3/duck-harness/ARC3-Inference/flash-next-a100-smoke-%j.log \
  /home/018270239/arc-agi-3/duck-harness/ARC3-Inference/scripts/smoke_flash_next_a100.sh'
```

Adjust the checkout path/account for another user. The server binds only to
loopback and shuts down after the test, on failure, or after a 25-minute cap.
It starts with a 32k context, four maximum sequences, prefix caching off, and
MTP disabled. It checks text arithmetic, a synthetic red image, and a structured
tool call; it does not measure game scores or sustained throughput.

Logs: `ARC3-Inference/flash-next-a100-*.log` and
`.cache/flash-next-a100/runs/<job-id>/` inside `ARC3-Inference`.

## Outcome

| Job | Test | Result |
| --- | --- | --- |
| 76279 | Hardware | Passed on `cs001`. |
| 76298 | Kernel preflight launcher | Exited before vLLM: Slurm spool path; fixed to use `--chdir`. |
| 76299 | Four-layer synthetic model | All ranks loaded (3.56 GiB each), selected Marlin/FlashAttention 2, and compiled; cancelled after shared-filesystem cache waits. |
| 76303 | Kernel preflight with local caches | Passed: warm-up and four generated tokens; random weights, not a quality test. |
| 76306 | Full FP8 checkpoint + PLE CPU offload | Passed text, vision, and structured tool calls. Slurm `COMPLETED`, exit `0:0`, elapsed `00:08:28`. |

### Full-model result

The real model passed all three HTTP smoke checks on `cs001`:

| Check | Verified response | End-to-end request time |
| --- | --- | --- |
| Text arithmetic | `4` | 1.73 s |
| Synthetic red image | `Red` | 1.78 s |
| Required structured tool call | `report_answer({"value": 4})` | 1.76 s |

These are tiny, sequential requests with thinking disabled, not sustained
throughput measurements, long-context validation, or an ARC-AGI score benchmark.
The first text request included additional shape-specific kernel compilation.

- Model loading: **32.45 GiB per GPU**.
- Post-request GPU usage: 36,385 / 36,307 / 36,311 / 36,315 MiB (about **35.5 GiB
  per card**, out of 40 GiB).
- PLE CPU weights loaded, materialized parameters verified, and all four GPU
  workers registered successfully.
- Available KV cache: **1.29 GiB**, with **99,433 token slots** and reported
  maximum concurrency **3.03x at 32,768 tokens/request**. Configuring four maximum
  sequences does not guarantee four simultaneous full-32k requests.
- Local-cache warm-up completed; full-model graph compilation took about 32 s.
- The launcher stopped the server after the checks, and `cs001` returned to
  `IDLE`. No serving allocation was left running.

The runtime printed an `EngineDeadError` during the intentional SIGTERM teardown,
after all three HTTP 200 responses and the passing test marker. It was not an
inference failure; Slurm still completed with exit `0:0`.

Evidence: `ARC3-Inference/flash-next-a100-smoke-76306.log`, plus
`ARC3-Inference/.cache/flash-next-a100/runs/76306/server.log` and
`responses.json`. The pinned SIF and full checkpoint remain cached for reuse.

## Full-context throughput benchmark

Append `benchmark` to the launcher command above, use `--time=01:10:00`,
and change the output name to `flash-next-a100-bench-%j.log`. The benchmark
mode preserves the smoke profile except for maximum sequences **28** and GPU
memory utilization **0.95**. Server lifetime is capped at 65 minutes and the
benchmark client at 45 minutes; both exit automatically.

The pinned runtime's `vllm bench serve` sends one finite burst of 28 synthetic
text-completion requests at client concurrency 28, after one untimed warm-up:

- Context limit: 32,768; target input 31,744 tokens and output 1,024 tokens.
- Fixed lengths, seed 42, temperature zero, EOS ignored to exercise the full
  output budget. Actual token counts are recorded in the result.
- Prefix caching and MTP remain disabled; max batched tokens remains 1,024.
- JSON results include per-request timing and failures; TTFT, TPOT, ITL, and
  end-to-end latency report p50/p90/p95/p99.
- Prometheus snapshots before and after the client capture cache/preemption
  counters (including warm-up). Server logs provide scheduling diagnostics.

This is a finite-batch, text-only capacity test, not a steady-state game replay.
Concurrency 28 means up to 28 outstanding client requests, not that 28 full
32k sequences fit in GPU KV cache. End-to-end throughput includes server queueing
and prefill, but excludes model startup and the client warm-up.

### Measured result: job 76328 (2026-09-13)

Completed successfully on `cs001`, Slurm exit `0:0`. The allocation lasted
11m54s including startup; the measured request burst lasted **213.10 s**.

| Metric | Measured value |
| --- | --- |
| Successful / failed requests | 28 / 0 |
| Input / output tokens per request | 31,744 / 1,024 (all requests verified) |
| Total input / generated tokens | 888,832 / 28,672 |
| Aggregate output throughput | **134.55 tokens/s** |
| Aggregate input + output throughput | 4,305.48 tokens/s |
| Request throughput | 0.1314 requests/s |
| Median / p95 time to first token | 100.71 / 186.97 s |
| Median / p95 end-to-end latency | 147.36 / 212.43 s |
| Median / p95 per-request time per output token | 45.55 / 46.46 ms |

The server allocated **3.23 GiB of KV cache**, **250,091 token slots**, reporting
**7.63x full-32k concurrency**. Logs showed seven running requests while the
remainder waited. The preemption counter stayed at **zero**: this run queued
requests instead of repeatedly discarding/recomputing their contexts.

Post-run memory usage was 39,201 / 39,125 / 39,125 / 39,125 MiB across the four
40,960-MiB GPUs. The server stopped automatically after collecting the results.

This is a **first-burst** measurement: one full-context warm-up was excluded,
but additional batch-shape JIT compilation occurred during measured inference.
It is not a fully warmed steady-state result. The 371 tokens/s short-window
output peak is not the whole-batch throughput. The client/server counters
agree on all tokens (server counters include the one additional warm-up).
Artifacts: `.cache/flash-next-a100/runs/76328/benchmark-32k-c28.json`,
`benchmark.log`, `server.log`, and `metrics-{before,after}.txt`.
