# Rolling extension design and experiments

The main inference configs select UCB. Rolling extension remains available for
experiments with broad initial coverage followed by more time for games with
promising progress. No model score improvement was measured by the original
implementation task.

Exact behavior, defaults, parameter mappings, and examples are maintained in the
[allocation reference](../../ARC3-Inference/docs/allocation.md). In particular, see
[budgets and timing](../../ARC3-Inference/docs/allocation.md#budgets-and-timing) and
[ranking and exploration](../../ARC3-Inference/docs/allocation.md#ranking-and-exploration).

## Design tradeoffs

Preserving sessions across pauses avoids restarting their reasoning and game
history. Keeping each session on its original serving endpoint also preserves
cache locality, but can leave slots idle when eligible games concentrate on one
replica. Small eligible pools offer little opportunity to measure reallocation
gains; a completed small public run does not establish a benefit on the larger
competition pool.

Cooperative slices preserve in-flight model responses and tool calls, at the
cost of imprecise slice durations. This makes timeout handling and graceful
worker shutdown important implementation concerns. See the reference for the
exact timing contract.

Broad initial coverage trades depth on promising games for an opportunity to
observe every game. For the historical 132-minute allowance, a 5/6 initial share
and 110 games at concurrency 28 imply about 440 minutes of full-cap initial
work, leaving 88 minutes of a 528-minute serving budget. This is illustrative
budget arithmetic, not measured performance or a current-config guarantee.

Nonzero exploration remains useful for reproducing earlier experiments:
adaptive-reasoning v3 used 0.5. Changing local defaults does not rewrite that
published version. A shorter scoring window and token-aware warm-up were
proposals, not changes implemented by this experiment.

For comparisons, explicitly match the shared serving budget across policies;
see the [parameter reference](../../ARC3-Inference/docs/allocation.md#parameters).

## Flash-Next FP8 feasibility: blocked before GPU allocation

Historical investigation, checked on 2026-09-12. The restriction below applies
to that pinned runtime. Later results with a different upstream runtime are in
[the H100 notes](../serving/flash-next-h100-2node.md).

Checked on 2026-09-12. The proposed topology is two separate nodes, each with
one H100. The dedicated runtime explicitly rejects multi-node PLE CPU offload;
the planned two-node/TP2/EP2 probe cannot proceed with this runtime.

Runtime provenance:

- Image: `vllm/vllm-openai:qwen38-flash-next`
- Linux/amd64 manifest: `sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`
- Installed package: `vllm-0.1.dev20073+g8e685d198`
- Package layer: `sha256:10dce885652a53a2e644a75d256b90830663330988b8599c3c0e56cd6c8a4675`

In that image, `usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py`,
`_validate_ple_offload_config()` (lines 212–251) checks
`parallel_config.nnodes != 1`, adds it to unsupported settings, and raises
`ValueError`. `spawn_ple_offload()` describes a node-local worker. The connector
uses shared CPU tensors and CUDA IPC outputs; the CPU worker expects every TP
rank's output buffers locally. This is an explicit runtime restriction, not a
memory-capacity estimate or evidence that distributed PLE is impossible in
principle.

The read-only inspection can be reproduced without pulling the full image:

```sh
crane blob vllm/vllm-openai@sha256:10dce885652a53a2e644a75d256b90830663330988b8599c3c0e56cd6c8a4675 | tar -xzOf - usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py | sed -n '212,281p'
```

The [official recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) uses the
dedicated runtime and reports a four-H100 PLE-offload configuration. That does
not validate two H100s on separate nodes. No model weights were downloaded, no
GPU allocation was submitted, and no throughput or game-score pilot was run.
Proceeding requires suitable single-node hardware or a runtime with supported
multi-node offload; implementing that support is outside this change. Full
uniform/rolling/UCB comparisons remain deferred until serving feasibility
and a pilot budget are agreed.
