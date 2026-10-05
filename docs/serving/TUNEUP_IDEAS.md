# Serving tune-up ideas: Daniel v3 plus the other Milestone 2 winners

Sources:

- The combined design `_checkpoint/winners/COMBINED-DESIGN.md` (kept outside Git). Every claim in it cites a file and a line. It was built offline: nothing ran on a GPU and nothing was submitted.
- The queued fp32-SSM run `matijaludvig/arc3-ssm-fp32-25x1b`. There are no results yet.
- See [MEMORY_BUDGET.md](MEMORY_BUDGET.md) for the GPU and host memory split each winner uses.

This note keeps only the ideas that matter for the serving engine. Behaviour ideas are listed briefly at the end.

## Base: Daniel (dfranzen) v3, unchanged with every new flag off

Most of the other winners' good serving ideas are already in v3:

- FP8 KV cache.
- The model's reasoning is sent back in the history.
- Long history with rare, large trims.
- The Mamba state stays on the server.

We keep all of Daniel's serving setup:

- His trim policy and his token estimate.
- How a game keeps its server slot between tool calls.
- His priority formula.
- His 128K context.
- His server patches.

His log shows 93.5% of prompt tokens served from the prefix cache. Any change has to keep that hit rate.

## Taken from the other winners

**lordhansolo**

- Keep game state in memory instead of rewriting the full history as JSON after every action. This gives about +1.5% tokens.
- A watchdog that restarts a crashed server.
- Metrics logging.
- His vLLM patches are dropped. They only apply to vLLM, or Pennyroyal already does the same thing.

**sirikilohit**

- His SGLang patch that moves the embedding table to host RAM. It frees GPU memory for KV, the same trick lordhansolo uses for `embed_tokens` (see MEMORY_BUDGET).
- Dropped: his 70K context, his 16 streams and his HiCache. His logs show no benefit from any of them.

## Our throughput ideas (step 0)

| Idea | Expected gain | Notes |
| --- | --- | --- |
| Prefill chunk 8K → 16K/32K | ~+1.1–1.4% | Check that prefill activations still fit next to the CUDA graphs. |
| Stop re-prefilling the model's own last output | up to +2–3% | The cause is timing, not eviction. Diagnose it offline first. |
| More KV room, spent on **one** of two options, chosen by measurement | | |
| ↳ One more resident game, with at most 10 running at once | +2–4% | |
| ↳ A larger context window, so trims happen less often | not estimated | |

Sources of more KV room: the embedding offload above, and the memory-fraction change in MEMORY_BUDGET.

## Build order

1. An offline replay that proves each change is a no-op when its flag is off.
2. Keep game state in memory.
3. Robustness: retry with auto-reset, checks at boot, the watchdog.
4. The first GPU run. Judge it on tokens per second. The score only has to not drop.
5. The throughput items above.
6. Behaviour changes, each tested alone with alternating runs: 8 passes against 8 control passes.

## Constraints

- **Boot budget.** Daniel's notebook runs under a 9-hour budget, not 12 hours. Anything that adds boot time has to fit in about 8 spare minutes.
- **One pool of GPU memory.** The fp32 SSM state (below), the extra resident game and the larger context all draw on the same freed memory. If fp32 helps, decide between them using measurements.
- **Upside is small.** All the throughput work together is estimated, not measured, at about +0.7 to +4 local points. That doesn't take Daniel's hidden 25–31 to 45. A bigger gain has to come from behaviour, mainly on games where the agent never clears level 1. No design addresses those games yet.

## Experiment: fp32 SSM state (`matijaludvig/arc3-ssm-fp32-25x1b`, queued)

This is Daniel v3 with one change: the recurrent (Mamba/GDN) state is stored in fp32 instead of bf16.

**Why test it**

- The model's `config.json` asks for fp32 (`mamba_ssm_dtype = float32`), but Daniel serves the state in bf16.
- The state covers 36 of the 48 layers. It is carried through every token of contexts up to about 116K, so rounding errors could build up over a long game.
- Daniel chose bf16 because the fast FlashInfer kernels on this GPU require it, not to improve quality.

**What the run changes** (cell 12 only, enabled in the notebook itself)

- Each game's state slot takes twice the GPU memory.
- During decode, MTP verify and prefill, those layers run on Triton kernels instead of FlashInfer.
- State slots drop from 60 to 48. Daniel's run used at most 40. The KV pool shrinks from ~1.01M to ~0.85M tokens.
- Two safety checks:
  - If the server crashes during startup, the run stops instead of playing without a server.
  - After startup, a check confirms the state really is fp32 and the kernel really is Triton.
- 25 games × 1 pass (`bm.n_passes = 1`).

**What it answers**

1. Does the server start with fp32 together with Daniel's other server settings (MTP drafting and the state-recovery mode)? This can't be checked offline.
2. Decode speed, compared with Daniel's 820 tok/s at 10 games. Triton is expected to be slower.
3. Score, compared with Daniel's single-pass scores of 45.80, 45.59, 47.46 and 47.12.

## Not serving (for reference)

These come from the same design but are harness or behaviour changes:

- sirikilohit's solved-level memory, which he credits with +2.8. Put it in user messages so it doesn't break the prefix cache.
- lordhansolo's catalog of the model's saved functions with their docstrings. Test it as a separate change.
- Choose behaviour changes on part of the 25 public games and confirm them on the rest, to avoid overfitting.
