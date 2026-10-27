# GRPO

Our own asynchronous GRPO loop for Qwen3.8-Flash-Next on the ARC-AGI-3 games. The evaluation harness plays levels 1–2 against vLLM; the trainers learn a LoRA (attention + GDN) from the exact tokens vLLM sampled; the merged weights go back into the running server after every wave.

**Status (2026-10-27):** built 2026-10-26, CPU-tested (34 tests and a CPU example), **never run on a GPU**. The first run's gates catch a broken setup in the first waves. The B200 container and the 10-game config are new and not run either.

## Read first

| Doc | What it holds |
|---|---|
| [grpo-pipeline.md](../../../docs/RL/grpo-pipeline.md) | The design: episodes, training sequences, reward, loss, FP8 caches, weight sync, settings, gates, CPU example |
| [decisions.md](../../../docs/RL/decisions.md) | Every decision with its date and reason: stack, budget, reward, games, hardware, when to stop |
| [grpo-game-ranking.md](../../../docs/RL/grpo-game-ranking.md) | All 25 games ranked by signal per group; the 10 proposed for the B200 run |
| [grpo-l1-l2-token-budget.md](../../../docs/RL/grpo-l1-l2-token-budget.md) | The episode budget min(1,750 × human actions of L1 + L2, 130K) context tokens |
| [grpo-training-throughput.md](../../../docs/RL/grpo-training-throughput.md) | Trainer speed (H200: 2,878 tokens/s at 110K) and why 6 of 8 rollouts per group |
| [h200-rollout-serving.md](../../../docs/RL/h200-rollout-serving.md) | vLLM flags and rollout benchmarks (TP=2, 64 rollouts: 3,920 generated tokens/s) |
| [rft-v1-kaggle-ab.md](../../../docs/RL/rft-v1-kaggle-ab.md) | Why GRPO starts from the base model, not the RFT model |
| [rft/README.md](../rft/README.md) | The trainer GRPO builds on (`train_rft`): model, patches, results |
| [rft/AGENTS.md](../rft/AGENTS.md) | Pitfalls: chat template, loading, QSA, memory, pods. Read before changing anything that runs on a GPU |
| [rft/Throughput.md](../rft/Throughput.md) | Trainer speed history, including B200 vs H200 (2026-10-20; Chinese) |
| [CONFIGURATION.md](../../CONFIGURATION.md) | Harness settings (`harness_env`) |
| [docs/rules](../../../docs/rules/README.md) | Repo rules: branches, reports, docs |

## Harness

The harness is **porting-advaith v6** (main a3b57e6e, 59.83 on Kaggle), the best run so far, plus the GRPO-mode hooks, which the model never sees (`rl` since 267f9780). Training and evaluation must see the same thing ([decisions.md](../../../docs/RL/decisions.md), Harness):

- Same as v6: prompts, tools, observations (`analyzer`, `multimodal`, model-facing `harness_env`, including `ARC3_GAME_PRIORS=1` and `ARC3_TOUCHING_SPLIT=0`).
- Different by design: no context trimming (an episode is one growing sequence, capped by its budget), no turn yield, 8 rollouts per game, 64–80 streams.

## Code

| File | Role |
|---|---|
| [loop.py](loop.py) | Waves, rewards, jobs for the trainers, weight syncs, gates, state and resume |
| [rl_record.py](rl_record.py) | Harness GRPO mode (`ARC3_RL_RECORD=1`): token ids and logprobs per request, prefix chain, budget, episode end |
| [episodes.py](episodes.py) | Training sequences from what vLLM saw: trained spans, forced `</think>` mask, images |
| [reward.py](reward.py) | Reward with the thinking factor, advantages, 6-of-8 selection, minibatches, trainer balance |
| [loss.py](loss.py) | The sampler's processed logprob, clipped objective with the ratio mask, stochastic BF16 |
| [train_grpo.py](train_grpo.py) | One trainer process per GPU: LoRA data parallel over NCCL, merged-weight export, probes |
| [vllm_sync.py](vllm_sync.py) | vLLM worker extension that loads the merged tensors in place |
| [cpu_example.py](cpu_example.py), [fake_vllm.py](fake_vllm.py) | The whole pipeline on a laptop, with a tiny model / a stand-in server |
| [pod/grpo_pod.sh](pod/grpo_pod.sh) | The run, end to end: model, venvs, server, CUDA tests, trainers, loop |
| [pod/patch_fla_sm100.py](pod/patch_fla_sm100.py) | fla's GDN backward on B200 |
| [container/Dockerfile](container/Dockerfile) | Image for a GPU node: vLLM nightly + trainer venv + harness venv |
| [bench_rollout.py](bench_rollout.py), [build_replay.py](build_replay.py), [pod/rollout_*_pod.sh](pod/), [pod/spec_bench_pod.sh](pod/spec_bench_pod.sh), [profile_summary.py](profile_summary.py), [rollout_env_report.py](rollout_env_report.py) | The H200 serving benchmarks |
| Configs | [configs/grpo/v1.json](../../configs/grpo/v1.json) (round 1: 8 games, 64 rollouts), [configs/grpo-b200/v1.json](../../configs/grpo-b200/v1.json) (top 10, 80 rollouts) |

## Run on a B200 node

**Needs:** 8 × B200 (assumed; 4 also work), NVIDIA driver ≥ 580 (the image is CUDA 13.0), Docker with the NVIDIA runtime, ≥ 400 GB disk for `/workspace`, Hugging Face access while the model downloads, and a Kaggle account that accepted the [competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) rules (game files). Host memory: the trainers abort below 120 GB of pinned arena each, after 163 GB for the shared PLE cache, processes and harness: with 4 trainers, ~650 GB must be free once the server is up.

1. Build the image from the commit to train (only tracked files go in):

   ```bash
   git checkout rl
   git archive --format=tar HEAD ARC3-Inference tufa-arc-agi-framework docs/RL | docker build \
     --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t arc3-grpo:$(git rev-parse --short HEAD) \
     -f ARC3-Inference/inference/grpo/container/Dockerfile -
   ```

   Add `--build-arg GPU_ARCH=sm90` for H200.

2. Get the game files:

   ```bash
   kaggle competitions download -c arc-prize-2026-arc-agi-3 -p /data/arc3/env
   cd /data/arc3/env && unzip -q arc-prize-2026-arc-agi-3.zip
   ls /data/arc3/env/environment_files   # one folder per game
   ```

3. Start the run. It downloads the checkpoint (169 GB, once), then works through the phases in [grpo_pod.sh](pod/grpo_pod.sh); nothing waits for input:

   ```bash
   docker run -d --name grpo --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
     -v /data/arc3:/workspace \
     -e CONFIG=configs/grpo-b200/v1.json -e SERVER_GPUS=0,1,2,3 -e TRAINER_GPUS=4,5,6,7 \
     -e MODEL_DIR=/workspace/w4a16 -e ENV_DIR=/workspace/env/environment_files \
     -e WANDB_PROJECT=arc3-grpo -e WANDB_API_KEY \
     arc3-grpo:<tag> bash -c 'mkdir -p /workspace/grpo && cd /workspace/grpo \
       && exec bash $REPO/ARC3-Inference/inference/grpo/pod/grpo_pod.sh > run.log 2>&1'
   ```

4. Watch `/data/arc3/grpo/`:
   - `run.log`: the phases, and why it aborted if it did (`ABORTED` holds the reason).
   - `loop_log.jsonl`: one row per wave and step: rewards per game, gates, clipping and ratio statistics, timings. Also in W&B.
   - `results/`: server, trainer and loop logs, the CUDA tests (`pytest_*.log`), server counters (`metrics.jsonl`), GPU use (`gpu.csv`).
   - `waves/local/*_wave_*/`: each wave's harness run, viewable with `make view VIEW_RUN_DIR=...`.

5. Stop with `touch /data/arc3/grpo/STOP`: the loop finishes the wave in flight and ends. Running step 3 again resumes from `loop_state.json` and `ckpt/` (a resume loses at most one wave).

### GPU layout

The loop drives one vLLM server (one URL) and any number of trainers. Rollouts set the pace, so spare GPUs go to the server.

| Layout | Server | Trainers | Status |
|---|---|---|---|
| 8 × B200, proposed | `SERVER_GPUS=0,1,2,3` (TP=4) | `TRAINER_GPUS=4,5,6,7` | Not measured. If the server fails to start or is slower than TP=2, use the next row |
| 8 × B200, fallback | `SERVER_GPUS=0,1` (TP=2, the H200-tested shape) | `TRAINER_GPUS=2,3,4,5` | Not measured on B200 |
| 4 × H200 pod | `0,1` (defaults) | `2,3` (defaults) | Built for it; [grpo-pipeline.md](../../../docs/RL/grpo-pipeline.md#running-it) |

Rough sizes for the 10-game wave (80 rollouts) from historical attempts: ~8.0M context tokens, ~4.7M generated, ~6.0M trained (6 of 8). At the H200 rates, rollouts take ~20 min on one TP=2 server, and training takes ~17 min on 2 trainers or ~9 min on 4. B200 rates are not measured. KV: 80 × 130K = 10.4M tokens. A TP=2 H200 pair held 9.34M (79 GiB per GPU after ~35 GiB of weights); a B200 has ~1.3× the memory, so a TP=2 pair should hold ~13.5M (estimate).

### B200 differences

- **Trainer:** torch 2.8 has no `torch._grouped_mm` on sm100, so experts use the Triton grouped GEMM. fla 0.5.2's GDN backward needs [patch_fla_sm100.py](pod/patch_fla_sm100.py), and tilelang is Hopper only. The RFT trainer ran on a B200 on 2026-10-20 (64K: 3,772 tokens/s vs 4,031 on H200; LoRA gradients cos 0.9997 per layer); the GRPO path has not.
- **Trainer venv inside the vLLM image:** torch 2.8 (CUDA 12.8 wheels) in a CUDA 13.0 image. The RFT runs used a torch 2.8 image instead. The CUDA tests in phase 2 check it before the trainers start, and switch off any optional kernel whose test fails.
- **Server:** vLLM on sm100 is untested here. On H200 the INT4 experts ran on the `HUMMING` MoE backend; check which backend the server log names on B200.
- The CUDA tests and the step-1 gate compare against references, not against H200 numbers.

### Settings of grpo_pod.sh

| Variable | Default | Meaning |
|---|---|---|
| `CONFIG` | `configs/grpo/v1.json` | Harness config: games, rollouts, `harness_env` |
| `SERVER_GPUS` / `TRAINER_GPUS` | `0,1` / `2,3` | Server GPUs (TP = their count) / one trainer per GPU |
| `MAX_NUM_SEQS` | the config's `environment.concurrent_jobs` | vLLM `--max-num-seqs` |
| `STEPS` | 30 | Optimizer steps (waves 0..STEPS) |
| `LR`, `KEEP`, `MINIBATCHES`, `OPTIMIZER` | 1e-4, 6, 4, adamw | See grpo-pipeline.md, Settings |
| `MAX_HOURS` | 12 | Stops the run |
| `MODEL_DIR`, `ENV_DIR`, `WORK` | `/workspace/w4a16`, `$WORK/env/environment_files`, `/workspace/grpo` | Checkpoint, game files, run directory |
| `ARENA_MAX_GB`, `ARENA_MIN_GB` | 200, 120 | Pinned host arena per trainer |
| `WANDB_PROJECT` | off | W&B logging (needs `WANDB_API_KEY`) |
| `POD_END` | stop | RunPod only: stop or delete the pod at the end |

## On a laptop

```bash
cd ARC3-Inference
make grpo-cpu-venv                   # .venv-grpo: torch 2.8, transformers 5.20.1, peft
make grpo-cpu-example                # the whole pipeline on a tiny model, about a minute
PYTHONPATH=.:../tufa-arc-agi-framework/src .venv-grpo/bin/python -m pytest -q tests/test_grpo_*.py
```

Breakpoints, in the order the data flows: [grpo-pipeline.md](../../../docs/RL/grpo-pipeline.md#cpu-example).

## Open

- First GPU run: the gates (wave 0, step 1, sync 1) decide whether the setup is sound.
- Games for the B200 run: the top 10 of [grpo-game-ranking.md](../../../docs/RL/grpo-game-ranking.md) are proposed, not decided.
- Held-out checks (L3 and untrained games) are not in the loop yet ([decisions.md](../../../docs/RL/decisions.md), When to stop).
- Two servers (e.g. 2 × TP=2) need the loop to sync and split waves across URLs: not built.
- The Dockerfile has not been built yet.
