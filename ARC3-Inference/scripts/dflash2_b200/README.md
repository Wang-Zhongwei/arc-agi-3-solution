# DFlash2 k=7 on a Frida B200

Boots the Kaggle serving stack of three runs on B200s instead of Kaggle and replays recorded episodes against
each server:

| Arm | Kaggle run | Drafter | `prefix_match_unit` |
|---|---|---|---|
| A | dflash2-k7 v3 (`configs/dflash2-k7/v3.json`) | DFlash2, 7 tokens | 16 |
| B | dflash2-k3 v2 | DFlash2, 3 tokens | 128 |
| C | porting-advaith v6 | Lord NVFP4 MTP, 3 tokens | 128 |

k=7 failed on Kaggle at boot (dflash2-k7 v2): 7 draft tokens add 4 rows to every GDN conv state, the
attention block grows from 1664 to 1744 tokens to stay at least as large as the GDN page, and 1744 is not a
multiple of 128. With `prefix_match_unit` 16 every group's block size divides.

What runs is what Kaggle ran: `make_arms.py` takes each run's rendered setup script from its
`setup_commands.json`, and `run_arm.py` replays its main flow (Lord's image layers, overlay e3a6fe0d, PLE
fixes, DFlash patches, server command, watchdog, smoke test). B200 changes: no RTX GPU-name check,
`TORCH_CUDA_ARCH_LIST=10.0`, a port per arm, and `--kv-cache-memory-bytes` set to the pool each Kaggle run
reported, so prefix-cache eviction matches. The model's SM120 GEMM path switches itself off on SM100.
Throughput on a B200 is not a Kaggle number; acceptance, prefix-hit rate, KV use and boot are comparable.

```bash
python3 make_arms.py runs/kaggle/jonathanwang2022 OUT         # setup_A.py, setup_B.py, setup_C.py
# copy OUT/setup_*.py and this folder to /shared/workspace/arc-agi-3/dflash2_b200/run on Frida
bash stage.sh                                                  # login node: layers, drafter, /kaggle/input
sbatch --reservation=maint_arc --nodelist=ixb2 run.sbatch      # or the open partition
```

Load: `bench.py` replays recorded GRPO episodes (porting-advaith v6 harness, Frida bench_b200 workload
208013) at 14 streams with the harness's sampling (T 0.7, top-p 0.95, top-k 20); prompts grow by the server's
own replies plus the recorded observations, so the prefix cache sees multi-turn reuse. Images are replaced by
a text id.

## Results (10-31, ixb2, 14 streams, 900-s window)

k=7 boots with `prefix_match_unit` 16 (groups 1744 x 6 + 16, KV 1,580,652 tokens as on Kaggle). Both DFlash
arms then ran into a prefix-cache storm, which two runtime patches in `DFLASH_RUNTIME_PATCHES` fix:

- **Draft-group rule.** vLLM applies the EAGLE drop only to the drafter's KV groups, and when it identifies
  none ("no KV cache group could be identified as the draft model's") to every group. The target's GDN groups
  then drop a recurrent checkpoint that align mode does not keep, and a turn's hit falls to 0 or to the
  system prompt. DFlash2's five sliding-window layers register last, as MTP's do, so adding `dflash` to
  `_uses_trailing_mtp_layers` flags exactly their group.
- **Replay-tail hold.** The drafter's sliding-window group frees the window blocks a hit at the replay
  boundary needs while the request runs, so they are evicted first. The request now keeps them until it
  finishes (`cpu_hold_test.py` checks the bookkeeping on CPU).

| Job | Arm | Patches | Prefix hit | Turns with total miss | Prompt tok/s | Gen tok/s | Acceptance per position |
|---|---|---|---|---|---|---|---|
| 209821 | A k=7 | none | 42.1% | 166/331 | 24.7K | 505 | .754 .559 .418 .324 .258 .211 .176 |
| 209824 | A k=7 | hold | 48.0% | 158/363 | 27.1K | 539 | .747 .555 .415 .320 .257 .209 .177 |
| 209829 | A k=7 | rule + hold | **90.1%** | **0/835** | 64.4K | **1,329** | .744 .552 .414 .318 .253 .207 .174 |
| 209821 | B k=3 | none | 48.3% | 147/358 | 26.9K | 545 | .791 .619 .485 |
| 209824 | B k=3 | hold | 53.9% | 142/385 | 29.5K | 588 | .777 .600 .461 |
| 209829 | D k=3 | rule | 62.6% | 116/448 | 33.9K | 681 | .786 .612 .478 |
| 209829 | B k=3 | rule + hold | **90.1%** | **0/814** | 63.2K | **1,315** | .780 .600 .463 |
| 209821 | C MTP-3 | n/a | 92.8% | 0/812 | 62.7K | 1,298 | .785 .610 .470 |

"Total miss": a continuing turn of an episode with at most 20% of its prompt cached. With both patches,
DFlash2 reaches MTP-3's hit rate and throughput on the B200; k=7 (mean acceptance length 3.66) is +2% over
MTP-3 (2.87), k=3 +1%. B200 tokens/s are not Kaggle numbers.

Patched setups: `make_arms.py RUNS OUT --repo-patches` (this checkout's `DFLASH_RUNTIME_PATCHES`), then
`SETUPS=$D/run/fix ARMS="A B" sbatch --reservation=maint_arc --nodelist=ixb2 --gpus-per-node=B200:2 run.sbatch`.
Arm D was setup_B with the hold entry removed and port 8004.
