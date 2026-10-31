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
