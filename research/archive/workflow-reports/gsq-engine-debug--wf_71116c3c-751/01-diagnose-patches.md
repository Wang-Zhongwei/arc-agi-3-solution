# gsq-engine-debug: diagnose:patches

Workflow run `wf_71116c3c-751`, phase Diagnose. Find and fix the CUDA illegal memory access in the GSQ engine self-test (both patched and int8 cases), add a production-path control case, and build a ~5-8 min GPU debug notebook

---

The self-test crash is a stock Pennyroyal Marlin MoE bug, not one of the gsq patches. On sm120, an 8-bit (uint8b128) GEMM with group size 64 at `moe_block_size` 64 writes 1024 bytes past its dynamic shared memory. In the self-test that is the `int8` case's down GEMM (Q2_0 to int8, group 64) at M=8192. Patches 01, 02 and 03 only steer the run into that configuration, so reverting 02 and 03 in run 1 was a misattribution. Nothing ran on a GPU here; this comes from reading the source and a CPU model of the host and kernel arithmetic.

**Why M=8192.** The fault must come at some M in {10, 40, 64, 8192}. `verify_gsq_engine.py:120-122` records `first_call_s` only after a successful sync, so M=1 passed. M=1 to 64 all use block size 8 and the same kernels, and the model gives those at least 1.7 KB of slack. M=8192 is the only M here with block size 64, which happens for M ≥ 1106 at E=256, top-10 (`fused_marlin_moe.py:240-242`: 8192·10/256/64 = 5 ≥ 0.9, so the loop ends at 64). Which M failed can't be confirmed from the logs, because `rel_err` is written only after the whole M loop (`verify_gsq_engine.py:125`).

**The overrun** (files under the unpacked wheel `sglang/kernels/jit/csrc/gemm/marlin_moe/`). Down GEMM: N=2560, K=640, 8-bit, group 64 (group_blocks 4), thread_m_blocks 4.

1. **Host config (`moe_wna16_marlin.cuh`).** The host picks the large-batch config 64x256 with 256 threads.
   - Its estimate (lines 219-240) adds up as follows:

     | Part | Bytes |
     |---|---|
     | sh_b: 4·(64·256/4)·4 | 65536 |
     | sh_a | 32768 |
     | sh_s | 2048 |
     | block metadata: `tb_m*4` (line 219) | 256 |
     | **Total** | **100608** |

   - The validity check at line 293 passes: 100608 + 512 ≤ 101376.
   - At line 560 the per-SM count is 101376 / 102144 = 0, clamped to 1. The 64x128 config would also get 1, so it does not win.
   - Lines 742 and 828-830 then launch with 101376 B of dynamic shared memory.
2. **Kernel layout (`marlin_template.h`).**
   - Lines 422-427: `sh_new = sh + 2*mbs` int4, which is 2048 B at block size 64. The host counted 256 B.
   - Line 827: `shm_size_used` = 64 + 128 + 4096 = 4288 int4.
   - Line 831: `sh_a_max_row` = ((101376-1024)/16 - 4288)/8 = 248. The code comment says this is at least 4 stages × 64 rows = 256, which does not hold here.
   - Lines 889-890: `max_num_stage_groups` = ((248-64)/64+1)/4 = 0, floored to 1.
   - Lines 894/901 (cp.async writes) and 981-984 (ldsm reads) use A stage 3 at int4 [5888, 6400), which is bytes [94208, 102400).
   - Bytes 101376 to 102399 are out of bounds: rows 56-63 of pipe 3. They are written for every full block of 64 tokens, which at M=8192 is most blocks.

**Scope of the bug** (CPU model at `<scratchpad>/diagB/smem_model.py`, run over register counts 32-255, block sizes 8-64, and bits/groups 8/32, 8/64, 4/16, 4/32, 4/64, 4/128):
- The only overruns are 8-bit group 64 at block size 64, for both gate_up and down, 1024 B each.
- Everything else fits:

  | Path | Kernel needs | Available |
  |---|---|---|
  | Production W4A16 (4-bit, group 128), block 64 | 70656 B | 101376 B |
  | 8-bit group 32, block 64 (host picks 64x128) | 69632 B | 101376 B |
  | LUT kernels, group 16 or 32 | fits | fits |

- So `int8_exact` and every LUT case pass at M=8192.
- In int8 serving mode, every Q2_0 GEMM (9 in this checkpoint) would crash on any prefill chunk of 1106 tokens or more.

**Where patch 01 triggers it.** The table is in `serving/runtime_patches/gsq/gsq_lut.py` (embedded in 01). Line 57 sets `GROUP["Q2_0"] = 64`. Lines 165-174 keep group 64 for int8 and permute scales at group 64.

**Fix (not applied).** In `marlin_weights`' int8 branch, expand Q2_0 scales to group 32 with `scales.repeat_interleave(2, dim=-2)` and set `group = 32`. This is exact and needs no kernel edit; the model then picks 64x128, which fits.

**Checked and in bounds:**
- **Repack and permute:** `gptq_marlin_moe_repack` output is [E, K/16, 4N]. The repack host checks the input as [K/4, N] via TensorMatcher. Scale shapes are [E, 80, 1280] and [E, 10, 2560]. `scales_expert_stride` matches.
- **Packing helpers:** `pack_gptq8`, `unpack_gptq4` and the IQ2_S merge are torch ops, so a shape error would raise rather than read out of bounds.
- **Workspace and locks:** the workspace is 752 ints, at least the minimum required. `locks_off` is at most gridDim (376 for block size 8), which is under 752.
- **Reduction buffer:** `c_tmp` is unused, because sm120 with BF16 takes the atomic-add path.
- **Token routing:** `sorted_token_ids` and `expert_ids` sizes cover every padded block for both the single-token path and the block-align path.
- **Patch 02:** with default arguments it is the stock call, and int8 mode passes no extra kwargs. The per-GEMM bits only change scalar types and asserts.
- **Patch 03:**
  - `kHasLut=false` code is byte-identical to stock, under a new JIT key; run 2's 99 s first call confirms a fresh stock compile.
  - The LUT is a 4×int32 device tensor that stays alive for the call; the host checks its size is 4 and the kernel reads exactly 16 bytes.
  - The LUT dispatch only instantiates uint4b8 at group_blocks 1 or 2. Shared-memory sizing agrees between host and kernel; at group 16 the scale tiles fit the thread count.
- **Server path:** `replace_parameter` re-registers a parameter when its size changes, so no 4-bit buffer is reused for 8-bit weights.

**Ways to confirm on the next GPU run:**
- Run `--m 1,10,40,64` and `--m 8192` separately: the first should pass and the second should fault.
- Or run the 8192 case under `compute-sanitizer` and look for an "Invalid __shared__ write" in Marlin.
- Recording progress after each M in `run_case` would make the failing M visible in future reports.

Nothing was committed to the branch. The model script is at `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/diagB/smem_model.py`, and the unpacked wheel sources are under `.../scratchpad/diagB/wheel/`.
