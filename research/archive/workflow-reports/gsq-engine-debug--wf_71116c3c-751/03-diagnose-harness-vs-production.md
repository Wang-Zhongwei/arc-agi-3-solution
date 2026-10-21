# gsq-engine-debug: diagnose:harness-vs-production

Workflow run `wf_71116c3c-751`, phase Diagnose. Find and fix the CUDA illegal memory access in the GSQ engine self-test (both patched and int8 cases), add a production-path control case, and build a ~5-8 min GPU debug notebook

---

The most likely cause is a shared-memory overflow inside the stock Marlin MoE kernel. It only happens in the `int8` case's down GEMM (Q2_0 expanded to uint8b128 at group 64), at the M=8192 step. Production uses 4-bit weights and never gets near it. This is from reading the source and a CPU emulation. Nothing has been run on a GPU yet.

**Which M failed.** `first_call_s` is only written after the M=1 `torch.cuda.synchronize()` returns (`verify_gsq_engine.py:117-122`). So M=1 was clean and the crash came at M ∈ {10, 40, 64, 8192}. The harness doesn't save which M was running.

## Ranked causes

**1. The host check underestimates the kernel's shared memory (high confidence).**
- **Trigger:** 8-bit weights, group size 64 or more, and `moe_block_size=64`. With E=256 and top-10, `fused_marlin_moe.py:240-242` picks block size 64 once M ≥ 1106, so the M=8192 step hits it.
- **Host config pick:** for the down GEMM (prob_m=81920, n=2560, k=640) it chooses thread config (64,256,256) at 1 block per SM. It launches with 101376 B, the sm_120 per-block opt-in maximum (`moe_wna16_marlin.cuh:705-742`, `828-830`).
- **Why the host thinks it fits:** `get_kernel_cache_size` (`moe_wna16_marlin.cuh:197-243`) counts the per-block metadata as `tb_m*4` = 256 B (line 219). It then checks with a 512 B margin (lines 159 and 293). That gives 100608 + 512 = 101120 ≤ 101376, so the config passes.
- **What the kernel really uses:**
  - Metadata: `sh_new` sits at 2·moe_block_size int4 = 2048 B (`marlin_template.h:422-427`).
  - Reserve: the A-cache size `sh_a_max_row` subtracts a 1024 B reserve (lines 827-831).
  - Result: `sh_a_max_row` is 248, but the pipeline needs 4 stages × 64 rows = 256.
- **Why it isn't caught:** `max_num_stage_groups = max(...,1)` (lines 889-890) hides the shortfall. The kernel uses 256 rows anyway and ends at 102400 B, 1024 B past the allocation.
- **Where it faults:** A-tile writes (`cp_async4_pred` at lines 894-901) and `ldsm` reads (lines 980-984) go past the end for rows 56-63 of the last stage. An out-of-range shared-memory access surfaces as error 700, "an illegal memory access was encountered".
- **Sweep results** (`smem_sweep.py`): the overflow happens only for 8-bit with group 64, group 128 or per-channel scales, at block size 64.
  - The int8 g32 GEMMs fall back to (64,128,128) and fit, so `int8_exact` (all g32) would pass at 8192.
  - All 4-bit configs fit. Production W4A16 at 8192 uses (64,256,256) and ends at 70656 B.
- **Serving impact:** int8 serving mode would hit the same fault. The 9 Q2_0 layers' down GEMM would crash on any batch of 1106 tokens or more, which includes the 8192-token prefill chunks.
- **Origin in our code:** `gsq_lut.py:98-99` and `:166` keep Q2_0 at group 64 in the int8 expansion. `verify_gsq_engine.py:45` (the `int8` case) and `:145` (default `--m` including 8192) are what exercise it.

**2. An unknown bug in the uint8b128 kernel on sm_120 (low).** No production path runs 8-bit Marlin MoE. My static read and the emulation found no out-of-bounds global access, so only a compiled-code problem remains possible.

**3. Small overruns that also happen in production and are absorbed by allocator padding (very low):**
- The fill block in `moe_align_kernel.cu:73-79` rounds `max_num_tokens_padded` up to a multiple of 4. At 98111 that writes one int past the end; production does the same at 114239.
- Lock indices stay within the `sms*4` workspace (`marlin_utils.py:284-292`). The highest the emulation saw was 562 of 752.

## How the harness call differs from production

Production is `GPTQMarlinMoEMethod` → `moe_runner/marlin.py:116-205` → `fused_marlin_moe`.

| | Harness `int8` case | Production W4A16 |
|---|---|---|
| Weight type | uint8b128, `num_bits=8` for both GEMMs | uint4b8, 4-bit |
| w1 / w2 shapes | [256,160,5120] / [256,40,10240] int32, contiguous | [512,160,2560] / [512,40,5120] |
| Experts | 256 | 512 |
| Scale shapes | [256,80,1280] (g32) / [256,10,2560] (g64) | [512,20,1280] / [512,5,2560] (g128) |
| Scale dtype | bf16 | Created as fp16 (`gptq_moe.py:239,247`), but the assert at `fused_marlin_moe.py:224-229` needs bf16. I couldn't find where it gets cast, so this is unverified. It can't cause an out-of-bounds access either way. |
| g_idx / sort_indices | `None` | Empty [E,0] tensors; both give `has_act_order=False` |
| CUDA graphs | Eager | Decode runs under CUDA graphs |
| Block size at M=8192 | 64, with config (64,256,256) on the down GEMM (64,128,128 on gate_up) | (64,256,256), kernel ends at 70656 B |

These match production: workspace (`marlin_make_workspace(max_blocks_per_sm=4)` = 188·4), `use_atomic_add=True` (so `c_tmp` is empty), the block-size rule, the moe_align paths, int32 `topk_ids`, float32 `topk_weights`, `is_k_full=True`, and the 8-bit GPTQ packing and repack layout.

All shapes pass the Python asserts and the host `RuntimeCheck`s. The emulation tracks every global index (A, B, scales, C, `topk_weights`, `sorted_ids`, `expert_ids`, locks) against its allocation. For M = 1, 10, 40 and 64, each one's maximum is allocation − 1, with nothing out of range.

## Checks to run on the GPU
- `int8` should pass with `--int8-only --m 1,10,40,64` and at M=1105, and fail at 1106 and 8192.
- `int8_exact` should pass at 8192.
- `compute-sanitizer` should report an invalid `__shared__` write or read in the uint8b128 kernel with group_blocks=4 at the 64-row block size (template args: 256 threads, thread_m_blocks 4, thread_n_blocks 16, thread_k_blocks 4).

## Possible fixes (not applied; this pass was read-only)
- **Exact and smallest:** expand Q2_0 to int8 at group 32 by repeating each g64 scale twice along K.
- **Kernel-side:** add a runtime patch so `get_kernel_cache_size` counts `2*moe_block_size*16 + 1024` bytes of metadata and reserve, which makes the host fall back to (64,128,128).
- **Harness:** save the M that is running before each `moe_forward` call in `run_case`, so a crash shows which M failed.

Files are in /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/diagA:
- marlin_moe_bounds.py
- smem_sweep.py
- run8192.txt
