# gsq-engine-debug: diagnose:int8-sm120

Workflow run `wf_71116c3c-751`, phase Diagnose. Find and fix the CUDA illegal memory access in the GSQ engine self-test (both patched and int8 cases), add a production-path control case, and build a ~5-8 min GPU debug notebook

---

**Diagnosis C: 8-bit (uint8b128) Marlin MoE on sm120 in Pennyroyal d00d88ef**

**Verdict.** Stock Pennyroyal does build and dispatch 8-bit Marlin MoE kernels, but on sm120 they are not safe for every group size. At group 32 they fit at every block size. At group 64, group 128 or per-channel, prefill batches (moe_block_size 64, which is M ≥ 1106 tokens at top-10 over 256 experts) launch a config whose shared memory overruns the 99 KiB per-block limit by 1024 B. That fits the self-test's illegal memory access. The int8 fallback serves Q2_0 down at 8-bit group 64, so the fallback must change. This is reasoned from source and replayed on CPU; it has not been run on a GPU.

**1. How the kernel is built: JIT, not prebuilt.**
- `fused_marlin_moe.py:18` imports `moe_wna16_marlin_gemm` from `sglang.kernels.ops.moe.moe_wna16_marlin`, not from sgl_kernel.
- `moe_wna16_marlin.py:18-26` builds it with `load_jit("moe_wna16_marlin", ...)` from `gemm/marlin_moe/moe_wna16_marlin.cuh`. The cache key is only (dtype, is_ep, has_bias), so one compile covers every quantization type, uint8b128 included. That is the 99 s first call.
- The target is the local GPU: `kernels/jit/utils/arch.py:72-93` picks `sm_120f` (nvcc 13.0 is at least 12.9), and `:114-136` adds `-DSGL_CUDA_ARCH=1200 -O3`.
- The JIT sources in the mirror match the wheel exactly (sha256 equal to `kaggle/tests/fixtures/pennyroyal-v253/SHA256SUMS`).

**2. 8-bit kernels and sm120-specific code.**
- `moe_wna16_marlin.cuh:461` (`COMMON_GET_IF(host::kU8B128)`) instantiates 8-bit kernels at group_blocks -1, 2, 4 and 8 (per-channel, g32, g64, g128). `:468` adds the act-order variants.
- There is no sm120-specific path in the kernel. The only effects of the GPU are `use_atomic_add` being true for capability 9 and up (`fused_marlin_moe.py:300-303`, so `c_tmp` is unused) and the opt-in shared memory size of 101376 B (99 KiB on SM120, 188 SMs on the RTX PRO 6000).
- Pennyroyal's own Marlin MoE tests cover only uint4 and uint4b8, never 8-bit (`test/registered/kernels/ops/moe/test_moe_wna16_marlin.py:209`, `test/registered/quant/test_marlin_moe.py:104-106`).

**3. Root cause: the host's shared-memory estimate is too small for the device layout.**
- The host undercounts the per-block metadata: `get_kernel_cache_size` uses `sh_block_meta_size = tb_m*4` bytes (`moe_wna16_marlin.cuh:219`), but the device reserves 2·moe_block_size int4, which is 32·moe_block_size bytes (`marlin_template.h:422-427`).
- The device then sizes its A-tile ring from what is left: `sh_a_max_row = ((max_shared_mem-1024)/16 - shm_size_used)/(2*thread_k_blocks)` (`:827-831`). Its comment assumes there is room for at least stages × moe_block_size rows.
- If there is not, `max_num_stage_groups` is clamped to at least 1 anyway (`:889-890`). A tiles are then written and read at `sh_a + moe_block_size*a_sh_stride*pipe_a` (`:894-901`, `:981-984`) past the end of the shared memory allocation.
- Concrete case, the self-test's int8 down GEMM (Q2_0 → int8 g64, K=640, N=2560, M=8192, moe_block_size 64):
  - The host accepts config (64,256,256): 100608 + 512 ≤ 101376 (`:151`, `:293`). allow_count is 1, so the launch gets the full 101376 B (`:742`, `:830`).
  - On the device, sh_a_max_row is 248, below the 256 rows needed (4 stages × 64). The kernel touches up to 102400 B, which is 1024 B over (A rows 56-63 of stage 3). Shared memory out of bounds normally reports as cudaErrorIllegalAddress.
- My CPU replay of the host config choice and the device layout, swept over register counts and every block size:
  - 8-bit at g64, g128 and per-channel overflows at moe_block_size 64 for both expert shapes (2560→1280 and 640→2560).
  - 8-bit g32 is fine at every block size: the host rejects (64,256,256) because it needs 102656 B and falls back to (64,128,128).
  - Every 4-bit case is fine, including g16 and g32 for the LUT kernel and production g128. That matches W4A16 running every day.
  - There is no overflow at sm90 or A100 shared memory sizes, so this is a 99 KiB-GPU problem (sm120, sm89).
- This matches the run: `first_call_s` 99.32 was recorded after the sync at M=1, so M=1 passed. The case died at a later sync (`verify_gsq_engine.py:120`) about 6 s later. M=8192 is the only M the replay predicts to crash, in the w2 GEMM. The w13 GEMM (IQ2_S → int8 g32) fits.
- Run 1, with patches 02 and 03 applied, failed on the same int8 case first. Reverting 02 and 03 was therefore triggered by a stock-kernel bug, and the LUT cases have never been tested.

**4. Upstream and known issues.**
- Upstream SGLang main has a byte-identical `marlin_template.h`. Its `moe_wna16_marlin.cuh` differs only by an sm90 heuristic and one MXFP4 config, so the bug is not fixed upstream (checked via raw.githubusercontent).
- vLLM's current kernel has no A-tile ring and counts the metadata as `tb_m*16` (`csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu`).
- I found no report of this exact bug. Related sm12x Marlin MoE illegal-memory-access reports have different mechanisms:
  - vLLM #36811, #36889 and #43730: `c_tmp` out of bounds on SM 12.0. Not applicable here, since SGLang uses atomic add on capability 9 and up.
  - vLLM #56064: an SM121 no-split-K crash at M=256.
  - vLLM #41955: vLLM's MarlinExperts refuses int8 MoE outright.

**5. What to change in the int8 fallback.**
- **Recommended:** expand Q2_0 to int8 at group 32 by repeating each g64 scale twice, so every 8-bit GEMM is g32. This is exact, needs only patch 01, and costs about 118 MB of extra scales. It affects the 9 Q2_0 down GEMMs in the converted checkpoint (`convert.log` type_counts). The change is in `gsq_lut.expand_int8` and the `group = 32 if typ == "IQ2_S"` line in `marlin_weights` (`gsq_lut.py:166`). Then update the mode tables in `gsq_lut.py`, `runtime_patches/gsq/README.md` and `CUSTOM.md`, and re-run `pin.py`.
- **Alternative:** serve Q2_0 down as stock uint4b8 g64. It is exact and fits at every size, but needs patch 02 for mixed bit widths, so the int8 mode would no longer be patch-01-only.
- Moving all GEMMs to uint4b8 g32 as you suggested is not needed and would be lossy, since the IQ tables are non-uniform.
- **Optional:** patch `is_valid_config` to count the metadata as 32·moe_block_size bytes and reject configs where sh_a_max_row is below stages × moe_block_size. That would make every 8-bit group safe.

**6. GPU confirmation for the next run.**
1. Have `verify_gsq_engine` record the M in progress in its report.
2. Run `--int8-only --m 1,10,40,64`; it should pass.
3. Run `--m 8192` on the unchanged code; it should fail.
4. Run `--m 8192` with the Q2_0 g32 change; it should pass.

Nothing was committed. Files are in /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/:
- marlin_smem_sim.py
- marlin_smem_sweep.py (the sweep; it can become a `kaggle/tests` check)
- web/ (fetched vLLM and SGLang sources and issues)
