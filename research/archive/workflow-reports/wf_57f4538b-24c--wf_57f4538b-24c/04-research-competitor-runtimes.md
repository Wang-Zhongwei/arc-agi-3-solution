# wf_57f4538b-24c: research:competitor-runtimes

Workflow run `wf_57f4538b-24c`, phase Custom research. 

---

Ranked optimizations for the custom GSQ-RCO Coder runner. Full notes with all file:line citations are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/runner-research/competitor-runtimes-NOTES.md`.

**Our operating point**, from the F001 v2 serve.log in `runs/kaggle/v4-fable-f001/v2/serve.log.xz`:
- **Decode speed:** 760 tok/s at batch 9 and 785 tok/s at batch 10. Accept length is 2.58–2.75.
- **Rows per step:** at batch 10, the verify pass and the draft-extend pass each run 10 streams × 4 draft tokens = **40 rows**.
- **Startup:** weight load 440.55 s for 69.85 GB, then 145.75 s to capture the target-verify CUDA graph.
- **Expert backend:** both target and draft experts run on Marlin.
- **Key finding:** every competitor's fast path stops before 40 rows. Gabriel's low-M GEMM and the rowwise-FP8 lm_head both cap at 32 rows (`0004:60`, Pennyroyal `sm120_online_fp8.py:20`). The fused HC mix caps at 16 (`hc_mix_triton.py:33`).
  - Above 16 rows the HC mix falls back to a compiled BF16 path. With online FP8 on, it also converts both FP8 mix weights back to BF16 on every call (`hyperconnection.py:286-315`).
  - Above 32 rows the FP8 lm_head is converted back to BF16 in 64 MB chunks.
  - This explains our −27% with online MXFP8 (555 vs 765, `arms.md:103`), against Pennyroyal's +28% at 1 stream and +15–39% at 4 streams (`README.md:158-162`, `RESULTS.md:67-70`).
- **No-op knobs:** our launcher sets `SGLANG_SM120_LOWM_FP8_WEIGHT` and `SGLANG_SM120_LM_HEAD_FP8` (`launch_sglang.py:281`), but nothing in our wheel reads them. Gabriel's 0005 and 0006 were never ported (0 hits in the builder and in the Pennyroyal source), and bench C4 is disabled.

**Ranked, with expected effect at batch 10:**
1. **Serve 256 experts natively on Marlin INT4 (symmetric).** This is the checkpoint's own gain, not a competitor's.
   - Expert reads per verify drop from 33.9 GB to 24.8 GB. That assumes uniform routing, so it is an upper bound; RCO kept the most-used experts, so the real saving is smaller.
   - Expected **+10–25% decode**, about 30 GB less VRAM (70 → ~40 GB) and about 75 s less load time.
   - Marlin needs no per-shape tuning table. It picks its block size from `M*topk/E` (`fused_marlin_moe.py:240`); with 256 experts that gives 8 at decode and 64 in prefill.
   - Avoid backends that need tuning: Triton fused-MoE has no table for 256 experts × N=640 on the RTX PRO 6000 Server Edition (it would run untuned), and FlashInfer CUTLASS autotunes at every fresh Kaggle start (Gabriel reports about 20 min on first start).
   - Before converting anything, measure the real routing skew on the W4A16 with `--expert-distribution-recorder-mode`.
2. **Gabriel 0005 + 0006: FP8 dense weights, FP8 lm_head and FP8 HC mix (weight-only, activations stay BF16).** Port them onto Pennyroyal's existing rowwise-FP8 helpers and raise the row limits to 64.
   - His gains: 1 stream 147.9 → 163.5 → 188.8 tok/s; 4 streams 428 → 477 → 529 (`0005:12-14`, `0006:12-13`).
   - After pruning, dense weights are about 25% of the bytes read per verify; this halves them from 8.5 to 4.3 GB. Expected **+8–15%**.
   - It is pure Triton/Python, so it can be a runtime patch.
   - The GSQ dense layers are already 4–6-bit K-quants, so FP8 costs essentially nothing in quality relative to the GGUF.
   - Prefer this over fixing online MXFP8: no activation quantization and no autotune. Fixing MXFP8's row caps is the alternative; its gate is beating 826 tok/s (`arms.md:336`).
3. **Relaxed MTP acceptance**, 0.5 then 0.3 (Gabriel `serve.sh:73-77`). His ladder at 1 stream and temperature 0.6: 179 → 203 → 231 tok/s. Expected **+10–20%**. It is lossy at our temperature 0.7, so it needs its own score A/B (bench C1/C2 and the gated `tput-accept05` arm already exist).
4. **Low-M BF16 GEMM limit from 32 to 64 rows, without split-K** (bench C3 `m64.py`/`nosplitk.py` already exist). Gabriel saw the verify graph go from 14.43 to 13.79 ms. Expected **+1–5%**; mostly replaced by item 2 where that lands.
5. **Fused HC mix limit from 16 to 64 rows**, with the persistent kernel's tile retuned. HC is 1.26 GB per step and 12% of Gabriel's verify graph. Expected **+2–4%**; item 2 needs it.
6. **Speculative path:** keep NEXTN 3 steps / 4 draft tokens with the albucino INT4 drafter. The GGUF has no NEXTN tensors, and the drafter keeps its own 512 experts.
   - Measure accept length on the Coder: every −0.1 costs about −4% throughput.
   - FR-Spec (+9.3%, Gabriel `serve.sh:78-80`) only works if the converted `tokenizer.json` is byte-identical to the one its token map is pinned to. Copy the base file; don't regenerate it from the GGUF (vocab 248,320 and merges 247,587 match).
   - Deeper MTP and DFlash are ruled out: QSA caps draft tokens at 4 (`PERF_CEILING.md:125`).
7. **More streams with the freed VRAM:** about 30 GB allows 12–14 streams at the full 136K context. This needs the 64-row limits from items 2, 4 and 5, and it is a harness decision.
8. **PLE table: don't expand the IQ4_NL table.** Either:
   - (a) link the base BF16 PLE shards from the mounted Intel model (no new code), or
   - (b) adapt mratsim's patch 0025 packed-PLE Triton gather to IQ4_NL. It keeps the table at 28.8 GB in pinned host memory instead of 102.4 GB of BF16. At our ~390 MB/s load rate that is about 150–190 s less startup. Startup only adds play time while the per-game clock is not capped.
9. **Startup:** take a JIT-cache snapshot for the new config (the 145.75 s capture) and avoid FlashInfer paths that autotune.
10. **Correctness:** set `SGLANG_FORWARD_UNKNOWN_TOOLS=true` as Pennyroyal does (`serve-flash-next-frspec.sh:9`). No throughput effect.

**Already in our stack:** RecoverSSM and the WY verify path, the FP8 QSA KV fix, the FP32 prefill state, the KV-budget fix, FR-Spec, about 6 Mamba slots per stream, the abort retry, the single-GPU sampler skip, the grammar warmup and the loader fix.

**Not recommended:**
- **NVFP4 experts:** our nvfp4 profile was −22.9% at 4 streams (`CUSTOM.md:93`), quantizing activations to 4 bits stacks on top of the re-quant, and it would autotune on every start.
- **torch.compile and HC block retuning:** Gabriel found both were dead ends.
- **extra_buffer_lazy or exact Mamba sizing:** only about 0.3 GB, and it risks Franzen's prefill-checkpoint patch.
- **mratsim's admission-latch fix:** our log shows a queued request with a free slot in only 2 of 1,011 decode lines.
- **HiCache/NIXL, NVMe PLE, TP2 and YaRN:** not useful for a single-GPU Kaggle run at 136K context.

**Combined:** items 1 + 2 + 3 stacked optimistically give about 1.6× (roughly 1,250 tok/s at batch 10); about 1.3× is more realistic. All of these are estimates from a bandwidth model, not measurements.
