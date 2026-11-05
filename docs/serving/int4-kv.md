# int4 KV with a Hadamard rotation

`kaggle.vllm.kv_cache_dtype: "int4_per_token_head"` on the Lord runtime. Config: `configs/int4-kv/v1.json`, which is
`configs/main-tuned-drafter/v3.json` plus the settings below. Base commit `7af5c93f`, the code of main-tuned-drafter
Kaggle v03 (65.68).

## What it adds

The e975732 QSA owner only reads BF16 and FP8-e4m3, so the flag alone is rejected.

- `inference/framework/lordhansolo/int4_kv/taaf_qsa_kv8.py`: kernels vendored unchanged from
  [gzatravkin/pathbreaker-arc-agi-3](https://github.com/gzatravkin/pathbreaker-arc-agi-3) `serving/qsa_kv8.py` @
  `04f0e36` (Apache-2.0 header; the 6th public solution, 54.55). Their fused and lean knobs stay off, so this runs the
  plain `sparse_attention_int4` path. Writes use vLLM's stock `reshape_and_cache_int4`.
- `int4_kv/runtime_patches.json`: anchored edits to the stock `nvidia/qsa.py` and QSA warmup. Their anchors target
  a BF16-only QSA; ours are rewritten for e975732's FP8 branch. A missing anchor stops the setup before the load.
- `int4_kv/preflight.py`: GPU self-test before the server starts. It checks the write-read round trip, the kernel
  against a Python reference on the same codes, and the kernel against unquantized attention. A failure stops the
  notebook.

## Settings and why

| Setting | main-tuned-drafter v3 | int4-kv v3 | Reason |
|---|---|---|---|
| `kv_cache_dtype` | `fp8_e4m3` | `int4_per_token_head` | 544 vs 1,024 B per token per layer |
| `max_num_seqs` | 14 | 25 | The pool grew 1.69×, not 2×; v2 collapsed at 30 |
| `ARC3_MAX_ACTIVE_STREAMS` | 15 | 25 | No extra gate stream; each live context takes pool |
| `prefix_match_unit` | 128 | 128 | Must divide the block size, which stays 1,664 |
| `cudagraph_capture_sizes` | default | 1, 2, 4, multiples of 8 to 96, and 100 | Full graphs up to 25 streams × 4 rows |
| `public_runtime_minutes` | 120 | 120 | Same as main-tuned-drafter v3, with 1 pass, so scores compare |

- **Block size:** 1,664 tokens, same as FP8. `_align_hybrid_block_size` sizes it from the first non-SSM backend, which
  reports 1,024 B per token. v1 predicted 3,120 and set `prefix_match_unit` to 120, so it stopped at boot; v2 only
  restores 128.
- **Pool:** 2,645,854 tokens in 21.3 GiB, against 1,564,172 in 21.73 GiB for v3 (v1 boot log). That is 8,644 vs 14,917 B
  per token, about 13 attention pages × 480 B less, so int4 pages are not padded: BLNHC takes the packed group path.
- **Drafter:** inherits int4. An FP8 draft would probably cost only its own page on this path; not tested.
  Drafter acceptance may fall; compare with v3.

## Results

- **v2 (30 streams, cancelled at 44 min):** 1,400 generated tok/s until minute 15, then KV 98%, about 230 generated and
  10,000 prefill tok/s, with no recovery. 30 streams outgrew the 2,645,854-token pool, so LRU evicted prefixes while
  requests queued. Run: `runs/kaggle/jonathanwang2022/int4-kv/v02`.
- **v3 (25 streams, 1 pass, 120 min, commit `85a273fa`):** score 58.32 against main-tuned-drafter v3's 65.68, diff
  −7.36 (±9.74, t −1.56, p about 0.13), inside the noise, and it mixes int4 with an allocation change (no queue gate).
  Levels 126 vs 140. Tokens 6.81M vs 5.98M, 1,028 vs 906 tok/s over the job (+13.5%). In minutes 5-45: 1,183 generated
  and 3,029 prefill tok/s at 24.0 running. KV peaked at 94.9% for one minute at minute 17.3, prefill passed 5,000 tok/s
  for four one-minute windows from minute 20.5, and it recovered. Acceptance 2.889 vs 2.882. Generated tok/s by running
  in low-prefill windows: 14 streams 1,037, 20 streams 1,278, 24 streams 1,357, against the FP8 baseline's 1,029 at 14,
  so int4 costs no measurable speed. Run: `runs/kaggle/jonathanwang2022/int4-kv/v03`.
- **v4:** adds a CPU KV tier, GEMM rows and token-exact history. See [KV offload](kv-offload.md).
