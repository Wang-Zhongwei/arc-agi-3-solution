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

| Setting | v3 | int4-kv v1 | Reason |
|---|---|---|---|
| `kv_cache_dtype` | `fp8_e4m3` | `int4_per_token_head` | 544 vs 1,024 B per token per layer |
| `max_num_seqs` | 14 | 30 | Doubled streams, as asked |
| `ARC3_MAX_ACTIVE_STREAMS` | 15 | 30 | Same |
| `prefix_match_unit` | 128 | 120 | Must divide the block size; 128 does not divide 3,120 |
| `cudagraph_capture_sizes` | default | 1, 2, 4, multiples of 8 to 120, and 100 | Full graphs at 30 streams (120) and 25 games (100) |
| `public_runtime_minutes` | 120 | 60 | A test; long enough for KV to fill and drain |

- **Block size:** predicted 3,120 tokens: `16 × ceil(mamba page / (16 × 544))`, from FP8's 1,664 and its 0.48% padding.
- **Drafter:** must inherit int4. An FP8 draft page does not divide an int4 page, so `unify_kv_cache_spec_page_size`
  would pad every int4 page to the FP8 size and lose the saving. Drafter acceptance may fall; compare with v3.
