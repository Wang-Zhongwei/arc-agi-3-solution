# wf_57f4538b-24c: research:gsq-weights

Workflow run `wf_57f4538b-24c`, phase Custom research. 

---

I found a lossless way to run the ISTA Coder's experts in our Pennyroyal Marlin MoE, and checked it on 12 real experts. Every expert weight in the GGUF is one of at most 16 values times a per-group scale. It can be stored as a 4-bit code plus a per-group scale and table. Re-encoded that way, the only error left is 0.17% from rounding the scales to BF16.

Speed should be about the same as today, not faster. The real gains are about 26 GiB of freed GPU memory and the chance to test GSQ quality. Nothing was run on a GPU or on Kaggle. Error numbers come from about 290 MB of HTTP range reads. Speed figures are byte-count estimates.

## 1. The weight format

- **Block layouts** (ggml b11476, verified by decoding the real blocks). Each type is "one of a few values × a per-sub-block scale":

  | Type | Weights / bytes per block | Value |
  |---|---|---|
  | Q2_0 | 64 / 18 B | {-1,0,1,2} × d |
  | IQ4_NL | 32 / 18 B | 16-entry table {-127…113} × d |
  | IQ4_XS | 256 / 136 B | same table × d·(ls−32), scale per 32 |
  | IQ3_S | 256 / 110 B | ±{1,3,…,15} × d·(1+2s), scale per 32 |
  | IQ3_XXS | 256 / 98 B | ±{4,12,…,52,62} × d·(0.5+s)/2, scale per 32 |
  | IQ2_S | 256 / 82 B | ±{8,25,43} × d·(0.5+s)/4, scale **per 16** |

- **Where each type is used** (out of 144 expert matrices):
  - gate/up: IQ2_S 20 layers, IQ3_XXS 17, IQ3_S 10, IQ4_XS 1.
  - down: IQ4_NL 39 layers, Q2_0 9.
  - There are no IQ1_M tensors; "IQ1_M" is only the folder name. Native size is 25.15 GB at 3.33 bits per weight.
- **Checked against the base model:**
  - The GGUF router equals the base router with the Coder mask applied, bit for bit, in layers 0 and 47. So the mask and the expert order are confirmed.
  - Expert orientation is confirmed: gate is `gate_up_proj[:640]`, up is `gate_up_proj[640:]`.
  - Experts are not reordered, but dense GDN layers are. In `attn_qkv` the V heads are permuted: GGUF head i is base head 3·(i%16)+i//16.
  - The n-gram table rows in the GGUF are IQ4_NL with 7–9% error against the BF16 base. Dense Q6_K tensors have about 2% error.

## 2. Conversion error, measured on 12 experts

Relative error, pooled across samples. "Conv" is the error the conversion adds to ISTA's weights; "tot" is the end-to-end error against BF16.

| GGUF type | GSQ vs BF16 | Intel W4A16 vs BF16 | 4-bit table re-encode (conv) | INT4 g32 conv → tot | INT4 g128 conv | NVFP4 conv | Humming INT3 g32 conv | Humming INT5 g32 conv |
|---|---|---|---|---|---|---|---|---|
| IQ2_S | 27.1% | 11.3% | 0.17% | 7.7 → 28.1% | 9.5% | 8.5% | 18.4% | 3.3% |
| IQ3_XXS | 22.3% | 11.3% | 0.17% | 10.0 → 24.4% | 11.7% | 8.9% | 17.5% | 2.1% |
| IQ3_S | 15.0% | 10.8% | 0.17% | 9.6 → 17.7% | 10.9% | 9.1% | 16.6% | 1.5% |
| IQ4_NL | 9.7% | 11.0% | 0.17% | 8.2 → 12.8% | 10.3% | 9.5% | 17.8% | 4.1% |
| IQ4_XS | 9.6% | 11.3% | 0.16% | 8.4 → 12.8% | 10.6% | 9.4% | 18.0% | 4.2% |
| Q2_0 | 43.6% | 10.9% | 0.16% | exact as INT4 g64 or Humming INT2 g64 | — | — | — | — |

- **The table re-encode is lossless.** Its maximum error is ≤0.36% of the tensor's largest value, and only 0.02% if the scales are FP16. MXFP4 (11.5–12.2%) is worse than INT4.
- **Some types need no new kernel to stay exact:**
  - Q2_0 is exact as stock Marlin INT4 g64, or Humming INT2 g64 with the scale negated.
  - IQ3_S is exact, and IQ3_XXS within 1.65% (its ±62 becomes ±60), as Humming uint4 with a BF16 zero point of 7.5. That costs 5.0 bits per weight.
  - INT8 g32 (stock Marlin) is exact for every type except IQ2_S.
- **IQ2_S needs a 16-wide group.** Its scales are per 16 weights. Humming needs groups of at least 32, and stock Marlin only has 16-wide groups for FP4.
- **Requantizing to any stock INT4 format undoes most of GSQ's advantage.** The IQ4 layers go from 9.7% to 12.8%. That is worse than Intel's 11.0%, and worse than quantizing the base directly to INT4 g32 (8.9%).
- **GSQ's experts are further from BF16 than Intel's,** by 1.4–4× on the same experts. GSQ optimizes output KL rather than weight error, so this alone doesn't settle quality.

## 3. Options

**(a) Recommended: re-encode offline to a 4-bit-code-plus-table format and add a table lookup to Marlin MoE.**
- **Format:**
  - Codes are packed exactly like GPTQ 4-bit, so the stock `gptq_marlin_repack` works on them.
  - Scales are stored FP16 on disk, one per 32 weights. IQ2_S gets one per 16; Q2_0 uses stock uint4b8 at g64.
  - Each layer records which type its gate/up and down use.
- **No sgl-kernel rebuild.** Pennyroyal JIT-compiles Marlin MoE from `sglang/kernels/jit/csrc/gemm/marlin_moe/*.cuh` and `marlin/dequant.h`, so the change is a source patch.
- **The kernel change** replaces the nibble-to-value step with a 16-entry byte lookup (llama.cpp's `get_int_from_table_16` trick), then reuses Marlin's existing INT8→BF16 conversion. My numpy emulation of it matches the stock element order on 24K random words with 0 mismatches.
- **Python side:** a new MoE quant method so one layer's gate/up and down can use different types and group sizes.
- **Effort:** about 4–7 days. The main risk is a 16-wide group with BF16 scales, which no stock Marlin MoE kernel uses yet (only FP4 uses 16-wide groups). If it doesn't work, IQ2_S falls back to INT4 g32, which raises its error from 27.1% to 28.1%.
- **Cost:**
  - Experts are 34.9 GB on GPU (4.62 bits per weight), against 62.8 GB today.
  - The lookup adds about 5 integer ops per 4 weights. That costs nothing at decode. I estimate +10–20% MoE prefill kernel time; not measured.

**(a') No kernel change, for a first test.** Stock Marlin: INT8 g32 for the IQ3/IQ4 types (exact), INT4 g64 for Q2_0 (exact), INT4 g32 for IQ2_S (adds 3.7% relative error).
- 104 of the 144 matrices are exact.
- It still needs the same Python mixed-type MoE method.
- Experts are 53.8 GB, so decode reads about 30% more expert bytes per step than today and is slower.

**(a'') Speed probe only.** Requantize everything to INT4 sym g32 (33.9 GB), using the existing `qwen38fn-gsq` profile path (`gptq_marlin`).
- It has the same bytes as (a), so it measures (a)'s speed on stock kernels today.
- Its quality is not representative of the Coder.

**(b) Port llama.cpp's IQ MoE kernels into SGLang: not recommended.**
- About 2–3 weeks and high risk:
  - int8 activations;
  - the expert-routing glue;
  - CUDA-graph capture;
  - a Q2_0 path that SGLang's existing ggml kernels lack.
- Its only advantage is reading the native 3.33 bits per weight, which is 26% fewer decode bytes than (a).
- The existing vec kernel reads each expert once per token instead of once per step, about 1.8× redundant.
- llama.cpp's prefill on this architecture is 3–5× slower than our stack.

**(c) Everything except the routed experts: take it from Intel's W4A16 checkpoint, which is the exact BF16 base.**
- That covers dense, attention, the BF16 n-gram table, the vision tower, embeddings, the tokenizer and the chat template.
- The router is base[mask], which equals the GGUF.
- Using the GGUF's quantized dense tensors instead would add their ~2% error, need the V-head un-permute above, and take the 7–9%-error n-gram table.
- **Expert name map:** `blk.L.ffn_{gate,up,down}_exps[:, :, j]` becomes `model.language_model.layers.L.mlp.experts.{j}.{gate,up,down}_proj`, where j = 0..255 is the original expert `mask[L][j]` in ascending order.
- **Config:** set `text_config.num_experts=256`. I found no hard-coded 512 in the qwen4_exp model code or MoE kernel ops.

## 4. Speculative decoding

- **The albucino drafter still works.** The launcher's draft view keeps the drafter's own config (512 experts, INT4 g32), so the "expert_count mismatch" that llama.cpp hits does not happen here.
- **Acceptance will probably drop.** Today it averages 2.66 accepted tokens per step with 3 drafts (v4-fable-f001 v2, 1011 samples), about 0.73 per draft. The drafter was trained on the full model's hidden states. My estimate for the Coder is 2.3–2.5 (−6% to −14%); this is unmeasured and the bench reports it.
- **The FR-Spec map is reusable.** We ship Intel's tokenizer (the GGUF has the same 248,320-token vocabulary), so the tokenizer hash `06b95…` and map `becfa41…` still pass the profile check.
- **Use the base chat template,** not the Unsloth one embedded in the GGUF.

## 5. Memory: 10 streams × 136K context on 96 GB

Using `check_checkpoint.py`'s reference numbers (13,312 bytes per token for the FP8 KV cache, MEMFRAC 0.96, 60 Mamba slots):

| Plan | GPU weights | KV pool (FP8) | KV pool (BF16) | Expert bytes per verify step |
|---|---|---|---|---|
| Today | 68.5 GiB | 1.01M tokens | — | 30.7 GB |
| (a) table re-encode | 42.5 GiB | 3.11M | 1.55M | 25.9 GB |
| (a'') INT4 g32 | 41.5 GiB | 3.19M | 1.59M | 25.1 GB |
| (a') INT8 mix | 60.1 GiB | 1.69M | 0.85M | 39.9 GB |

- 10 streams × 136K needs 1.36M tokens of KV.
- Bytes per step assume uniform routing at 34 tokens per step.
- **With (a), a BF16 KV cache becomes affordable,** or about 2× the streams at FP8.
- **Net throughput for (a): about −7% to +1% versus today.** Decode steps should be about 7% faster from fewer expert bytes, offset by the lower acceptance.
- The n-gram table stays in host RAM in BF16 (102.4 GB), as today.

## 6. Where to convert

- **Inputs:** only GGUF shard 1 is needed (29.6 GB). Shard 2 and the mmproj are not used. The non-expert tensors come from the mounted Intel model.
- **Encode speed** in numpy is 56M weights/s per core: about 18 core-minutes for all 60.4B expert weights, about 1–2 minutes with 16 processes. RAM is streamed per layer, under 3 GB.
- **Output is about 45.6 GB:** 34.9 GB of experts plus about 10.7 GB of non-expert tensors copied from Intel. The 102.4 GB BF16 n-gram table is linked from the Intel mount at serve time.
  - Franzen's Kaggle copy is resharded, so the converter must work from the index, not from shard names.
  - Kaggle Models takes this size: Franzen's Intel model instance is 181 GB.
- **Write to `/tmp`** (about 1 TB free); `/kaggle/working` is capped at 19.5 GB.
- **For the Fable test,** conversion can run at load time in the internet-enabled test notebook. That adds about 5–8 minutes, mostly the HF download, which I estimated and did not measure.
- **For an offline submission,** convert once in an internet-enabled notebook and publish it as a Kaggle Model. That upload needs your approval and Kaggle API credentials in the notebook.

## 7. A cheap control to run alongside

Apply the Coder mask to the BF16 base and quantize to INT4 g32 (about 8.7–9.0% error, stock Marlin) or reuse Intel's experts (11%). This separates the effect of pruning from the effect of GSQ's quantization.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/runner-research/gsq/`:
- `remote.py`: HF range reader for GGUF and safetensors
- `fetch_samples.py`: fetches the samples into `data/`, listed in `data/manifest.json`
- `ggufq.py`: block decoders, including Q2_0, which gguf-py lacks
- `fmt.py`: the candidate formats
- `analyze.py`: the error study; full table in `results.json`
- `verify_side.py`: router, n-gram table and V-head checks
- `convert_gsq.py`: re-encoder and GPTQ-style packing, with `--selftest`
- `lut_dequant_emu.py`: the proposed kernel lookup, with an emulation that checks it
- `memplan.py`: byte and memory calculations
- `.venv`: Python environment with numpy, safetensors and pyyaml
- `ref/humming`: Humming source at commit 763c3b7
