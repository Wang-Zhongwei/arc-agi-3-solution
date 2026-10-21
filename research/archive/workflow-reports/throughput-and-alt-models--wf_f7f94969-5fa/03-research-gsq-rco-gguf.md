# throughput-and-alt-models: research:gsq-rco-gguf

Workflow run `wf_f7f94969-5fa`, phase Research. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

**Verdict:** we can't serve the Coder GGUF as published at competition speed, and every path that keeps the GSQ quant needs new work.
- **SGLang 2.5.3 (Pennyroyal):** it has a GGUF loader, but for this model it hits five blockers (listed below). It would mean about 1 to 2 weeks of loader and kernel work, and the result would still run slower than our Marlin W4A16.
- **llama.cpp:** it reads the file natively, but it has open multi-slot, MTP and tool-call bugs on qwen4exp. Prefill runs about 5x slower than our stack.

**The cheapest credible path:** ISTA published exactly which experts the Coder keeps. We can apply that same expert mask to our current W4A16 model with a small, env-gated runtime patch. This tests the Coder's pruning on our unchanged stack, with no new model upload. It does not keep GSQ's quantization. Keeping GSQ quality on our engine needs the friend to export the model in a format SGLang can run (path B below).

**Correction to the brief:** the architecture is `qwen4_exp` / llama.cpp `qwen4exp`, not `qwen3_next`. It has 36 Gated-DeltaNet layers plus 12 full-attention layers with a sparse-attention indexer, hyper-connections, a 51.2B-entry per-layer n-gram table, 512 experts and top-10 routing (Qwen `config.json`; llama.cpp PR #27742).

## What the model is
- **Not a fine-tune.** "Coder" means half the routed experts were removed (256 of 512 per layer, 48 layers), then GSQ-quantized. RCO picked the experts to keep by minimising KL divergence to the full model on code, agentic, vision and spatial calibration data. Base is Qwen/Qwen3.8-Flash-Next. Sources: [card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF), [API](https://huggingface.co/api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF).
- **GSQ** ([arXiv 2604.18556](https://arxiv.org/abs/2604.18556), [repo](https://github.com/IST-DASLab/GSQ), Apache-2.0) is post-training scalar quantization. It learns which grid value each weight takes and the per-group scales together, using a Gumbel-Softmax relaxation.
- **RCO** ([arXiv 2605.00649](https://arxiv.org/abs/2605.00649), [repo](https://github.com/IST-DASLab/RCO)) solves "pick one of K options per group under an exact budget." Here it chose the experts; in the unpruned builds it chose the quant type per tensor.
- **Files:**

| File | Size (bytes) | Contents |
|---|---|---|
| Shard 1 | 29,608,446,496 | Transformer weights |
| Shard 2 | 28,800,138,432 | n-gram table at IQ4_NL; identical in every ISTA build, "neither pruned nor searched" |
| mmproj | 907,543,008 | Vision projector, BF16 |

  The folder is named "IQ1_M", but no tensor is IQ1_M; it is a size label.
- **No MTP head.** There are 0 nextn/MTP tensors in the allocation file, so it can't use our MTP drafting from the file itself.
- **Bits per weight:** the per-tensor types are exactly those of the unpruned IQ3_S build (I diffed the two allocation files; they match).
  - Gate/up experts: IQ2_S, IQ3_XXS, IQ3_S or IQ4_XS, depending on the layer.
  - Down experts: IQ4_NL in 39 layers, Q2_0 in 9.
  - Average expert bits per weight is 3.33, against 4.125 for our W4A16 at group size 128.
  - Attention and dense layers are Q4_K to Q6_K; hyper-connections, the sparse-attention indexer and the router stay BF16.
  - The headline "1.89 bpw" spreads the removed experts over the original parameter count.
- **The keep-list is published.** Section 2 of the allocation file gives, per original expert index, whether it was kept: "so the mask can be reapplied to the original weights". I extracted it: all 48 layers keep 256, and gate/up/down keep the same set.
- **Quality claims** (all at xhigh effort):
  - Coder: SWE-bench Verified 75.60 vs 82.80 for BF16 (91.3% retained); LiveCodeBench v6 86.28 vs 87.43.
  - There is no comparison to AutoRound and nothing agentic beyond SWE-bench.
  - The unpruned [IQ3_S build](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) scores AIME25 100, GPQA-Diamond 92.93 (BF16 91.92) and LCB v6 86.86.
  - The unpruned Q2_0 build scores 96.67 / 89.39 / 81.14.
- **Our model for comparison:** Intel's W4A16 quantizes only the experts (INT4, symmetric, group 128) and keeps everything else BF16. Its card reports 99.64% of BF16 on GSM8K, MMLU, PIQA and HellaSwag ([card](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)).
- **License:** the HF metadata says apache-2.0, but the card says the weights "inherit the license of the base model". The base is under the Qwen Community License 1.0, which allows internal use; offering it as a service or AI work-assistant commercially needs a separate licence ([LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE)).
- **No non-GGUF GSQ-RCO export of Flash-Next exists.** ISTA's only other Flash-Next release is P48NVFP4-MoESQ, a different method that needs a patched vLLM 0.30 and scores 67.0 vs 79.8 on SWE-bench Verified. ISTA has published Humming/compressed-tensors GSQ exports for other models (Qwen3.8-27B-3Bit-GSQ, Qwen3.6-35B-A3B-2Bit-GSQ, Kimi-K2.5/K2.6-2Bit-GSQ). The public GSQ repo has no wrapper for this architecture.

## Can our engines load it?
**SGLang 2.5.3.** I read the local Pennyroyal source (`competitors/sglang-rtxpro6000`, f9c4a89; earlier research found its Python identical to d00d88ef). It has `--load-format gguf` (`loader.py:3108`), but for this model:
1. **No name map.** The model type `qwen4_exp` doesn't match the GGUF arch name `qwen4exp`, and there is no builder in `gguf_name_maps.py`, so loading raises "Unknown gguf model_type" (`loader.py:3150-3168`).
2. **Single file only.** The loader reads one file (`_prepare_weights` checks `isfile`), so the two shards would need merging first.
3. **Q2_0 is unsupported.** Type 42 isn't in the sgl-kernel type sets, so it raises `NotImplementedError` (`gguf.py:148-221`).
4. **IQ types run on the slow path.** They have no fast batched kernel, so the MoE falls back to the per-token `ggml_moe_a8_vec` kernel at every batch size (`gguf.py:164-180,243-300`). Prefilling 8K-token chunks this way would be very slow.
5. **Converter permutations must be undone.** llama.cpp's converter reorders the linear-attention value heads: rows of `in_proj_qkv/z/a/b` and columns of `out_proj` (`conversion/qwen.py:454-492`). The `out_proj` columns can only be put back by dequantizing them.

**vLLM:** its GGUF support is "highly experimental and under-optimized" ([docs](https://docs.vllm.ai/en/latest/features/quantization/gguf.html)). It's also a different stack, so I didn't pursue it.

**llama.cpp:**
- Architecture support was merged in #27742 on 2026-08-27. #29751 ("Qwen4exp is broken on current master") was merged on 2026-10-01, and MTP support (#29761) the same day. MTP needs NextN tensors, which these GGUFs don't have.
- Open issues that block our use:
  - #29166: with more than one sequence in a shared KV cache, "the model stops seeing its last messages".
  - #28019: multi-sequence rollback corrupts the recurrent state.
  - #28286: MTP with more than one parallel slot mixes content across slots.
  - #29148, #29174, #29811: MTP fails to load.
  - #30078, filed today: on this exact kind of GSQ-RCO GGUF under CUDA llama-server, tool calls come back as plain text in about 30-50% of chats.
- Speed on an RTX PRO 6000 at one slot without MTP: prefill 2,559 tok/s and decode 59.7 tok/s; another user reports decode falling from 60 to 30 tok/s by 128K context ([HF discussion](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3)). Unsloth reports 100 → 170 tok/s single-stream with MTP ([docs](https://unsloth.ai/docs/models/qwen3.8-next)). I found no multi-slot numbers.
- Our SGLang does about 11-14K tok/s prefill and 72 tok/s per stream at 10 streams.

**Converting to a format SGLang already runs:**
- Q2_0 stores {-1,0,1,2}×d with a group-64 fp16 scale (PR #24448). That converts exactly to INT2 group-64 for Humming or INT4 group-64 for Marlin.
- The IQ2_S, IQ3_XXS, IQ3_S and IQ4_NL types are lookup-table grids. They can't be represented exactly in any kernel we have, so requantizing them throws away GSQ's learned grid choices.
- Pennyroyal already includes Humming: `HummingConfig` / `HummingMoEMethod` (`humming.py:447,987`) and the dependency `humming-kernels[cu13]==0.1.12` (`pyproject.toml:38`). Humming runs symmetric INT1-8 weights for dense and MoE layers ([repo](https://github.com/vllm-project/humming)). It compiles kernels at run time, so it needs nvcc. Whether the package is in our offline wheelhouse is unverified.

## Paths, ranked
**A. Apply the Coder's expert mask to our W4A16 (cheapest; keeps our quant, not GSQ's).**
- What it is: an env-gated runtime patch that sets the router logits of the removed experts to -inf. This is mathematically the same routing as deleting those router rows; the GGUF's parameter count shows only the 31,457,280 router weights changed shape.
- The mask file is 59 KB and ships in the notebook. No new model upload, and our MTP drafter, prefix caching, tool-call parser and patches stay as they are.
- A follow-up could physically drop the experts to free about 31 GB of VRAM.
- Effort: 0.5-1 day. Locally I can only check that the patch applies and the mask parses, since the laptop has no CUDA. It needs a 20-min panel run and then a 121-min run on Kaggle when quota returns.
- Expected decode gain: fewer distinct experts read per step, about −8% at 10 tokens per step and −27% at 40 tokens per step under uniform routing; real routing is skewed, so less. These are my estimates, not measurements.
- Quality risk is high: experts16 cost 7.6 points, and rellik13 found a 25% prune gave "worse play".

**B. Ask the friend for a Humming or compressed-tensors GSQ export of Flash-Next (the only path that keeps GSQ quality on our engine).**
- What to ask for: symmetric INT3 or INT4 experts, dense layers in BF16, optionally with the Coder mask. That is the format GSQ natively targets (`save_model.py`, `convert_to_humming.py`).
- Our side is about 1-2 days to bring it up on Pennyroyal. Humming MoE speed on sm120 vs Marlin is unmeasured, and ISTA's own Humming exports needed a vLLM patch (`patch_vllm.py`).

**C. Unpruned Q2_0 experts converted exactly to Humming INT2 group-64.**
- This cuts expert bytes about 45% vs W4A16, which could speed up decode.
- But ISTA's own numbers show the quality loss: LCB v6 −6.3 points and GPQA −2.5 vs BF16. Losses have converted to score for us before.

**D. Native llama.cpp: reject** for the reasons above.

**Against the "+2 games via throughput" goal:** none of these gives a large proven speedup. Throughput gains haven't converted to score so far (streams14: +15% tok/s for +0.13), and both the pruning and the 2-bit paths carry measurable quality risk. If you want to try one, A is the cheap A/B; run it with `make significance` against the control.

Notes, raw sources and the extracted mask are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/altmodel-research/gsq/`. The main ones are `NOTES.md`, `coder_expert_mask.json` and `gsq_alloc.txt`.
