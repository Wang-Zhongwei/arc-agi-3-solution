# W6: Toolchain, compute and deployment path for training Qwen3.8-Flash-Next (collected 2026-10-18)

Method: primary sources only where possible (GitHub repos, issues and PRs read with `gh`; HF model API and file listings; the Azure Retail Prices API; Microsoft Learn size pages; framework docs). Everything was read on 2026-10-18. "UNVERIFIED" means no primary source was found. "[I]" marks my inference. This builds on `research/data/finetune-research-2026-10-17/models.md` and `research/reports/finetune-plan-2026-10-17.md` and corrects them in a few places (see the Implications section).

## 1. Which training stacks support Flash-Next (Oct 2026)?

### Takeaway
ms-swift (Megatron backend, through ModelScope's own `mcore-bridge`) is still the only stack that documents **LoRA** on **one 8-GPU node**. Its code has gained MTP-head training and context parallelism since the 10-17 notes. Axolotl has a new **single-GPU QLoRA** recipe. Every RL stack that supports the model (NeMo RL, miles, prime-rl) does full-parameter training on 24-64 GPUs at 4k context. verl, LlamaFactory, TRL, Unsloth and NVIDIA Megatron-Bridge have no support.

### Cited Findings
- **ms-swift best-practice page: no change since 10-17.** The doc file has exactly one commit, `06c7d80d`, dated 2026-09-07 ([commits](https://github.com/modelscope/ms-swift/commits/main/docs/source_en/BestPractices/Qwen3_8-Flash-Next-Best-Practice.md)). The numbers are the same as before: LoRA r8 on `in_proj out_proj linear_proj linear_qkv`, TP2/EP4/PP2.
  - Peak memory per GPU: 75.6 GiB at 2k, 77.8 at 4k, 93.3 at 8k.
  - With PLE offload: 63.6 GiB at 4k and 77.7 at 8k; with PLE offload plus padding_free, 72.5 GiB at 8k.
  - Other layouts at 8k: TP2/EP4/PP1 142.5 GiB; TP4/EP2/PP2 105.7 GiB at 2.58x the step time.
  - GRPO-LoRA is commented "8*135G" and needs a vLLM build that includes PR #53896 ([page](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)).
  - The page counts "125B parameters in total (including an additional 51B N-gram embedding table)". The NeMo AutoModel doc instead says "125B backbone … plus the 51.2B Engram table", and the HF safetensors hold 180.0B params. The naming conflicts across sources.
- **ms-swift example script (newer than the doc page).** `examples/models/qwen4_exp/megatron_sft.sh` lines 4-5 read: "To train the checkpoint's MTP head, add `--mtp_num_layers 1` and `--mtp_loss_scaling_factor 0.1`". It was updated 2026-09-21 in commit `eeeaefb3` ([file](https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen4_exp/megatron_sft.sh)).
- **mcore-bridge history** (`src/mcore_bridge/model/gpts/qwen4_exp.py`, [commits](https://github.com/modelscope/mcore-bridge/commits/main/src/mcore_bridge/model/gpts/qwen4_exp.py)):
  - #174: model added, 2026-08-26.
  - #180: QSA with sequence parallelism, 09-05.
  - #187: "fix qwen4-exp thd cp", 09-08.
  - #202: "bound QSA memory", 09-21.
  - #204: "support qwen4_exp & glm5_next MTP", 09-21.
  - #220: "exclude PLE n-gram base state from PEFT saves", 10-09.
  - The latest release is v1.6.5 (2026-10-13).
- **Context parallelism in mcore-bridge.** The code routes thd (packed) batches, and any CP>1, to a sparse QSA kernel. It requires `cp_comm_type == 'all_gather'` when CP>1 (`qwen4_exp.py` lines 166-188, [file](https://github.com/modelscope/mcore-bridge/blob/main/src/mcore_bridge/model/gpts/qwen4_exp.py)). No ms-swift doc gives a CP memory or throughput number for this model (UNVERIFIED in practice).
- **A real full-parameter user run on ms-swift** ([issue #10376](https://github.com/modelscope/ms-swift/issues/10376), closed 2026-10-15):
  - Settings: `--tuner_type full`, TP2/PP2/**EP16**, `--max_length 32768`, `--packing true`, `--optimizer_cpu_offload true`.
  - Versions: transformers 5.19.0, megatron-core 0.16.1, mcore_bridge 1.6.5, ms_swift 4.6.0.
  - EP16 with PP2 implies at least 32 GPUs [I].
  - Its bug: the exported config renamed `full_attention` to `indexed_attention`. That is a transformers ≥5.18 normalization, which broke vLLM ([vLLM #59756](https://github.com/vllm-project/vllm/issues/59756)). The user fixed it by hand-editing config.json.
- **NVIDIA Megatron-Bridge does not support the model.** A maintainer says Megatron Core lacks "An N-gram / hashed embedding table", "QSA", and "Gated residual (GR)", with "no firm date" ([#6060](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/6060)).
- **NeMo AutoModel** ([coverage doc](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/llm/qwen/qwen3-8-flash-next.mdx)):
  - Scope: SFT only, FSDP2 + EP, no TP/PP.
  - Validation: on "8 nodes / 64 H100" (EP64) and 128 H100 for the CP comparison, at 4,096 tokens.
  - "the checkpoint's 4B MTP head is not loaded". MTP, TP and PP are listed as unsupported.
  - QSA indexers are frozen.
  - No LoRA recipe is documented (UNVERIFIED whether PEFT works).
  - The Flex QSA mask cost 40 GiB extra at 64k; draft PR #4334 cuts that to 1.5 GiB ([#4147](https://github.com/NVIDIA-NeMo/Automodel/issues/4147)).
- **NeMo RL** (branch `qwen3-8-flash-next-support`, [guide](https://github.com/NVIDIA-NeMo/RL/blob/qwen3-8-flash-next-support/docs/guides/models/qwen/qwen3-8-flash-next.md)):
  - Status: "Functionally Ready".
  - Recipe: GRPO full-model on AutoModel (DTensor), EP64, vLLM TP8/EP8, **8 nodes × 8 GPUs**, validated at 4k for 182 steps.
  - "longer context lengths have not yet been validated".
- **miles (RadixArk)** ([doc](https://github.com/radixark/miles/blob/main/docs/models/qwen/qwen3-8-flash-next.md)):
  - Recipe: GRPO full model on **32 GPUs (8 × 4)**, TP2/PP8/EP4, with SGLang TP8/EP8 rollouts colocated.
  - Validation: only a 4-layer smoke slice runs on one 8×H200 node (CI with rollout-routing replay).
  - "MTP (`--mtp-num-layers` is omitted, the MTP tensors are not mapped yet)"; the PLE table is frozen and host-resident (~102 GB).
  - The pipeline is pinned to an SGLang branch `sglang-miles-qwen38next` and Megatron-LM#89.
- **prime-rl** added native Flash-Next on 2026-10-03 ([PR #3434](https://github.com/PrimeIntellect-ai/prime-rl/pull/3434)). It was validated with "16 trainer B300 GPUs and TP8 inference", context 4096, 20 math RL steps, mismatch KL <0.015.
- **verl**: only open feature requests ([#7653](https://github.com/verl-project/verl/issues/7653), 2026-09-01; #8202, 2026-10-12).
- **No Flash-Next support found** in LlamaFactory (repo search), TRL, Unsloth (only GGUF/inference issues), torchtitan, SkyRL or slime. The repo-tree searches matched no `qwen4_exp` or Flash-Next file.
- **Axolotl** ([doc](https://docs.axolotl.ai/docs/models/qwen3.8-flash-next.html); [qlora.yaml](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/qwen3.8-flash-next/qlora.yaml), added 2026-09-01):
  - Memory: "QLoRA (1x B300 @ ~120 GiB w offload, else ~216 GiB)" at `sequence_len: 4096`, micro-batch 2.
  - Settings: `quantize_moe_experts: true`, `ple_cpu_offload: true` ("needs ~100 GB free system RAM").
  - Limits: `attn_implementation: sdpa` only, because "the QSA indexer's 4D selection mask only reaches sdpa or eager". LoRA kernels and Liger are unsupported, and "Full finetuning: Untested. The bf16 weights alone are 329.6 GiB".
  - An NVFP4 MoE-LoRA path ("1x B300 @ ~223 GiB", `nvfp4_merge_aware: true`) merges "back into a plain NVFP4 checkpoint losslessly".
- **The official Qwen README** only generically advises "Unsloth, Swift, Llama-Factory" for SFT/DPO/GRPO ([README](https://github.com/QwenLM/Qwen3.8-Flash-Next), Finetuning section). It names no Flash-Next-specific recipe.
- **Kaggle practitioners** (sibling notes `W5b-kaggle-intel.md` lines 19 and 25; secondary):
  - One user built a miles-based RL pipeline on "a full H200 node" with no gain.
  - Another ran "QLoRA 4-bit + FSDP2 + PLE CPU Offload … in 2*96G GPU".

### Inferences
- [I] For one node, the realistic choices are:
  - (a) **ms-swift Megatron LoRA**, the main path;
  - (b) **Axolotl QLoRA, one job per GPU**, as a real-model wind tunnel (8 parallel hyperparameter runs on 8×H200). The ND H200 v5's 1,850 GiB of host RAM holds eight ~95 GiB PLE copies.
- [I] Online RL on one node means ms-swift's GRPO-LoRA with colocated **vLLM**, not our SGLang stack. The miles/NeMo-RL/prime-rl recipes need 3-8 nodes.

### Gaps
- No published tokens/s for any Flash-Next trainer. ms-swift gives relative step times only.
- Whether NeMo AutoModel's PEFT path works on this model. Whether ms-swift CP has been run beyond unit tests.

## 2. Kernel support: GDN backward and QSA backward

### Takeaway
GDN backward comes from flash-linear-attention's Triton `chunk_gated_delta_rule`. QSA backward exists in three independent implementations: the miles/mcore-bridge Triton block-sparse kernel, AutoModel's FlexAttention and FA4 (SM90), and the HF sdpa dense-mask path. The QSA **indexer** has no gradient anywhere, so it stays frozen.

### Cited Findings
- **mcore-bridge GDN** imports `fla.ops.gated_delta_rule.chunk_gated_delta_rule` and has CP slicing helpers (`gated_delta_net.py` lines 21-23 and 53-90, [file](https://github.com/modelscope/mcore-bridge/blob/main/src/mcore_bridge/model/modules/gated_delta_net.py)).
- **mcore-bridge QSA training kernel** (`kernels/qsa_block_sparse_attn.py` lines 1-42, [file](https://github.com/modelscope/mcore-bridge/blob/main/src/mcore_bridge/model/modules/kernels/qsa_block_sparse_attn.py)):
  - It is a "VENDORED FILE" from radixark/miles: "Tensor-core QSA sparse attention for training: forward + backward".
  - The earlier gather form cost "2.5 s forward and 13.9 s forward+backward per layer … ~330x slower than a DENSE causal flash kernel" at T=25k. It notes that "on the 16-node agentic run those three QSA layers per pipeline stage were ~49 s of a ~52 s micro-batch".
  - So long-context agentic RL on this model has been run by RadixArk, at 16-node scale.
- **NeMo AutoModel** uses FlexAttention sparse GQA, plus an FA4 QSA backend that requires "SM90/BF16/head-dimension-256" (tracker [#3971](https://github.com/NVIDIA-NeMo/Automodel/issues/3971), PR #3925).
- **HF transformers QSA is inference-only.** "the Transformers implementation only supports inference" for the indexer; the top-k truncates its gradient ([#48826](https://github.com/huggingface/transformers/issues/48826)).
- **fla issue history.** There is a closed GDN-backward bug on B200 (#612) and non-determinism on B300/GB300 sm_103 (#945, #1228), all fixed or closed ([issues](https://github.com/fla-org/flash-linear-attention/issues?q=gated+delta)).

### Inferences
- [I] H200 (sm90) is the best-trodden target: miles CI runs on H200, and FA4 QSA is SM90-only. B200/B300 work with Triton kernels, with past determinism bugs.
- [I] On the HF/Axolotl path, QSA under sdpa with a 4D mask is dense-cost attention. Expect it to be practical only at ≤8-16k.

### Gaps
- No kernel-level speed numbers for the Triton block-sparse QSA kernel at 32-64k.

## 3. Training on the quantized checkpoint, and merging without re-quantization

### Takeaway
The Intel W4A16 checkpoint we serve quantizes **only the routed experts**. Every module the ms-swift LoRA recipe targets (GDN in/out projections, QSA q/k/v/o), plus the shared expert and router, is stored **BF16** in it. A LoRA restricted to those modules can therefore be merged straight into our Kaggle checkpoint's BF16 tensors: no AutoRound re-run and no change to the int4 experts.

### Cited Findings
- **Intel AutoRound checkpoint.**
  - Quantization config: `quantization_config` is bits 4, group_size 128, sym, `auto_round:auto_gptq` packing, autoround 0.15.0 ([config.json](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/blob/main/config.json)).
  - BF16 modules: `extra_config` keeps 16 bits for `.*linear_attn.*`, `.*self_attn.*`, `.*shared_expert.*`, `.*mlp\.gate.*`, `.*hyper_connection.*`, `.*indexer.*`, `.*mtp.*`, `.*ple.*`, `.*embed_tokens.*` and `.*visual.*`.
  - Generation command: `auto-round --scheme W4A16 --iters 200 --ignore_layers lm_head,embed_tokens,visual,linear_attn,self_attn,hyper_connection,mlp.gate,shared_expert,in_proj_a,in_proj_b,ple,mtp,indexer` ([README](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)).
- **Checkpoint sizes** (HF API `?blobs=true`, 2026-10-18):
  - BF16 `Qwen/Qwen3.8-Flash-Next`: 360.0 GB, 179,999,981,424 BF16 params, ungated.
  - FP8: 185.6 GB.
  - Intel W4A16: 181.2 GB (I32 15.2B, BF16 59.2B, F16 0.94B).
  - Our Kaggle copy `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`: totalUncompressedBytes 181,240,184,163, "Resharded … original weights are unchanged" (`kaggle models instances get …/Transformers/default`).
- **Parameter counts from config.json** (my arithmetic from [config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json): hidden 2560; 36 GDN layers with 16×128 q/k and 48×128 v; 12 QSA layers with 24 heads × 256 and a sigmoid output gate, 2 KV heads; 512 experts × 640 wide):
  - Routed experts: 120.8B.
  - GDN projections: 2.09B. QSA projections: 0.60B. Shared experts: 0.24B.
  - LoRA size on GDN+QSA: r8 = 13.1M params (26 MB), r16 = 26.1M, r64 = 104M (209 MB).
- **Axolotl.**
  - Its QLoRA quantizes the experts to bnb 4-bit (`quantize_moe_experts`) and puts LoRA on GDN/QSA/shared expert, plus `lora_target_parameters: mlp.experts.gate_up_proj, down_proj`.
  - The QSA indexer must not be targeted: "`lora_target_linear` … expands to the QSA indexer projection incorrectly" ([doc](https://docs.axolotl.ai/docs/models/qwen3.8-flash-next.html)).
- **Gotchas from ms-swift users.**
  - Transformers drops MTP weights in a LoRA merge-export, so one must copy the original MTP weights or export through Megatron with `mtp_num_layers` ([#9802](https://github.com/modelscope/ms-swift/issues/9802)).
  - Before #220 (10-09), mcore-bridge PEFT saves could include the PLE base state ([commit](https://github.com/modelscope/mcore-bridge/commits/main/src/mcore_bridge/model/gpts/qwen4_exp.py)).
- **Memory math [I, extrapolated from the ms-swift table; measure it before planning on it]:**
  - Base footprint: 8k on TP2/EP4/PP2 with PLE offload + padding_free = 72.5 GiB. H200 has ~131 GiB usable.
  - Weights: routed experts are ~30 GB per GPU at EP4×PP2 (120.8B BF16 / 8); the non-expert trunk is ~2 GB per GPU.
  - The sequence-dependent terms are activations under full recompute, plus fp32 logits. Full-vocab fp32 logits are 7.6 / 15.2 / 30.3 / 60.6 GiB at 8k / 16k / 32k / 64k before the TP split. So `--cross_entropy_loss_fusion` matters.
  - Fit per sequence length:
    - **16k** should fit at CP1 on H200.
    - **32k** probably needs CP2 (TP2/CP2/PP2/EP4 = 8 GPUs).
    - **55-64k** (our median prompt) needs CP4 on 8 GPUs (PP1 doubles expert weights per GPU), or a second node. It is borderline on H200 and more comfortable on B200 (180 GB).
  - LoRA on experts (Axolotl-style `target_parameters`) adds negligible adapter memory, but it makes the merge touch int4 tensors. See §5.
- **FLOP estimate [I].**
  - Cost per token: about 8 × ~7.4B active (miles counts "~7.4 B" active from config shapes) ≈ 6e10 FLOP per token for forward + backward + recompute.
  - Node throughput: 8×H200 at 989 TFLOPS BF16 dense. At 3-10% MFU (UNVERIFIED for 512 small experts) that is ~4-13k tokens/s, or **100M processed tokens ≈ 2-7 node-hours**.
  - A 55k-token decision sample with loss on ~900 reply tokens wastes ~98% of that compute. Packing all assistant turns of an append-only segment (between trims) into one sequence would cut cost by roughly the turns per segment. Test-first: check that our harness prompts really are append-only between trims.

### Gaps
- Whether ms-swift/mcore-bridge can load the AutoRound int4 experts directly (QLoRA-style on Megatron). Nothing found; assume no.
- Real QLoRA speed on Axolotl (sdpa, bnb dequantisation, HF MoE loop). Not published.

## 4. Serving: LoRA adapters at serve time vs merging

### Takeaway
Merge. SGLang's LoRA serving is generic, and nothing documents it for `qwen4_exp` + NEXTN speculation + AutoRound on sm120. A merged BF16 delta costs nothing at decode.

### Cited Findings
- SGLang LoRA supports Triton and csgmv backends. It says "enabling LoRA on additional modules introduces a minor performance overhead", and the docs say nothing about MoE experts, speculative decoding or AutoRound ([docs](https://docs.sglang.io/advanced_features/lora.html)).
- Our serving changes go through `serving/runtime_patches/` (CUSTOM.md lines 54-56). CUSTOM.md §6 (line 92) treats the W4A16 model as fixed by default.

### Inferences
- [I] Two zero-cost deployment forms:
  - (a) **Offline merge.** Write the merged BF16 tensors into the copies of the affected Intel shards and upload them as a small Kaggle dataset (GDN+QSA+shared expert ≈ 2.9B params ≈ 5.9 GB). At start-up, build a symlink overlay over the 181 GB model.
  - (b) **Load-time merge.** Add W += (α/r)·B·A in SGLang's weight loader through a runtime patch, so the upload is the adapter only (26-209 MB).
  - Both keep the original `config.json`. That avoids the `indexed_attention` rename (ms-swift #10376) that our pinned SGLang 2.5.3 may not accept (UNVERIFIED).

### Gaps
- No measured decode-speed cost of unmerged LoRA on Flash-Next in SGLang.

## 5. Re-quantization (AutoRound) and QAT

### Takeaway
With non-expert LoRA there is nothing to re-quantize. If experts are trained, a full AutoRound pass takes roughly 9 h on 1-2 large GPUs. RTN takes minutes but is unmeasured on this model. QAT is a frontier-lab recipe and not feasible by 11-02.

### Cited Findings
- **AutoRound on Flash-Next** ([#2323](https://github.com/intel/auto-round/issues/2323), 2026-09-08, Intel maintainer, with PR #2303):
  - It builds a meta skeleton and "materializ[es] weights per block".
  - Calibration defaults to `NeelNanda/pile-10k`.
  - On 2 cards: GDN blocks took ~4 min each, but block 3 (the first QSA layer) took ~32 min (20:34 → 21:06). Peak was 121 GB of RAM and 92.7/100.3 GB of VRAM. That rules out a 96 GB RTX PRO 6000 for the PLE-adjacent blocks [I].
  - [I] Total ≈ 36×4 + 12×32 min ≈ 9 h per 200-iteration pass.
- The albucino MTP draft we serve is "MTP routed experts locally RTN-quantized to INT4 symmetric group-32 with min/max scales; other MTP tensors unchanged", sourced from RadixArk's NVFP4 repo ([compact_sources.json](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE/blob/main/runtime/mtp-int4-g32/compact_sources.json)). So re-quantizing the drafter is cheap RTN.
- Kimi K2-Thinking adopts "Quantization-Aware Training (QAT) during the post-training phase, applying INT4 weight-only quantization to the MoE components", for a ~2x speed-up ([card §4](https://huggingface.co/moonshotai/Kimi-K2-Thinking)). Axolotl's `nvfp4_merge_aware` is the LoRA-scale analogue for NVFP4 ([nvfp4-lora.yaml](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/qwen3.8-flash-next/nvfp4-lora.yaml)).

### Inferences
- [I] If expert LoRA is ever wanted, use AutoRound with `--iters` reduced, or RTN g128, calibrated on agent transcripts. Gate it on equal offline score.
- [I] A "merge-aware" int4 fake-quant on the experts would be the W4A16 analogue of Axolotl's NVFP4 trick. It is not built anywhere for AutoRound (UNVERIFIED).

### Gaps
- No measured quality loss of RTN vs AutoRound for this model's experts.

## 6. MTP head after trunk changes

### Takeaway
Keep the albucino drafter first and measure accept length. If it drops, ms-swift can train the MTP layer jointly (`--mtp_num_layers 1`). SpecForge MoE-drafter training is weeks of GPU time and is not servable on Qwen4Exp in SGLang anyway.

### Cited Findings
- ms-swift: `--mtp_num_layers 1 --mtp_loss_scaling_factor 0.1` (example lines 4-5). The mcore-bridge MTP support landed in #204 (2026-09-21). NeMo AutoModel and miles do not load or map the MTP.
- **SpecForge.**
  - The Flash-Next DSpark MoE drafter, with "MTP warm start", is an open PR, #983 (2026-10-16) ([PR](https://github.com/sgl-project/SpecForge/pull/983)).
  - For the 27B, one drafter epoch took "36.3 s/step … 50.7 h" on 4×GB300 ([PR #943](https://github.com/sgl-project/SpecForge/pull/943)).
  - SGLang "Qwen4Exp … cannot emit DFlash aux hidden states: DFlash/DSpark drafters cannot be served" ([#38589](https://github.com/sgl-project/sglang/issues/38589)).
- A warning from ms-swift #9283: after an MTP-enabled SFT, a Qwen3.5-35B scored 94% with MTP off but 74% with it on under vLLM. MTP should be lossless, so that points to a serving bug ([issue](https://github.com/modelscope/ms-swift/issues/9283)).
- Local: SPEC_STEPS=3; "4 can't run (Qwen QSA caps at 4 draft tokens)" (CUSTOM.md line 24).

### Inferences
- [I] Gate every trained model on token-identical greedy output with spec on vs off, plus accept length and tok/s vs today's ~2.68 / ~690.
- [I] If the MTP is retrained: keep its LoRA to its BF16 attention/dense parts, then copy them into `mtp-dense.safetensors`, with no re-quant. Alternatively RTN its experts to int4 g32 as albucino did.

## 7. Azure SKUs, prices, quota, storage, and the Kaggle upload

### Takeaway
ND H200 v5 costs **$84.80 per node-hour** ($10.60 per GPU-hour) in eastus2, westus2 and westus3. Its spot meter is priced the same as on-demand, so plan for no spot discount. The prior plan's 450-950 GPU-h comes to about **$4.8-10.1k** on-demand. Azure also sells the exact Kaggle GPU (RTX PRO 6000 Blackwell Server Edition) at **$5.50/h on-demand and ~$1.02/h spot**, which makes cheap exact-stack rollout boxes.

### Cited Findings (Azure Retail Prices API, Linux, `priceType eq 'Consumption'`, read 2026-10-18; e.g. [H200 query](https://prices.azure.com/api/retail/prices?$filter=armSkuName%20eq%20%27Standard_ND96isr_H200_v5%27))

| SKU | GPUs | On-demand $/h (cheapest US) | Spot $/h | $/GPU-h on-demand |
|---|---|---|---|---|
| Standard_ND96isr_H200_v5 | 8×H200 141 GB | 84.80 (eastus2/westus2/westus3) | 84.80 (same as on-demand) | 10.60 |
| Standard_ND96isr_H100_v5 | 8×H100 80 GB | 98.32 (eastus/eastus2/westus3) | 18.17 | 12.29 |
| Standard_ND96isr_MI300X_v5 | 8×MI300X 192 GB | 48.00 (eastus2/westus3) | 8.87 | 6.00 |
| Standard_ND128isr_NDR_GB200_v6 | 4×GB200 192 GiB (Grace aarch64) | 108.16 (eastus/eastus2/westus2/3) | 108.16 | 27.04 |
| Standard_NC144lds_xl_RTXPRO6000BSE_v6 | 1×RTX PRO 6000 BSE 96 GB, 144 vCPU, 264 GiB | 5.50 | 1.02 (eastus/westus2) | 5.50 |
| Standard_NC144ds_xl_RTXPRO6000BSE_v6 | 1×, 516 GiB | 6.38 | 1.18 | 6.38 |

- **Node specs.**
  - ND H200 v5: 96 vCPU Sapphire Rapids, **1,850 GiB RAM**, 1×1,024 GiB local disk, 8×400 Gb/s IB ([Learn](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-h200-v5-series)).
  - ND GB200 v6: "two NVIDIA Grace CPUs and four NVIDIA Blackwell GPUs", 900 GiB RAM ([Learn](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-gb200-v6-series)).
  - RTX PRO 6000 BSE v6: NC144 sizes are 1 GPU, NC288 sizes are 2 GPUs, smaller sizes are fractional (1/4, 1/2) ([Learn](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-rtxpro6000-bse-v6-series)).
- **Storage and egress.**
  - A Premium SSD P40 (2 TiB) costs $235.52 per month in eastus2.
  - Internet egress is $0.087/GB after the first 100 GB (routing preference MGN; Retail API `serviceName eq 'Bandwidth'`).
  - A 181 GB checkpoint upload costs ≈ $7; a ≤6 GB delta is free [I].
  - Disk need [I]: base BF16 (360 GB) + Megatron conversion + merged export already exceed the 1 TiB local disk, so attach ≥2 TiB.
- **Quota.** No official ND H200 v5 approval time was found.
  - A Microsoft Q&A answer says high-demand GPU SKUs may be unavailable for new subscriptions ([Q&A](https://learn.microsoft.com/en-us/answers/questions/2152497/issues-with-gpu-vm-quota-increase-on-azure-startup)).
  - A vendor blog estimates 1-4 weeks ([Spheron](https://www.spheron.network/blog/azure-h200-pricing-2026/), secondary, UNVERIFIED).
- **ROCm (MI300X/MI355X).** SGLang inference on gfx950 is an open optimisation tracker ([#44304](https://github.com/sgl-project/sglang/issues/44304)). ms-swift's AMD image is pinned to swift 4.1.0 / torch 2.10 ([AMD-support.md](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/AMD-support.md)), older than the transformers ≥5.16 / mcore-bridge 1.6.x that Flash-Next needs. No Flash-Next training on ROCm was found.
- **Kaggle.**
  - A 181.24 GB model instance exists and serves today (template-metadata.json lines 23-25), so a full-size derivative is uploadable.
  - Kaggle's official size limits were not found (UNVERIFIED). The search returned no Kaggle doc.

### Gaps
- Whether the H200 "Spot" meter is actually allocatable. The price equals on-demand, which is unusual.
- Azure ML vs plain VM overhead. No primary source checked; plain VMs avoid an extra layer [I].

## 8. Fallback compute

### Cited Findings
- **Lambda on-demand** ([pricing](https://lambda.ai/pricing), read 2026-10-18):
  - 8× B200 SXM6 180 GB: **$6.69/GPU-h** ($53.52/h per node), 2,900 GiB RAM, 22 TiB SSD.
  - 8× H100: $3.99/GPU-h.
  - No H200 or RTX PRO 6000 listed.
- **RunPod** (secondary trackers only, UNVERIFIED): H200 $5.29/h and B200 $7.99/h after a 2026-10-12 increase ([pricingsaas](https://pricingsaas.com/companies/runpod/diffs/2026W42)). RTX PRO 6000 is quoted at $1.69-2.09/h, with sources conflicting.
- Kaggle practitioners: "rtx 6000 pro can not be used for training of any ~100B model" (`W5b-kaggle-intel.md` line 19).

### Inferences
- [I] A Lambda 8×B200 node costs **37% less per GPU-hour than Azure H200**, with 28% more memory per GPU and a 22 TiB local disk. It is the best fallback if Azure quota stalls. The risk is sm100 kernel maturity (past fla bugs on B200).
- [I] For rollouts on the exact serving stack: Kaggle quota (~11 GPU-h/day) gives ~27M generated tokens/day at ~690 tok/s. Azure NC144lds RTX PRO 6000 spot at ~$1.02/h gives ~2.5M tokens/h, about **$0.41 per million tokens**. It needs the Pennyroyal wheelhouse running outside Kaggle's image (test-first).

## Implications for our plan

1. **DO: LoRA only on BF16 modules, merged into the Intel W4A16 shards.**
   - Target GDN in_proj/out_proj, QSA linear_qkv/linear_proj and optionally the shared expert. **Never the routed experts or the QSA indexer.**
   - This deletes the re-quantization step and the 50-100 GPU-h the 10-17 plan reserved for it (§3, §5).
   - Build the overlay or load-time merge first, and prove that a zero-delta merge reproduces today's offline score and tok/s.
2. **DO: keep ms-swift Megatron as the main trainer.** Pin the versions from #10376: ms_swift 4.6.0, mcore_bridge 1.6.5, megatron-core 0.16.1, transformers 5.19.0. Use `--padding_free true` (sparse QSA kernel), PLE offload and `--cross_entropy_loss_fusion`.
   - Measure peak memory at 16k and 32k on day 1 before choosing a sequence length.
   - Try CP2 (`cp_comm_type all_gather`) for 32k. It exists in code but has no doc numbers.
3. **TEST-FIRST: Axolotl QLoRA as a real-model wind tunnel.** Run 8 single-GPU jobs on the H200 node at 4k (~120 GiB each per the B300 measurement). This replaces the "small sibling" wind tunnel with the real model. Check that LoRA learned against bnb-NF4 experts transfers to AutoRound int4 experts after merge.
4. **DON'T plan online RL on our SGLang stack within the deadline.** Every SGLang-rollout RL recipe (miles) needs 32 GPUs and doesn't map the MTP. The only one-node RL is ms-swift GRPO-LoRA with vLLM rollouts and a gym-env interface ([gym_env.md](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/gym_env.md)) and R3 router replay (Megatron only, megatron-core ≥0.16, [router_replay.md](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/AdvancedResearch/router_replay.md)). Keep rollouts off-policy on Kaggle, or on Azure RTX PRO 6000 spot, and train offline (RFT/ReST-EM).
5. **DO: keep the albucino MTP drafter. Gate every candidate on:** spec-on vs spec-off greedy identity; accept length ≥ ~2.6; real tok/s within 5% of ~690.
   - If acceptance falls, retrain MTP jointly in ms-swift (`--mtp_num_layers 1`) and copy the dense part into the drafter.
   - Don't use SpecForge drafters, which can't be served on Qwen4Exp.
6. **DO: keep config.json untouched.** Overlay the tensors only, so that transformers ≥5.18 doesn't rename `full_attention` → `indexed_attention` (#10376).
7. **Compute ask (Azure).**
   - Node: one ND96isr H200 v5 in eastus2/westus2/westus3 at $84.80/h, plus a 2 TiB P40 ($236/month).
   - Budget: 450-950 GPU-h ≈ 56-119 node-hours ≈ **$4.8-10.1k**. Spot gives no discount.
   - Fallback: Lambda 8×B200 at $53.52/h if quota isn't granted within ~3 days.
   - Don't use MI300X (no Flash-Next training stack on ROCm) or GB200 (aarch64, $27/GPU-h).
8. **TEST-FIRST: append-only segment packing.** If our harness prompts are append-only between trims, one sequence can supervise many assistant turns, cutting training FLOPs by up to the turns per segment. At 55k tokens and loss on ~900 reply tokens, single-decision samples spend ~98% of compute on prompt tokens.

## Fact-check (independent verifier, 2026-10-18)

Method: each claim was re-read from its primary source on 2026-10-18, with no browser. Tools: `gh api` for repos, issues, PRs and commit history; `curl` for HF `config.json`, `model.safetensors.index.json` and `?blobs=true`; the Azure Retail Prices API, all pages fetched; the Microsoft Learn size pages, raw HTML including the storage tabs; `trafilatura` for the ms-swift, Axolotl, SGLang and Lambda pages.

| # | Claim (short) | Verdict | Correction / detail |
|---|---|---|---|
| 1 | Intel W4A16 keeps linear_attn, self_attn, shared_expert, mlp.gate, hyper_connection, indexer, mtp and ple at 16 bits; only routed experts are int4 (4-bit, g128, sym) | **Confirmed** | `quantization_config`: bits 4, group_size 128, sym true, `auto_round:auto_gptq`, autoround 0.15.0. `extra_config` has exactly those 16-bit patterns, plus `in_proj_a/b`, `embed_tokens`, `visual` and per-layer `mlp.gate`. The README `--ignore_layers` list matches. In the index, `qweight` exists only for `model.language_model.layers.N.mlp.experts.N.{gate,up,down}_proj` (24,576 each = 48 × 512). Every ms-swift LoRA target is a plain `.weight`. Detail: the MTP layer's 512 routed experts are also listed as 16-bit, so int4 covers only the 48 trunk layers' experts. `lm_head` is not quantized. |
| 2 | ms-swift recipe: `in_proj out_proj linear_proj linear_qkv`, TP2/EP4/PP2 on 8 GPUs; 77.7 GiB at 8k with PLE offload (72.5 with padding_free); only commit 06c7d80d (2026-09-07) | **Confirmed** | The table rows match exactly. The commit list for the English doc holds only `06c7d80d 2026-09-07 [doc] qwen3.8-flash-next best-practice (#10054)`. The readthedocs page carries the same numbers. |
| 3 | ms-swift MTP training via `--mtp_num_layers 1 --mtp_loss_scaling_factor 0.1` (example updated 2026-09-21); mcore-bridge #204 | **Confirmed** | `examples/models/qwen4_exp/megatron_sft.sh` lines 4-5 say exactly this. Last commit: `eeeaefb3`, 2026-09-21, "update qwen4_exp/glm5_next mtp example & glm5.3 support (#10219)". mcore-bridge `ec23691b`, 2026-09-21, is "support qwen4_exp & glm5_next MTP (#204)", which touches `qwen4_exp.py`, `mtp_layer.py` and `tests/test_qwen4_exp_mtp.py`. |
| 4 | Every SGLang- or vLLM-rollout RL recipe needs multiple nodes and is validated at 4k only (NeMo RL 8×8; miles 32 GPUs, MTP unmapped; prime-rl 16 B300 + TP8) | **Partially confirmed** | The three named recipes hold. NeMo RL: 8 nodes × 8 GPUs, vLLM TP8/EP8, 4k, 182 steps, "longer context lengths have not yet been validated". miles: 32 GPUs (8 × 4), SGLang TP8/EP8, "MTP tensors are not mapped yet". prime-rl #3434 (merged 2026-10-03): "16 trainer B300 GPUs and TP8 inference" (vLLM 0.30), context 4096. **But "every vLLM-rollout recipe" is false.** ms-swift's own best-practice page documents GRPO-LoRA on **one 8-GPU node** with **colocated vLLM** rollouts (`# 8*135G`, `max_completion_length 8192`, `vllm_max_model_len 9216`). The multi-node rule holds only for SGLang rollouts (miles). Also, the miles doc never says "validated at 4k": its launch example uses `--rollout-max-response-len 4096`, and the miles QSA kernel that mcore-bridge vendors cites a "16-node agentic run" at T=25k. |
| 5 | Axolotl QLoRA on 1× B300 at ~120 GiB with PLE offload (~216 without), sequence_len 4096, sdpa only | **Confirmed** | Doc: "QLoRA (1x B300 @ ~120 GiB w offload, else ~216 GiB)". The limitations table says `attn_implementation`: "`sdpa` only". `qlora.yaml` (added 2026-09-01, #3965): `sequence_len: 4096`, `sample_packing: true`, `micro_batch_size: 2`, r16. Caveats for the "8 jobs per H200" use: the figure was measured on a B300 (288 GB). That recipe also puts LoRA on the routed experts (`lora_target_parameters`). No H200 (~131 GiB usable) number exists, so the fit is unmeasured. |
| 6 | ND96isr H200 v5: $84.80/h on-demand in eastus2/westus2/westus3 (Linux), Spot meter also $84.80; 8× H200 141 GB, 1,850 GiB RAM | **Confirmed** | All 265 API rows were read. Linux Consumption is 84.80 for both the on-demand and Spot meters in eastus2, westus2 and westus3, which are the cheapest US regions. Learn lists 96 vCPU, 1,850 GiB, 8× H200 (141GB). The budget arithmetic is right: 450-950 GPU-h ≈ $4,770-10,070. Note: the API also lists a Linux "Low Priority" meter at $16.96/h (eastus2, westus3). Low-priority VMs were retired (Azure Batch on 2025-09-30, Azure ML on 2026-03-31; such nodes are now allocated as Spot), so this is not a usable discount. |
| 7 | Standard_NC144lds_xl_RTXPRO6000BSE_v6: 1 GPU, 144 vCPU, 264 GiB, $5.50/h on-demand, ~$1.02/h spot in eastus/westus2 | **Confirmed** | API: on-demand 5.50 (eastus, eastus2, eastus3, centralus, westus2, westus3); Spot 1.0164 (eastus, westus2 and also eastus2). Learn compute-optimized table: 144 vCPU, 264 GiB, 1 accelerator, 96 GB. |
| 8 | Transformers ≥5.18 normalizes `full_attention` → `indexed_attention`, which broke vLLM; an ms-swift full-SFT user hand-edited config.json | **Partially confirmed** | vLLM #59756 confirms the 5.18 rename broke loading. #10376 confirms the user fixed it by hand ("目前就是手动sed了", i.e. sed on the config). **Version scope is off.** Every transformers release with `qwen4_exp` rewrites `full_attention` in `configuration_qwen4_exp.py`: 5.16.0, 5.16.1 and 5.17.0 rewrite it to `qwen_sparse_attention`, and 5.18.0 and 5.19.0 to `indexed_attention`. So any `save_pretrained` export changes `layer_types`, not just ≥5.18. The vLLM break was a main/nightly regression after #57387 (2026-10-02), fixed by PR #59621 ("Bump Transformers version to 5.18.0", merged 2026-10-04). Implication 6 (keep the original config.json) stands and covers all versions. |

### Other factual errors found in the notes
- **§1, ms-swift #10376 "closed 2026-10-15".** It was opened 2026-10-15 and closed 2026-10-16.
- **§7 and §8, ND H200 v5 local disk.** The Learn storage tab lists a 1 × 1,024 GiB temp disk **plus 8 NVMe disks, 28 TiB**. The disk-need inference ("exceed the 1 TiB local disk, so attach ≥2 TiB") and the "22 TiB local disk" advantage given to Lambda both rest on the incomplete spec. The NVMe is ephemeral, so a persistent disk may still be wanted for checkpoints, but scratch space is not the constraint.
- **§4(a), overlay upload size.** In Intel's layout, the LoRA-target tensors (GDN, QSA and shared-expert weights) sit in 14 of 17 shards, which total about 72.3 GB. Copying the affected shards is therefore about 72 GB, not about 5.9 GB. A ~5.9 GB upload needs new shard files holding only the changed tensors, plus an overlaid `model.safetensors.index.json`. Our Kaggle copy is resharded, so its layout should be checked separately.
- **§5, auto-round #2323 date and setup.** The issue was opened 2026-09-09 UTC; the log timestamps are 09-08. The timed run was default `auto_round` ("quantized 1546/1551 layers in the block"), which quantizes the GDN and attention linears too, not Intel's experts-only ignore list. So the ~32 min QSA block and the ~9 h total are not measurements of an experts-only re-quant.
- **§1 Takeaway, "Every RL stack that supports the model … 24-64 GPUs".** ms-swift's GRPO-LoRA (vLLM colocated, 8 GPUs) is also a supported RL path. The notes cite it elsewhere (§1 Inferences, Implication 4), so the takeaway contradicts them.
- **§2, "HF transformers QSA is inference-only".** The maintainer's quote in #48826 is about the indexer: top-k truncates its gradient. QSA attention itself backpropagates under sdpa, which is how Axolotl trains with the indexer frozen. §2's own takeaway lists the HF sdpa path as a QSA backward.
- **§3, ms-swift #9802.** It concerns Qwen3.6-35B-A3B, not Flash-Next. The MTP-drop-on-merge behaviour is plausible for Flash-Next but was not reported for it.
- **§3, LoRA size (minor).** "r8 = 13.1M" assumes split HF modules (`in_proj_qkv/z/a/b`, `q/k/v/o_proj`). With ms-swift's fused Megatron targets (`in_proj`, `linear_qkv`), the same config arithmetic gives ≈10.4M at r8 [verifier's arithmetic].
- **§7 table (minor).** ND96isr H100 v5's $98.32 also applies in westus2, so the cheapest US regions are eastus, eastus2, westus2 and westus3.
- **Spot-checked and correct:** the mcore-bridge history and v1.6.5 date; mcore-bridge CP `all_gather` (lines 185-188) and the GDN fla imports; the NeMo AutoModel scope, validation scale and MTP note; Megatron-Bridge #6060; AutoModel #4147/#4334 (40.00 → 1.50 GiB) and #3971 (FA4 SM90/BF16/hd256); the verl, fla and SpecForge (#983, #943) and SGLang (#38589, #44304) issues; HF sizes (BF16 360.0 GB, 179,999,981,424 params, ungated; FP8 185.6 GB; Intel 181.2 GB with I32 15.2B, BF16 59.2B, F16 0.94B); the routed-expert (120.8B), GDN (2.09B), QSA (0.60B) and shared-expert (0.24B) arithmetic; the albucino `compact_sources.json` quote; the Kimi K2-Thinking QAT quote; the SGLang LoRA doc quote; the Lambda prices; Azure MI300X, GB200, P40 ($235.52) and egress ($0.087/GB, MGN); the ms-swift AMD image (swift 4.1.0, torch 2.10.0); router_replay (megatron-core ≥ 0.16.0); the Kaggle model-instance overview ("original weights are unchanged"); and the CUSTOM.md and template-metadata line references.
