# Open-weight model candidates for ARC-AGI-3 fine-tuning (research as of 2026-10-17)

Method: web search plus fetches of HF cards / docs. Secondary aggregator sites often disagree; "VERIFIED" = seen on an official card or doc, "UNVERIFIED" = aggregator/blog only.

## 1. Qwen3.8-Flash-Next (current model)

Facts (HF card https://huggingface.co/Qwen/Qwen3.8-Flash-Next, fetched):
- VERIFIED: 125B language-model params, 6B active, plus 51B "n-gram embedding" (the PLE/engram table) and 4B MTP. Safetensors metadata shows ~176-180B total (so older notes of "176B-A6B" are the full file count). SGLang cookbook: 176B-A6B.
- VERIFIED: 48 layers, hybrid Gated DeltaNet + Qwen Sparse Attention (QSA), each followed by MoE; 512 experts, 10 routed + 1 shared; hidden 2560; 4 hyper-connection streams (4 x 2560 = 10,240 flat state, per NeMo AutoModel doc); 1 MTP layer. Class name in NVIDIA NVFP4 card: `Qwen4ExpForConditionalGeneration`. Qwen calls it an early public preview of the Qwen4 architecture.
- VERIFIED: context 262,144 native, 1M with YaRN; text+image+video input; thinking by default; license `qwen-community-1.0` (not Apache; read LICENSE file for terms).
- VERIFIED: release 2026-08-26 (letsdatascience dateline). Only ONE post-trained checkpoint (plus -FP8). No separate Base / Instruct repos in the Qwen org list (https://huggingface.co/Qwen shows only Qwen3.8-2.4T-A95B(+FP8), Qwen3.8-27B(+FP8), Qwen3.8-Flash-Next(+FP8)).
- Card benchmarks (Qwen-reported): GPQA-D 91.7, LiveCodeBench v6 91.9, SWE-bench Pro 62.5, HLE 35.9, Toolathlon 73.5, IFBench 81.3. No ARC-AGI numbers found on arcprize.org results for it.
- Quants: Qwen FP8; nvidia/Qwen3.8-Flash-Next-NVFP4 (routed experts W4A4 NVFP4, attention/shared in BF16, PLE FP8; listed for B200/B300 only, needs vLLM PR #55513 for MTP); RadixArk NVFP4; Intel W4A16 AutoRound (ours); community NVFP4 (lesj0610, sdworld SDnvfp2).
- Serving on sm120: community report (Level1Techs) single RTX PRO 6000 NVFP4: 81-88 tok/s single stream, ~341 tok/s with 6 clients, 120-173 tok/s with MTP, needs ~100 GB host RAM for the 51B table; vLLM has no KV offload on sm120. SGLang cookbook lists only GB300/B300/B200/H200 (NEXTN MTP 3-1-4). Our ~690 tok/s at 14 streams is consistent with this being a well-tuned setup.

Trainability:
- ms-swift has an official best-practice page (https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html): Megatron-backend (mcore-bridge, Transformer Engine, flash-linear-attention>=0.5.2, causal-conv1d), transformers >=5.16. LoRA SFT recipe: rank 8, alpha 32, targets `in_proj out_proj linear_proj linear_qkv`, TP2/EP4/PP2 on 8 GPUs, ~70-77 GiB/GPU at seq 4-8k with PLE offload (93 GiB at 8k without; 142 GiB for TP2/EP4/PP1). GRPO-LoRA example: 8 GPUs, vLLM colocate, ~135 GB/GPU per script comment, max completion 8192, needs vLLM built from source with a specific PR.
- PLE_CPU_OFFLOAD=1: table frozen. Harmless for LoRA; silently freezes it under full-param training. PLE sits on pipeline stage 0 -> set decoder_first_pipeline_num_layers 12 to balance.
- NVIDIA NeMo AutoModel / NeMo RL: day-0 support claimed (NVIDIA post via traeai mirror); AutoModel doc: pipeline and tensor parallelism NOT supported, context parallelism supported, FlexAttention / SM90 CuTe sparse GQA kernels (SM90 = Hopper: check Azure GPU type), no KV/SSM cache in training model, loss outside model.
- Unsloth: only inference/GGUF guide found; no fine-tuning support evidence. LLaMA-Factory, verl, TRL, Megatron-Bridge: no evidence found either way (UNVERIFIED, check issue trackers). HF Transformers has the model (card says so). PEFT on HF path unverified.
- Published fine-tunes: none found beyond quants.
- Gotchas: huge frozen table (51B, ~95 GiB bf16) must be on host or sharded; MoE + hyper-connection + GDN kernels mean you need the Megatron path (8 GPUs for LoRA) rather than a single-GPU QLoRA; no base checkpoint means you are fine-tuning an RL-post-trained model; MTP head will drift unless you retrain/keep it (ms-swift page does not cover MTP); after fine-tuning you must re-quantize (W4A16 AutoRound again) and re-attach the albucino drafter or retrain it.
- Hardware needed for LoRA SFT: roughly 8x 80-96 GB GPUs (Azure ND H100 v5 / H200 / B200 class). Full-param training much more.

## 2. Candidate replacements

Fit rule: 96 GB VRAM, 177 GB host RAM, 14 streams, ~55k prompts.

| Model | Release | Total / active | Arch | Ctx | License | Quants / spec-decode | Base ckpt | Trainability | Fits 96 GB? | ARC-AGI evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 2026-08-26 | 125B(+51B table) / 6B | GDN+QSA hybrid MoE, HC, PLE | 262k | qwen-community-1.0 | FP8, NVFP4, W4A16; MTP | No | ms-swift Megatron LoRA, NeMo AutoModel; heavy | Yes (ours) | Our runs: 26.9-38.2 LB, 54.69 on 25 public |
| Qwen3.8-27B | 2026-08-14 | 27B dense | GDN+gated attn hybrid dense | 262k | Apache-2.0 | FP8, NVFP4 (community), MTP | Not listed | Best of all: same family as Qwen3.5/3.6 (Unsloth/TRL/PEFT/ms-swift/verl paths well trodden; UNVERIFIED for 3.8 specifically) | Yes, easily | Ours: 14.51 offline (FAILS 30 bar). arcprize.org: ARC-AGI-2 42.4%, ARC-AGI-3 none |
| Qwen3.6-27B | ~Apr 2026 | 27B dense | same | 262k | Apache-2.0 | many | Not listed on card | Excellent; community QLoRA (e.g. star-ga ARC-AGI-3 adapter) | Yes | Metir blog (secondary, UNVERIFIED) says top Kaggle entry (Tufa Labs, 55.89 public LB) runs local Qwen3.6-27B FP8 |
| Qwen3.6-35B-A3B | Apr 2026 | 35B / 3B | GDN+gated attn MoE 256 experts | 262k | Apache-2.0 | many, MTP | Not listed | Good (mainstream arch) | Yes | None; Terminal-Bench 2.0 51.5, GPQA 86.0 |
| Gemma 4 31B-it | ~Apr-Jun 2026 | 31B dense | sliding-window dense | 256k | Apache-2.0 | GGUF etc. | Yes (google/gemma-4-31B) | Excellent | Yes | Kaggle Milestone #1 winners (Reki, forge) built on Gemma-4-31B (Metir, secondary), low scores (winner 1.21%). GPQA 84.3, LCB 80.0, AIME26 89.2, tau2 76.9 |
| Gemma 4 26B-A4B | same | 25.2B / 3.8B | MoE | 256k | Apache-2.0 | yes | likely | Good (MoE LoRA less mature) | Yes | None |
| Mistral Small 4 119B-A6B | 2026-03 | 119B / ~6.5B | MoE, 128 experts, 4 active | 256k | Apache-2.0 | NVFP4 checkpoint, EAGLE head officially (mistralai/...-eagle), FP8 | Not listed | Axolotl supported (card) | NVFP4 ~65 GB: yes | None; GPQA-D 71.2 (below Flash-Next 91.7) |
| Nemotron 3.5 Super VL 120B-A12B | 2026-10-15 (HF repo not yet public per one report) | 120B / 12B | Mamba2-Transformer LatentMoE, MTP, RADIO vision | 1M (UNVERIFIED) | not found | NVFP4 (nvidia/...-NVFP4) | n/a | NeMo / Megatron-Bridge likely (UNVERIFIED) | NVFP4 yes | AA Intelligence Index 26.5 (low vs Flash-Next); none for ARC |
| Nemotron 3.5 Bolt 74B-A9B | 2026-10-15 | 74B / 9B | hybrid, native 2-bit | ? | ? | 2-bit native (24 GB) | ? | ? | Yes | AA index 21.4; Terminal-Bench 2.1 51.0 (NVIDIA) |
| Nemotron 3.5 Lightning 30B-A3B | 2026 | 30B / 3B | Mamba2+MoE+attn | 262k-1M | OpenMDW 1.1 | BF16 "for customization/post-training", NVFP4 | BF16 intended for post-training | NeMo | Yes | GPQA ~75, TB2.1 24.6 (weak) |
| gpt-oss-120b | 2025-08 | 117B / 5.1B | MoE, MXFP4 | 128k | Apache-2.0 | MXFP4 native (~63 GB); EAGLE3 community | No base | LoRA via TRL/Unsloth/ms-swift (from prior knowledge, not re-verified); no successor found | Yes | Older than Flash-Next; weaker agentic; no ARC-AGI-3 evidence |
| Step 3.7 Flash | May 2026 | 198B / 11B (AA) | MoE | 256k | Apache-2.0 | ? | ? | ? | 4-bit ~100 GB: NO (marginal) | none |
| Kolibri-1 (Aleph Alpha) | 2026-10-03 | 78B / 3.46B | MoE | ~1M claimed | Apache-2.0 | ? | ? | ? | likely | German/English only, UNVERIFIED, no evals |
| GLM-5.3-Flash | 2026-08-26 | 320B / 18B | MoE | 1M | MIT | - | - | - | NO | ARC-AGI-1 91.0, ARC-AGI-2 65.8 (arcprize); AGI-3 none |
| DeepSeek V4.1 Flash | 2026-09-10 | ~552-763B (disputed) / 16B | MoE+engram | 1M | MIT | - | - | - | NO | ARC-AGI-3 3.26% on arcprize.org |
| Kimi K3, MiniMax M3 (428B/23B), Muse Spark Open (816B/42B), Qwen3.8-2.4T-A95B, Tencent Hy3 (295B/21B), Nemotron 3 Ultra (550B) | 2026 Q2-Q4 | huge | MoE | - | mixed | - | - | - | NO (all need multi-GPU) | Kimi K3 ARC-AGI-2 60.4 |
| GLM-4.7-Flash | Jan 2026 | 30B / 3B | MoE | 200k | MIT | - | - | OK | Yes | Old, weaker |
| IBM Granite 4.1 (3/8/30B dense) | 2026-04 | dense | dense | 512k | Apache-2.0 | - | - | good | Yes | Enterprise-oriented, not frontier reasoning |
| Qwen 4 / Qwen 4 27B | announced 2026-09-22 | unknown | unknown | - | unknown | - | - | - | - | No weights as of 2026-09-29; a prediction market reportedly priced release before Nov 1 at 74% (secondary). Kaggle deadline is Nov 2, so too late to matter. |

Not found / no 2026 release surfaced: gpt-oss successor, Llama successor, Phi, LFM2 successors, ERNIE 5, Seed-OSS, MiniCPM, Intern-S (search returned nothing specific; UNVERIFIED either way). No "Qwen3.9" exists in results.

## 3. ARC-AGI-3 evidence
- Official arcprize.org results (https://arcprize.org/results): DeepSeek V4.1 Flash ARC-AGI-3 = 3.26%; Claude Opus 5 = 30.16%; Grok 4.6 = 2.11%. No open-weight small model has an official ARC-AGI-3 score. (Official benchmark is raw-LLM, not Kaggle harness.)
- Kaggle (RHAE scoring, offline): Milestone #1 winner 1.21%; Milestone #2 high 27.9% (Sep 30); by Oct 4 public top 55.89% (Tufa Labs, open-sourced harness; Manifold https://manifold.markets/Terminator2/arcagi3-kaggle-competition-what-wil and Metir https://www.metirai.com/blog/arc-agi-3-kaggle-scores-surge-harnesses-arc-prize-2026). Metir says Tufa runs local Qwen3.6-27B FP8 (secondary source; contradicts nothing in our data but is surprising next to our 14.51 for Qwen3.8-27B; likely harness/token-budget difference). Final deadline 2026-11-02. Market resolves on PUBLIC leaderboard.
- No evidence any entry fine-tuned a model and gained. Only a small QLoRA adapter (star-ga/naestro-agi3-27b on Qwen3.6-27B, synthetic sims, no reported scores). A Terminal-Bench write-up found harness-specific SFT can score below base when traces are poor.
- The rules require all code/methods open-sourced for prizes; no hardware limit published on the competition page.

## 4. Ranked shortlist

1. **Qwen3.8-Flash-Next (stay)** - the only model with evidence >=30 (our LB mean ~31, best 38.22). Fine-tuning is feasible but heavy: Megatron LoRA on 8x80GB+ (ms-swift) or NeMo AutoModel; no base checkpoint; MTP/quant/drafter must be redone. MEETS constraint.
2. **Qwen3.8-27B (dense, fine-tune target / distillation student)** - easiest to train (single-node QLoRA/LoRA, mainstream trainers), same hybrid family, Apache-2.0, fits 96 GB. But base 14.51 offline fails the >=30 rule as measured; only viable if fine-tuning is the very thing that lifts it (SPECULATIVE). Token-per-action 2x is the killer for time budget.
3. **Qwen3.6-27B** - Metir (secondary) claims top Kaggle score with it; if true, harness not weights explains the gap and this is cheap to train. SPECULATIVE; verify Tufa's actual notebook.
4. **Mistral Small 4 119B-A6B** - closest in size/active params and has official NVFP4 + EAGLE head + Axolotl; GPQA 71 vs 92 suggests substantially weaker reasoning. SPECULATIVE, likely <30.
5. **Gemma 4 31B** - has base checkpoint, very trainable, but Kaggle Milestone #1 entries on it scored ~1%; weaker reasoning. SPECULATIVE, likely <30.

Nothing else released through Oct 2026 clearly beats Flash-Next within the 96 GB envelope. Larger open models (GLM-5.3-Flash, DeepSeek V4.x Flash, Kimi K3, MiniMax M3) do not fit.

## Caveats
- Flash-Next param naming varies across sources (125B vs 176-180B); numbers above are from the HF card.
- Many numbers are from vendor cards or aggregators; no independent evaluation for several models.
- The claim that a 27B dense model wins the Kaggle leaderboard is unverified; if true, it undermines the assumption that model upgrade is the main lever.
- SM120: NVIDIA/SGLang official NVFP4 recipes target B200/B300/GB300; sm120 relies on community patches. FlexAttention/CuTe training kernels in NeMo are SM90 for CuTe.

## Sources
- HF card Qwen3.8-Flash-Next: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- HF card Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B
- HF Qwen3.6-35B-A3B: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- HF Gemma 4: https://huggingface.co/google/gemma-4-31B-it
- HF Mistral Small 4: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
- HF NVIDIA NVFP4 Flash-Next: https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
- ms-swift: https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html
- NeMo AutoModel: https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/models/qwen3_8_flash_next/model
- SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- Level1Techs sm120: https://forum.level1techs.com/t/qwen-3-8-flash-next-nvfp4-on-a-single-rtx-pro-6000-and-system-memory/254579
- arcprize results: https://arcprize.org/results ; competition page https://arcprize.org/competitions/2026/arc-agi-3
- Metir: https://www.metirai.com/blog/arc-agi-3-kaggle-scores-surge-harnesses-arc-prize-2026
- Manifold: https://manifold.markets/Terminator2/arcagi3-kaggle-competition-what-wil
- Nemotron 3.5: https://letsdatascience.com/news/nvidia-expands-nemotron-35-with-super-vl-and-bolt-ff81d36b
- Qwen4 status: https://www.yottalabs.ai/post/qwen-4-27b-release-date-specs-hardware-what-is-known-2026
- Naestro adapter: https://huggingface.co/star-ga/naestro-agi3-27b
- Dropped: OrcaRouter, Pasquale Pillitteri, benchlm and similar aggregators except for existence/date hints.
