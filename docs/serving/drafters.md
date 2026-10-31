# Flash-Next draft heads

Which drafter we serve for Qwen3.8-Flash-Next, and why. Log paths starting with `runs/` are relative to `ARC3-Inference/`.
That is the Google Drive folder (see [AGENTS.md](../../AGENTS.md)); Frida outputs are copied into `runs/local/frida/`.
GitHub links and `jonathanwang2022/*` Kaggle notebooks are private; Hugging Face and NVIDIA links are public.

## Summary

**Served: the fine-tuned MTP head with an FP8 drafter KV** ([decision](#decision)). Kaggle acceptance length is 2.805
against v6's 2.662 over the same 15 minutes, with v6's KV pool and no prefill storm.

| Drafter | Acceptance length (Kaggle) | Gen tok/s vs its baseline | Verdict | Section |
|---|---|---|---|---|
| Intel BF16 MTP, 3 tokens | 2.752 | 737 vs 820 (Lord) | 5.2 GB; replaced by Lord's | [Quantization](#expert-quantization-costs-no-acceptance) |
| Lord NVFP4 MTP, 3 tokens (porting-advaith v6) | 2.743 | 894 | baseline | [Quantization](#expert-quantization-costs-no-acceptance) |
| Lord MTP, BF16 drafter KV | 2.910 (Frida) | step cost unchanged | +0.13 acceptance, −6.9% pool | [Drafter KV](#drafter-kv-precision) |
| **Fine-tuned MTP, FP8 drafter KV** | **2.805** (v6: 2.662, first 15 min) | **+3.3%** | **served** | [Kaggle](#on-kaggle) |
| Fine-tuned MTP, BF16 drafter KV | 2.882 (first 15 min) | +5–7% until a storm at minute 13 | rejected | [Kaggle](#on-kaggle) |
| Fine-tuned MTP, FP8 KV, 4 tokens | 3.055 (Kaggle 0–15 min; k = 3: 2.805); Frida 3.249 | Kaggle gen −1.2% vs k = 3 (Frida −9.9% time per token) | no gain on the RTX: a step costs ~10% more | [k = 4](#k--4) |
| DFlash2 k = 3 (B200 replay) | 2.843 (B200) | +1% (B200) | no gain | [DFlash2](#dflash2-block-drafter-incoai) |
| DFlash2 k = 7 | 3.336 | −10.1% (first 30 min) | slower | [DFlash2](#dflash2-block-drafter-incoai) |
| DFlash2 k = 4, BF16 drafter KV | 2.902 | storm from minute 10 (pool −35.5%) | rejected; k = 7 ≥ k = 4 | [k = 4 vs k = 7](#k--4-vs-k--7) |

## Checkpoints

The four MTP drafters are the same Qwen3.8-Flash-Next MTP layer: one decoder layer with QSA attention, hyper-connections
and a 512-expert MoE, plus `fc_embedding`, `fc_hidden`, two pre-norms and a final mixer. They differ only in how the
routed experts are stored, and whether the checkpoint carries its own `embed_tokens` / `lm_head`.

| Drafter | Source | Routed experts | Rest of the layer | Own embed + lm_head | Bytes |
|---|---|---|---|---|---|
| Intel BF16 MTP | [`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`](https://www.kaggle.com/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround) = HF [`Intel/…-W4A16-AutoRound@4c67bf6`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/tree/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897) `model_extra_tensors.safetensors` | BF16, 5.033 GB | BF16, 0.181 GB | no | **5.214 GB** |
| Lord NVFP4 MTP | [`lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8`](https://www.kaggle.com/models/lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8) `hf-mixed-mtp-nvfp4/1` = HF [`primitive-ai/…-mixed-NVFP4-FP8@07915ee`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/tree/07915ee79ec217c117e8a57bf2557a4a1418c10f/mtp_nvfp4) `mtp_nvfp4/` | NVFP4 W4A4, 1.416 GB | BF16, **bit-identical to Intel's** (29/29 tensors) | no | **1.597 GB** |
| Daniel's albucino INT4 | [`dfranzen/albucino-qwen3-8-flash-next-drafter`](https://www.kaggle.com/models/dfranzen/albucino-qwen3-8-flash-next-drafter) `default/1` | INT4, 1.416 GB | BF16, 0.181 GB | yes, 2.542 GB | **4.140 GB** |
| Fine-tuned MTP (mtp-v1-50m, step 1300) | [`jonathanwang2022/qwen3-8-flash-next-mtp-tuned`](https://www.kaggle.com/models/jonathanwang2022/qwen3-8-flash-next-mtp-tuned) `PyTorch/default/1` (private) | Lord's, unchanged | BF16, retrained | byte copy of the target's, replaced at load | **1.597 GB** |
| DFlash2 (block drafter) | HF [`incoai/Qwen3.8-Flash-Next-DFlash2@a22a174`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/tree/a22a1744e85c17364dcc54a6a420af7135d4c7f6); our copy `jonathanwang2022/qwen3-8-flash-next-dflash2/Transformers/default/1` (private) | none (dense) | 5 layers, BF16 | no | **1.206 GB** |

Quantization:
- **Intel:** `extra_config` keeps `.*mtp.*` at 16 bits. Only the target's routed experts are quantized (AutoRound INT4,
  symmetric, group 128).
- **Lord** ([`mtp_nvfp4/config.json`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/blob/07915ee79ec217c117e8a57bf2557a4a1418c10f/mtp_nvfp4/config.json)):
  - Only `mlp.experts.N.{gate,up,down}_proj` are `nvfp4-pack-quantized`: FP4 E2M1 codes, an FP8 scale per 16 inputs
    and an F32 global scale, with activations quantized the same way at runtime. Gate and up share one global scale.
  - Attention, `mtp.fc_*`, norms, router, shared expert, hyper-connections and indexer are on the `ignore` list (BF16).
  - Dequantized experts match Intel's at cosine 0.995 (experts 0, 7, 300). So Lord's head is Intel's layer with
    experts rounded to NVFP4.
- **Albucino** ([`compact_sources.json`](https://www.kaggle.com/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1)):
  from `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225`, with routed experts RTN-quantized to INT4 (symmetric, group 32,
  weights only). It carries 31 dense tensors, including its own `embed_tokens` and `lm_head`.
- **DFlash2** ([`config.json`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/blob/a22a1744e85c17364dcc54a6a420af7135d4c7f6/config.json)):
  5 non-causal `sliding_attention` layers (window 2048, hidden 2560, 32 Q / 8 KV heads × 128). It drafts a block of
  8 (`selector_top_k` 16, `selector_rank` 256) from target taps at layers [5, 15, 25, 35, 45], and shares embed and
  `lm_head` with the target. sha256 `d6723f2a…3186f`.

The bit-identity and dequantization checks were run interactively on Frida (2026-10-30), not with a repo script. The
same decode is `dequant_nvfp4` in `inference/rft/mtp_train.py` (branch `mtp-data`).

## MTP head

### Expert quantization costs no acceptance

3 draft tokens, as served:

| Run | Drafter | Engine | Acceptance length | Per position | Gen tok/s | Log |
|---|---|---|---|---|---|---|
| [`matijaludvig/arc3-vcintel-25x1`](https://www.kaggle.com/code/matijaludvig/arc3-vcintel-25x1) v1 | Intel BF16 | vLLM, 13 streams | 2.752 | .767 / .572 / .418 | 737 | `runs/kaggle/matijaludvig/arc3-vcintel-25x1/v01/vllm-openai-server.log` |
| [`matijaludvig/arc3-vcintel-lorddraft-25x1`](https://www.kaggle.com/code/matijaludvig/arc3-vcintel-lorddraft-25x1) v1 | Lord NVFP4 | vLLM, 14 streams | 2.726 | .763 / .562 / .406 | 820 | `runs/kaggle/matijaludvig/arc3-vcintel-lorddraft-25x1/v01/vllm-openai-server.log` |
| [`jonathanwang2022/porting-advaith`](https://www.kaggle.com/code/jonathanwang2022/porting-advaith) v6 | Lord NVFP4, 32K draft vocab | vLLM, 14 streams | 2.743 | .767 / .568 / .412 | 894 | `runs/kaggle/jonathanwang2022/porting-advaith/v06/vllm-openai-server.log` |
| [`dangthuygiang/arc3-v207-danibase-base-dfranzen`](https://www.kaggle.com/code/dangthuygiang/arc3-v207-danibase-base-dfranzen) v5 | albucino INT4 | SGLang EAGLE, top-1 | 2.688 | not logged | — | `runs/kaggle/dangthuygiang/arc3-v207-danibase-base-dfranzen/v05/serve.log` |

The gaps between rows are within run-to-run variation. Lord's head pays off in memory: 1.6 GB against 5.2 GB, and the
difference went to the KV pool (a 14th stream). It also decodes faster, from smaller expert reads. Switching vcintel
to lorddraft raised throughput from 737 to 820 tok/s at the same score (56.92 vs 56.89;
[lord-vllm-daniel-weights.md](lord-vllm-daniel-weights.md#lords-mtp-draft-head)). GPU memory per setup is in
[MEMORY_BUDGET.md](MEMORY_BUDGET.md).

### Drafter KV precision

The target's KV is FP8 throughout. Keeping the drafter's own KV in FP8 costs about 0.13 acceptance length, all at draft
positions 2–3, where the head reads back its own drafts.

**Offline** (transformers, 48 held-out `mtp_v1` samples, Lord's untrained head). K/V and QSA-indexer caches are rounded
to e4m3 where vLLM's caches would be FP8:

| FP8 simulated in | Per position | Acceptance length | Frida job | Log |
|---|---|---|---|---|
| nothing | .813 / .642 / .489 | 2.943 | 209705 (step 0) | `runs/local/frida/mtp-eval-fp8/none/train_log.jsonl` |
| target only | .813 / .642 / .490 | 2.945 | 209816 | `runs/local/frida/mtp-eval-fp8/target/train_log.jsonl` |
| drafter only | .804 / .598 / .407 | 2.809 | 209815 | `runs/local/frida/mtp-eval-fp8/mtp/train_log.jsonl` |
| both (as v6 serves) | .804 / .606 / .432 | 2.842 | 209817 | `runs/local/frida/mtp-eval-fp8/both/train_log.jsonl` |

The drafter-only row reproduces v6's served third position (.407 vs .412). "Both" scoring above "drafter only" is not
explained. Code: [`qsa.set_kv_fp8`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/97c3655d8db698c685140aae948eb1d1beb89c60/ARC3-Inference/inference/rft/qsa.py#L43-L49),
switched on by [`mtp_train.py --fp8-kv`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/97c3655d8db698c685140aae948eb1d1beb89c60/ARC3-Inference/inference/rft/mtp_train.py#L235-L237)
at `97c3655d` (`--eval-only --no-compile-layers`).

**In vLLM on Frida** (one B200; the [replay](#frida-vllm-replay) of v6's 141 requests). The step cost is the same in
all four arms (time per token × tokens per step ≈ 103 ms), so the gains are pure acceptance:

| Head | Drafter KV | Acceptance length | Per position | Time per output token | KV pool (B200) | Job | Log |
|---|---|---|---|---|---|---|---|
| Lord | FP8 (v6) | 2.777 | .776 / .578 / .422 | 37.10 ms | 7,447,052 | 209838 | `…/runs/replay3-draft-fp8/`, `arc3_vllm_lord-209838.txt` |
| Lord | BF16 | 2.910 | .790 / .624 / .496 | 35.23 ms | 6,930,768 (−6.9%) | 209834 | `…/runs/replay3-draft-bf16/`, `arc3_vllm_lord-209834.txt` |
| fine-tuned | FP8 | 2.898 | .799 / .623 / .476 | 35.38 ms | 7,447,052 | 209926 | `…/runs/replay3-tuned-fp8/`, `arc3_vllm_lord-209926.txt` |
| fine-tuned | BF16 | **3.020** | .820 / .664 / .537 | **34.14 ms** | 6,930,768 | 209913 | `…/runs/replay3-tuned-bf16/`, `arc3_vllm_lord-209913.txt` |

`…` = `runs/local/frida/vllm-replay-draft-kv`.
- The head and the drafter KV add up independently. Each is worth about +0.12; together +0.243 (+8.8%) over v6.
- The FP8 Lord arm reproduces Kaggle v6's 2.743 to within 0.03.
- Wall time is 946 s in every arm, because the replay is prefill-bound.
- The B200 pool is 7× Kaggle's, so the smaller BF16 pool is never under pressure here.

**KV cost:** the MTP layer stores 2 KV heads × 256 dims, 1,024 B per token in FP8. BF16 adds 1,024 B to v6's ~14.9 KB
per token: −6.9% pool, 1.455M instead of 1.564M tokens on Kaggle.

**Serving a BF16 drafter KV** needs a runtime patch. The launcher sets `speculative_config.kv_cache_dtype` through
`kaggle.vllm.draft_kv_cache_dtype` (`auto` = BF16).
- Without the patch it fails at startup:
  [`draft-bf16-kv`](https://www.kaggle.com/code/jonathanwang2022/draft-bf16-kv) v1 died with "211.88 GiB KV cache is
  needed, which is larger than the available KV cache memory (21.71 GiB)"
  (`runs/kaggle/jonathanwang2022/draft-bf16-kv/v01/vllm-openai-server.log` line 190).
- Cause, traced in Frida job 209822 (`…/runs/arc3_vllm_lord-209822.txt`, `…/runs/draftkv-auto-debug/server.log`):
  - The draft dtype gives the drafter a copied cache config. The platform raises the attention block to 1664 only on
    the engine's config, so the drafter's QSA indexer cache stays a separate group of 16-token blocks.
  - Lord's packed layout strides every block by the widest page (24.5 MB), so one 147K-token request needs 9,192 of them.
- Fix: [`use_engine_block_size_for_draft_qsa`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6c8bffd173c72b657e10052c755d749411614f6f/ARC3-Inference/inference/framework/lordhansolo/kaggle.py#L1464-L1501)
  (`6c8bffd1`). It makes 3 anchored edits to `qsa_cache.py` so the QSA caches read the engine's block size, and is
  applied only when a draft KV dtype is set. With it, a full-length request needs 2.19 GiB, against 2.04 GiB for v6
  (job 209823: `…/runs/arc3_vllm_lord-209823.txt`, `…/runs/draftkv-auto-fix1/server.log`).

### Fine-tuned head (mtp-v1-50m)

- **What is trained:** Lord's 29 dense tensors only. Experts, `embed_tokens` and `lm_head` are frozen.
- **Trainer:** [`mtp_train.py`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/e0d1f88e36af708602cc026a973bd70ccbf1529f/ARC3-Inference/inference/rft/mtp_train.py)
  at `e0d1f88e` (branch `mtp-data`), Frida job 209705 on one B200. The frozen Intel W4A16 target runs forward only.
  The loss is a soft cross-entropy at the 3 draft positions, against the target's sampling distribution (T 0.7,
  top-k 20, top-p 0.95).
- **Data:** `mtp_v1` from [`mtp_data.py`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/e0d1f88e36af708602cc026a973bd70ccbf1529f/ARC3-Inference/inference/rft/mtp_data.py),
  every request of 54 Intel-W4A16 runs. Train: 3,821 samples (137M tokens). Eval: 842 samples from 5 held-out games
  (ar25, cd82, cn04, g50t, sk48).
- **Run:** 1,400 samples round-robin over games (~50M tokens), LR 3e-5, eval on 48 samples every 100 steps.
- **Offline result:** 2.943 → **3.080** at step 1300 (.839 / .686 / .555), the best of 15 evals. Log:
  `runs/local/frida/mtp-v1-50m/train_log.jsonl` (job output `arc3_mtp_train-209705.txt`, commit `commit_209705.txt`).
- **Served checkpoint:** the trainer's export cannot be served as it is. Its hard-linked `carry-model-bf16-0001{0,1,2}`
  files still hold Lord's original dense tensors, and vLLM loads tensors by name, so load order would pick the winner.
  [`build_tuned_draft.py`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/68b1a5c63fc476652f69e82f077a1bcdfd603285/ARC3-Inference/inference/rft/frida/build_tuned_draft.py)
  writes each tensor once:

  | File | Tensors | Bytes |
  |---|---|---|
  | `mtp-experts-nvfp4.safetensors` | 6,144 Lord experts, unchanged | 1,416,330,416 |
  | `mtp-finetuned-bf16.safetensors` | 29 retrained dense tensors (names, dtypes and shapes checked against Lord's) | 181,140,352 |
  | `draft-shared-bf16.safetensors` | `lm_head`, `embed_tokens`: byte copies of the target's (SHA-256 `40bddd25…`, `62288a3f…`) | 2,542,797,048 |

  At load the runtime swaps in the target's embedding and `lm_head`. The 32K draft-vocabulary head is cut from an
  identical copy, so VRAM grows by 1.597 GB, the same as Lord's ("Model loading took 69.54 GiB" in v6 and both
  fine-tuned Kaggle runs).

### On Kaggle

Both runs are v6 with only the draft changed, 30-minute public runs. The table covers the first 15 minutes after the
server's first metrics line (90 lines each; the FP8 run winds down after that).

| | v6 | **fine-tuned + FP8 KV** | fine-tuned + BF16 KV |
|---|---|---|---|
| Notebook | [`porting-advaith`](https://www.kaggle.com/code/jonathanwang2022/porting-advaith) v6 | [`tuned-mtp-fp8-kv`](https://www.kaggle.com/code/jonathanwang2022/tuned-mtp-fp8-kv) v1 ([config](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/b35f2e9715d09f9bc80a1315c4a4d15bd774f059/ARC3-Inference/configs/tuned-mtp-fp8-kv/v1.json) @ `b35f2e97`) | [`tuned-mtp-bf16-kv`](https://www.kaggle.com/code/jonathanwang2022/tuned-mtp-bf16-kv) v1 ([config](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/d2532fe17c3e2ea709bd23126739a62b48752269/ARC3-Inference/configs/tuned-mtp-bf16-kv/v1.json) @ `d2532fe1`) |
| Acceptance length (`accept --until 15`) | 2.662 | **2.805** (+5.4%) | 2.882 (+8.3%) |
| Per position | .744 / .539 / .382 | .778 / .590 / .443 | .793 / .618 / .486 |
| KV pool | 1,564,172 tokens | 1,564,172 | 1,455,400 (−7.0%) |
| KV maximum | 86.5% | 82.5% | 94.7% |
| Prefill storm | no | **no** | from minute 13 |
| Mean gen tok/s, minutes 0–14 | 922 | **952** (+3.3%) | 941 (977 over minutes 0–12) |
| Mean prompt tok/s, minutes 0–14 | 1,787 | 1,887 | 2,335 |
| Log (`vllm-openai-server.log` in) | `runs/kaggle/jonathanwang2022/porting-advaith/v06/` | `runs/kaggle/jonathanwang2022/tuned-mtp-fp8-kv/v01/` | `runs/kaggle/jonathanwang2022/tuned-mtp-bf16-kv/v01/` |

- **FP8:** each 5-minute block is 0.15–0.21 above v6. Throughput gains less than acceptance (+3.3% against +5.4%),
  because faster decoding brings more turns and prefill rises 5.6%. 15 minutes of play says nothing about the score.
- **BF16 storm:** the pool was 90.5% full at minute 12, against v6's 79.8%. From minute 13, at 93–95%, prompt
  throughput jumped from ~1.7K to 8–10K tok/s, generation fell to 190–430 tok/s, and prefix hits slid from 93% to 83%.
  v6's first-hour maximum is 88.5%. The run ended inside the storm. The head's weights are the same size as Lord's,
  so the smaller pool is the cause.

### Decision

Serve the fine-tuned head with the FP8 drafter KV.
- BF16 adds only +0.08 acceptance on Kaggle (2.882 vs 2.805) and +0.12 on Frida (3.020 vs 2.898).
- BF16 costs 7% of the pool and storms at 14 streams. Avoiding the storm takes 13 streams (1,455,400 / 13 ≈ 112.0K
  tokens each, as v6's 111.7K). Losing a stream likely costs more than the gain; that is not measured.
- FP8 keeps v6's pool and leaves room for a fourth draft token.

### k = 4

[`tuned-mtp-k4`](https://www.kaggle.com/code/jonathanwang2022/tuned-mtp-k4) is `tuned-mtp-fp8-kv` v1 with
`mtp_tokens: 4`. CUDA graphs go to 14 × 5 = 70, rounded to 72. Position 4 reuses the head untrained.
- **Guess:** if it decays like 2 → 3 (×0.75), position 4 is ≈ 0.33, giving ≈ 3.14 on Kaggle. It pays off if one more
  verify row per sequence costs less than that.
- **v1** ([config](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/a4da31c0340096ef827985b007e441ad8214d7ac/ARC3-Inference/configs/tuned-mtp-k4/v1.json)
  @ `a4da31c0`) **cannot start.** k = 4 raises the attention block to 1680, which v6's `prefix_match_unit` 128 does not
  divide: "Invalid prefix_match_unit=128 … Got group block sizes=[1680, 1680, 1680, 8, 1680]" (Frida job 209959,
  `runs/local/frida/vllm-replay-draft-kv/runs/arc3_vllm_lord-209959.txt`).
- **v2** ([config](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/ece901afb5ef701260b1a6e4d12798a4251e0a5c/ARC3-Inference/configs/tuned-mtp-k4/v2.json)
  @ `ece901af`) uses `prefix_match_unit: 112`.
  - The unit only sets where a hit may land. It must divide every prefix-cacheable block size and be a multiple of
    each cache's per-state compression, which divides 16.
  - 112 = 1680 / 15 is the valid value closest to 128.
- **Frida replay with 112** (job 209961, the [same replay](#frida-vllm-replay); the pool is 7,452,680 tokens):

  | k | Acceptance length | Per position | Time per output token | Step cost | Log (`runs/local/frida/vllm-replay-draft-kv/runs/`) |
  |---|---|---|---|---|---|
  | 3 | 2.898 | .799 / .623 / .476 | 35.38 ms | 102.5 ms | `replay3-tuned-fp8/`, `arc3_vllm_lord-209926.txt` |
  | 4 | **3.249** | .799 / .620 / .473 / .357 | **31.87 ms** (−9.9%) | 103.5 ms | `replay3-tuned-fp8-k4/`, `arc3_vllm_lord-209961.txt` |
  | 4, v3's graph list | 3.256 | .801 / .622 / .475 / .359 | 31.90 ms | 103.9 ms | `replay3-tuned-fp8-k4-graphs/`, `arc3_vllm_lord-209964.txt` |

  Position 4 is 1 + .357 more tokens per step for about 1% more step cost. `analyze.py` prints 3 positions; the fourth
  is the length minus the rest. The job's tail ("unexpected EOF" in `inside.sh`) came after the replay finished.
- **v3:** k = 4 needs its own CUDA graph list.
  - FULL decode graphs round each capture size up to a multiple of 1 + k and keep it within `max_num_seqs` requests
    (`vllm/v1/worker/gpu/cudagraph_utils.py`). Lord's runtime adds {44, 52} only for 4 tokens per request at 14
    sequences (`_set_cudagraph_sizes` in `vllm/config/vllm.py`), which gives k = 3 FULL graphs for 1, 2, 4, 6, 8,
    10–14 requests.
  - The default k = 4 list (1, 2, 4, 8, 16, …, 64, 72) covers 1, 2, 4, 5, 7, 8, 10, 12 and 13 requests. 72 rounds to
    75, past 14 requests, so a full 14-request decode step (v6 averages 13.9 running) runs PIECEWISE.
  - `kaggle.vllm.cudagraph_capture_sizes`
    ([`7f240b3f`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/commit/7f240b3f9cd64b1f2aea6b546fdd24485df323ea))
    passes an explicit list through `--compilation-config`, sets the maximum to its top, and refuses a list without
    the full decode step (70). [`configs/tuned-mtp-k4/v3.json`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/7f240b3f9cd64b1f2aea6b546fdd24485df323ea/ARC3-Inference/configs/tuned-mtp-k4/v3.json)
    = v2 + [1, 2, 4, 5, 10, 20, 30, 40, 50, 55, 60, 65, 70], the same request counts as k = 3.
  - On the B200 it changes nothing measurable (job 209964, the table's last row): 10 FULL decode graphs instead of
    9, the same pool, 31.90 against 31.87 ms per token, total decode time 6,640 against 6,666 s (−0.4%). Whether the
    RTX PRO 6000 gains more is not known; Kaggle v3 against v2 measures it.
  - Kaggle v3 was pushed at 13:53 EDT, 2026-10-31 (30-minute public run), after v1 and v2 held both GPU sessions.
    v2 keeps the default list; its acceptance numbers are unaffected.
- **Kaggle v3** (completed 2026-10-31; `runs/kaggle/jonathanwang2022/tuned-mtp-k4/v03`): the server was ready after
  10.5 minutes and served 18 minutes; the harness began its tail fade at 14.6 minutes, so the comparison is minutes
  0–15 of stats lines (`drafter_stats accept --until 15`, `timeline --bucket 5`, `steps --until 15`). Same pool
  (1,565,155 tokens), FULL 10 decode graphs.

  | Minutes 0–15 | tuned-mtp-k4 v3 (k = 4) | tuned-mtp-fp8-kv v1 (k = 3) | porting-advaith v6 |
  |---|---|---|---|
  | Acceptance length | **3.055** | 2.805 | 2.662 |
  | Per position | .769 / .572 / .423 / .310 | .778 / .590 / .443 | .744 / .539 / .382 |
  | Gen tok/s per 5 min | 875 / 967 / 961 | 909 / 975 / 950 | 862 / 957 / 959 |
  | Mean gen tok/s | **934** (−1.2%) | 945 | 926 |
  | KV max | 92.4% | 82.5% | 86.5% |
  | `steps` fit: mean step / fixed part | 47.5 / 34.7 ms (R² 0.58) | 41.1 / 31.6 ms (R² 0.95) | 40.3 / 31.9 ms (R² 0.96) |

  - +8.9% tokens per step at −1.2% generation means a k = 4 step costs about 10% more than a k = 3 step on the RTX
    PRO 6000 ((3.055 / 934) / (2.805 / 945) = 1.10). On the B200 it cost 1% more (103.5 vs 102.5 ms). The extra verify
    row is far more expensive on the RTX, as DFlash2 k = 7 already showed (decode part 1.21× MTP-3 vs 1.09–1.13×).
  - The untrained position 4 accepts 0.310 on Kaggle against 0.357 on Frida; positions 1–3 are also a little below
    k = 3's run, which played different turns.
  - KV peaked at 92.4% in minute 14–15, near where BF16 drafter KV started storming (93–95%); no storm in 15 minutes.
  - **v2 is not usable:** its server took 25 minutes to come up (17:25 → 17:50 UTC), leaving about 2.5 minutes of
    play, so the graph-list A/B on the RTX did not happen.
  - One run, 15 minutes: k = 4 does not raise throughput on Kaggle; k = 3 with the fine-tuned head stays the setup.

## DFlash2 block drafter (incoai)

DFlash2 drafts its whole block in one pass, so drafting is cheaper than MTP's three sequential passes. But the target
verifies k + 1 rows per sequence, and for this MoE more rows cost more. Even with both prefix-cache patches, it does
not beat MTP-3:
- **k = 3 (B200):** same acceptance as MTP-3 at about the same step time (0.98×).
- **k = 7 (B200):** acceptance 3.66 against 2.87 (+28%), but steps are 1.25× longer. Net +2%.
- **k = 7 (Kaggle):** acceptance 3.34 against 2.74, but generation is 10% slower over the first 30 minutes.
- **k = 4 (Kaggle, BF16 drafter KV):** acceptance 2.90, the same as k = 7's first four positions, so BF16 adds
  nothing and costs a third of the pool. A shorter block saves only verify rows, so k = 7 ≥ k = 4.

### Serving it

Lord's runtime needs setup-time patches,
[`DFLASH_RUNTIME_PATCHES`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/inference/framework/lordhansolo/kaggle.py#L1621-L1822)
(branch `dflash2-drafter`). They admit `dflash` for Qwen4Exp, guard the MTP hidden-state buffer, capture the five taps,
and round the QSA key ring for k > 4. They also include two prefix-cache fixes, both needed:
- **Draft-group rule**
  ([L1722–1733](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/inference/framework/lordhansolo/kaggle.py#L1722-L1733)).
  vLLM did not recognise DFlash2's group as the drafter's, so it applied the EAGLE drop to every group. The target's
  GDN groups then missed down to 0 tokens. Adding `dflash` to `_uses_trailing_mtp_layers` fixes it.
- **Replay-tail hold**
  ([L1734–1821](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/inference/framework/lordhansolo/kaggle.py#L1734-L1821)).
  A hit at the previous prompt's end needs the sliding-window blocks just below it, but the window frees them during
  decode. The request now holds them until it finishes, at the cost of up to 3 blocks per sliding-window group.

KV pool, from "GPU KV cache size" and "kv cache group sizes" in each run's `vllm-openai-server.log`:

| Run | KV pool | Group block sizes | Log |
|---|---|---|---|
| porting-advaith v6 (MTP-3) | 1,564,172 tokens / 21.73 GiB | 1664 × 4, 8 | `runs/kaggle/jonathanwang2022/porting-advaith/v06/` |
| [`dflash2-k3`](https://www.kaggle.com/code/jonathanwang2022/dflash2-k3) v2 | 1,616,364 / 22.23 GiB | 1664 × 6, 8 | `runs/kaggle/jonathanwang2022/dflash2-k3/v02/` |
| [`dflash2-k7`](https://www.kaggle.com/code/jonathanwang2022/dflash2-k7) v3 | 1,580,652 / 21.91 GiB | 1744 × 6, 16 | `runs/kaggle/jonathanwang2022/dflash2-k7/v03/` |

- DFlash2 weighs less than Lord's MTP (model loading 69.11 vs 69.54 GiB), but it adds two sliding-window groups.
  Each sequence's ~3 window blocks per group cost full 24.6-MB pages, though they hold only 2048 tokens.
- k = 7 grows the attention block to 1744, which 128 does not divide, so dflash2-k7 v2 died at startup
  (`runs/kaggle/jonathanwang2022/dflash2-k7/v02/vllm-openai-server.log` line 199). v3 uses `prefix_match_unit` 16
  ([config](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/configs/dflash2-k7/v3.json)).

### B200 replay

Frida `ixb2`, Lord's runtime, each arm with its Kaggle pool size. The load is recorded GRPO episodes (v6 harness)
replayed at 14 streams from 40K-token contexts over a 900-s window
([README](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/scripts/dflash2_b200/README.md),
[`bench.py`](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/scripts/dflash2_b200/bench.py)).
The step time is acceptance length ÷ gen tok/s, relative to MTP-3, prefill included.

| Drafter | Patches | Prefix hit | Gen tok/s | Acceptance per position | Acceptance length | Step vs MTP-3 | Log (`runs/local/frida/dflash2-b200/`) |
|---|---|---|---|---|---|---|---|
| MTP-3 (v6) | n/a | 92.8% | 1,298 | .785 .610 .470 | 2.865 | 1 | `209821/C/` |
| DFlash2 k = 3 | none | 47.4% | 545 | .791 .619 .485 | 2.894 | | `209821/B/` |
| DFlash2 k = 3 | hold only | 53.5% | 588 | .777 .600 .461 | 2.838 | | `209824/B/` |
| DFlash2 k = 3 | rule only | 62.1% | 681 | .786 .612 .478 | 2.876 | | `209829/D/` |
| DFlash2 k = 3 | rule + hold | 90.1% | 1,315 | .780 .600 .463 | 2.843 | 0.98 | `209829/B/` |
| DFlash2 k = 7 | none | 41.5% | 505 | .754 .559 .418 .324 .258 .211 .176 | 3.700 | | `209821/A/` |
| DFlash2 k = 7 | rule + hold | 90.0% | **1,329** | .744 .552 .414 .318 .253 .207 .174 | **3.662** | **1.25** | `209829/A/` |

Each arm folder holds `bench/summary.json` (these numbers) and `vllm-openai-server.log`. The job output is
`<job>.txt`. Unpatched hit rates differ from the README's by up to 0.9 points.

### On Kaggle

**k = 3 without the prefix-cache patches**: [`dflash2-k3`](https://www.kaggle.com/code/jonathanwang2022/dflash2-k3) v2
([config](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/fc12dafdeaada5ab6fa1f1cf73ae08a15f51bc81/ARC3-Inference/configs/dflash2-k3/v2.json)
@ `fc12dafd`). Acceptance was 2.676 against v6's 2.743, but the prefix cache collapsed. Mean gen tok/s / last
prefix-hit rate per 10 minutes:

| Minutes | 0–10 | 10–20 | 20–30 | 30–40 |
|---|---|---|---|---|
| dflash2-k3 v2 | 764 / 74% | 382 / 58% | 283 / 47% | 359 / 43% |
| v6 | 909 / 92% | 930 / 93% | 916 / 93% | 869 / 93% |

Logs: `runs/kaggle/jonathanwang2022/dflash2-k3/v02/vllm-openai-server.log`, `…/porting-advaith/v06/vllm-openai-server.log`.

**k = 7 with both patches**: [`dflash2-k7`](https://www.kaggle.com/code/jonathanwang2022/dflash2-k7) v3
([config](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6fd992579012872e5f8926468d78f4c7fb53137f/ARC3-Inference/configs/dflash2-k7/v3.json)
@ `6fd99257`). The games ran for 30 minutes before the time cap, so the comparison covers the first 30 minutes of stats
lines (180 each).

| First 30 min | dflash2-k7 v3 | v6 | change |
|---|---|---|---|
| Acceptance length (whole run) | **3.336** | 2.743 | +21.6% |
| Per position | .740 .529 .376 .270 .197 .148 .111 | .767 .568 .412 | |
| Gen tok/s | **825** | 918 | −10.1% |
| Prompt tok/s | 3,106 | 1,923 | +62% |
| Prefix hit at 2 / 10 / 20 / 30 min | 61 / 85 / 88 / 89% | 92% from 10 min | |
| Harness tok/s, minutes 0–10 / 10–20 / 20–30 (`stdout.log`) | 702 / 744 / 847 | 806 / 956 / 852 | |

Logs: `runs/kaggle/jonathanwang2022/dflash2-k7/v03/{vllm-openai-server,stdout}.log`, `…/porting-advaith/v06/{…}.log`.
- Kaggle accepts less than the B200 (3.34 vs 3.66), almost all at positions 4–7. MTP-3 loses less (2.74 vs 2.87).
  The cause is not known; the replay skips early-game turns.
- Prefix hits work now, but they level off at 88–89% against 92–93%, which adds prefill.
- Most of the shortfall is in the first 20 minutes. In minutes 20–30 the harness rates are equal (847 vs 852).
- This is one run; 30 minutes of play is not a score comparison.

### Step time: drafting vs verification

From `drafter_stats steps` (10-s intervals with ≥ 12 running requests), fitted as fixed cost + cost per prefill token.
The logs are each run's `vllm-openai-server.log` above.

| Run | GPU | Mean step | Prefill tokens per step | Fit | R² |
|---|---|---|---|---|---|
| v6, MTP-3 | RTX PRO 6000 | 45.0 ms | 130 | 35.5 ms + 73.0 µs | 0.82 |
| dflash2-k7 v3 | RTX PRO 6000 | 57.4 ms | 199 | 44.9 ms + 62.8 µs | 0.80 |
| 209821 C, MTP-3 | B200 | 32.0 ms | 166 | 25.2 ms + 40.8 µs | 0.93 |
| 209829 B, DFlash2 k = 3 | B200 | 31.3 ms | 210 | 20.4 ms + 52.0 µs | 0.81 |
| 209829 A, DFlash2 k = 7 | B200 | 40.1 ms | 283 | 25.2 ms + 52.4 µs | 0.96 |

dflash2-k3 v2 is left out: its storm made most steps mixed prefill/decode. Because the fixed term trades off against
the slope, the steadier split subtracts prefill at one per-token cost per GPU, taken across that GPU's fitted slopes:

| Decode part of a step | MTP-3 | DFlash2 k = 3 | DFlash2 k = 7 |
|---|---|---|---|
| B200 (prefill 41–52 µs/token) | 23.4–25.2 ms | 20.4–22.7 ms (0.87–0.90×) | 25.4–28.5 ms (1.09–1.13×) |
| RTX PRO 6000 (prefill 63–73 µs/token) | 35.5–36.8 ms | not measured | 42.9–44.9 ms (1.21–1.22×) |

- **Drafting:** at k = 3 both drafters verify 4 rows, so the 10–13% gap is DFlash2's single pass against MTP's three.
- **Verification:** k = 3 → 7 adds 25% to the decode part. The target runs 14 × (k + 1) rows. With 10 of 512 experts
  per token, a layer touches ~342 experts at 56 rows and ~456 at 112 (+33% expert reads).
- **Prefill:** k = 7 prefills more per second (B200: 7.1K vs 5.2K tok/s), the cost of its lower hit rate.
- Requests in prefill do not draft, so steps are slightly undercounted.

### k = 4 vs k = 7

[`dflash2-k4-bf16-kv`](https://www.kaggle.com/code/jonathanwang2022/dflash2-k4-bf16-kv) v1
([config](https://github.com/Wang-Zhongwei/arc-agi-3-solution/blob/6a3a741ff60b07babf2d2918354a28855f51410c/ARC3-Inference/configs/dflash2-k4-bf16-kv/v1.json)
@ `6a3a741f`, branch `dflash2-k4-bf16kv` = `dflash2-drafter` + `draft-bf16-kv`) is dflash2-k7 v3 with `mtp_tokens` 4 and
`draft_kv_cache_dtype` `auto` (BF16). The target KV stays FP8.

| | dflash2-k4-bf16-kv v1 | dflash2-k7 v3 | v6 (MTP-3) |
|---|---|---|---|
| Acceptance, whole run | 2.902 (.743 .533 .378 .270) | 3.336; first 4 positions 2.915 | 2.743 |
| Acceptance, first 10 min | 2.890 | 3.203; first 4 positions 2.874 | 2.638 |
| KV pool | **1,019,208** / 22.21 GiB (1680 × 5, 8) | 1,580,652 / 21.91 GiB | 1,564,172 / 21.73 GiB |
| Gen tok/s, KV max, prefix hit: min 0–10 | 738, 97.8%, 75.0% | 790, 68.1%, 68.8% | 909, 67.0%, 85.6% |
| min 10–20 | **261**, 99.7%, 64.3% | 820, 89.1%, 87.8% | 930, 87.5%, 93.1% |
| min 20–30 | **231**, 99.7%, 45.9% | 864, 79.4%, 88.2% | 916, 85.1%, 92.9% |
| Min 2–10: mean step / prefill tokens per step | 60.2 ms / 286 | 53.1 ms / 150 | 39.4 ms / 69 |
| Min 2–10: decode part (prefill at 68 µs/token) | 40.8 ms | 42.9 ms | 34.7 ms |
| Log (`vllm-openai-server.log` in) | `runs/kaggle/jonathanwang2022/dflash2-k4-bf16-kv/v01/` | `…/dflash2-k7/v03/` | `…/porting-advaith/v06/` |

- **A BF16 drafter KV gives DFlash2 nothing.** Every position is within .004 of k = 7's FP8 one, because DFlash2 never
  reads its own drafts back.
- **It costs a third of the pool.** Doubling the sliding-window pages raises the per-block stride from ~26.0 to
  ~39.3 MB. KV is 97.8% full within 10 minutes, then a prefill storm follows (hits 75 → 46%, generation 738 → 231).
- **Drafting is flat in k.** One pass reads the 1.206 GB drafter and at most the 1.271 GB BF16 `lm_head`: ~1.4 ms at
  the RTX PRO 6000's 1.79 TB/s, whatever k is. Compute is 0.17–0.28 TFLOP. MTP, by contrast, runs one pass per token.
- **Verification grows less than acceptance.** k = 7 verifies 112 rows against 70. With 10 of 512 experts per token,
  a layer touches 456 against 383 experts (+19% routed weights), while positions 5–7 add 0.456 tokens (+15.7%).

Time per accepted token = decode part / acceptance + prefill per accepted token. Prefill is taken at k = 7's
150 × 68 µs, since the games, not k, set it. A clean FP8 k = 4 accepts k = 7's first four positions.

| k = 4 decode part | Basis | k = 7 vs k = 4, whole run | First 10 min |
|---|---|---|---|
| 40.8 ms | measured (BF16 run, before the storm) | **+7.5%** | **+4.4%** |
| 37.4 ms | B200's k = 3 → 7 growth interpolated to 383 experts | −0.2% (tie) | −2.3% |
| 36.1 ms | bound: whole decode part scales with routed experts | −3.0% | −5.1% |

So k = 7 ≥ k = 4. The bound is pessimistic for k = 7, because drafting, attention, GDN and dense weights do not grow
with rows. Neither k reaches MTP-3's 34.7 ms. These are one run per arm, compared on step times, not scores.

## Method notes

### Acceptance and throughput from server logs

([`drafter_stats.py`](../../ARC3-Inference/inference/tools/drafter_stats.py)):
- vLLM: token-weighted from "SpecDecoding metrics", `1 + accepted / (drafted / k)`. Per-position rates are the mean
  of the logged rates.
- SGLang: unweighted mean of `accept len`.
- tok/s: mean of the 10-s "Avg generation / prompt throughput" lines.

### Frida vLLM replay

(`runs/local/frida/vllm-replay-draft-kv/`: `run_vllm.sbatch`, `inside.sh`, `setup_runtime.py`,
[`replay.py`](../../ARC3-Inference/runs/local/frida/vllm-replay-draft-kv/replay.py),
[`analyze.py`](../../ARC3-Inference/runs/local/frida/vllm-replay-draft-kv/analyze.py)):
- Lord's runtime is rebuilt on one B200 from `vllm/vllm-openai@sha256:e0eee5c5…` plus his 32 overlay files
  (sha256-checked against `runtime-manifest.json`), with the launcher's patches and v6's exact `vllm serve` command.
  `LORD_DIR` swaps the draft; `MTP_TOKENS`, `PREFIX_MATCH_UNIT` and `CAPTURE_SIZES` (a comma list for
  `cudagraph_capture_sizes`) change the server command, and `CLIENT` runs the replay.
- The load is v6's own 141 `mtp_v1` samples, one request each: the context before the last trained reply, mean 104K
  prompt tokens. Sampling is the harness's, with a fixed seed per request, up to 2,048 generated tokens, 14 at a time.
- Numbers are differences of `/metrics` counters before and after the replay (`metrics_before.txt` /
  `metrics_after.txt`).

### Offline eval vs vLLM

- The offline target is transformers Qwen4Exp with the RFT patches, and the head is vLLM's `Qwen4ExpMultiTokenPredictor`
  rebuilt from transformers modules.
- FP8 is a scale-1.0 rounding of stored K/V. Draft steps 2–3 attend to keys from the same step.
- The eval set is 5 held-out games and leaves out `uncertain` requests and requests over 124K tokens. So offline stays
  above served even with FP8.
- The two implementations were not compared layer by layer.

## Reproduce

From `ARC3-Inference/`. `drafter_stats bytes` reads safetensors headers by HTTP range requests (Kaggle needs
`~/.kaggle/access_token`).

```sh
P=https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/resolve/07915ee79ec217c117e8a57bf2557a4a1418c10f
python -m inference.tools.drafter_stats bytes $P/mtp_nvfp4/mtp-experts-nvfp4.safetensors \
  $P/mtp_nvfp4/carry-model-bf16-00011.safetensors $P/carry-model-bf16-00012.safetensors \
  $P/carry-model-bf16-00010.safetensors --prefix mtp.                       # Lord: 1.597 GB
python -m inference.tools.drafter_stats bytes \
  https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/resolve/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/model_extra_tensors.safetensors \
  --prefix mtp.                                                             # Intel: 5.214 GB
K=https://www.kaggle.com/api/v1/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1/download
python -m inference.tools.drafter_stats bytes $K/mtp-dense.safetensors $K/mtp-routed-experts-int4.safetensors  # 4.140 GB
curl -sIL https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/resolve/a22a1744e85c17364dcc54a6a420af7135d4c7f6/model.safetensors \
  | grep -i x-linked                                                        # DFlash2: 1,206,140,008 B, sha256

# Kaggle runs (make pull-kaggle NOTEBOOK=jonathanwang2022/<name> VERSION=<n> if missing)
L=runs/kaggle/jonathanwang2022
python -m inference.tools.drafter_stats accept $L/porting-advaith/v06/vllm-openai-server.log           # 2.743
python -m inference.tools.drafter_stats accept $L/dflash2-k3/v02/vllm-openai-server.log                # 2.676
python -m inference.tools.drafter_stats accept $L/dflash2-k7/v03/vllm-openai-server.log                # 3.336
for r in tuned-mtp-fp8-kv/v01 tuned-mtp-bf16-kv/v01 porting-advaith/v06; do
  python -m inference.tools.drafter_stats accept $L/$r/vllm-openai-server.log --until 15; done          # 2.805 / 2.882 / 2.662
for r in tuned-mtp-bf16-kv/v01 porting-advaith/v06; do                                                 # storm timeline
  python -m inference.tools.drafter_stats timeline $L/$r/vllm-openai-server.log --bucket 5; done
for r in porting-advaith/v06 dflash2-k7/v03 dflash2-k3/v02; do                                         # step-time table
  python -m inference.tools.drafter_stats steps $L/$r/vllm-openai-server.log; done
for r in dflash2-k4-bf16-kv/v01 dflash2-k7/v03 porting-advaith/v06; do F=$L/$r/vllm-openai-server.log  # k = 4 vs k = 7
  python -m inference.tools.drafter_stats accept $F; python -m inference.tools.drafter_stats accept $F --until 10  # 2.902 / 3.336 / 2.743; 2.890 / 3.203 / 2.638
  python -m inference.tools.drafter_stats timeline $F --bucket 10
  python -m inference.tools.drafter_stats steps $F --from 2 --until 10 --prefill-us 68                  # decode 40.8 / 42.9 / 34.7 ms
  grep -E "GPU KV cache size|kv cache group sizes" $F; done                                             # k = 4: 1,019,208 tokens
python -m inference.tools.drafter_stats bytes \
  https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/resolve/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897/model-00017-of-00017.safetensors \
  --prefix lm_head                                                          # target lm_head: 1.271 GB BF16
for a in 209821/C 209829/B 209829/A; do                                                                # B200 step-time rows
  python -m inference.tools.drafter_stats steps runs/local/frida/dflash2-b200/$a/vllm-openai-server.log; done

# Frida vLLM replay
for a in draft-fp8 draft-bf16 tuned-fp8 tuned-bf16 tuned-fp8-k4 tuned-fp8-k4-graphs; do                # 2.777 / 2.910 / 2.898 / 3.020 / 3.249 / 3.256
  python3 runs/local/frida/vllm-replay-draft-kv/analyze.py runs/local/frida/vllm-replay-draft-kv/runs/replay3-$a; done
```
