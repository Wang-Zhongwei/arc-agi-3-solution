# G-D (gap G10): existing post-trained Flash-Next variants as free evidence

Scope: public post-trained Qwen3.8-Flash-Next variants. For each: what tensors it changed (checked at file level against `Qwen/Qwen3.8-Flash-Next` rev `de4b8e4d43b9`), its method, licence, claims and Kaggle evidence, whether it can be overlaid on our Intel W4A16 shards without re-quantization, and what a smoke test in our harness would show.

Method. No weight files were downloaded. Shard comparisons used the HF API (`?blobs=true` LFS sha256) and `model.safetensors.index.json`. Tensor-level checks used HTTP range reads of each safetensors header, plus 4 KB byte windows hashed per tensor: one window for most tensors, eight spread across each routed-expert `down_proj`. That came to ~8,000 windows, about 33 MB in total. Scripts and outputs are in the scratchpad `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/`:
- `scripts/cmp_shards.py`, `scripts/shard_class.py`: shard sha256.
- `scripts/sample_tensors.py` → `samples/results.json`, then `scripts/analyze_samples.py` → `samples/analysis.txt`.
- `scripts/magcheck.py`: delta size.
- `scripts/intel_check.py`: Intel vs Qwen.
- `scripts/agnes_check.py`.
- `scripts/felix_bench.py`: Kaggle A/B.
- `lb/…2026-10-18T10:46:31.csv`: leaderboard.

## 1. What exists, and which tensors each variant changed

### Takeaway
Of 547 HF repos with `base_model: Qwen/Qwen3.8-Flash-Next`, only about 8 are real post-trains. Two of them change no routed expert: **Swift 1.5** and **Tinfield-1**. In both, every changed tensor is one that the Intel W4A16 checkpoint stores in BF16, so each can be overlaid on our Kaggle model with no re-quantization (Swift 5.82 GB, Tinfield 6.64 GB). AZ-Focus, Agnes-3.0, CYBER-FROST, Victoria, REAP and the abliterations all change or remove routed experts.

### Cited Findings
- **The listing.** The HF API returns 547 repos (2 pages) with `base_model: Qwen/Qwen3.8-Flash-Next`: 457 quantized, 82 finetune, 6 merge and 2 adapter. Most "finetune" entries are abliterations, prunes, format conversions or engines ([api](https://huggingface.co/api/models?filter=base_model:Qwen/Qwen3.8-Flash-Next&limit=500); scratch `hfapi/all_derivs.json`).
- **Base layout** (`Qwen/Qwen3.8-Flash-Next`, 131 shards, 360,000,192,888 B, 1,658 tensors; [index](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/model.safetensors.index.json)):
  - Each layer's `mlp.experts.gate_up_proj` sits alone in a shard of 3,355,443,368 B (48 shards).
  - The 512-expert routed MLP is fused into `experts.gate_up_proj` and `experts.down_proj`, with `down_proj` shaped [512, 2560, 640].
  - `experts.down_proj` shares a shard with that layer's router, shared expert and hyper-connection (HC) tensors and the next layer's attention.
  - The PLE n-gram table has 31 shards of its own, and `lm_head` has one.
- **Shard sha256 vs base** (`cmp_shards.py`, `shard_class.py`):
  - Tinfield-1 (rev `f7c714a97b`), Swift1.5 (rev `0bd4fe2243`), cyberneurova and the Splice-QFN merge each keep exactly 80 of 131 shards sha-identical: all 48 `gate_up_proj` shards, all 31 PLE shards and `lm_head`.
  - Outside those 80 base shards, no two of these variants share a shard, so each is a distinct checkpoint.
  - AZ-Focus keeps 103 of 131 shards identical. CYBER-FROST keeps only the 31 PLE shards, so all its `gate_up_proj` changed. huihui-abliterated keeps none.
- **Tensor-level check of the 51 changed shards** (`analyze_samples.py`; per-family table in `samples/analysis.txt`):
  - **Tinfield-1** changed 518 tensors (6.64 GB), all BF16 in Intel:
    - GDN `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` and `out_proj` in all 36 linear-attention layers;
    - QSA `q/k/v/o_proj` in all 12 full-attention layers;
    - HC `input_mix_weight_down/up` and `block_inject_weight` (attn and mlp) in all 48 layers, plus the global `hyper_connection_mixer` mix weights.
    - **Unchanged:** routed experts (`gate_up_proj` by sha; `down_proj` by 8 windows × 48 layers), shared experts, router `mlp.gate`, norms, `A_log`/`dt_bias`/`conv1d`, the QSA indexer, PLE, `embed_tokens`, `lm_head`, all MTP tensors and vision.
  - **Swift 1.5** changed 300 tensors (5.82 GB):
    - GDN `in_proj_qkv`, `in_proj_z` and `out_proj` (36 layers);
    - QSA `q/k/v/o_proj` (12 layers);
    - `shared_expert.gate/up/down_proj` (48 layers).
    - **Unchanged:** routed experts, router, HC, `in_proj_a/b`, MTP, PLE, embed, `lm_head`.
  - **AZ-Focus** (BF16) changed 75 tensors, 42.81 GB in total, of which 41.94 GB is routed `experts.down_proj`:
    - `linear_attn.out_proj`, `self_attn.o_proj`, `shared_expert.down_proj` and **routed `experts.down_proj`**, in layers 20-44 only.
    - Its card's "some output-projection weights changed" ([card](https://huggingface.co/AxonZeta/Qwen3.8-Flash-Next-AZ-Focus)) therefore includes the routed experts.
  - **cyberneurova** ("uncensored") changed `embed_tokens` and every output-writing matrix: GDN `out_proj`, QSA `o_proj`, shared and **routed** `down_proj` (145 tensors, 83.47 GB). That is an abliteration pattern.
  - **Agnes-3.0-Qwen** (119 shards in its own layout, so checked by tensor name with `agnes_check.py`): changed GDN `in_proj_qkv`/`out_proj`, QSA `q`/`o`, `shared_expert.down_proj`, and **routed `down_proj`** (all 8 windows in layers 30 and 47, some windows in layer 0). Unchanged: router, HC, embed and MTP.
- **Size of the deltas** (4,096 mid-tensor BF16 values, layer 30; `magcheck.py`):

  | Variant | Tensor | Values changed | Relative RMS delta |
  |---|---|---:|---:|
  | Tinfield | GDN `out_proj` | 601/4096 | 0.04% |
  | Tinfield | `in_proj_qkv` | 850/4096 | 0.08% |
  | Tinfield | HC `input_mix_up` | 1433/4096 | 0.12% |
  | Swift | GDN `out_proj` | 2068/4096 | 0.28% |
  | Swift | `shared_expert.down` | 3285/4096 | 1.42% |
  | AZ-Focus | routed `down_proj` | 2302/4096 | 0.43% |
  | AZ-Focus | GDN `out_proj` | 2913/4096 | 0.63% |
- **Intel W4A16** ([repo](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)):
  - It quantizes only the routed experts. The index has 73,728 `qweight` = 48 layers × 512 experts × 3 projections, stored per expert (`experts.N.down_proj.qweight`). Every other tensor is `.weight` in BF16 (`quant_idx.py`).
  - 139 of its BF16 tensors (layers 0, 3, 20, 30 and 47, embed, mixer, MTP attention) are byte-identical to Qwen's BF16 in the sampled windows (`intel_check.py`).
  - So overlaying the trunk on Intel gives exactly "variant trunk + Intel int4 experts".
- **Configs.** `chat_template.jinja` has the same blob (`c0c686f9c3`) in base, Intel, Tinfield, Swift, Swift-W4A16 and AZ-Focus. Tinfield's `config.json`, `generation_config.json` and tokenizer files are blob-identical to the base (HF siblings `blobId`).

### Inferences
- Swift touches exactly the modules of ms-swift's Flash-Next LoRA recipe (GDN in/out projections, QSA q/k/v/o) plus the shared expert. That is "all-linear except routed experts, router and HC", the scope W6 §3 recommends for our own LoRA, so Swift is a live example of the deployable shape.
- Tinfield's deltas are ~4-30× smaller than Swift's and also touch HC. That fits a low-LR or low-rank adapter (method undisclosed), so expect small behavioural shifts.

### Gaps
- Window sampling cannot rule out a change confined to rows that no window hit. Any dense update would be caught; a few-expert sparse edit might not.
- Agnes `gate_up_proj` was not checked (its layout differs).

## 2. Method, licence, claims and Kaggle/LB evidence per variant

### Takeaway
There is no public evidence that any of these variants beats stock Flash-Next on ARC-AGI-3:
- Swift users land at 26-35.6 on the leaderboard.
- The one Tinfield ARC probe was +5.6 pp on a small first-level test and counted by its author as NO_GO.
- An 8-minute public Kaggle A/B of Swift's own W4A16 in a Franzen-style harness used ~15% fewer tokens per action, decoded ~16% slower and cleared fewer levels.

### Cited Findings
- **Swift 1.5** (UkisAI, 2026-09-22; [card](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)):
  - **Method.** "figuring out which tokens were linked to pathological overthinking and penalizing them … then regained the accuracy with RL and OPD", with data "re-sampled, turned into proper RL environments" from `ukisai/Qwen3.8-27B-multi-turn-agent-sft`.
  - **Claims.** Mean thinking tokens fall 31-57% on GPQA, MMLU-Pro, AIME, LiveCodeBench and ERQA at accuracy −2 to +2 pts. On Terminal-Bench 2.1, mean output tokens rose 40,591 → 45,428 (+11.9%) while the score went 67.64 → 69.66.
  - **Licence.** Swift Open License v1.0 plus Qwen, with a US$1M gross-revenue commercial threshold (L2 §4).
  - **ukisai's own W4A16** ([repo](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound)) differs from Intel's:
    - AutoRound 0.16.0 at `iters 50`, against Intel's 0.15.0 (iters 200 per Intel's README command, W6 §3);
    - 73,884 `qweight`, i.e. 156 extra int4 linears = GDN qkv/z/out × 36 + QSA q/k/v/o × 12;
    - 82 shards, 177.2 GB.
    - phuongncn's Kaggle model is a "verbatim copy" of it ("v188m-swift source"; `kaggle models get phuongncn/arc3-qwen38-swift-w4a16-autoround`).
- **Swift in Kaggle runs:**
  - **felixwang1m** ran public commit A/Bs on the same Franzen-style harness. The Swift W4A16 run used `--model-path /kaggle/working/swift15_merged`, the 82 shards reassembled. Numbers are summed over 100 game runs (`felix_bench.py` on `benchmark.json`, downloaded with `kaggle kernels output … --file-pattern`; [Swift](https://www.kaggle.com/code/felixwang1m/arc-agi-3-swift15-w4a16-model-swap), [Franzen](https://www.kaggle.com/code/felixwang1m/arc-agi-3-m2-open-source-reproduction-franzen), [12-stream](https://www.kaggle.com/code/felixwang1m/arc-agi-3-throughput-12-streams)):

    | Run | Model | Minutes | Actions | Levels | Generated tokens | Tokens/action | Tokens/s | Levels/min |
    |---|---|---:|---:|---:|---:|---:|---:|---:|
    | Swift | Swift W4A16 | 7.98 | 315 | 7 | 231,217 | 734 | 483 | 0.88 |
    | Franzen repro | Intel W4A16 | 10.86 | 436 | 13 | 374,709 | 859 | 575 | 1.20 |
    | 12-stream | Intel W4A16 | 11.11 | 404 | 11 | 452,558 | 1,120 | 679 | 0.99 |

  - Felix's team "hierarchical" is at **26.00** after 26 submissions. Its last submission (2026-10-12 21:02:13) came 6 minutes after the Swift run ended (20:55:52). Which notebook produced 26.00 is UNVERIFIED.
  - Fususu's team: 35.62 after 196 submissions. rellik13: 29.28 after 55; his write-up says Swift was "No better per token and about 35% slower to decode" (`competitors/rellik13-milestone2/WRITEUP.md:148`).
  - Leaderboard source: scratch `lb/arc-prize-2026-arc-agi-3-publicleaderboard-2026-10-18T10:46:31.csv`.
- **Tinfield-1** (badtheorylabs, 2026-09-21; [card](https://huggingface.co/badtheorylabs/Tinfield-1)):
  - "agentic model for terminal work and long-horizon software engineering"; method not disclosed.
  - Self-reported, mini-swe-agent k=5: Terminal-Bench 4.0 33.0 vs 29.0 for the base, DeepSWE v1.1 62.0 vs 58.7.
  - Licence: Qwen Community License 1.0.
  - No HF discussions; web search found no independent write-up.
- **Tinfield NVFP4 probe** (Natapong Nitarach, LB 30.12 after 120 submissions; [HF](https://huggingface.co/natnitaract/tinfield-1-nvfp4-probe), [Kaggle](https://www.kaggle.com/models/natnitarach/tinfield-1-nvfp4-probe)):
  - "No additional SFT or RL training"; an NVFP4 export with BF16 non-experts on vLLM 0.30 / B200.
  - `evaluation-summary.json` reports a first-crack probe of 20/48 = 41.67% against a historical stock baseline of 26/72 = 36.11%: "+5.56 pp … NO_GO", and "runtimes differ; … does not isolate a causal model effect".
  - My two-proportion z-test gives ≈0.6 (p≈0.54).
- **AZ-Focus** (AxonZeta, 2026-10-09; Qwen licence):
  - "optimised the model's weights against [hesitation]".
  - BF16 card: −39% thinking, 109 vs 112 correct of 128, SWE-bench-Verified first 100: 98 vs 96. These were measured on EXL3 4.05 bpw, not BF16.
  - Its INT4 build is on a different 5-of-10-expert healed base (`azampatti/…-125B-A5B-INT4-AutoRound`), so it cannot be used with our stack ([INT4 card](https://huggingface.co/AxonZeta/Qwen3.8-Flash-Next-AZ-Focus-INT4)).
- **Signal 3.8 Flash Next** (agentionai; [card](https://huggingface.co/agentionai/Signal-3.8-Flash-Next-GGUF)):
  - Self-distillation on the model's own answers "under an instruction to be direct".
  - Claims: MTP draft acceptance 69% → **90%**; MATH-500 mean reasoning tokens −26%; Terminal-Bench 2.0 10/10 vs 8/10 on 10 tasks, one trial each; AIME at 32k 23/30 vs 25/30.
  - GGUF only (Q8_0 188 GB, AP-Q4_K_XL 101 GB, IQ4_XS 90 GB), so we could not diff it.
- **Tong Hui Kang** (LB 32.48 after 106 submissions):
  - Instance card: "LoRA checkpoint 09-29-21-15, merged into its pinned NVFP4 base" (RadixArk rev `7b719225…`); "ModelOpt NVFP4 expert weights and BF16 dense weights"; PLE borrowed at assembly time.
  - Listed as Apache 2.0, with the base licence file included (`kaggle models instances get huikang/qwen-38-flash-next-finetune/transformers/default`).
  - The file listing is not available, so the tensor scope is unknown.
- **REAP-k448** ([HF](https://huggingface.co/lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448); Kaggle mirror `boristown/…-reap-448e` from 暗黑AGI, LB 31.81 after 90 submissions):
  - 64 of 512 experts pruned per layer by REAP; "kept expert weights are bit-identical … router … rows subset"; −7.9 GiB in NVFP4.
  - The MTP keeps 512 experts and needs a vLLM `sitecustomize` patch. Licence: NVIDIA Open Model plus Qwen.
  - rellik13: pruning "a quarter of the model's experts … More room, worse play" (`WRITEUP.md:146`).
- **Others:**
  - **Victoria** ([card](https://huggingface.co/rmonsurate/Victoria)): experts 512 → 288, retrained at 4-bit (QAT), NVFP4/vLLM on B300/B200, 48 GiB of weights, self-reported Terminal-Bench 2.1 70.0% avg@3 under its own protocol.
  - **CYBER-FROST** (security fine-tune): all experts changed.
  - **Splice-QFN**: task arithmetic `swift + (orcarouter − base)`.
  - **Whittle-Next-27B-A3B** (HF `logic65`, Apache): its base is Qwen3.6-35B-A3B, so it is not a Flash-Next derivative. Its own card: long-context 66.7% at 26-75k vs the parent's 100 ([card](https://huggingface.co/logic65/Whittle-Next-27B-A3B)).

### Inferences
- Felix's 8-minute screen fits Swift doing what it was tuned for (fewer tokens per action) while losing on two things we care about.
  - **Decode speed.** It plausibly comes from ukisai's int4 GDN/QSA linears (INFERENCE).
  - **Levels per action.** 0.022 vs 0.030, on 7 vs 13 levels: noise-level.
  - The confound means the clean test is Swift's BF16 trunk on *our* Intel experts, not ukisai's W4A16.

### Gaps
- Which notebooks produced each team's best leaderboard score.
- Tong Hui Kang's `training_run.json`.
- Tinfield's and Agnes's methods and data.
- Whether Signal's MTP-acceptance gain holds under SGLang NEXTN.

## 3. Overlay onto the Intel W4A16 shards: feasibility and file-level recipe

### Takeaway
An overlay with no re-quantization works for **Swift 1.5 and Tinfield-1 only**. Each needs one small BF16 overlay file with the same names, dtypes and shapes as Intel's tensors. Their MTP is unchanged, so the albucino drafter (built from the base MTP) still matches.

### Cited Findings
- Overlay feasibility by variant:

  | Variant | Routed experts changed? | Overlay with no requant? | Overlay size |
  |---|---|---|---|
  | Swift 1.5 BF16 | no | **yes** | 5.82 GB, 300 tensors |
  | Tinfield-1 | no | **yes** | 6.64 GB, 518 tensors |
  | AZ-Focus | `down_proj` in layers 20-44 | no: needs int4 re-quant of 25 × 512 `down_proj` | 41.9 GB BF16 to re-quant |
  | Agnes-3.0 | `down_proj` | no | n/a |
  | cyberneurova / CYBER-FROST | yes | no | n/a |
  | huikang LoRA (NVFP4) | unknown | only if its BF16 dense deltas can be pulled out against RadixArk's dense tensors | n/a |
  | REAP-k448 | experts removed, kept ones bit-exact | possible as an expert *subset* of Intel (drop 64/layer, subset router rows); the mask is not published, but it can be recovered by matching `mlp.gate.weight` rows (2.6 MB/layer) | n/a |
  | Signal | unknown (GGUF) | no clean path (Q8_0 rounding error vs small deltas) | n/a |
  | Victoria | yes (288 experts, QAT) | no | n/a |

  Sources: §1 tables and the cards cited in §2.
- Our Kaggle model `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` is "Resharded … original weights are unchanged" (W6 §3). Its shard layout differs from Intel's 17 HF shards, so an overlay must be keyed by tensor name, not by file.

### Inferences (recipe; nothing here was run)
1. **Build the overlay with no GPU quota.** Use a CPU Kaggle notebook (internet on) or the laptop.
   - Parse the variant's headers (already in scratch `samples/headers.json`) and range-download only the changed tensors' bytes from HF.
   - Write `overlay-<variant>.safetensors` (≤6.7 GB), plus a manifest of per-tensor sha256 and source revision. Pin `f7c714a97b…` (Tinfield) and `0bd4fe2243…` (Swift).
2. **Apply at start-up, so no SGLang change is needed.**
   - Build `/kaggle/tmp/model-<variant>/` from symlinks to every file of our Kaggle model, plus *patched copies* of only the shards that hold overlaid tensors.
   - Patch each copy in place: dtypes and shapes match, so the header and byte offsets don't change.
   - Leave config, tokenizer, chat template and PLE untouched, and point `--model-path` at the new directory.
   - If the trunk is spread over many large shards, the copy cost grows. The fallback is a `runtime_patches/` loader hook that swaps tensors by name. Don't rely on "last duplicate wins" loading (UNVERIFIED in Pennyroyal).
3. **Verify before playing.**
   - After load, compare a few overlaid parameters through SGLang's `get_weights_by_name` against the manifest, and check that a base tensor is unchanged.
   - Then compare greedy output on a fixed prompt against base: it must differ. Feeding the base its own reply must give identical logprobs.
4. This is exactly the deploy step our own GDN/QSA/shared-expert LoRA needs (W6 §4a), so building it now de-risks FT-1.

### Gaps
- How our resharded Kaggle copy places the BF16 trunk tensors, which sets how big the patched copies are. To find out, run `kaggle models instances versions files`, or read the index in a CPU notebook.

## 4. What to smoke-test, and what result would show that post-training raises levels per wall-clock

### Takeaway
Smoke-test the **Swift trunk overlay first**, then the **Tinfield overlay**, each as a default-off flag on the leader stack (`mem-repair-10x`), with only the weights changed. Swift's lever, fewer tokens per action, is the one our compute-bound harness can turn into levels. Tinfield's tiny deltas give it a low prior. AZ-Focus comes third, and only if the first two show a token effect. Everything else is out.

### Cited Findings
- Noise: an offline run has sd ≈3.15-3.2 and a leaderboard draw sd ≈3.5 (`_critic.md` corrections). So one full run cannot resolve changes below ~6 points; mechanism metrics have to decide the screens.
- `arms.md:674` already budgets a 20-minute Swift smoke. Felix's data (§2) show the confounded version.

### Inferences (proposed gates; thresholds are mine and are calibrated on public games, so the PR must name them)
- **Stage 0 (done here):** weight-level scope verified. **Stage 1:** overlay built and verified (§3, steps 1-3).
- **Stage 2: one 20-minute panel per variant, paired with a base panel on the same notebook build.**
  - Measure:
    - decode tokens/s from `serve_report`;
    - MTP accept length;
    - generated tokens per action;
    - tool-call or parse error rate;
    - actions per minute;
    - levels per action;
    - the levels-vs-wall-clock curve.
  - **Kill** if any of these holds: decode tokens/s below 0.95× base, accept length below 0.9× base, or parse errors up more than 2 pp.
  - **Promote** if tokens per action ≤0.85× base and levels per action ≥0.95× base, i.e. levels per generated token up by ≥10%.
- **Stage 3: one full 121-minute run, stacked on the leader** (never a repeat).
  - Signal of an alpha: levels cleared at matched wall-clock (20/60/121 min) above the leader's run distribution by more than 1 sd, with a favourable paired per-game `make significance`.
  - Then the user's repeated-run rule.
- **How to read the results.**
  - A tokens-per-action cut with unchanged levels per wall-clock means token savings alone don't buy levels here: wall-clock goes to prefill, the environment or slots. That argues against R3-style "think less" training.
  - A drop in levels per action means shortening hurts exploration, which is Signal's own stated trade-off on search-heavy problems. Any R3 would then have to be on-distribution, using shortest-*correct* harness decisions.
  - A Tinfield null says little about our own training, because its update is tiny and off-distribution.

## Implications for our plan
1. **Cheapest real evidence on "does post-training Flash-Next help in our harness": the Swift-1.5 BF16 trunk overlay on our Intel experts.** It costs 5.82 GB, has no requant and keeps the drafter valid. Run it instead of the `phuongncn` W4A16 copy, which confounds the test with a different AutoRound run (iters 50 vs 200) and int4 attention. The Tinfield overlay (6.64 GB) is the second arm. GPU cost: two 20-minute panels plus a base panel, then at most one full run.
2. **Build the overlay tooling now.** Range-pull the tensors, patch shard copies by name, and verify with `get_weights_by_name`. It is FT-1's deploy path, and W6's "LoRA on BF16 modules, no requant" now has two working real examples (Swift, Tinfield) that touch exactly those modules.
3. **Scope our own LoRA like Swift:** GDN in/out, QSA q/k/v/o and the shared expert, without routed experts, router or the QSA indexer. Then deployment stays a file overlay. Training routed experts (as AZ-Focus and Agnes did) forces an int4 re-quant of `down_proj`.
4. **Lower expectations.** No public variant has a positive ARC-AGI-3 result:
   - Swift users score 26-35.6 on the leaderboard;
   - the Tinfield probe was +5.6 pp, NO_GO;
   - in Felix's A/B, Swift W4A16 used −15% tokens per action but decoded −16% slower, with 7 vs 13 levels.

   General agentic or "think less" post-training is not shown to transfer; on-distribution training (R3/R4) remains the bet.
5. **Ideas to add to `arms.md`:**
   - Signal's self-distillation "be direct" recipe claims +21 pp MTP acceptance, a throughput lever that fits R1/R3.
   - The REAP-448 expert *subset* applied to Intel W4A16 (no requant; mask recoverable from the router rows) is a KV-capacity arm, not a post-training one. rellik13's negative pruning result ranks it low.
6. **Licence check before any final pick.** Swift's licence adds a revenue threshold and redistribution terms on top of Qwen's. Whether that is compatible with the ARC Prize open-sourcing rules is UNVERIFIED. Tinfield and AZ-Focus add no terms beyond the base's Qwen Community License.

## Fact-check (independent verifier, 2026-10-18)

I re-checked all 11 key results and skipped none. I fetched my own HF API data (`?blobs=true`, indexes, safetensors headers) pinned to the same revisions, re-downloaded Felix's kernel outputs and the leaderboard, and wrote new scripts. My sampler hashed windows at 33% and 75% of each tensor, and 6 windows for each routed `down_proj`, which are different offsets from the original sampler's. Scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/gdv-scripts/` and data in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/gdv/`.

| # | Result | Verdict | Correction or note |
|---|---|---|---|
| 1 | Shard sha256 (80/131, 103, 31, 0) | confirmed | Exactly reproduced (`shards.py`). Minor slip in §1: the 48 `gate_up_proj` shards are 38 × 3,355,443,376 B plus 10 × 3,355,443,368 B, not all 3,355,443,368 B. |
| 2 | Tinfield: 518 tensors, 6.64 GB | confirmed | Same families, count and size at different windows (`sample.py`, `analyze.py`). All 518 exist in Intel's index as BF16 with the same shape (`overlay_names.py`). Routed `down_proj` (6 windows × 48 layers), router, shared experts, embed, PLE and MTP are unchanged. |
| 3 | Swift: 300 tensors, 5.82 GB | confirmed | Same as 2. All 300 exist in Intel's index as BF16 with the same shape. |
| 4 | AZ-Focus layers 20-44, 75 tensors, 42.81 GB; Agnes routed `down_proj` in layers 30/47 (+ some of layer 0) | partially_confirmed | AZ-Focus is exactly reproduced. Agnes is **understated**. With 7 windows per tensor (`agnes.py`): routed `down_proj` changed in every window in layers 0, 10, 20, 30, 40 and 47, layer 0 included. Routed `gate_up_proj` also changed in all six layers, which fills the §1 gap. QSA `k_proj` (layer 3) changed as well. Router, embed and `lm_head` are unchanged. Dtype and shape are the same, and the deltas are real: layer-0 `down_proj` changed 3,914 of 4,096 values (3.9% relative RMS). The "no overlay" conclusion stands and is stronger. |
| 5 | Layer-30 delta sizes | partially_confirmed | All six numbers reproduce exactly at the mid-tensor window (`mag.py`), but a single window is noisy. At the 20% and 80% offsets: Tinfield `out_proj` 0.030-0.045%, HC `input_mix_up` 0.04-0.12%; Swift `out_proj` 0.23-0.33%, `shared_expert.down` 1.4-1.9%; AZ routed `down_proj` 0.36-1.10%, AZ `out_proj` 0.41-2.03%. Comparing the same tensor (`out_proj`), Tinfield is ~5-10× smaller than Swift. The "4-30×" in §1 Inferences compares different tensors. |
| 6 | Intel quantizes only routed experts; Swift W4A16 config | confirmed | 73,728 `qweight` (48 × 512 × 3, per expert). Swift W4A16: 73,884 (+156 = 36 × 3 + 12 × 4), AutoRound 0.16.0, `iters` 50, 82 shards, 177,237,111,728 B. My own check of 122 other non-routed tensors (layers 5, 11, 26, 38 and 44, `lm_head`, MTP fc/shared/gate/HC), 2 windows each (`intel.py`): all BF16 and byte-identical to Qwen. Minor: Intel also keeps the MTP's 512 experts unquantized, as per-expert BF16 `.weight`. "Every other tensor is `.weight`" leaves out vision biases, `A_log` and `dt_bias`. |
| 7 | Felix A/B numbers | partially_confirmed | Every number reproduces from a fresh `benchmark.json` (`--model-path /kaggle/working/swift15_merged` vs the dfranzen Intel model). Four caveats. (a) In each run only **10 of the 100** game runs took any action; all 100 are `cancelled` and 90 have an empty history, so this is a 10-game screen. (b) The two runs differ in wall-clock length (7.98 vs 10.86 min). (c) **483 vs 575 tok/s is tokens from completed actions per wall-clock second, not decode speed.** Inside the benchmark window, SGLang's `serve.log` shows Swift decoding at a mean of 745.8 tok/s (median 786.5) vs Intel's 693.7 (757.4). Accept length is 2.72 vs 2.63, with 9.5 vs 9.3 running requests (`tput.py`). So Swift decoded ~4-8% *faster*. The "decoded ~16% slower" in §2 and Implication 4, and the inference blaming ukisai's int4 GDN/QSA linears, are refuted by these logs. (d) The `solver_note` totals (280,755 vs 405,353 tokens) give 891 vs 930 tokens per action (−4%). The −15% holds only if the generations still running at the cut-off are left out. |
| 8 | Tinfield NVFP4 probe | confirmed | `evaluation-summary.json` reads 20/48 vs 26/72, +5.56 pp, `NO_GO`. The gate required +10 pp. The probe ran with MTP disabled and PLE offloaded to CPU. z = 0.61, p = 0.54 (Fisher p = 0.57). Episodes sit in 4 vs 6 arms, so the z-test is if anything optimistic. |
| 9 | Leaderboard rows | confirmed | Same in the 10:46 snapshot and in a fresh 11:01 UTC download. The only change is that hierarchical moved from rank 1366 to 1367. Felix's last submission came 6 min 21 s after the benchmark's `end_time`. Caveat: these teams used a variant at some point. That any of their scores came from a variant is UNVERIFIED, so "Swift users land at 26-35.6" is not evidence about Swift. |
| 10 | REAP-k448: subset possible with no requant; rellik13 quote | partially_confirmed | The card (rev `8d565c90ad`) says: kept experts "bit-identical to the source (only renumbered 0..447)", router rows "bit-exact rows of the original", MTP's 512-expert bank "preserved verbatim". No mask file is published. I recovered layer 30's mask (`reap_mask.py`): all 448 REAP router rows (BF16 [448, 2560]) match base BF16 rows bit-exactly, distinct and in order. Nuances. The "source" is `nvidia/Qwen3.8-Flash-Next-NVFP4`, so the bit-identity is to NVIDIA's NVFP4 experts. Dropping the same 64 experts from Intel's per-expert int4 tensors still needs no requant. SGLang would also need a 448-expert main model with a 512-expert MTP, and the card's patch is for vLLM only. The rellik13 quote is at `WRITEUP.md:146`, but he pruned "a quarter" of the experts (25%), twice k448's 12.5%, so it is weak evidence about k448. |
| 11 | Chat template and Tinfield configs | confirmed | `chat_template.jinja` blob `c0c686f9c3` appears in all six repos. Tinfield's `config.json`, `generation_config.json`, `tokenizer.json` and `tokenizer_config.json` are blob-identical to the base. Side note: Swift BF16's `tokenizer.json` and `tokenizer_config.json` *differ* from the base's, but its `tokenizer.json` is byte-identical to Intel's (same vocab and added tokens, re-serialised), and its `tokenizer_config.json` differs from Intel's only in `processor_class`. Harmless for the overlay. |

Other errors in the note:
- "Decoded ~16% slower" / "−16% slower" (§2 Takeaway, §2 Inferences, Implication 4) is contradicted by Felix's own `serve.log` (row 7). The levels-per-wall-clock gap is not a decode-speed gap.
- "Numbers are summed over 100 game runs" (§2): only 10 runs per notebook took any action.
- §1 Gaps, "Agnes `gate_up_proj` was not checked": it has now been checked, and it changed. Agnes also changes QSA `k_proj`.
- §1 Inferences, "~4-30× smaller": the range depends on the window and compares different tensors (row 5).
- Implication 5 and §2: rellik13's pruning result was for 25% of experts, not REAP-k448's 12.5%.
