# throughput-and-alt-models: plan

Workflow run `wf_f7f94969-5fa`, phase Plan. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

PLAN: alt-model serving and throughput (2026-10-07)

Nothing here ran on a GPU. Today I checked these locally:
- Branch ancestry, and test merges and cherry-picks with `git merge-tree`.
- The wheelhouse contents, with `kaggle datasets files dfranzen/pennyroyal-v253`: it contains `wheels/humming_kernels-0.1.12` and `wheels/gguf-0.19.0`.
- `arm/nvfp4` contains `arm/stack-v1`.

All other figures come from the research reports.

**(A) Verdict per candidate**

**1. GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit: not feasible, and not worth building.**
- **It doesn't fit.**
  - The checkpoint is 175.6 GB; the card has 96 GB.
  - Every format SGLang can load is 181.5 GB or more.
  - Only two builds fit, and neither works well:
    - A 62 GB NVFP4 prune that keeps 80 of 288 experts. Its output is "repetition-heavy" and it needs 2 NaN fixes to boot.
    - A 2.05 bpw EXL3 build, which runs only on ExLlamaV3 and has no 10-stream speed data.
- **Our engine can't run it.**
  - Pennyroyal 2.5.3 has no GLM-5.3-Flash model code.
  - SGLang has no EXL3 loader.
  - Upstream SGLang on sm120 is blocked: #37105 (DSA backends), #42012 (Triton prefill is about 140 tok/s) and #36830 (FP8 KV cache).
- **It would cost throughput, not add it.**
  - It has 18B active parameters, 3 times our 6B.
  - The best sm120 build needs 2 GPUs and gets 381.5 tok/s at 4 streams. We get about 444 tok/s at 4 streams and 705–789 at 10, on 1 GPU.
  - The Ablit weights are tuned for thinking off.
  - There is no ARC-AGI-3 data for it.
- **Cheapest path if forced:**
  - A new upstream SGLang wheelhouse, built in an internet-enabled GPU notebook.
  - Plus the 62 GB prune.
  - Plus porting the launcher and harness to GLM's parsers and template.
  - 1–2 weeks, with high quality risk.
- **Action:** add an arms.md entry under "Not pursuing" with these numbers. No profile.

**2. Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF: not servable as published. The friend's work is usable only as an export SGLang can load.**
- **Pennyroyal's GGUF loader has 5 blockers for this model:**
  - no `qwen4exp` name map;
  - single-file loading only;
  - Q2_0 is unsupported;
  - the IQ types run on the slow per-token MoE kernel;
  - the converter reorders linear-attention heads, and that has to be undone.
- **llama.cpp is rejected** for its multi-slot, MTP and tool-call bugs (#29166, #28019, #28286, #30078) and roughly 5× slower prefill.
- **The file has no MTP tensors.**
- **"Coder" means 256 of 512 experts were pruned**, plus IQ quantization at about 3.33 bpw on the experts. It keeps 91.3% of BF16's SWE-bench score.
- **Path B (recommended; the only path that keeps GSQ quality).** The friend re-runs GSQ with uniform symmetric INT weights, in a format Pennyroyal already loads:
  - **B4: INT4 at group size 128**, in AutoRound or compressed-tensors format.
    - Runs on our current Marlin kernels with no engine change.
    - About the same speed as now, so it's a quality-only test.
  - **B3: INT3, or RCO's mixed INT3/INT4**, through Humming.
    - The Humming wheel is in the wheelhouse.
    - About 25% fewer expert bytes. The decode gain is unmeasured; I'd guess 0–15%.
    - Humming's speed against Marlin on sm120 is also unmeasured.
  - Our side is 1–2 days, and all of it can be built now. No internet notebook is needed: SGLang and Humming are already in pennyroyal-v253.
- **What to ask the friend for:**
  - **The unpruned model, with no Coder mask.** arms.md already lists pruning Flash-Next experts under "Not pursuing". Cutting to 8 experts per token gave no speed-up, and 16 experts per token cost 7.6 points.
  - **Only the experts quantized.** Attention, shared experts, dense layers, PLE and router stay BF16, as in Intel's W4A16.
  - **Tokenizer and chat_template byte-identical to the base.** That keeps the FR-Spec SHA gate (06b95093) and `preserve_thinking` working.
  - **A `quantization_config`** stating bits, group_size and sym.
  - **Safetensors shards plus an index.**
  - **Weights no larger than the Intel W4A16**, so the KV and mamba sizing stays the same.
- **Path A (apply the Coder mask to our W4A16 by setting the dropped experts' router logits to -inf; 0.5–1 day): deprioritized.**
  - It contradicts the "Not pursuing" entry.
  - Its only upside is measuring how much the prune costs in quality.
  - If built: a `mask` variant in arm/moe-routing's `serving/runtime_patches/20-moe-routing.py`, on branch `arm/coder-mask` off `arm/moe-routing`. Run it last.
- **Bottom line:** neither candidate adds throughput this week. B3 is the only alt-model path with throughput upside, and it depends on the friend's export plus one bench.

**(B) Serving refactor to build now: `arm/model-profiles`**

**Base:** `arm/fix-6-cold-start`, which is custom-serving plus the JIT cache plus the launcher stub tests, merged with `arm/serving-bench` (test merge: clean).

**Why not `arm/nvfp4`:** it carries all of stack-v1: D′, ledger-on-trim and the #2 fix, a run that scored 43.32 and lost. stack-v2 leaves D′ and the #2 fix out on purpose.
- Bring NVFP4 over by cherry-picking `fee2a6c`, its launcher commit (test: clean on fix-6).
- Re-implement `30c7ddf`'s `"model"` key instead of cherry-picking it, because it conflicts with fix-6 in `kaggle/build_notebook.py`.

The branch touches only `serving/` and `kaggle/`, so it merges into any harness stack.

**Commits:**
0. **Record the goldens.** Write `kaggle/tests/golden/launch_<name>.json` with the server argv, server env and runtime-patch list, captured from the current serving cell:
   - for every arm in arms.json;
   - for bench configs C0–C7 (C4 is disabled: its FP8 port isn't written yet);
   - for arm/nvfp4's NVFP4 launch.

   Use the stubs from `test_launcher_jit_cache.py`: a fake `MODEL_DIR` with config, tokenizer and index, and Popen and urllib stubbed.
1. **Refactor only.**
   - Move the builder's `_replace_once` edits into `launch_sglang.py`: `DRAFT_KV_OLD`; `ATTN_ARGS_ANCHOR`, which becomes the CFG keys `TEXT_CONFIG_OVERRIDE` and `SPEC_ADAPTIVE`; `SERVER_DIED_OLD`; `SERVE_ENV_ANCHOR`; `SERVE_ANCHOR`; and the wait loop.
   - Split the launcher into `install()`, `prepare_views()`, `build_args()`, `build_env()` and `launch()`.
   - **Gate:** every golden is identical.
2. **Profile loader.** `serving/profiles/qwen38fn-w4a16.json` holds every value that is hard-coded per model today:
   - `inputs`: model, draft, wheelhouse.
   - `validate`: auto-round, bits 4.
   - `args`: quantization; reasoning parser `qwen3`; tool parser `qwen3_coder`; chat kwargs `{preserve_thinking: true}`; JSON model override; chat template.
   - `hybrid`: the `--mamba-*`, `--linear-attn-*`, PLE and `--gdn-mtp` flags; null for a full-attention model.
   - `spec`: NEXTN, 3 steps, draft view (albucino, native or none), draft quantization.
   - `frspec`: the map SHA and tokenizer SHA, or null.
   - `cfg` defaults, `serve_env`, `harness_env` (empty) and `startup_timeout_s` 720.

   The launcher keeps the W4A16 profile inline, so it can still be pasted into a cell. The builder injects the chosen profile as one JSON line. CFG = the profile's `cfg` with the arm's `"serve"` keys on top. **Gate:** goldens identical.
3. **NVFP4 profile.** `qwen38fn-nvfp4.json`: `modelopt_fp4`, the native NEXTN head with draft quantization `unquant`, tokenizer SHA 0997f410, and the RadixArk rope override.
   - An arms.json `"profile"` key replaces `"model"`.
   - **Gate:** matches the nvfp4 golden.
4. **Builder generalization.**
   - Inputs and metadata come from the profile.
   - The drafter is optional: `_resolve_input`, `precache` and the drafter's metadata source are dropped when `inputs.draft` is null.
   - The wheelhouse line can be swapped, and `extra_datasets` is supported.
   - Add `--profile` for bench builds.
   - Assert that the harness context window plus 8Ki equals CTX. gate11's 11 streams over MAXREQ 10 stays legal.
5. **GSQ profile slot.** `qwen38fn-gsq.json`:
   - `quant_method` is one of auto-round, compressed-tensors or gptq, with 3 or 4 bits.
   - INT4 uses the Marlin path (auto-round or compressed-tensors); INT3 uses `--quantization humming`. Whether Humming works this way with our drafter is unverified; bench C8 checks it.
   - The albucino drafter.
   - FR-Spec only on a SHA match, through the existing gate.
   - The source is "TBD", and the builder refuses to push until it is set.
6. **Checkpoint checker.** `serving/tools/check_checkpoint.py` runs on the CPU, on the laptop or the friend's machine.
   - It reads config.json, `quantization_config`, the safetensors headers (no tensor data) and the tokenizer/template SHAs compared with the base.
   - It picks the matching profile.
   - It estimates weights + FP8 KV (CTX × MAXREQ) + mamba cache (MAMBA_CACHE slots) against the 0.96 × 96 GB pool.
7. **Bench C8.** Humming on our current W4A16 (`args_set --quantization humming`) compared with C0. This prices Humming against Marlin on sm120 before anyone bets on INT3.
8. **Docs.**
   - `serving/CUSTOM.md` §6: base-model changes now go through profiles. It also lists what a non-Qwen model would still need: an engine switch, a tool-markup regex, the vision patch size, and the token scale.
   - `kaggle/README.md`: the `profile` key, `--profile` and `check_checkpoint`.
   - The overview in `README.md`.
   - `CONFIGURATION.md` is unchanged, because no harness flag is added.

**Tests** (run under `nice`, one at a time):
- `test_launch_golden.py`: every arm and bench config renders identically.
- `test_profiles.py`:
  - W4A16 renders auto-round, the draft path and the token map.
  - NVFP4 renders modelopt_fp4, `unquant` and the override, with no draft path.
  - GSQ INT4 renders auto-round; GSQ INT3 renders humming.
  - `hybrid=null` drops every hybrid flag.
  - SPEC off works.
  - An FR-Spec SHA mismatch raises.
- `test_build_profiles.py`: every arm builds, the metadata sources match the profile, and the drafter is dropped when it should be.
- `test_check_checkpoint.py`: runs on synthetic headers.

**Not building:**
- a GLM profile;
- a llama.cpp or ExLlamaV3 engine switch;
- `ARC3_TOOL_MARKUP` or `ARC3_VISION_PATCH_PIXELS`, which no feasible profile needs;
- any internet build or convert notebook.

PR: "Serve the model from per-model profiles". Log it in arms.md under In progress.

**(C) Throughput stack to build now: `arm/throughput-stack` off `arm/stack-v2`**

**Merges, in order (all test-merged):**
1. `arm/serving-bench`: clean. Brings the editable launcher, `runtime_patches/` and the bench.
2. `arm/fix-6-cold-start`: conflicts only in arms.json.
3. `arm/gate11`: arms.json only. Flag `ARC3_MAX_ACTIVE_STREAMS=11`.
4. `arm/stub-dead-reasoning`: arms.json only. Flag `ARC3_STUB_DEAD_REASONING=1`.
5. `arm/fix-3-prefix-reuse`: one real conflict, in `tool_agent.py` near line 7082. Keep both the ledger lines and `reply_message = assistant_message`.
   - Also apply the one-block fix the research report flagged: the replayed history keeps blank lines.
   - Flags: `ARC3_VERBATIM_HISTORY=1`, plus `ARC3_PREFIX_REUSE_DIAG=1` in the first run only.
   - Keep `MAMBA_CACHE` at 60. 84 costs about 1.3 GB of KV.

**Own commits:**
6. A CFG key `CUDAGRAPH_BS_EXTRA`, default `[]`, so control's argv doesn't change. The arm sets `[3,5,6]`.
7. An arms.json `"runtime_patches"` key, reusing the bench's mechanism, so an arm can pick `serving/bench_patches/m64.py` and `nosplitk.py`.
8. Tests: the full suite, a flags-off identity check against stack-v2, and the launch goldens.
9. Later, merge `arm/model-profiles`. It shares the fix-6/serving-bench base, so the 30c7ddf conflict doesn't come back.

**Arms:**
- **`tput-safe`:**
  - env: `ARC3_LEDGER_ON_TRIM=1`, `ARC3_MAX_ACTIVE_STREAMS=11`, `ARC3_STUB_DEAD_REASONING=1`, `ARC3_VERBATIM_HISTORY=1`, `ARC3_PREFIX_REUSE_DIAG=1`.
  - serve: `CUDAGRAPH_BS_EXTRA=[3,5,6]`, and `JIT_CACHE_DIR` once the cold-start dataset exists.
- **`tput-safe-lowm`:** `tput-safe` plus the m64 and nosplitk runtime patches. Only if bench C3 gains at least 1% at batch 10 and loses nothing at batch 8 or below.
- **`tput-accept05`:** `tput-safe` plus `SPEC_ACCEPT_SINGLE=0.5` and `SPEC_ACCEPT_ACC=0.5`. Only if bench C1 gains at least 3%. It changes outputs at temperature 0.7, so it needs its own score comparison.
- **Baseline:** `stack-v2`, on the same branch. It has never been run.

**sim-v2 estimate.** This is an upper bound: no real run has confirmed the simulator's throughput-to-score slope.

| Arm | Throughput | Offline (25 games) | Hidden set (110 games) |
|---|---|---|---|
| `tput-safe` | ×1.06 (×1.03 to ×1.09) | +1.7 points (0.8–2.5), +0.5 games (+0.25 counting only games some real run has won) | +1.4 games, +0.1 more from the JIT cache |
| `tput-accept05` | ×1.12 | +3.2 points, +0.9 games | about +2 to +3 games |

**"+2 games":**
- **On the hidden 110:** needs ×1.09–1.10, which is reachable per the simulator.
- **On the offline 25:** needs ×1.27. That means acceptance 0.3/0.3 plus the FP8 dense-weight port, which isn't written. Counting only games some real run has won, it isn't reachable even then.

Offline gaps under 3 points are noise. So judge first by the serve report's tok/s, and judge score only over at least 2 paired runs with `make significance`.

**(D) Kaggle run order when quota returns, cheapest first**
0. **Now, without quota:** build and test everything above locally. The friend runs `check_checkpoint` on his export.
1. **Bench C0, C3, C1 and C8** (no games; about 25 min each, mostly the 10-min startup). Answers:
   - whether low-M GEMM, acceptance 0.5 and Humming are faster at batch 10;
   - the accept length.

   These gate `tput-safe-lowm`, `tput-accept05` and GSQ INT3.
2. **`coldstart-save`** (5-min demo, about 20 min). Publishes the JIT cache and prices the 145 s startup saving, which only counts in the competition run.
3. **`tput-safe` as a 20-min panel** (about 35 min). Answers whether the merged stack runs clean, and how much prefix is reused. A panel runs 4 streams, so it doesn't test gate11.
4. **`stack-v2` and `tput-safe`, full 121-min runs** (about 2.3 h each). Answers the throughput gain (tok/s, accept length, prefill stalls) and gives the first score pair.
5. **`tput-accept05`, full run**, only if step 1's gate passes. Answers whether lossy acceptance costs score.
6. **Repeat step 4**, then run `make significance` on the pairs.
7. **The GSQ profile**, once the export exists:
   - Bench C0 on `qwen38fn-gsq`: load time, VRAM, tok/s, and the albucino drafter's accept length on the new model.
   - Then a 20-min panel.
   - Then a full run against the best of steps 4–6.
8. **Optional backlog, last:**
   - `nvfp4-stack` on profiles. Its native draft head measured −22.9% at 4 streams.
   - `coder-mask`.

**Key paths:**
- Worktrees: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/{fix-6,serving-bench,nvfp4,stack-v2,gate11,stub-dead,fix-3,moe-routing}`
- Coder expert mask: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/altmodel-research/gsq/coder_expert_mask.json`
- Wheelhouse file listing: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/pennyroyal_files.txt`
