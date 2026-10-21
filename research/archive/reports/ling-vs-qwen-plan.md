# Ling-3.0-flash vs Qwen3.8-Flash-Next: test plan

2026-10-10. The user asked: "this model seems to be a good contender to test out as well: informally, do parallel tests on equivalent simple runners on both qwen 3.8 flash next and https://huggingface.co/inclusionAI/Ling-3.0-flash-int4 — tell me what results in an higher score."

This is a plan, not a result. Nothing below has been run. Figures marked ESTIMATE are derived, not measured.

## Verdict: run the OpenRouter path first

Run **(B) OpenRouter** first, then the Kaggle path only if Ling wins clearly there.

- **It answers the user's question cheaply.** The question is which model scores higher in our harness. A text-only 10-game panel pair costs about $2–6 at a 60-request cap per game, or $10–19 at a high-end 100–200 requests per game (no cache). It needs no GPU quota and can start today, once the runner is fixed (stage B0).
- **The Kaggle path is expensive and risky before it plays a single game.** The fork named on the model card (`inclusionAI/sglang_ling_v3`) is gone (404). The 77 GB checkpoint has to be staged through about five CPU notebooks. The engine has never run on sm120, and our last custom engine (GSQ) died at self-test with an illegal memory access after 13 GPU minutes. Our estimate is that Ling decodes at about 55–70% of our Qwen engine's speed (ESTIMATE), and Kaggle scores by wall-clock, so Ling has to win per decision just to break even.
- **OpenRouter tells us whether the Kaggle path is worth paying for.** If Ling does not beat Qwen at an equal number of requests, it cannot beat Qwen on Kaggle while decoding more slowly. If it does beat Qwen, the Kaggle path becomes a serving problem worth solving.

One caveat bounds everything below. **OpenRouter compares the models, not our serving.** Both models run on someone else's stack, at unknown or full precision. There is no W4A16, no MTP drafter, no FR-Spec, no prefix cache under our control, and no wall-clock pressure. An OpenRouter win is a reason to try Kaggle, not a Kaggle result.

### Facts checked for this plan (2026-10-10)

- **`qwen/qwen3.8-flash` is Flash-Next.** OpenRouter's model record gives `hugging_face_id: Qwen/Qwen3.8-Flash-Next`, and that HF repo exists (created 2026-08-24, `image-text-to-text`). There is no separate `flash-next` slug.
  - The only provider is Alibaba. Quantization is listed as `unknown`, with a 1M context and image and video input.
  - So it is the same base model by OpenRouter's mapping, served differently: probably not 4-bit, and with none of our serving stack. Whether Alibaba's hosted "Flash" is byte-identical to the open weights is not verifiable.
  - `qwen/qwen3.8-omni-flash` is a different model at the same price; don't use it.
- **`inclusionai/ling-3.0-flash` (text) has two providers:**
  - Novita: quantization `unknown`, 262K context, $0.021 in / $0.063 out per M tokens, $0.0042 per M for cache reads.
  - DeepInfra: BF16, 131K context, $0.06 / $0.18.
  - Both cap output at 32,768 tokens.
- **Other listings:**
  - `inclusionai/ling-3.0-flash-vl` (image input) is $0.021 / $0.0616 on Novita.
  - `inclusionai/ling-3.1-flash` is a different model (560B total / 25B active). It is too big for one RTX PRO 6000; ignore it.
- **Neither model has a `:free` variant.** The free key's 50-requests-per-day cap applies only to `:free` models, so it neither limits nor helps here. Paid calls need OpenRouter credits.
- **Ling's chat template reads `reasoning_content` and `enable_thinking`.** Our template default `ARC3_REASONING_HISTORY_KEY=reasoning_content` and the `enable_thinking=False` sent for summary requests therefore work unchanged on a Ling SGLang server. `preserve_thinking` is ignored, because Ling hard-codes preserved thinking, which matches our Qwen setting. User turns are joined as plain strings. With `MULTIMODAL_CONTEXT=none`, our user messages are plain strings (`tool_agent.py:5054`), so the template should render them.
- **The INT4 text checkpoint ships its own MTP head** (layer 42, enabled with `--speculative-algorithm NEXTN`). The "no MTP drafter" risk applies only to the VL model.
- **Our measured reference panel.** `v4-panel-uc10x` v1 is Qwen with images on the balanced 10 games, 25 min, 4 streams. It scored 9.28 (15 levels) and sent 349 requests (10–56 per game, median about 38). Totals were 19.2M prompt tokens and 0.62M completion tokens, with a median prompt of about 54K tokens and a median completion of 0.4–1.4K. The prefix-cache hit rate was 86–97% per game. The notebook took 36 min end to end, including 10.8 min to reach the first game.

## (B) OpenRouter path: informal, quick

Target: the same harness and games on both models, text-only, with an equal request budget per game.

### B0. Runner fixes (laptop, no spend, about 1–2 h)

- **Rebase the testbed onto current main.**
  - `tooling/openrouter` (PR #143) is 183 commits behind `main` and lacks `ARC3_LEDGER_ON_TRIM`, `ARC3_ANIM_TRUTH`/`ANIM_DIGEST` and `ARC3_OPENER_DEDUP`, all of which the panel-uc10x environment uses.
  - Merge main into it, or cherry-pick its 8 commits onto a branch off main, so the harness matches the Kaggle panel's code.
- **Let paid models through the quota gate.** `quota_allows_launch` checks the free-model counter before every launch. With a 60-request cap and a 50-per-day limit, `remaining < needed` is always true, so the runner would sleep until the UTC reset every day and never launch.
  - Fix: skip the gate when the model ID has no `:free` suffix, or add a `--paid` flag.
  - Add a dollar guard instead. Read `usage.cost` from each response, or poll the credit balance through `/api/v1/key`, and stop at `--max-dollars`.
- **Add a config `configs/inference.openrouter-panel.json`** that is identical for both models:
  - `context_window` 131072, so 128K harness plus output fits both DeepInfra's 131K and Novita's 262K;
  - `max_output` 12288;
  - temperature 0.65, top_p 0.95, top_k 20 (matching Ling's card and our template);
  - thinking on;
  - `multimodal.context` "none".
- **Shared env for both runs.** This is the panel-uc10x set plus text-only:
  - `MULTIMODAL_CONTEXT=none`
  - `ARC3_LEDGER_ON_TRIM=1`
  - `ARC3_ANIM_TRUTH=1`, `ARC3_ANIM_RING=8`, `ARC3_ANIM_DIGEST=1`
  - `EXPOSE_RESET=on`
  - `ARC3_OPENER_DEDUP=1`
  - `LOCAL_ANALYZER_TEMPERATURE=0.65`
  - `ARC3_TOOLCALL_INVOKE_FALLBACK=1`
  - `OPENROUTER_STREAM=0`

  Diff the two runs' resolved env (`harness.log`) before reading any score. Only `MODEL` and the provider pin may differ.
- **Pin one provider per model.** Use `OPENROUTER_PROVIDER_ORDER` with fallbacks off, so a run does not switch precision halfway through:
  - Qwen: `Alibaba`.
  - Ling: `Novita`, which is cheapest and has 262K context but unknown quantization. Use `DeepInfra` (BF16) if Novita errors.

### B1. Smoke (about 15 min, under $0.30)

- **What to run:** one game (ft09) per model, capped at 10 requests.
- **Pass if all of these hold:**
  - 10 of 10 replies are structured tool calls, or at most one "You did not call a tool" nudge.
    - Our text fallback recovers only Qwen XML and `<invoke>` markup, not Ling's `<arg_key>`/`<arg_value>`. A leaked Ling call therefore costs a request, not a crash.
    - Count these nudges per model as a parity metric.
  - Reasoning is present on at least 9 of 10 replies.
  - There are no 4xx errors and no list-form content in the request log.
  - The prompt token count grows across turns in a way that shows earlier reasoning being replayed.
    - If one provider drops replayed reasoning and the other keeps it, the comparison is asymmetric.
    - Note this rather than fix it; it is the provider's template.
- **No-go:** a provider that strips tool calls or reasoning.
  - Switch the provider pin first, then the model.

### B2. Panel pair (about 3–4 h wall, about $1–6)

- **Games:** the balanced 10 (sk48 bp35 tn36 sp80 lf52 ft09 vc33 tu93 cd82 r11l), one pass each, on both models.
- **Budget: 60 requests per game.** This is the request cap, and it is the matched budget.
  - It is just above the Kaggle panel's per-game maximum of 56, so both models get at least as many decisions as Qwen gets on Kaggle in 25 min.
  - Set `--minutes 120` as a safety limit only. On OpenRouter, wall-clock reflects provider speed, which says nothing about Kaggle, so it must not be the binding limit.
- **Concurrency:** both runners in parallel, under `nice`.
  - The games are network-bound, but keep the laptop rule of at most 3 heavy jobs: for example, Ling `--concurrency 2` and Qwen `--concurrency 1`, swapped on the repeat.
  - Concurrency does not affect scores here, because each model has its own provider and the budget is requests.
- **Record per game:** score, levels, requests used, prompt and completion tokens, dollars, tool-call misses, and a level-clear time series indexed by request number.

### B3. Read-out and repeats

- **Primary:** mean panel score at equal requests, plus `make significance` (paired per game) on the two score files.
- **Kaggle projection (secondary):** cut Ling's per-game curve at 60% of Qwen's request count. This approximates Ling's estimated decode speed on our hardware (ESTIMATE, see stage A3) and asks whether Ling still wins.
  - Also report score per million completion tokens, since a model that thinks longer costs wall-clock on Kaggle.
- **Repeats (user's rule: one run is a screen):**
  - If Ling is at or above Qwen in the screen, run 2 more pairs (3 in total).
  - Panel sd on Kaggle is large. With 10 games and 3 pairs, only differences of about +3 points or more on the panel mean are likely to show up.
- **Optional B4, images on, the question that matters for the competition:**
  - `qwen/qwen3.8-flash` with `MULTIMODAL_CONTEXT=current_grid` against `inclusionai/ling-3.0-flash-vl` with images, same cap.
  - Franzen credits images with real gains (10x upscale was his biggest image gain). Ling text-only only has to beat Qwen *as we run it*, which is with images.
  - Caveat: Ling-VL has no MTP head, so it would decode even more slowly on Kaggle.

### Go/no-go into the Kaggle path

- **Go** if, over 3 pairs, Ling text is above Qwen text by at least about 2 panel points and still at or above Qwen at the 60% request cut, and Ling text or Ling-VL is at or above Qwen *with images* (B4, or the 9.28 Kaggle reference as a rough anchor).
- **Stop** if Ling is at or below Qwen at equal requests. Then the slower decode on Kaggle can only make it worse. Log the result as **Didn't work** in `arms.md`.

### B cost

Per-request cost = prompt × input price + completion × output price.

| Basis | Ling ($0.02 / $0.06) | Qwen3.8-flash ($0.15 / $0.47) | Pair |
|---|---|---|---|
| Per request, 50K prompt + 2K completion, no cache | $0.00112 | $0.00844 | $0.0096 |
| Panel, 100–200 req/game (high-end assumption), no cache | $1.12–2.24 | $8.44–16.88 | $9.6–19.1 |
| Same, with a 90% cache-read share at the listed cache prices (Ling $0.0042, Qwen $0.016) | $0.41–0.82 | $2.41–4.82 | $2.8–5.6 |
| Panel at the recommended 60-req cap (600 req), no cache | $0.67 | $5.06 | $5.7 |
| Panel with the measured Kaggle token mix (349 req, 19.2M in, 0.62M out), no cache | $0.42 | $3.17 | $3.6 |

- **Pinning Ling to DeepInfra (BF16)** triples Ling's cost (about $2.0 per panel at the 60-request cap). That is still small.
- **Thinking length is the main uncertainty.** At 6K completion tokens per request instead of 2K, add about $0.0019 per request for Qwen and $0.0002 for Ling.
- **The whole B plan:** 3 pairs plus the B4 image pair is about $20–25 uncached at the 60-request cap. **Buy $25 of credits.**
  - Buying $10 or more also raises the free-model cap to 1,000 per day, which is irrelevant here.
- **Without credits:** these models cannot run at all (no `:free` variant). Even if they could, 600 requests per panel at 50 per day would take 12 days per model.
- **Wall-clock:** about 30–60 s per request (50K prefill plus 1–2K decode at provider speed; ESTIMATE). That gives 30–60 min per 60-request game, and about 3–4 h per panel pair at 3 concurrent games.

## (A) Kaggle path: the real comparison on our hardware

The target is the same text-only panel on one RTX PRO 6000, offline, with Ling INT4 against Qwen W4A16 under identical harness code. Every stage has a gate, and nothing moves on until its gate passes.

### A0. Code prep (laptop, no quota, about 1 day)

- **Branch:** create `arm/ling-vs-qwen` off main.
  - Merge in `arm/model-profiles` (PR #68, still open and not in main).
  - Copy the `panel-uc10x` arm definition from `arm/traj-arms` (`kaggle/arms.json:380`) twice:
    - `panel-uc10x-text`: profile `qwen38fn-w4a16` plus `MULTIMODAL_CONTEXT=none`;
    - `panel-ling-text`: profile `ling3f-int4` plus `MULTIMODAL_CONTEXT=none`.
  - Build both arms from one commit in one batch.
- **Launcher engine switch.** Add a profile key `engine: "pennyroyal" | "upstream"`.
  - For `upstream`, `build_args` emits `python -m sglang.launch_server` and drops the Pennyroyal-only flags. These are `--model-loader-extra-config`, `--cuda-graph-*-decode`, `--mm-feature-transport`, `--image-processor-backend`, `--warmups structured_output`, `--enable-request-time-stats-logging` and `--weight-loader-*`.
  - Keep `--served-model-name flashnext` so the harness model ID is unchanged.
  - The installer must accept a wheelhouse without Pennyroyal's bundled `nvidia/cu13` nvcc, or the new wheelhouse must be built in Pennyroyal's shape.
- **Profile `ling3f-int4`:**
  - `validate`: `compressed-tensors`, bits [4]; `quantization`: `compressed-tensors`. Leave the MoE backend on auto; the cookbook says not to force it. sm120 gets Marlin, because `is_sm100` matches only major 10.
  - `reasoning_parser` and `tool_call_parser`: `ling3`.
  - `chat_template_kwargs`: `{}`. Preserved thinking is hard-coded, and `enable_thinking` comes per request.
  - `spec`: `native` (NEXTN, using the checkpoint's own layer 42), with `SGLANG_ENABLE_SPEC_V2=1` in `serve_env`.
  - `frspec`: null, because the pinned map is tied to Qwen's tokenizer.
  - `hybrid`: new keys `max_mamba_cache_size: 64` (the vendor's 320 would be about 22 GiB of fp32 KDA state; ESTIMATE) and `linear_attn_backend: triton`. Drop the PLE and gdn-mtp keys.
  - `cfg`: `MEMFRAC` 0.88, `MAXREQ` 10, `CTX` 136K, `CHUNK` 8192, KV dtype `auto` (MLA KV is about 1 GiB per 128K tokens; fp8 saves nothing worth the risk).
  - Also pass `--enable-fp32-lm-head` and `--disable-shared-experts-fusion`, as recommended.
  - `startup_timeout_s`: 2400.
- **Builder:** let profile `inputs` mount `kernel_sources` (notebook outputs), not only models and datasets. Alternatively, copy the `build_fable_gguf.py` approach of replacing the serving cell.
- **`check_checkpoint`:** keep only the header and quant-config checks for this profile; its memory model is fitted to Qwen. Measure the pool in A3 instead.
- **Tests:** a golden argv for both profiles, and a check that `qwen38fn-w4a16`'s argv stays byte-identical to main's.

### A1. Engine wheelhouse (CPU notebook with internet, 0 GPU quota, 20–40 min wall)

**The named fork cannot be built.** `inclusionAI/sglang_ling_v3` (branch `ling_v3_support`) returns 404 from the GitHub API and the website, so its source is unavailable. The support has been absorbed upstream: SGLang PR #33561 merged on 2026-08-27, and v0.5.19 (2026-09-05) is the first release with `bailing_moe_v3.py`. Build from that instead, in this order:

1. **Preferred: upstream sglang v0.5.19's pure-Python wheel on Pennyroyal's binary stack.** Upstream v0.5.19 pins the same torch 2.13.0 and sglang-kernel 0.4.6.post1 as our Pennyroyal 2.5.3.
   - Recipe (CPU notebook, as `mellamoa/gguf-engine-build` did):
     - copy Pennyroyal's wheelhouse;
     - replace the `sglang-*.whl` with `pip download sglang==0.5.19 --no-deps`;
     - resolve the extra pure-Python dependencies with `pip download` against `requirements.lock`;
     - bump flashinfer 0.6.17 to 0.6.18 only if the resolver insists (sm120 Ling uses Triton MLA, Triton KDA and sgl-kernel Marlin, so flashinfer is off the hot path);
     - rewrite `requirements.lock`.
   - Nothing compiles, so the build should take about 20 min. Publish it as dataset `mellamoa/sglang-v0519-ling`.
   - **Loss:** our sm120 runtime patches (`runtime-patches/*`) and FR-Spec do not apply to upstream. The bench decides whether that matters.
2. **Fallback: overlay onto Pennyroyal 2.5.3.** As a runtime patch, add the following:
   - `bailing_moe_v3.py` and `bailing_moe_nextn` from upstream (copies in `scratchpad/ling/up_*`);
   - `ling3_detector.py` and the `ling3` reasoning-parser entry;
   - the arch-list entries in `server_args` / `overrides.py`.

   This keeps our launcher and sm120 patches, but it may hit upstream internal APIs (spec v2, KDA radix cache) that 2.5.3 lacks. Expect half a day to two days of porting.
3. **Last resort: vLLM 0.28 or later** (Ling since v0.28.0, Marlin INT4 seen in the DGX Spark log). This needs a whole new launcher and harness provider checks. Only worth it if 1 and 2 both fail.

**Gate A1.** These checks run on CPU in the build notebook:

- `import sglang.srt.models.bailing_moe_v3` succeeds;
- `--help` lists `ling3` under both parser options;
- `uv pip install --offline` from the wheelhouse succeeds in a clean venv.

Any failure means moving to the next option in the list above.

### A2. Stage the 77 GB checkpoint (CPU notebooks with internet, 0 GPU quota, 30–60 min wall)

- **Plan:** 24 shards, 77.04 GB in total. Shard 1 is 8.29 GB, shard 24 is 3.33 GB, and the rest are 2.96–3.01 GB. Whole shards per notebook (each notebook's `/kaggle/working` holds at most about 19.5 GB), so nothing needs reassembling:

| Notebook | Shards | Size |
|---|---|---|
| `ling3f-int4-stage-p1` | 1–4 | 17.2 GB |
| `-p2` | 5–10 | 17.8 GB |
| `-p3` | 11–16 | 17.9 GB |
| `-p4` | 17–22 | 17.8 GB |
| `-p5` | 23–24 plus config, tokenizer, index, chat template, `*.py` | 6.3 GB |

- **Verify:** check each shard's sha256 against the HF LFS `oid` from the tree API, and write a manifest.
- **Time:** download at the measured 140–168 MB/s takes about 2 min per notebook. Committing the output takes most of the wall time.
- **Mounting:** keep all five notebooks under `mellamoa`, so the GPU notebook can mount them as `kernel_sources` without making them public. Run them sequentially if CPU session limits bite.
- **At load time:** a symlink view in `/tmp/ling3f` joins the five mounts. The read from `/kaggle/input` is expected to take about 6 min (GGUF measured 59 GB in 261 s).
- **Gate A2:** all 24 checksums match, `config.json` quant check passes, and `model.safetensors.index.json` resolves every tensor to a mounted file.

### A3. GPU smoke and bench, no games (1 GPU session, 30–45 min, budget 2 attempts, so 60–90 GPU-min)

- **Command:** `python kaggle/build_notebook.py --bench baseline --profile ling3f-int4 --push`. If the queue allows, run `--bench baseline --profile qwen38fn-w4a16` alongside on the second account, so the two use the same corpus.
- **Checks, in order** (the bench stops at the first hard fail):
  1. **Load.** The server comes up within the 40-min timeout and logs its startup phases:
     - weights read;
     - Marlin repack of 68 GB of INT4 experts;
     - Triton JIT for KDA, MLA and Marlin;
     - CUDA-graph capture.

     It must not hit an illegal memory access (the GSQ failure mode). Record peak memory and the KV and mamba pool sizes at `MEMFRAC` 0.88. If startup OOMs, retry at 0.90 with `max_mamba_cache_size` 48.
  2. **One request.** A 2-turn chat with the harness's `python` tool definition. Checks:
     - `tool_calls` is structured, not text;
     - `reasoning_content` is non-empty;
     - the second turn reports `cached_tokens > 0` (radix cache on KDA works).
  3. **Parse rate over the bench corpus.** Structured tool calls on at least 95% of tool turns and reasoning present on at least 95%. Count any string argument that is valid JSON and gets rewritten (upstream bug #42778; our `code` argument is a string, and Python code that parses as JSON is rare, but log it).
  4. **Throughput.** Decode tok/s at 1, 4 and 10 streams, with NEXTN on and off, plus accept length and TTFT on a 50K-token prompt.
     - Qwen reference: about 731–760 tok/s aggregate at 9–10 streams.
     - Ling ESTIMATE: about 100 tok/s single-stream and 400–500 tok/s at 10 streams without MTP, more with NEXTN.
  5. **Context-overflow error wording.** Send a deliberately oversized prompt and check that `_is_context_length_error` matches it.
- **Gate A3:**
  - **Go to A4:** load OK, parse rates at least 95%, cache hits work, and at least 450 tok/s aggregate at 10 streams (about 60% of Qwen).
  - **Hold** between 300 and 450 tok/s. Ling would need a large per-decision lead from B. Try FlashInfer MLA (`--attention-backend flashinfer`, as the DGX Spark recipe does) before deciding.
  - **No-go:** an illegal memory access that survives one attempt with `--disable-cuda-graph`, below 300 tok/s, or parse rate below 90%.

### A4. Text-only panel pair (about 40 + 50–60 GPU-min per pair)

- **What to run:** push `panel-uc10x-text` and `panel-ling-text` together.
  - Same commit and batch.
  - Balanced 10 games, 25 min, 4 streams.
  - A byte-identical harness except the profile; diff the resolved env in both logs.
  - Ling's notebook runs longer because of startup (estimate 15–25 min against Qwen's 10.8).
- **Compare at matched wall-clock** with `score_timeline` and `make significance` (paired per game). Report per-token results as a secondary view.
- **Reference:** `v4-panel-uc10x` v1 (Qwen with images, 9.28, 15 levels) is the free third arm. It shows how much going text-only costs Qwen.
- **Gate A4 (user's rule: repeats decide):**
  - One pair is a screen. If Ling text is at or above Qwen text, push 2 more pairs.
  - Go to full runs only if the 3-pair mean has Ling at or above Qwen text *and* within about 1 panel sd of Qwen with images (otherwise Ling only beats a handicapped Qwen).
  - Then the next arm is Ling-VL, which needs SGLang v0.5.21 and has no MTP.

### A5. Full runs, only if Ling is close (about 2.3 GPU-h each)

- **What to run:** 121-min offline runs on the 25 public games.
  - Ling, text-only or VL depending on A4.
  - Against our current Qwen stack (user-combo-10x, with images).
  - 2–3 runs per model, then `make significance`.
- **Recorded in:** `arms.md` and a PR from `arm/ling-vs-qwen`, as usual.

### A time and quota

| Stage | Wall | GPU quota |
|---|---|---|
| A0 code prep | about 1 day of work | 0 |
| A1 wheelhouse (option 1) | 20–40 min (overlay fallback: 0.5–2 days) | 0 |
| A2 staging (5 CPU notebooks) | 30–60 min | 0 |
| A3 smoke and bench | 30–45 min per attempt | 1–1.5 GPU-h (2 attempts) |
| A4 one panel pair | about 1.7 h | about 1.7 GPU-h |
| A4 two repeat pairs | about 3.4 h | about 3.4 GPU-h |
| A5 full runs (2–3 per model) | about 9–14 h | about 9–14 GPU-h |
| **Total to a screened answer (A0–A4, one pair)** | **2–3 days** | **about 3 GPU-h** |
| **Total to a repeated answer (A4 x3)** | | **about 6 GPU-h** |

Each account allows at most 2 concurrent GPU sessions. A pair fits on one account; repeats can split across `mellamoa` and `snowiers`.

### Main failure points on the Kaggle path

1. **The engine on sm120 (the most likely failure).** Nobody has run Ling on sm120. The closest evidence is inclusionAI's DGX Spark (sm121) notebooks, where INT4 with NEXTN decoded at about 42 tok/s on slow unified memory. Triton KDA, Triton MLA and Marlin are all generic paths, so it should run, but our GSQ engine looked equally plausible and died at self-test.
2. **The wheelhouse.** Upstream sglang 0.5.19 may need dependency versions that Pennyroyal's lock does not have, or may import kernel symbols that Pennyroyal's sgl-kernel build lacks. The overlay fallback can hit missing internal APIs.
3. **Decode speed.** Ling's KDA attention is kept in BF16 (4.41 GB read per step) and the KDA short convolutions waste 2x bandwidth on hardcoded fp32 (issue #40504). Our estimate is 55–70% of Qwen's tok/s (ESTIMATE). Our GGUF runner had per-token quality on par with Qwen but decoded 4x slower and scored 2.33 against 24.29: speed decides on Kaggle.
4. **Memory.** Weights take about 71.7 GiB against Qwen's 68.5 GiB, and the vendor recipe's mamba cache of 320 would not fit. The pool size is only an estimate until A3 measures it.
5. **Tool-call parsing.** The ling3 (GLM-4.5 XML) parser has open bug #42778, and our harness cannot recover leaked `<arg_key>` markup. A1–A3 measure the miss rate. If misses are at or above 2%, add a flag-gated `<arg_key>` recovery regex to `tool_agent.py` before A4 (applied to both arms).
6. **Prefix cache.** The radix cache supports KDA, but the host-RAM HiCache tier does not (issue #33713). Our Qwen setup does not use HiCache either, so the cache setups match.
7. **Harness differences dominate model differences.** The Tycho port scored 3.53 against 29.22 on the same server. Keep both arms byte-identical apart from the profile and the image flag.
8. **Images.** Text-only handicaps Qwen as well, so a Ling text win over Qwen text is necessary but not sufficient.

## Where this lands

- `arms.md`: add **ling-vs-qwen** under Not yet implemented, with OpenRouter (B) as its first step and this file as the plan.
- Source notes, downloaded headers and upstream source are in the session scratchpad (`scratchpad/ling/`). Upstream model and parser files are copied there as `up_*.py` for the A1 overlay fallback.
