# L1: Our own training data, environments and tooling (local repo audit, 2026-10-18)

Local audit plus three web checks (ms-swift docs and source, SGLang cookbook). Paths are relative to `daniel-franzen-sol/` unless absolute.

Scripts (uncommitted, ephemeral): `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/scripts/` (`decisions2.py`, `bench_join.py`, `reqstats.py`, `windows.py`, `pair.py`). Their outputs are one folder up. They ran under `nice -n 15` with at most 3 processes. One sample (ft09, `v4-mem-repair-10x/v1`) is unpacked in `scratchpad/sample/`.

Corrections to earlier notes, found here:
1. The ~50–60% figure is the share of tokens on **stalled** levels: levels of about 100k tokens, cleared or not (`research/reports/rsi-proposals-2026-10-15.md:30`). Levels **never cleared** take only 22–28% of tokens in the leader runs, and 32% across all 85 logged runs. The CLAUDE.md phrase "50–60% of tokens go to levels never cleared" mixes the two.
2. The local DreamTeam clone has **no** `solution.json` files. `find competitors/dream-team -iname '*solution*'` returns 0. Each game dir has only `<id>.py` and `metadata.json` (with `baseline_actions`). So the claim in `research/data/finetune-research-2026-10-17/data.md` §2, "each has a solution.json that replays to WIN", is not true of our copy.
3. Event rows give the level **after** each action; a win has `state: WIN` and `level_completed: false`. The counts below attribute each action to the level before it (6,034 clears from events vs 5,997 in `benchmark.json`).

## Q1. What the run archives contain, per run

### Takeaway
85 runs give full per-call request logs: Franzen-harness Qwen3.8-Flash-Next runs, 1,642 game-runs and 141,483 logged model responses. Every game-run in them also has a per-action event log with exact boards. All of it is on the 25 public games.

### Cited Findings
- **Layout.** Each run sits in `runs/kaggle/<kernel>/v<N>/`. It holds `benchmark.json` (per-action history with `generated_tokens` and `wallclock_seconds`), `*_p0_requests.jsonl` (request payloads), `artifacts/*_p0_events.jsonl`, `serve.log` and transcripts. Files over 64 KB are xz-compressed — `runs/README.md:12-17,31`.
- **Which runs have which files** (counted with `find`; the 157 kernel dirs hold 151 runs with a `benchmark.json`):

  | File | Runs | Files |
  |---|---|---|
  | `*_requests.jsonl*` | 89 | 1,710 |
  | `*_events.jsonl*` | 149 | 3,022 |
  | `*_viewer_data.json*` | 87 | 1,770 |
  | `transcripts/` | 87 | — |
  | `score_timeline.jsonl` | 60 | — |
  | `serve.log` | 101 | — |
  | `*_signals.jsonl` (old-harness logprobs) | 17 | — |

- **Of the 89 request-log runs:**
  - 4 are not Franzen + Flash-Next: `taaf-d1-thinkecho` (old harness), `v4-gguf-final`, `v4-gguf-smoke` and `v4-q27-dflash2-10x` (other models).
  - That leaves 85 usable runs: 65 `v4-*` and 20 `fz-*`.
- **Size.** All request logs are 0.40 GB xz and **65.7 GB raw**; one leader run (`v4-mem-repair-10x/v1`) is 1.13 GB raw (`xz --robot --list`). Every request repeats the whole conversation.
- **Request-log row format** (sample: ft09 in mem-repair-10x):
  - Each row has `messages`, `tools`, `chat_template_kwargs` (`{"preserve_thinking": true}`), `tool_choice`, `event` (request or response), `analysis_step`, `action` and `request_index_within_turn`.
  - Response rows add `usage` (prompt, completion, reasoning, cached and image tokens) and `finish_reason`.
  - Messages use the OpenAI chat format:
    - a system prompt of 23,100 characters;
    - user turns with `image_url` parts holding base64 PNG data URIs (640×640);
    - assistant turns with `reasoning_content` and `tool_calls` (one `python` tool);
    - `tool` results as JSON strings;
    - an internal `_arc3_control` key on some user turns.
  - Source: the writer at `ARC3-Inference/inference/agent/tool_agent.py:4764-4808`, plus my sample inspection.
- **Event-log rows.**
  - One `action` row per action: the raw 64×64 int `board`, `board_ascii`, `level`, `score`, `state`, `action_name`, `level_completed`, `game_over`, `analysis_step`, `batch_index`.
  - Plus `analysis` rows whose `transcript` holds `[MODEL RESPONSE META]` (`finish_reason`, `reasoning_chars`, `raw_tool_calls`) and `[THINKING]` sections for **every** call. In the sample there are 60 such sections for 60 responses.
  - Source: `scratchpad/sample/ft09-0d8bbf25_p0_events.jsonl`.
- **Fleet totals, 85 runs** (`reqstats.json`):

  | Measure | Value |
  |---|---|
  | Requests | 142,572 |
  | Responses | 141,483 |
  | finish = `tool_calls` | 139,883 |
  | finish = `length` | 1,514 |
  | finish = `stop` | 86 |
  | Completion tokens | 269.6M (235.9M reasoning, 87%) |
  | Prompt tokens | 8.04B (7.28B cached, 90.5%; 0.97B image tokens, 12%) |
  | Prompt tokens per call | median 57,313, p90 84,800, max 120,756 |

- **Decision counts, 85 runs** (`decisions2_joined.json`; level attributed to the board before each action):
  - **Volume.** 1,642 game-runs, 289,520 actions, **6,034 level clears** (`benchmark.json`: 5,997) and **354 wins**.
  - **Wins by game.** ft09 50, sb26 50, lp85 49, ar25 45, tr87 35, cd82 33, r11l 31, vc33 22, sc25 14, tu93 8, m0r0 6, tn36 6, cn04 4, su15 1. The other 11 games have none.
  - **Distinct levels.** **151 distinct (game, level) pairs** were ever cleared, out of 183 levels in the 25 games. That is about 40 clears per distinct level.
  - **Tokens on cleared levels.** 176.7M (68%); 224,639 actions; 47,693 analysis steps.
  - **Tokens per cleared level.** Median 20.3k, p75 39.9k, p90 64.2k. 1,017 levels took ≥50k tokens and 191 took ≥100k.
  - **Never-cleared levels.** 1,170 of them hold 82.1M tokens (32%), 64,881 actions and 15,717 steps.
  - **Model calls per analysis step.** 2.08 (141,483 / 68,004).
- **Leader runs, share of tokens on never-cleared levels.** mem-repair-10x 26%, user-combo-10x 24%, stall-fix-10x 28%, mem-unified-10x 22% (same script).
- **All harness families, 149 runs with events.**
  - 2,991 game-runs, 458,828 actions, 7,773 clears, 366 wins.
  - The 60 extra runs (old taaf/duck/wang harness and runs without request logs) add actions and boards but **no exact prompts**.

### Inferences
- Event and benchmark logs are complete enough to label every call: its level, whether the level was later cleared, its tokens, and deaths. The request logs give the context the model saw.
- Coverage is the real limit, not volume. There are 6k clears but only 151 distinct puzzles, and wins come from 14 games. A "level-clear SFT" set built from these logs is mostly repeats of easy early levels.

### Gaps
- Request and response rows were paired by file order and by keys. The keys `(analysis_step, action, request_index)` are not unique: in mem-repair they gave 1,758 unique keys for 2,636 requests. An exporter should pair rows by file order.

## Q2. Can we rebuild the exact token sequence of each call, and is the thinking kept?

### Takeaway
The **prompt side** can be rebuilt almost exactly: the messages, tools, kwargs and images are logged as sent, and a per-call `prompt_tokens` is available to check against. The **completion side** cannot be rebuilt exactly:
- the response row omits the reply;
- the reply survives only inside the next request, after the normaliser has run (on the leader stack);
- no token ids or logprobs are logged anywhere.

M0(d), "save exact rendered conversations" (`arms.md:644`), was therefore **only partly done**.

### Cited Findings
- **The response row repeats the request's messages and adds `usage` and `finish_reason`, but not the generated message.**
  - The `event="response"` call passes `messages=latest_request_messages` (`tool_agent.py:9673-9687`).
  - Request logging is on in every built notebook: `bm.solver.save_request_logs = True`, `kaggle/template.ipynb` cell 16.
- **The reply is recoverable from the next request's history.** In all 25 games of mem-repair-10x (2,636 requests):
  - 2,572 of 2,611 transitions contain the previous reply after the previous tail (98.5%);
  - 34 have no assistant message after it (retries or yields);
  - 5 were lost at a trim;
  - the final reply of each game (25) is never re-sent.

  All of these can be filled from the event transcripts (`[MODEL RESPONSE META] raw_tool_calls` + `[THINKING]`) — `windows.py` output.
- **The thinking is kept.**
  - Server: `default_chat_template_kwargs: {'preserve_thinking': True}`, `reasoning_parser qwen3`, `tool_call_parser qwen3_coder`, and the chat template is read from `/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1/chat_template.jinja` (`runs/kaggle/v4-mem-repair-10x/v1/serve.log.xz`).
  - In the ft09 sample, reasoning chars per reasoning token have a median of 3.12, and 0 of 58 replies fall below 2.0. So there is no sign of truncated thinking in history (`pair.py`).
- **The reasoning in history is normalised, not verbatim.**
  - By default the normaliser removes empty lines and literal `<think>` tags, and rejoins line splits, so "the next request then renders a reply that differs from the tokens the model generated" (`ARC3-Inference/CONFIGURATION.md:878`).
  - `ARC3_VERBATIM_HISTORY=1` keeps blank lines (`CONFIGURATION.md:880-886`). It is set only in the prefix-reuse, tput-safe, tput-safe-lowm, tput-accept05 and panel-tput-safe arms (`kaggle/arms.json:45-91`), **not** in `mem-repair-10x` (`kaggle/arms.json:1223`, env block).
- **Tool-call arguments are stored exactly as the server returned them** (`CONFIGURATION.md:884`). The model's raw emitted tool-call text (qwen3_coder markup) is not stored, only the parsed call.
- **No `logprobs`, `token_ids` or `return_logprob` appear anywhere** in `ARC3-Inference/inference` (grep). A `seed` field exists in `inference/utils/openai_compat.py:95`.
- **Evidence that the prompt can be replayed.** State-lab's OpenRouter replay of logged requests "matched Kaggle's prompt tokens to within 11" — `git show archive/tooling/state-lab:analysis/state_lab/README.md`, line 108.
- **Internal keys.** `_arc3_control` (`tool_agent.py:1528`) is stripped before sending (`_strip_control_keys`, `tool_agent.py:2209`). An exporter must strip it too.
- **ms-swift side.**
  - The `qwen3_8` template keeps historical `<think>` content by default, unlike Qwen3.5/3.6. It supports `reasoning_effort` (default `xhigh`) through `chat_template_kwargs` in the dataset — [ms-swift Qwen3_8-Best-Practice.md](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_8-Best-Practice.md) (top section, read 2026-10-18).
  - ms-swift ships agent templates `qwen3_coder` and `qwen3_5` — [swift/agent_template/mapping.py lines 20,37-38](https://github.com/modelscope/ms-swift/blob/main/swift/agent_template/mapping.py).
  - The dataset format supports `tools` (JSON string), `tool_call` / `tool_response` roles, a per-assistant `loss: true/false` (and `loss_scale` from v4.2), and `images` with `<image>` tags. OpenAI `image_url` blocks are auto-converted, and base64 data URIs are accepted — [Custom-dataset doc](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html).
  - The Flash-Next best-practice page shows **text-only** SFT at `max_length 8192`, with LoRA on `in_proj out_proj linear_proj linear_qkv` — [Qwen3_8-Flash-Next-Best-Practice](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html).

### Inferences
- **For SFT / RFT this is enough:**
  - render prompt + reply through `chat_template.jinja` (or ms-swift's `qwen3_8` template, which must be shown to match);
  - check every example against the logged `prompt_tokens`, and the reply against `completion_tokens`.
- The normaliser means we would train on replies without blank lines. Harmless for SFT, but it is not the exact on-policy sample.
- **For RL with importance ratios, the logs are not usable:** there are no token ids or logprobs. New rollouts are needed for that, from a logger that records the raw generated text and token ids.

### Gaps
- ms-swift's support for `reasoning_content` as a field is unverified. Embedding `<think>…</think>` in the assistant content is the documented route.
- Unverified whether `qwen3_8` + `qwen3_coder` renders byte-identical to SGLang's `chat_template.jinja`.
- Image training on Flash-Next under Megatron is not shown in the recipe.
- `chat_template.jinja` is not on the laptop (`~/.cache/huggingface/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound` has only `tokenizer.json`). Pull it from the Kaggle model or Hugging Face.

## Q3. The context-length problem: what a training example looks like

### Takeaway
Only **2.2% of logged calls have prompts ≤8k tokens, and 6.8% ≤16k**. The published recipe is measured only up to 8k. Exact-context SFT on our logs therefore needs either 60–100k-token training sequences or a shortened context that does not match deployment. This is the main data-format decision.

### Cited Findings
- **Prompt length, all 85 runs** (`reqstats.json`):

  | Prompt length | Share of calls |
  |---|---|
  | ≤8,192 | 2.2% |
  | ≤16,384 | 6.8% |
  | ≤32,768 | 16.5% |
  | ≤49,152 | 37.4% |
  | ≤65,536 | 62.8% |
  | ≤98,304 | 94.8% |

  The first call of a game has a 6,233-token prompt (median; range 5,472–6,955).
- **Windows** (requests whose messages extend the previous request; mem-repair-10x):
  - 163 windows over 25 games, median 15 requests per window;
  - the final prompt has a median of 79.9k tokens (max 85.2k);
  - a window ends with a median of 23 assistant turns and 21 images;
  - a window after a trim starts at about 42.0k tokens (min 31.3k);
  - the sum of final window lengths is **11.5M tokens per 25-game run** (`windows.py`).
  - Arm config: `LOCAL_ANALYZER_CONTEXT_WINDOW 98304`, `ARC3_CONTEXT_DRAIN_TOKENS 43008` (`kaggle/arms.json`, the mem-repair-10x env).
- **ms-swift Flash-Next memory per GPU:** 75.6 GiB at 2k, 93.3 GiB at 8k with full recompute, and 72.5 GiB at 8k with PLE offload + padding-free (TP2/EP4/PP2, all 8 GPUs) — [best-practice page](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html), memory table.
- **Images:** a median of 9 per request in the ft09 sample, at about 400 tokens each (median 3,600 image tokens per request). Fleet-wide, 12% of prompt tokens are images (`reqstats.json`, `pair.py`).

### Inferences
- **Option A: whole-window multi-turn examples.** Use exact contexts with `loss` only on the chosen assistant turns. Each run gives 160–170 sequences of about 40–85k tokens, about 11M tokens per run (roughly 0.9B over all 85 runs if every window is used). This needs context parallelism that the 8-GPU recipe doesn't have room for. **Test first:** one 64k sequence on the node before planning around it.
- **Option B: rebuilt short contexts** (≤8–16k): system prompt, memory card, the last few turns and the current images. This fits the recipe, but trains on a context the deployed harness never shows. **Test first:** check that a short-context-trained LoRA doesn't regress at 57k.
- **Option C: game-start and post-trim decisions only.** These are the calls the logs already have at ≤16k (6.8%), mostly level-1 calls, so the coverage is too narrow.
- **Images:** plan to train with images, extracted to deduplicated PNG files. Dropping them changes the input distribution.

### Gaps
- No measured training throughput (tokens/s) for Flash-Next LoRA at any length.
- Unknown whether ms-swift Megatron supports context parallelism together with EP for this model.

## Q4. Counts of useful decision categories

### Takeaway
There are many level-clear positives, few genuine "stall then breakthrough" positives (about 190–270), and many deaths (874). R3 candidate states run to tens of thousands, but all of them are on public games.

### Cited Findings
All from the 85 Franzen + Flash-Next runs, `decisions2_joined.json`. "New board" means the interior hash, with the outer 2 cells of each edge cropped, has not been seen before on this level. This is my approximation of a HUD mask.

| Category | Count | Volume |
|---|---|---|
| Level clears (positives) | 6,034 (354 game wins) | 224,639 actions, 47,693 decision steps (≈99k calls at 2.08 calls/step), 176.7M completion tokens |
| R3 candidate states (action-taking decisions on later-cleared levels) | ≈47.7k steps; ≈81% are on the hindsight-shortest route (`research/archive/reports/graph-exploration-bandits-memory.md:220`), so ≈38k on-route states | 151 distinct levels only |
| Hard clears (≥100k tokens on the level) | 191 | (≥50k: 1,017) |
| Clears after a stall of ≥15 actions without a new board | 191 levels, 240 breakthrough events | All levels: 347 breakthrough events |
| Never-cleared levels (R4 targets) | 1,170 | 82.1M tokens, 64,881 actions |
| Deaths (GAME_OVER) | 874 | 540 on levels later cleared, 334 on never-cleared levels; 415 cleared levels had at least one death |
| Stuck-vs-cleared pairs for the same (game, level) | 123 levels have at least one clear and at least one stuck attempt; 129 of the 151 cleared levels have ≥10 clears | Pairs exist only on public games |

- The earlier inventory's 6,083 clears and ~180k calls (`research/data/finetune-research-2026-10-17/data.md` §1) included 4 non-Franzen runs and extrapolated the call count. The measured values are 5,997–6,034 clears and 141,483 responses.

### Inferences
- **R4 data is thin.** About 200 real breakthroughs over 85 runs, many on the same handful of levels. Hindsight relabelling (R4) gives at most a few hundred examples per pass over the logs, so new rollouts on new games are the only way to scale it.
- **Deaths make good negatives for DPO** (state, fatal action vs the action that later cleared it), but only 874 exist.

### Gaps
- The stall threshold (15 actions) and the HUD crop are mine. The archive analysis used single-colour segmentation plus status-bar masking, so the counts will differ somewhat.
- I didn't build actual DPO pairs or check whether the two members share a state.

## Q5. Tooling that exists

### Takeaway
State-lab is the single most useful asset. It rebuilds any logged decision exactly in the local engine, runs model replies through the real sandbox, and samples on our own Kaggle server at about 1.2–1.6k samples per GPU-hour. It lives only on an archive tag. The RSI loop writes ms-swift JSONL, but it is single-turn and text-only.

### Cited Findings
- **State-lab** (`git show archive/tooling/state-lab:analysis/state_lab/README.md`).
  - **What it does.** It rebuilds a state from a logged request by replaying the run's actions through `_HarnessGameSession` on the exact engine version, and verifies the board, the image pixels, the step/level line and the retained functions.
  - **Fidelity.**
    - On mem-v2-10x: 100 of 100 states rebuild, the image matches in 71 of 71, and the logged code executes the same actions in 99 of 99 with a byte-identical tool result (line 59).
    - On `v4-pf-solved-10x`: 15 of 15 states, 14 of 14 actions (line 99).
  - **Kaggle backend.** `--export-kaggle` writes the exact server bodies. `kaggle/build_lab_notebook.py` swaps only the game cell, and `--import-kaggle` replays the samples through execute and grade (lines 100–102).
  - **Branch dependencies.** It needs the run's harness branch (`STATE_LAB_INFERENCE_ROOT`) and, for one variant, `arm/memory-main`.
- **Measured state-lab throughput on the Kaggle GPU:**
  - `v4-lab-round2`: 400 of 400 samples, 637,707 completion tokens, 14.9 min of sampling (`runs/kaggle/v4-lab-round2/v1/lab_summary.json`);
  - `v4-lab-round3`: 300 of 300 samples, 735,583 tokens, 15.5 min (`.../v4-lab-round3/v1/lab_summary.json`).
  - That is ≈1,610 and ≈1,160 samples per GPU-hour, at ≈713 and ≈791 tok/s.
  - Sample rows keep the full OpenAI response.
  - Server cold start: "READY after 523s" (`v4-lab-round2.log.xz` line 224). The notebook ran 1,495 s in total.
- **RSI loop** (`arm/rsi`, PR #177):
  - `gamereason/train/sft.py` collects k samples per task, keeps the shortest verified one, and writes `{"messages": [...]}` "(ms-swift's format)" plus a `.meta.jsonl`. It trains with transformers + peft, with loss on assistant tokens only (docstring, `git show arm/rsi:game-reasoning/gamereason/train/sft.py`).
  - It is single-turn and text-only, with no tools or images.
  - The Kaggle test: Qwen3-4B went 0.10 → 0.20 in 118 GPU-min (`arms.md`, RSI entry).
- **Local CPU engines:**
  - the 25 public games at their exact versions in `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/<game>/<version>/` (do not use the polyphony copy — auto-memory note);
  - DreamTeam: 25 games, Apache-2.0 (NVIDIA copyright line), same layout, `metadata.json` with `baseline_actions`, **no solutions**;
  - arc-interactive: 252 `metadata.json`, MIT;
  - sonpham `arc-agi-3`: 86 `metadata.json`, no licence.
  - Counts are from `find`.
- **Remote endpoint and offline games.**
  - The harness talks to any OpenAI-compatible server through `CONFIG_PATH` (provider `vllm` = OpenAI-compatible mode) — `ARC3-Inference/README.md:58-67`.
  - Offline play uses `arc_agi.Arcade(OperationMode.OFFLINE, environments_dir=...)`, but the notebook **hardcodes** the competition `environment_files` (`kaggle/template.ipynb` cell 20; `taaf-duck-harness-kaggle-share.ipynb:581-582`).
  - DreamTeam games have never been run in our harness: M0(c) was not done, and the "DreamTeam out-of-distribution panel" is marked "not built" (`arms.md:578,643`).
- **Analysis tools.** `make traces` (`ARC3-Inference/inference/tools/traces.py`) scores runs; it does not export training data. There is no SFT exporter on `main`.

### Inferences
- **Yes, our harness can play games on CPU against a remote endpoint.** Kaggle already does this, with the harness and engine in the notebook process talking to `127.0.0.1:8001`. Pointing `LOCAL_ANALYZER_BASE_URL` at an Azure SGLang server and setting `environments_dir` to an external game folder is configuration plus one notebook/builder change.
- **Per-game cost** (leader run mem-repair-10x, `benchmark.json`):
  - 91k–362k completion tokens per game-run (median 224k);
  - 59–632 actions;
  - won games: median 142k tokens and 5,268 s;
  - about 105 calls per game at a median 57k-token prompt, 90% prefix-cached.

### Gaps
- No CPU-load measurement for many concurrent games off Kaggle. No base pass rate on any external game.

## Q6. Rollout throughput: free Kaggle quota vs one 8×H200 node

### Takeaway
Kaggle (measured): about 2.5–2.9M completion tokens per GPU-hour, i.e. about 12 full game-runs or 1,200–1,600 state-lab resamples per GPU-hour. With 11 GPU-h a day, that is about 120–130 game-runs or 12–17k resamples a day, before any evaluation runs. The 8×H200 figure is an UNVERIFIED estimate: about 5–10× a day's Kaggle quota, if the node is used for rollouts.

### Cited Findings
- **One full run:** 25 games in 121 min, 5.03–5.26M completion tokens (mem-repair-10x: 5,256,956 tokens summed over `solver_note`). That is ≈12.4 game-runs per GPU-hour, at ≈690–720 tok/s.
- **Cold start:** 523–581 s per session (`v4-lab-round2.log.xz` lines 224, 230).
- **Daily caps:** mellamoa 6 h and snowiers 5 h (CLAUDE.md).
- **SGLang cookbook:** verified Flash-Next configs are GB300/B300/B200 NVFP4 on 1 GPU and **H200 FP8 TP4/EP4** (`Qwen/Qwen3.8-Flash-Next-FP8`, 4 GPUs, MTP 3-1-4). It gives no tok/s figures. Data generated 2026-10-16; H200 run verified 2026-08-27 — [SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.md).
- **Our build:** the Pennyroyal build is sm120 (`serve.log`: W4A16 auto-round, fp8 KV cache, EAGLE MTP).

### Inferences
- **Kaggle:** (11 h − ~0.5 h of startups) × ~2.6M ≈ 27M tokens a day. That is ≈120 game-runs, or ≈14k resamples at ~1.9k tokens each. Five days gives ≈600 game-runs or ≈70k resamples, if no quota goes to evaluation.
- **8×H200 (UNVERIFIED):**
  - two FP8 TP4 replicas;
  - H200 bandwidth is 4.8 TB/s vs about 1.8 TB/s on the RTX PRO 6000;
  - a guess of 3–7k tok/s aggregate gives ≈11–25M tokens per node-hour, i.e. ≈50–125 game-runs per node-hour.
  - Benchmark it on day 1 with our real request shapes (`kaggle/bench_corpus.json.gz` exists for that).
- **On-policy caveat:** FP8 rollouts are close to, but not exactly, our deployed W4A16 policy.

### Gaps
- No measured H200 throughput.
- Unverified whether the Intel W4A16 checkpoint runs on sm90 in stock SGLang.
- Azure node availability dates are unknown.

## Q7. Engineering needed, file by file

### Takeaway
About 2–3 engineer-days for an SFT/RFT exporter with token-count validation, and 2–4 days for a rollout service. The Kaggle path for both is mostly configuration of things that already exist.

### Cited Findings and design
**(a) SFT/RFT dataset in ms-swift multi-turn format.** New `training/export_sft.py` on an `arm/` branch:
1. **Read.** Per game-run, read `*_p0_requests.jsonl.xz`, `artifacts/*_p0_events.jsonl.xz` and `benchmark.json.xz`, streaming with `lzma`, under nice.
2. **Pair.** Pair request and response rows by **file order**. Recover reply *i* from request *i+1*'s messages after the previous tail (98.5% of replies). Otherwise fill it from the events transcript (`raw_tool_calls` + `[THINKING]`).
3. **Label each call** by mapping its `action` field ("number of the NEXT action", state-lab README) to the event row. Labels:
   - the level before the action (`prev.level`);
   - whether the level was later cleared (`level_completed` or `state == WIN`);
   - whether the call is on the hindsight route;
   - deaths;
   - tokens.
4. **Convert to ms-swift format:**
   - strip `_arc3_*` keys;
   - put `reasoning_content` into `<think>\n…\n</think>\n` in the assistant content;
   - turn `tool_calls` into `tool_call` messages with JSON `{"name","arguments"}`;
   - map `tool` to `tool_response`;
   - write images to content-hashed PNGs, with `<image>` tags and an `images` list;
   - write `tools` as a JSON string;
   - add `chat_template_kwargs` with `preserve_thinking`;
   - set a per-assistant `loss`.
5. **Validate.** Render with the model's `chat_template.jinja` + the Qwen processor and check:
   - the total equals the logged `prompt_tokens`;
   - the image token count equals `image_tokens`;
   - the reply length is close to `completion_tokens`.

   Then repeat the check through ms-swift's `qwen3_8` template (`swift sft --check_dataset`-style encode).
6. **Split and screen.**
   - Split by game id: the 10 held-out public test games plus a family list.
   - Hash games against `experiments/arc-agi-3/data/environment_files_manifest.json`.
   - Emit the variants: whole-window (Option A), short-context (Option B) and R3 single-decision.

**(b) Rollout service for RL / ReST rounds.**
- **Logging fixes** (each behind a default-off flag, per the project rules):
  - in `tool_agent.py` around 9673–9687, add the raw `result` message (content, reasoning, raw tool-call text) to the response row;
  - optionally request token ids and logprobs from SGLang;
  - turn `ARC3_VERBATIM_HISTORY=1` on in every data-collection arm.
- **External games on Kaggle:**
  - add an `environments_dir` override to `kaggle/build_notebook.py` and cell 20 of `kaggle/template.ipynb`;
  - upload DreamTeam and arc-interactive games as a private dataset;
  - add arms to `kaggle/arms.json`.
- **Azure:** an SGLang launcher (FP8 TP4/EP4 ×2, or W4A16 ×8 if it runs) and a `CONFIG_PATH` JSON pointing the harness at it. A driver that runs `inference.framework.run` offline over a game list with K seeds per game and writes the standard run-dir layout, so `make eval` and the exporter work unchanged.
- **ReST selection script:** keep rollouts above the group mean, cut at the last level-up, then pass them to the exporter (M2 in `arms.md:649-655`).
- **R3:**
  1. restore `analysis/state_lab/` from `archive/tooling/state-lab` onto a branch;
  2. write a states file of on-route decisions on cleared levels;
  3. `--export-kaggle` → upload → `build_lab_notebook.py` → import;
  4. a new grade rule: same executed actions, or the same post-action board, as the logged decision;
  5. keep the shortest passing sample;
  6. send it to the exporter.

### Gaps
- Effort estimates are mine.
- `kaggle/build_lab_notebook.py` is on the archive tag; I didn't check that it still builds against `main`'s template.

## Implications for our plan

- **Do (day 1): build the exporter with token-count validation first.** It turns 141k logged calls (269.6M reply tokens, 6,034 clears) into ms-swift JSONL. Without that check, every dataset rests on assumed template equivalence. The ground truth is already logged in every row: `usage.prompt_tokens` and `image_tokens`.
- **Test first (day 1, on the Azure node): context length.** 93% of real calls have prompts above 16k. Run one 64k-token Flash-Next LoRA step, with images, under ms-swift Megatron.
  - If it fails, the plan must commit to rebuilt short contexts (Option B) and add a check that a short-context-trained LoRA doesn't regress at 57k.
  - "Train on trimmed windows ≤8–16k" (`finetune-plan-2026-10-17.md:136`) is not a minor detail: it changes what every example looks like.
- **Do: turn on `ARC3_VERBATIM_HISTORY=1` and response logging in every data-collection run from now on.** Old logs are fine for SFT but useless for RL ratios: no token ids or logprobs, normalised reasoning, no raw reply.
- **Do: run R3 through state-lab on Kaggle.** It is the only verified, exact, on-policy, $0 pipeline we have:
  - ≈1.2–1.6k samples per GPU-hour, measured;
  - ≈38k candidate on-route states, which is far more than needed;
  - but only 151 distinct public levels, so it must be split by held-out game.

  Restore the tag, add a "same action / same board, shortest" grade rule, and spend about 2–3 Kaggle GPU-h a day.
- **Don't count on R4 from logs.** There are about 200 genuine stall-then-clear breakthroughs, and they can't scale. R4 needs new games.
- **Don't rely on DreamTeam solutions.** There aren't any in our copy. The engine's WIN and score signal is the verifier, so rollouts (ReST) are the way to get data from those games.
- **Don't train on the 25 public games without the held-out split.** 100% of our logs are public games. The plan's 15-train / 10-test split (`finetune-plan-2026-10-17.md` §4) is the minimum. Contamination concentrates in the 14 winnable games.
- **Test first: external games on Kaggle.** Add the `environments_dir` override and run one 25-game DreamTeam panel (≈2.3 GPU-h) to get the base pass rate (M0(c)) before committing quota to ReST.
- **Budget:** Kaggle covers about 120 game-runs or about 14k resamples a day in total. Holding back evaluation (2–3 full runs per checkpoint) leaves roughly half for data. Large-scale ReST (thousands of game-runs) needs the 8×H200 node serving; benchmark its tok/s on day 1 before planning around it.
- **Correct the context fact:** "~50–60% of tokens go to levels never cleared" should read "go to stalled levels (≥~100k tokens)". Never-cleared levels take 22–28% of tokens in the leader runs.

## Fact-check (independent verifier, 2026-10-18)

Checked against the files on `main` (92e3efb7), the run archives in `runs/kaggle/`, the archive tags, and the live web pages (fetched with curl, no browser). The recounts used new scripts, not the researcher's: they are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/fc-L1-verifier/` (`pt.py`, `clears.py`, `clears_sum.py`, `resp_rows.py`, `blank.py`, `first.py`).

| # | Claim | Verdict | Correction / note |
|---|---|---|---|
| 1 | The response row has no generated message; it repeats the request messages plus `usage` and `finish_reason`. | **Confirmed** | `tool_agent.py:9673-9687` passes `messages=latest_request_messages`, and `_append_request_snapshot` (4764-4808) has no reply field. In the ft09 mem-repair sample, 60 of 60 response rows have messages identical to the preceding request. They add only `usage` and `finish_reason`; the code also writes `served_by` when it is set, but it is absent in the sample. |
| 2 | mem-repair-10x does not set `ARC3_VERBATIM_HISTORY`, so history reasoning is normalised. | **Confirmed** | The env block at `arms.json:1223` has no such key. The flag defaults off (`tool_agent.py:2274`). The exact notebook `kaggle/notebooks/v4-mem-repair-10x-v1.ipynb` does not mention it. In the sample, 962 history reasoning messages contain 0 blank lines and 0 `<think>` tags. The normaliser also rejoins line splits (`CONFIGURATION.md:878`). |
| 3 | 2.2% of calls ≤8,192, 6.8% ≤16,384; median 57,313, p90 84,800. | **Confirmed** | An independent stream of 1,642 request files in 85 runs gives 142,572 requests and 141,483 responses with 0 parse failures. Shares: ≤8k 2.19%, ≤16k 6.76%, ≤32k 16.45%, ≤48k 37.43%, ≤64k 62.77%, ≤96k 94.81%. Median 57,313, p90 84,800, max 120,756. The first-call median is 6,233 (5,472–6,955). |
| 4 | The ms-swift Flash-Next page shows only text SFT at max_length 8192 (LoRA on in_proj/out_proj/linear_proj/linear_qkv, TP2/EP4/PP2), with 93.3 GiB/GPU at 8k under full recompute. | **Partially confirmed** | The SFT script matches: text dataset `swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000`, LoRA rank 8, those modules, TP2/EP4/PP2, `max_length 8192`. But the page also has a **GRPO** LoRA recipe (`max_length 1024`, `max_completion_length 8192`, `vllm_max_model_len 9216`), so it is not SFT only. The 93.3 GiB row is 8k with full recompute and **without** PLE offload or padding-free. The published script turns both on (`PLE_CPU_OFFLOAD=1`, `--padding_free true`), and that row is **72.5 GiB** (the script is commented "8*70G"). The measurements use synthetic equal-length samples at 2k, 4k and 8k only, and the page does not name the GPU type. |
| 5 | 85 runs: 6,034 clears (bench 5,997), 354 wins, 151 distinct (game, level) pairs, 25 public games, wins from 14 games. | **Confirmed** | The independent recount gives events (level increments + WIN) 6,034; `benchmark.json` `levels_completed` 5,997; bench wins 354 over 14 games, with the same per-game split as the notes; 151 of 183 levels; the 25 game ids equal `arc-agi-3/data/environment_files/`. Nuance: the event logs show **357** WIN game-runs (ft09, sb26 and cd82 each have one more than bench). So the 6,034 events-based clears go with 357 wins, while 354 is the bench count. |
| 6 | State-lab sampled 400 requests (637,707 tokens) in 14.9 min and 300 (735,583) in 15.5 min, 0 failures; cold start 523 s. | **Confirmed** | `lab_summary.json` matches exactly. Round 2 log line 224: "READY after 523s". The served model is `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (round-2 `serve.log`). Caveat for the "1.2–1.6k samples per GPU-hour" figure: it counts sampling minutes only. All-in, the notebooks ran 1,495 s (round 2) and 1,474 s (round 3), which is about **960 and 730 samples per GPU-hour**. Round 3's server was READY after **487 s**. |
| 7 | State-lab rebuilds logged states with full fidelity: 100/100 boards, 71/71 images, 99/99 same actions, byte-identical tool result. | **Partially confirmed** | The numbers are exact for **mem-v2-10x only** (`validate.py --per-game 4`, 100 states, README line 59). This is after fixing two mismatches the first version had in the retained-function pool. On the second run checked (`v4-pf-solved-10x`, line 99), 15/15 states and 14/14 actions match, but the tool result is byte-identical in only **13 of 14**. It has not been validated on the mem-repair-10x leader stack. `analysis/state_lab/` and `kaggle/build_lab_notebook.py` are absent from `main` and exist only on `archive/tooling/state-lab`. So "full fidelity" on every logged state is not established. |
| 8 | The DreamTeam clone has no solution files; each game has only `<id>.py` and `metadata.json`. | **Partially confirmed; the conclusion is refuted** | The file claim is true: `find -iname '*solution*'` returns 0, and the 25 game dirs hold only 25 `<id>.py` and 25 `metadata.json`. But **all 25 `<id>.py` files define `solve_level(level)`**, which returns a fixed per-level action sequence (e.g. `environment_files_generated/al7306/000afcc7/al7306.py:819`, "Return a fixed sequence of actions that solves this level"). DreamTeam's `README.md:207` says "every game's exported solution replays to a WIN through the SDK (`--verify`)". The scripted solutions are embedded in the game code; the earlier note had only the file name (`solution.json`) wrong. "DreamTeam data must come from rollouts, not scripted solutions" does not follow. (I did not execute the game code to replay them.) |

**Other errors found in these notes:**
- **DreamTeam solutions.** These lines are wrong for the same reason as row 8: Q5 "**no solutions**", the Implications bullet "Don't rely on DreamTeam solutions. There aren't any in our copy", and correction 2 at the top. Each game's `.py` ships a `solve_level`.
- **Correction 1 misattributes the phrase.** `arc-agi-v4/CLAUDE.md` does not contain "50–60% of tokens go to levels never cleared", and there is no `daniel-franzen-sol/CLAUDE.md`. The nearest source is `research/reports/finetune-plan-2026-10-17.md:98`, which already says "~50–60% of tokens go to **stuck** levels". That matches `rsi-proposals-2026-10-15.md:30` (2.4–3.2M of ~5M on stalled levels).
- **Q2: the `ARC3_VERBATIM_HISTORY` arm list omits `prefix-reuse-60`** (`arms.json:48-49`). Six arms set it, not five.
- **Q1: calls per analysis step.** The figure is 2.08 (141,483 / 68,004), but the table's own step counts sum to 63,410 (47,693 cleared + 15,717 never-cleared, as recomputed from `decisions2_joined.json`). That gives 2.23 calls per step, so the cleared-level estimate of "≈99k calls" would be about 106k. The 68,004 denominator has no stated source.
- **Q6: cold start.** "523–581 s per session (lines 224, 230)" mixes two measures. Line 230 is "581s from notebook start", not server boot. Server boot was 487–523 s (round 3 line 223, round 2 line 224).
- **Q4: the ≈81% on-route share.** It is a per-decision (action) share from an older archive sample (`graph-exploration-bandits-memory.md:220`; 2,815 of 17,477 off-route), here applied to 47.7k analysis steps. So the ≈38k on-route states is an extrapolation across units, not a count.
- **Q5: state-lab "rebuilds any logged decision exactly"** overstates the evidence; see row 7.
- **Q1: token totals.** 56 of the 1,642 game-runs have `benchmark.json` history misaligned with their event actions (`decisions2.py` pads or truncates the tokens). Per-level token totals are approximate for those 56.
- **Stale doc (not an error in these notes).** `runs/README.md:31` says "41.5 GB of raw output". The request logs alone are 65.71 GB raw (0.398 GB xz), as measured with `xz --robot --list`; the notes' figure is right and the README is stale.

**Spot checks that held:**
- 157 kernel dirs and 151 runs with a `benchmark.json`.
- 89 request-log runs; the 4 excluded are `taaf-d1-thinkecho`, `v4-gguf-final`, `v4-gguf-smoke` and `v4-q27-dflash2-10x`.
- Leader request logs: 1.130 GB raw.
- Sample: system prompt 23,100 chars; median 9 images and 3,600 image tokens per request.
- `solver_note` sum 5,256,956.
- `template.ipynb`: cell 16 sets `save_request_logs`, cell 20 has `environments_dir`.
- `_arc3_control` (`tool_agent.py:1528`), `_strip_control_keys` (2209), `seed` (`openai_compat.py:95`); no logprob or token_ids anywhere in `inference/`.
- `arms.md:578,643,644,649-655`.
- ms-swift `qwen3_8` keeps historical `<think>` and `reasoning_effort` defaults to `xhigh`.
- Agent templates `qwen3_coder` and `qwen3_5` (`mapping.py:20,37-38`).
- Custom-dataset `loss` and `loss_scale` (≥4.2.0), and `image_url`/base64 conversion.
- SGLang cookbook: NVFP4 on GB300/B300/B200 at TP1; H200 FP8 TP4/EP4 MTP 3-1-4, verified 2026-08-27; generated 2026-10-16; no tok/s figures.
- arc-interactive: 252 `metadata.json` (MIT). sonpham: 86, no licence. DreamTeam: Apache-2.0 with an NVIDIA copyright line.
