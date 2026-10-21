# G-C: training-render parity, history normalisation and server logprobs (2026-10-18)

Closes critic gaps G6 and G12 and conflict C8 (`_critic.md:82-89, 154-158, 217`).

**Method.**
- Everything ran locally under `nice -n 15`, one process at a time. No Kaggle, no GPU.
- Scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/gc/scripts/` (`parity.py`, `summ.py`, `swift_parity.py`, `jinja_backend_test.py`, `cache_scan.py`). Their outputs are in `../out/`.
- Downloads went into empty directories:
  - `gc/dl-hf-qwen`: Qwen/Qwen3.8-Flash-Next @ `de4b8e4d`.
  - `gc/dl-hf-intel`: Intel/Qwen3.8-Flash-Next-W4A16-AutoRound @ `4c67bf68`. Small files only.
  - `gc/dl-swift`: ms-swift `main` @ `80e55df9` (2026-10-16).
  - ms-swift 4.6.0 itself was pip-installed into a scratch venv.
- The SGLang source is `competitors/sglang-rtxpro6000` at HEAD `f9c4a89`. It is one commit after the Pennyroyal pin `d00d88ef` (`serving/README.md:3`). `git diff --stat d00d88ef HEAD` over serving_chat.py, qwen4_exp*.py, speculative/, logits_processor.py and the scheduler output processor is empty, so the line numbers below hold for the pinned build.

## Q1 (G6). Do our renders reproduce the server's token counts, and does ms-swift's training render match?

### Takeaway
**Parity: 200 of 200 logged requests match exactly (100%)**, both `usage.prompt_tokens` and `image_tokens`, once SGLang's own preprocessing is emulated. The only cause of mismatch is the tool-JSON serialisation, a constant −11 tokens per request if the client's tool dict is used. That explains state-lab's "within 11" replay (L1).

ms-swift 4.6.0 `qwen3_8` (swift backend) keeps historical thinking exactly as the server's `preserve_thinking=True` does. Taken as-is, though, it diverges from the served sequence in four ways. With four exporter transforms it reproduced **25 of 25 full logged windows token for token**, in both inference and training mode.

### Cited Findings
- **Template files.**
  - Qwen's and Intel's `chat_template.jinja` are byte-identical (sha256 `c3cf9e34…`).
  - The server read the template from the Kaggle copy of the Intel model, with `reasoning_parser qwen3`, `tool_call_parser qwen3_coder` and `default_chat_template_kwargs {'preserve_thinking': True}` (`runs/kaggle/v4-mem-repair-10x/v1/serve.log.xz`, server_args).
  - The two repos' `tokenizer.json` files differ in hash (`06b95093…` Intel vs `0997f410…` Qwen) but gave identical counts on 25 rows.
  - The preprocessor config is patch 16, merge 2, shortest_edge 65,536, longest_edge 16,777,216 (`gc/dl-hf-intel/preprocessor_config.json`). Each 640×640 frame is therefore 40×40 patches / 4 = **400 tokens**.
- **The template with `preserve_thinking`** renders every assistant turn as `'<think>\n' + reasoning|trim + '\n</think>\n\n' + content` (`chat_template.jinja:116-117`). With no system `reasoning_effort` it defaults to `xhigh`, which injects a ~48-token instruction (`:46-52`). Tools are rendered as `tool | tojson` inside `<tools>` (`:62-67`).
- **What SGLang does before templating** (all in `python/sglang/srt/`):
  - `messages = [msg.model_dump() …]`, with assistant `arguments` JSON-parsed to dicts (`entrypoints/openai/serving_chat.py:128-160, 1246-1249`).
  - None-valued keys are dropped (`parser/jinja_template_utils.py:149`). `image_url` parts become `{"type":"image"}` (`:178`).
  - Tools become `[item.model_dump() for item in request.tools]` (`serving_chat.py:1156`). The pydantic `Function` model has fields `description, name, parameters, strict=False` (`entrypoints/openai/protocol.py:742-756`), and `Tool` adds `defer_loading: None` (`:759-764`). So the template sees `{"type":"function","function":{"description":…,"name":"python","parameters":…,"strict":false},"defer_loading":null}`.
  - The prompt is `apply_chat_template(tokenize=False)`, then encoded (`:1402-1411`). For multimodal requests it is decoded back to text for the multimodal processor (`:1448`).
  - Images go through the HF Qwen2-VL processor with no extra resize (`multimodal/processors/qwen_vl.py:731-785`).
- **Parity script** (`parity.py`): it emulates the steps above, strips `_arc3_control` as the harness does (`tool_agent.py:2209-2219, 7561`) and expands each `<|image_pad|>` to (h/32)·(w/32) tokens.

  | Sample | Rows | prompt_tokens exact | image_tokens exact | Encode→decode→encode identical |
  |---|---|---|---|---|
  | `v4-mem-repair-10x/v1`, 4 per game | 100 | 100/100 | 100/100 | 100/100 |
  | `v4-mem-unified-10x/v1`, 2 per game | 50 | 50/50 | 50/50 | 50/50 |
  | mem-repair with Qwen's tokenizer.json | 25 | 25/25 | 25/25 | 25/25 |
  | `v4-tput-safe/v1` (verbatim history) | 25 | 25/25 | 25/25 | 25/25 |

  - Prompts in the 100-row set range from 7,671 to 83,336 tokens (median 56,358), with a median of 13.5 images (max 33).
  - 91 of the 100 rows contain `_arc3_control` messages.
  - Every image is 640×640.
  - With the client's tool dict instead (key order name, description, parameters; no `strict` or `defer_loading`), all 200 rows come out **exactly 11 tokens short** (`out/parity_*.json`, `summ.py`).
- **ms-swift 4.6.0 `qwen3_8`** (source @ `80e55df9`, installed 4.6.0):
  - It is registered with `preserve_thinking=True` and `agent_template='qwen3_5'` (`swift/template/templates/qwen.py:719-729`).
  - In the swift backend, `_get_preserve_thinking` returns True, so `_remove_history_thinking` is skipped in training too (`swift/template/base.py:173-190, 1408-1410`). The `qwen3_8` docs say historical thinking is kept ([Qwen3_8-Best-Practice.md](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_8-Best-Practice.md), line 11).
  - Divergences found with `swift_parity.py`, which compares the swift encode against the HF-jinja ids that matched the server:
    1. **`enable_thinking` defaults to False** for `qwen3_8`, because `non_thinking_prefix` is set (`base.py:133`). This drops the xhigh reasoning instruction, so `--enable_thinking true` is required.
    2. **Null keys are stripped from tool schemas** by `remove_arrow_padding` (`base.py:344`; `swift/utils/schema_utils.py:18-40`). That loses `"defer_loading": null` (−6 tokens).
    3. **Consecutive user messages are concatenated with no turn break** (`base.py:1398`; −6 tokens). The leader sends a memory-head user message followed by the opener ("suua…" in 20 of 25 rows).
    4. **A tool result followed by a user message raises** `AssertionError: response_role: "user"` (`base.py:1449`). This adjacency is in 25 of 25 rows, a median of 11 times per request, because every turn boundary and yield-resume produces it.
  - **The `qwen3_coder` agent template does not match.** Its `_format_tools` writes XML `<function><name>…` descriptions (`swift/agent_template/qwen3_coder.py:75-130`). Measured on 3 rows: +16 to +19 tokens, first divergence at token 41, the start of the tools block. The default `qwen3_5` agent template writes `json.dumps` tools and matches (`:178-212`).
  - **The jinja backend trains on everything.** `_jinja_encode` returns `[text], [1.], answer_len=1` (`base.py:1285-1286`). Toy test (`jinja_backend_test.py`): 85 of 86 tokens are trained, including system and user, versus 22 (default) or 11 (`last_round`) in the swift backend. The rendered strings are identical.
  - **Result with four transforms** (`--enable_thinking true`; tools moved into system text; user messages joined with a literal `<|im_end|>\n<|im_start|>user\n`; each tool group that is followed by a user message turned into a user message of literal `<tool_response>\n…\n</tool_response>` blocks): **25 of 25 full requests are id-identical in inference and in training mode.**
    - These requests have a median of 18 assistant turns.
    - The jinja count equals the server's `prompt_tokens` on all 25.
    - The number of label runs equals the number of assistant turns on all 25 (loss on every turn under `loss_scale default`).
    - 788,888 of 1,400,416 training-mode tokens (56%) are assistant tokens (`out/swift_full.json`).
  - Per-response `loss`/`loss_scale` keys are honoured (`swift/loss_scale/base.py:98-99`).

### Inferences
- The logged rows are a complete, exact record of what the model was fed. Count parity on 200 long multimodal prompts makes token-level divergence very unlikely, but it was not checked id by id against server ids (see Gaps).
- Historical thinking is not the issue: ms-swift preserves it as the server does. The issues are the harness's turn structure (user→user, tool→user) and two serialisation details. All of them are fixable in the exporter without patching ms-swift.
- The 2 tokens `<think>\n` at the start of each trained turn are part of the server's generation prompt, so the model never generates them. Training on them is harmless. `ignore_think_prefix` exists if wanted (`swift/loss_scale/mapping.py:10`).

### Gaps
- Server token **ids** were never logged, so parity is checked on counts, not ids. The smoke test in Q4 closes this with `return_prompt_token_ids`.
- Whether the Kaggle copy of the Intel model's `tokenizer.json` equals the HF Intel file is not checked. Counts match, so any difference is immaterial.
- Rare shapes were not seen in the 200 rows: consecutive assistant messages (the stub path, where swift merges them as list content) and video. The exporter must assert parity on every row.
- Megatron-swift rendering is assumed to use the same template classes. UNVERIFIED.

## Q2. How history is normalised (ARC3_VERBATIM_HISTORY off), how yields and `_arc3_control` show up, and what that means for packing

### Takeaway
On the leader, every stored assistant turn is the normaliser's output, and the raw reply is not stored anywhere. Packed and final-turn-only training therefore both learn normalised targets, and each packed turn's context equals what the server saw for that turn on 93.8% of requests. The C8 conflict ("context normalised, target generated") is moot for existing logs. It only matters for future verbatim, logprob-logged data, where packing becomes exactly on-policy.

### Cited Findings
- **What is stored.**
  - History stores `history_reasoning, history_content = reasoning, content` (the normalised text) unless the flag is on (`tool_agent.py:9742-9748`).
  - The normaliser drops whitespace-only lines and literal `<think>` tags, and rejoins line splits with `\n` (`_normalize_message_content`, `tool_agent.py:4856-4875`; `CONFIGURATION.md:878`).
  - Tool-call markup found in text is stripped (`:9737-9738`).
  - Reasoning goes under `ARC3_REASONING_HISTORY_KEY=reasoning_content`, set in the leader notebook (`kaggle/notebooks/v4-mem-repair-10x-v1.ipynb`, cell 4); the code default is `reasoning` (`tool_agent.py:2243-2259`).
  - Measured: 0 of 122 history reasoning messages carry a blank line in mem-repair, against 95 of 96 in verbatim `v4-tput-safe` (first 5 games, last response row each).
- **Yields.**
  - Leader cell 4 sets `LOCAL_ANALYZER_YIELD_TOKENS=2048` and `YIELD_SECONDS=0`.
  - The budget is checked between completions (`tool_agent.py:9582-9602`; `CONFIGURATION.md:853`), so a yield never truncates a reply. The turn ends after the completion that crossed 2,048 tokens, and that completion's tool result stays in history.
  - The next turn opens with a full rebuilt opener (`ARC3_YIELD_RESUME_PROMPT` is unset, so `full`; `tool_agent.py:1639-1656`), tagged `_arc3_control: "resume"` (`:9484-9492`). `ARC3_OPENER_DEDUP=1` then removes duplicated static and image parts (`CONFIGURATION.md:857`).
  - Leader: **625 of 2,624 responses (23.8%) are the first request of a resumed turn**. Other control kinds seen are `stub` and `nudge` (tput-safe, which has `STUB_DEAD_REASONING`). Source: `cache_scan.py`.
  - The tag is stripped on the wire (`tool_agent.py:2209-2219, 7561`). It stays in the logs and must be stripped by the exporter, as parity confirms.
- **Append structure.** 2,462 of 2,624 leader responses (93.8%) extend the previous request's messages exactly, with the reply inserted (`cache_scan.py`). L1 counts 163 windows per run with a median of 15 requests and 11.5M tokens of final windows, against 145.3M prompt tokens over all requests (this scan).

### Inferences
- **Segment packing** (one forward per window, loss on every assistant turn) costs about 11.5M tokens per run against 145M for final-turn-only. That is about **12.6× (92%) less**, not 98%. On the 93.8% of append requests, each turn's context equals that turn's served prompt exactly.
- **What packing trains.** The targets are normalised replies, and so are final-turn-only targets. The model learns to write reasoning without blank lines. Under normalised history this is self-consistent: the history normaliser then becomes nearly a no-op, which should also lift prefix reuse.
- **Window boundaries.** Trims, memory folds and the 6.2% of non-append requests start new windows. The exporter must cut there, as L1's `windows.py` does.
- **RL/IS and SDPO** need the raw reply tokens. Existing logs cannot supply them under either option.

### Gaps
- How much the normalised target distribution itself (no blank lines) changes behaviour is unmeasured.

## Q3. The ARC3_VERBATIM_HISTORY arms vs matched controls

### Takeaway
Six arms set the flag, but only `v4-tput-safe` ever ran: 42.09 against same-batch `v4-control` 50.62. That run bundles four other levers and lost through queue wait, so **there is no clean score test**. The mechanism is measured: verbatim history roughly doubles cache reach into the previous reply. On theory, switching is plausibly score-neutral or slightly positive, but per the alpha rule it is unproven.

### Cited Findings
- The arms that set it are `prefix-reuse`, `prefix-reuse-60`, `tput-safe`, `tput-safe-lowm`, `tput-accept05` and `panel-tput-safe` (`kaggle/arms.json`, `arms` block). The leader `mem-repair-10x` does not.
- `runs/index.csv` has only `v4-tput-safe` v1, scored 42.09, among them (`index.csv:158`).
- **The one comparison, from `arms.md:544`:**
  - `v4-tput-safe` 42.09 (6 wins, 591 tok/s) against `v4-control` 50.62 (7 wins, 602 tok/s), same batch `519c4f5`.
  - P(better) 0.03, paired t p = 0.074.
  - tput-safe also turned on gate11, ledger-on-trim, stub-dead-reasoning and extra graphs.
  - The loss was traced to queue wait (66.6 vs 7.1 min) from gate11 exceeding the server's slots.
- **Mechanism** (`cache_scan.py`; share of append requests whose cache reached past the previous prompt end into the reply):

  | Run | History | Reached into reply | Reply tokens re-prefilled | Share of uncached input |
  |---|---|---|---|---|
  | mem-repair | normalised | 44.3% | ≈2.79M | 17.7% of 15.78M uncached |
  | v4-control | normalised | 43.4% | not computed | — |
  | tput-safe | verbatim | 90.9% | ≈0.84M | 6.6% of 12.65M |

  - tput-safe's own `PREFIXDIAG totals` give `cache_reply_end=1766` of 2,000 requests (`v4-tput-safe.log.xz`).
  - The overall cache share was 0.891 (leader), 0.939 (control) and 0.911 (tput-safe). Slot pressure dominates the overall figure.
- What changes for the model: blank lines and paragraph breaks come back in retained reasoning and content; nothing else changes (`CONFIGURATION.md:880-886`).

### Inferences
- **Compute.** Verbatim history saves about 2M prefill tokens per 25-game run on the leader configuration. That is small: the reply re-prefill is about 1.9% of prompt tokens.
- **Behaviour.** The text the model sees becomes its own distribution, which is closer to how the model was trained. Retained history grows slightly, so trims come marginally earlier.
- **Neutrality and evidence.** No mechanism predicts a large loss. Still, "neutral" has to be shown with paired runs, and the user's rule forbids an exact repeat. So stack it as a flag in the next combined leader run, ideally alongside raw-reply and logprob logging. Collection runs need it regardless of score.

### Gaps
- No isolated run of the flag exists. `prefix-reuse`, `prefix-reuse-60` and `-raw` never ran.

## Q4 (G12). Input and top-k logprobs on Pennyroyal for `qwen4_exp` under MTP

### Takeaway
The code paths exist and nothing in the build blocks them. Everything rests on source reading; it has not been run.
- **Input logprobs** come only from native `/generate` (`return_logprob`, `logprob_start_len`, `top_logprobs_num`, `token_ids_logprob`; `max_new_tokens=0` allowed). The target prefill under EAGLE v2 is an ordinary target forward. The values are T=1 log-softmax.
- **The chat endpoint** gives output logprobs only (`logprob_start_len` is hard-coded to −1), plus `return_token_ids` and `return_prompt_token_ids` when not streaming. Under speculative decoding, output logprobs are temperature-scaled unless `SGLANG_RETURN_ORIGINAL_LOGPROB=1`, and are never top-p/top-k truncated.

### Cited Findings
- **Serving setup.** The leader serves with `speculative_algorithm EAGLE`, steps 3, topk 1, 4 draft tokens, overlap schedule on (spec v2), `speculative_token_map hot_tokens_64k`, `kv_cache_dtype fp8_e4m3`, `mamba_radix_cache_strategy extra_buffer`, `mamba_track_interval 64` and `enable_fp32_lm_head False` (`serve.log.xz`, server_args).
- **Chat endpoint.** `return_logprob=request.logprobs, logprob_start_len=-1, top_logprobs_num=request.top_logprobs` (`serving_chat.py:1061-1066`). `return_token_ids` and `return_prompt_token_ids` (`protocol.py:890-891`) are rejected with streaming (`serving_chat.py:986-995`). The harness posts with `stream=false` (`openai_compat.py:52-57`).
- **Native API.** Fields at `managers/io_struct.py:171-226`; `max_new_tokens ≥ 0` (`sampling/sampling_params.py:184-186`); meta keys `input_token_logprobs`, `input_top_logprobs` and `spec_verify_ct` (`managers/tokenizer_manager.py:2594-2672, 2861`). Upstream documents `logprob_start_len` with default −1, meaning output tokens only ([sampling_params.mdx](https://github.com/sgl-project/sglang/blob/main/docs/docs/basic_usage/sampling_params.mdx)).
- **Speculative path.**
  - Target prefill is `target_worker.forward_batch_generation(batch, capture_hidden_mode=FULL)` (`speculative/eagle_worker_v2.py:1274-1286`). The model's forward ends in the standard `logits_processor` (`models/qwen3_vl.py:1501-1509`); `Qwen4Exp` only swaps in the hyper-connection hidden states for the drafter (`models/qwen4_exp.py:1952-1958`).
  - Verify computes accepted-token logprobs, top-k and token-id logprobs via `compute_spec_logprobs` (`speculative/eagle_worker_common.py:626-627`; `layers/logprob_processor.py:352-410`), using `log_softmax(logits / T)` unless greedy or `SGLANG_RETURN_ORIGINAL_LOGPROB` (`:383-392`).
  - The draft's logprobs are switched off (`eagle_worker_v2.py:951`).
  - Input logprobs use plain log-softmax or the fused logsumexp path, with no temperature (`logprob_processor.py:437-460, 590-620`; `SGLANG_ENABLE_FAST_INPUT_LOGPROBS` defaults on, `environ.py:1148`).
- **Prefix cache.** With `logprob_start_len=s`, prefix reuse is capped at s (`managers/schedule_batch.py:1426-1431`).
- **Documented limits.** Upstream's only listed "returned logprobs not supported" limit is for the UNO algorithm, not EAGLE ([speculative_decoding.mdx](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/speculative_decoding.mdx), line 547). The fork's `LIMITATIONS.md` does not mention logprobs. None of our serving patches touch logprobs (grep of `serving/`).
- **The LM head is BF16** in the leader run: the online-FP8 head is gated on `SGLANG_SM120_ONLINE_MXFP8` (`models/qwen4_exp.py:1997-2025`), and its "rowwise FP8 weights ready" log line is absent from serve.log.

### Smoke-test spec (one Kaggle session, about 15 min after boot; not run)
1. **Boot.** Boot the leader server unchanged, adding only `SGLANG_RETURN_ORIGINAL_LOGPROB=1`. Pick one logged leader request of 8k–20k tokens that has at least 2 images (from `out/parity_mem_repair.json`), and ship its row plus our offline jinja-rendered text and expanded ids.
2. **Request A.** `POST /v1/chat/completions` with the row's messages (strip `_arc3_control`), tools, `chat_template_kwargs`, `temperature 0.65`, `top_p 0.95`, `max_tokens 512`, `logprobs true`, `top_logprobs 5`, `return_prompt_token_ids true`, `return_token_ids true` and `stream false`.
   - Pass if `prompt_token_ids` equals our offline ids **element-wise**. If image positions carry pad hashes, compare only the non-image positions.
   - Pass if len(output ids) = `completion_tokens` = number of logprob entries, all finite, with reasoning tokens included.
   - Record `spec_verify_ct`.
3. **Request B.** `POST /generate` with `text` = A's rendered prompt plus the decoded output, `image_data` = the same data URIs, `sampling_params {max_new_tokens: 0, temperature: 1.0}`, `return_logprob true`, `logprob_start_len = len(prompt_token_ids)` and `top_logprobs_num 20`.
   - Pass if `input_token_logprobs` covers exactly the output positions with the same token ids.
   - Report the mean and max |lp_B − lp_A| and the mean of (lp_A − lp_B), a k1 KL estimate. Expect a small nonzero gap from batch composition, FP8 KV and verify-versus-prefill kernels. The W2 gate is ≤ 0.003–0.004.
   - Record the wall time, and the `cached_tokens` that show how much was re-prefilled.
4. **Request C.** As B but `logprob_start_len = 0` and `top_logprobs_num 0`. This checks that the scoring cost of a full window is one prefill, and that the response size is manageable. With top-20 at 80k positions the response would be about 1.6M entries, so use chunked retrieval or `token_ids_logprob` instead.
5. **Fail modes to watch:** a 500 error or assertion in spec v2 with `return_logprob`, NaN, length mismatches, or a Mamba prefix match falling back to 0.

### Inferences
- **Online uses.** Prefill-only scoring (`/generate`, `max_new_tokens 0`) is the right primitive for W10's teacher logits, for W2's trainer↔server KL gate (T=1 input logprobs), and for after-the-fact behaviour logprobs.
- **Output logprobs during play** need `SGLANG_RETURN_ORIGINAL_LOGPROB=1`. They come from the untruncated distribution, while sampling uses top-p 0.95, so IS ratios must account for the truncation.

### Gaps
- No run on this build, the wheel or `qwen4_exp`. Mamba checkpoint alignment at an arbitrary `logprob_start_len` is UNVERIFIED and may force a longer prefill.
- Whether `prompt_token_ids` for multimodal requests holds `image_pad` ids or pad hashes is UNVERIFIED (it is captured after tokenisation, `tokenizer_manager.py:826-827`).

## Implications for our plan
1. **G6 is closed for SFT data.**
   - Build the exporter on `parity.py`'s rules: SGLang tool dump, None-drop, args→dict, `_arc3_control` strip, 400-token images.
   - Add the four ms-swift transforms: `--enable_thinking true`; the tools block embedded in system text; joined user messages; tool→user groups as literal `<tool_response>` user text.
   - Keep the default `qwen3_5` agent template. **Do not use `--agent_template qwen3_coder`**, and do not use the jinja backend.
   - Assert, per row, that the count equals the logged `prompt_tokens` and that the swift ids equal the jinja ids.
2. **Segment packing is valid on existing logs within append windows.** It costs about 12.6× less than final-turn-only, with loss on every turn plus per-turn `loss:false` masking (for example on never-cleared levels or stubs). Final-turn-only buys nothing in fidelity, because no raw replies exist either way.
3. **Turn on `ARC3_VERBATIM_HISTORY=1`, raw-reply logging and logprob logging** as stacked flags in the next combined leader run. The single confounded run shows no clean loss; the mechanism saves about 2M prefill tokens per run; and future data then becomes exactly on-policy for RL and SDPO.
4. **G12 → spend one cheap Kaggle smoke** (the spec above) before committing to SDPO, KL-gated RL or IS-RL. The code supports it.

## Fact-check (independent verifier, 2026-10-18)

Method: every key result was re-checked. Cheap computations were re-run with new code or on rows the original scripts did not sample, all under `nice -n 15`. Scratch files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/gcv/` (`verify_parity.py`, `swift_parity_v.py` (the same script re-seeded to `swift-verify:`), `blank_scan.py`, `append_scan.py`). Cited source lines were opened in `competitors/sglang-rtxpro6000`, in ms-swift (installed 4.6.0 and the `80e55df9` copy; `base.py`, `qwen.py`, `qwen3_coder.py` and the loss_scale files are byte-identical between the two) and in the harness. Both HF `chat_template.jinja` files were downloaded again into a new empty directory. Nothing was pushed or submitted.

| # | Result | Verdict | Notes |
|---|---|---|---|
| 1 | Prompt-token parity 200/200 | **Confirmed** | `summ.py` on the four `parity_*.json` files reproduces every figure: 100/50/25/25 exact, all round trips equal, 7,671–83,336 tokens (median 56,358), median 13.5 images, every image 640×640. An independent re-implementation on 150 new rows (the median and last response row of each game) also matched: mem-repair 50/50, mem-unified 50/50, and tput-safe 50/50 with Qwen's tokenizer.json. Prompt and image tokens were exact and the client-dict variant was −11 on every row. |
| 2 | Tool JSON is the only mismatch (−11) | **Confirmed** | `serving_chat.py:1156` dumps `item.model_dump()`. `protocol.py:742-770`: `Function` also has a `defer_loading` field, but its serializer pops it when None, so only the `Tool`-level `defer_loading: null` survives. −11 on all 150 new rows. That this explains state-lab's "within 11" (an OpenRouter replay) is an inference, though a plausible one. |
| 3 | Template and tokenizer identity, preprocessor | **Confirmed** | Fresh HF downloads: both templates are sha256 `c3cf9e34…`. The Qwen API sha is `de4b8e4d43b9…`; Intel's `4c67bf68` comes from the saved `_api.json`. The tokenizer.json hashes differ (`06b95093…` and `0997f410…`), but counts match on the 50 new tput-safe rows run with Qwen's tokenizer. The preprocessor is patch 16, merge 2, 65,536 / 16,777,216. |
| 4 | ms-swift `qwen3_8` keeps historical thinking | **Confirmed** | `qwen.py:719-728` and `base.py:173-190, 1408-1410`. One caveat: an explicit `preserve_thinking=False` (chat_template_kwargs or template arg) still overrides the template meta. |
| 5 | Four ms-swift divergences | **Confirmed** | (1) `get_template(..., 'qwen3_8')` with no argument gives `enable_thinking=False`. With it off, the render has no xhigh instruction (net −36 on one row). (2) and (3) decomposed on 4 rows: plain gave −12 on `suu…` rows and −6 on `sua…` rows; with `defer_loading` dropped it gave −6 and 0. (4) `FULL=1` with no fixes raises `AssertionError: response_role: "user"`. The "25 of 25 rows" figure depends on the sample: in my re-seeded 25 rows, 24 have tool→user (median 11) and 18 start `suu`. The exception is an early 4-message row. |
| 6 | Exporter fixes give id-identical renders | **Confirmed** | 25 new rows (re-seeded): inference and training both 25/25 id-identical, jinja = server `prompt_tokens` on 25/25, label runs = assistant turns on 25/25. 805,748 of 1,440,882 tokens (55.9%) are trained. The median is 16 assistant turns (18 in the original sample). |
| 7 | qwen3_coder +16 to +19, jinja backend trains 85/86 | **Partially confirmed** | On 4 new rows qwen3_coder gave **+15**, +19, +15 and +17, so across 7 rows the range is +15 to +19. The first divergence is at token 41 on all of them. The default `qwen3_5` template gives −6 there (`defer_loading` only). The jinja toy test reproduces 85/86 against 22 (default) and 11 (`last_round`). Line refs: `Qwen3_5AgentTemplate` starts at `qwen3_coder.py:178` and its json.dumps `_format_tools` is at :197-198. |
| 8 | Leader history is normalised; raw reply not logged | **Confirmed** | `tool_agent.py:4856-4875, 9729-9748, 9814` and `CONFIGURATION.md:878`. Notebook cell 4 sets `ARC3_REASONING_HISTORY_KEY=reasoning_content`. New scan: 0/122 against 95/96 on the first 5 games, and **0/468 against 486/498 over all 25**. The `response` log event (`tool_agent.py:9673-9687`) carries no reply text. "Line splits rejoined" means split on any line boundary and joined with `\n`; it does not mean lines are merged. |
| 9 | Yields, resume share, append share | **Confirmed** | The yield check runs at the top of each completion loop (`tool_agent.py:9582-9602`), the opener mode defaults to `full` (`:1639-1656`) and the `resume` tag is at `:9484-9492`. An independent scan (raw dict equality, not md5) gives 625/2,624 (23.82%) last-message `resume` and 2,462/2,624 (93.83%) appends. Caveat: the scan checks only that the next message is an assistant message, since the reply itself is not logged, so "with the reply inserted" is assumed rather than checked. |
| 10 | Six verbatim arms, only tput-safe ran | **Confirmed** | The six arms in `arms.json` match, and all six also set `ARC3_PREFIX_REUSE_DIAG`. A scan of all 221 non-serve run logs finds `PREFIXDIAG` only in `v4-tput-safe/v1`. `index.csv:157-158` gives 50.62 and 42.09. `arms.md:544` gives P(better) 0.03, paired t p = 0.074 and queue wait 66.6 vs 7.1 min. The extra levers are MAX_ACTIVE_STREAMS 11, LEDGER_ON_TRIM, STUB_DEAD_REASONING and CUDAGRAPH_BS_EXTRA [3,5,6]. |
| 11 | Cache reach into reply 44.3/43.4 → 90.9% | **Confirmed** | The independent scan reproduces 1,091/2,462, 976/2,248 and 1,768/1,945, plus 2,793,446 tokens (17.71% of 15,775,749) against 837,634 (6.62% of 12,654,120). The result holds at other thresholds: with cached > prev prompt it is 44.9 / 43.7 / 92.1%, and at ≥ +256 it is 38.3 / 37.0 / 83.2%. The log has `cache_reply_end=1766 … requests=2000`, in the last totals line, at 2,000 of 2,044 responses. Additions: the v4-control re-prefill estimate can be computed and is about **2.54M (24.0% of 10.59M uncached)**. The runs differ in request count, so per append request the re-prefill is about 1,135 (leader) vs 431 (tput-safe) tokens. |
| 12 | Chat vs /generate logprobs | **Partially confirmed** | Every cited line holds: `serving_chat.py:985-996, 1061-1063`, `protocol.py:858-859, 890-891`, `io_struct.py` fields, `sampling_params.py:183-186` and `schedule_batch.py:1426-1431`. Correction to the key-result wording: **output top-k logprobs are available on the chat endpoint** (`top_logprobs_num=request.top_logprobs`, `serving_chat.py:1063`). Only input (prompt) logprobs, input top-k and `token_ids_logprob` need `/generate`. The body text and smoke-test spec already treat it that way. |
| 13 | EAGLE v2 logprob semantics | **Confirmed (source only)** | `spec_info.py:287-299`: EAGLE always selects `EAGLEWorkerV2`, and the leader has `enable_multi_layer_eagle False`. The other cited lines also hold: `eagle_worker_v2.py:1274-1286, 951`, `eagle_worker_common.py:626-627`, `logprob_processor.py:383-392` (log_softmax(logits/T) unless greedy or `SGLANG_RETURN_ORIGINAL_LOGPROB`, no top-p), `environ.py:1135, 1148` and `schedule_batch.py:1426-1431`. The docs check (`speculative_decoding.mdx:547`, UNO only) used the earlier downloaded copy in `gc/dl-sglang-docs`, not a fresh fetch. Not run on the build. |

Other errors found:
- `chat_template.jinja:46-52`: the xhigh instruction is **37 tokens (38 with the `\n\n` separator)**, not "~48".
- Q1 "None-valued keys are dropped (`jinja_template_utils.py:149`)": line 149 covers string content only. List content drops None at `:214-216`. The behaviour is the same either way.

Skipped (not load-bearing): the `qwen_vl.py:731-785` processor path, the `tokenizer_manager.py` meta-key lines, the `qwen4_exp.py:1997-2025` BF16 LM-head gating (I saw the gate exists but did not check the log), and the Megatron-swift assumption.
