# wf_57f4538b-24c: research:server-contract

Workflow run `wf_57f4538b-24c`, phase Research. 

---

**Verdict: no proxy is needed.** The harness talks to the server through one endpoint, `POST /v1/chat/completions`, plus the notebook's `/health` poll. It never calls `/model_info`, `/get_server_info`, `/flush_cache`, `/freeze_gc`, `/metrics` or `/tokenize`. It never sends `priority`, and it uses no tokenizer.

Swapping to llama-server needs three things:
- **Server flags (required):** `--mmproj`, `-np 10` and a 10x `-c`. Add `--min-p 0` to match SGLang's sampling.
- **A small harness patch (strongly recommended):** without it, a too-long prompt makes a game give up instead of trimming history.
- **Cell 14 replaced:** keep its `/health` wait loop.

F001 does not depend on anything server-specific.

I checked the client code against the exact harness the notebook runs. The worktree at `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/model-profiles/ARC3-Inference` matches the post-patch state of the notebook's cell-2 harness patch: `git apply -R --check` of that patch passes, and the branch has no `ARC3-Inference` diff against `main`. Line numbers are before F001's runtime edits; F001 shifts lines below `_build_user_prompt`.

The llama-server side was checked in the b11476 source at `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/runner-research/llama.cpp-b11476`. The GGUF's embedded chat template is identical to `runner-research/chat_template.jinja` (checked against the shard-1 header).

How to read the references:
- **"tool_agent", "openai_compat", "solver", "vision_context"** are files under `ARC3-Inference/inference/`.
- **"cellN:L"** is line L inside code cell N of `push-fable-f001/v4-fable-f001.ipynb`. Extracted copies are in `scratchpad/ls-contract/`.
- **"tmpl:L"** is line L of the GGUF template.

## Adaptation table

| # | Dependency | Where (file:line) | llama-server b11476 | Change needed |
|---|---|---|---|---|
| 1 | `POST {base}/chat/completions`, base `http://127.0.0.1:8001/v1` | tool_agent:5265-5278; tool_agent:80, 3023-3041; cell4:76-78, 88-89 | `server.cpp:273-274` (both `/chat/completions` and `/v1/chat/completions`) | None. Keep port 8001. |
| 2 | `GET /health` readiness (status 200) | cell14:350-395 (url at 374) | `server.cpp:263`. Returns 503 "Loading model" while loading (`server-http.cpp:308-320`), then 200 `{"status":"ok"}` | None. Reuse the cell14:322-395 wait loop as is. |
| 3 | `/model_info`, `/get_server_info`, `/flush_cache`, `/freeze_gc`, `/metrics`, `/v1/models`, `/tokenize` | Not on the run path. Only `inference/tools/chat.py:44` (a dev CLI) and `framework/kaggle.py:290` (Tufa's vLLM setup, which this notebook never runs) | n/a | None |
| 4 | `model: "flashnext"` | cell4:25, 92-93 | Ignored in single-model mode (`server-context.cpp:4680`) | Optional `--alias flashnext` |
| 5 | Message shapes: assistant `content: None` (tool_agent:6916); assistant with only `tool_calls` (7046-7052); `role: tool` with `tool_call_id` (7085-7091) | as listed | Accepted (`server-common.cpp:1306-1324`; `common/chat.cpp:373-460`) | None |
| 6 | Assistant `tool_calls[].function.arguments` re-sent in history as a JSON string | tool_agent:7051 | The GGUF template raises on string arguments (tmpl:148-151). llama.cpp parses them into objects first for this template (`chat.cpp:1041-1057, 1303-1305`). A malformed string would make every later request fail with 400. | None. The harness already keeps malformed calls out of history (tool_agent:6855-6887). |
| 7 | Previous reasoning sent back under `reasoning_content` | env `ARC3_REASONING_HISTORY_KEY` (cell4:97); tool_agent:1254-1272, 6906-6908 | Read at `chat.cpp:453-455`; the template renders it (tmpl:114-120) | None. Keep `reasoning_content`. |
| 8 | `image_url` parts as `data:image/png;base64` in user messages | vision_context:89-101, 232, 282, 328-333; tool_agent:4661-4694 | Accepted only with an mmproj (`server-common.cpp:1165-1208`). Without one: runtime_error, HTTP 500, which the harness retries, so every turn fails. | **Required:** `--mmproj mmproj-Qwen3.8-Flash-Next-BF16.gguf` |
| 9 | `tools` (one `python` function, tool_agent:5176-5198) and `tool_choice: "auto"` (2502-2513) | as listed | Needs jinja, which is on by default (`common.h:644`); uses a lazy tool grammar (`server-common.cpp:1268-1281, 1391-1396`) | None |
| 10 | `chat_template_kwargs`: `enable_thinking` (openai_compat:94), `preserve_thinking` (tool_agent:1317-1330, 5200-5216, 5247-5258), `reasoning_effort` only if `ARC3_REASONING_EFFORT_LADDER` is set (1301-1315; F001 doesn't set it) | as listed | Request kwargs merge with the CLI ones (`server-common.cpp:1398-1412`). The template reads all three (tmpl:58-71, 119, 178-182) and raises on any effort other than xhigh, high, medium or low. | None. Optional `--chat-template-kwargs '{"preserve_thinking":true}'` to mirror SGLang's server-side default. |
| 11 | Sampling: temperature 0.7, top_p 0.95, top_k 20 (cell4:103-105; openai_compat:54-60, 91-93). `seed` only if `LOCAL_ANALYZER_SEED` ≥ 0 (unset). | as listed | All accepted, but llama.cpp's default `min_p` is 0.05 (`common.h:235`); SGLang's is 0 | **Add `--min-p 0`** for a clean A/B |
| 12 | `max_tokens` 12288 | cell4:109; tool_agent:3497 | Maps to n_predict. Running out of context gives `finish_reason: length` (`server-context.cpp:2018-2025`) | None |
| 13 | `stream: false` (`stream_options` only sent when streaming) | openai_compat:57, 72-73 | Normal non-streaming response, with usage | None |
| 14 | `priority` | openai_compat:63-65 supports it, but `_chat_completion` never passes it. Scheduling is the client-side `_PriorityGate` (tool_agent:1976, 2128-2145), with `ARC3_MAX_ACTIVE_STREAMS=10` (cell4:153) | llama.cpp has no request priority | None |
| 15 | `Authorization` header, only sent if an API-key env is set | tool_agent:3589-3603; openai_compat:18-35 | No key set; don't pass `--api-key` | None |
| 16 | `LOCAL_ANALYZER_PROVIDER=vllm` decides whether `top_k` and `chat_template_kwargs` are sent | cell4:90-91; openai_compat:91-96 | n/a | Keep `vllm`. Any other value silently drops both fields. |
| 17 | Response `message.reasoning`, falling back to `reasoning_content` | tool_agent:3431-3435, 6842 | Returns `reasoning_content` with reasoning-format auto/deepseek (`common.h:426-428, 655`; `chat.cpp:233-234`) | None (never use `--reasoning-format none`) |
| 18 | Response `tool_calls[]` (id, name, arguments as string) | tool_agent:6844-6887, 7051, 7089 | Returned; ids are generated (`server-task.cpp:175`). Issue #30078 (closed today): calls sometimes leak into content. The XML form is already recovered by the harness (tool_agent:85-91, 140-171, 6843-6851). The JSON form `<tool_call>{"name":…,"arguments":…}</tool_call>` is not, and becomes a wasted "you did not call a tool" turn (7001-7004). I found no #30078 fix in b11476's `common/chat*.cpp`. | Optional: add JSON-form recovery to the harness, or lift it in a proxy |
| 19 | `finish_reason` length / stop / tool_calls | tool_agent:6828-6843, 6887 | `server-task.cpp:415-428` | None |
| 20 | `usage.completion_tokens` feeds generated-token totals, the 2048-token yield budget, priority token scale, F001's thresholds and the run's "tokens" | tool_agent:3632-3675 | `server-task.cpp:365-372` (counts all decoded tokens, reasoning included) | None |
| 21 | `usage.prompt_tokens` calibrates the chars-per-token estimate, after subtracting 402 tokens per image | tool_agent:2856-2971, 5962-6008, 6778 | `prompt_tokens` includes image tokens (`server-context.cpp:2200`). A 640×640 image is 20×20 merged patches plus vision_start/end = 402, within the 8-4096 token limit (`clip.cpp:1668-1680`; `mtmd.cpp:696-702`) | None. Don't pass `--image-min-tokens` (it would upscale images); otherwise set `ARC3_IMAGE_TOKENS_FLAT`. |
| 22 | `usage.total_tokens` | tool_agent:3660-3666 | Provided | None |
| 23 | `prompt_tokens_details.cached_tokens` | Never read by the harness; only `serving/kaggle/bench.py:500-502` | Provided (`server-task.cpp:370`) | None |
| 24 | Context-overflow detection: `_is_context_length_error` matches only vLLM and SGLang wording | tool_agent:3438-3449, 6801-6826 | Returns HTTP 400 "request (N tokens) exceeds the available context size (M tokens)…", type `exceed_context_size_error` (`server-context.cpp:3418-3425`; `server-common.cpp:68-70`). The harness would treat it as a generic failure and resend the same prompt; after 10 failures in a row the game gives up (solver:105, 691-700). | **Strongly recommended:** add "exceeds the available context size", "exceed_context_size_error" and "is larger than the max context size" to the matcher with a one-anchor notebook patch (same method as F001's `apply()`), or rewrite the error in a proxy. Also keep per-slot context ≥ 139264. |
| 25 | Retries and timeouts: retryable statuses (tool_agent:521); 900 s first-request grace (cell4:18; tool_agent:557-662); 10 s connect timeout (526); 900 s read timeout from `analyzer_timeout` (cell18:3, solver:1834-1836, tool_agent:5272-5276); a read timeout yields with history kept (856-877, 7108) | as listed | The HTTP server is up during loading and answers 503, which is retryable. Default server timeout is 3600 s (`common.h:628`). A non-streaming request is cancelled when the client disconnects (`server-context.cpp:4703`). | Nothing for the protocol. For the test: raise `SERVER_STARTUP_TIMEOUT` (cell4:16) and the 900 s grace if download plus load can exceed about 12-15 min. Consider `analyzer_timeout` 1800, since 12K-token replies across 10 slots may approach 900 s. |
| 26 | Non-retryable 400s, e.g. #30078's mid-request "type_error.302" | solver:691-700 | Can occur | Optional: a proxy that retries once |
| 27 | 10 concurrent streams (SGLang `--max-running-requests 10`, cell14:15, 286) | cell4:153 | Defaults to 4 slots with a unified KV cache (`server.cpp:168-173`); the unified cache is where bug #29166 shows up | **Required:** `-np 10` (an explicit value turns the unified cache off) and `-c 1392640` (10 × 139264, because `-c` is split across slots; `server-context.cpp:4376-4385`) |
| 28 | Context window comes from env `LOCAL_ANALYZER_CONTEXT_WINDOW=131072`; the server is never asked | cell4:108; tool_agent:187, 3506 | n/a | Keep per-slot context ≥ 139264 |
| 29 | Chat template: SGLang uses `MODEL_DIR/chat_template.jinja` (cell14:303-304) | — | Uses the GGUF's embedded template. Against Intel's template it only differs in merging leading system/developer messages, mapping "high" to xhigh, raising on string arguments (converted, see row 6), and dropping the "No user query" raise. For our message shapes it renders the same, except llama.cpp joins adjacent text parts with "\n" (`chat.cpp:200-220`). | None. Optionally pass Intel's template with `--chat-template-file` for exact parity. |
| 30 | Reasoning and tool parsers (SGLang `qwen3` / `qwen3_coder`, cell14:297) | — | llama.cpp builds a parser from the template (`chat.cpp:1340-1376`) | None |
| 31 | Prefix cache (SGLang radix cache plus mamba extra_buffer, cell14:290-299) | Harness only keeps history byte-stable; it reads no cache stats | Per-slot prompt cache is on by default. For this hybrid model it keeps up to 32 context checkpoints per slot, at least 8192 tokens apart, plus an 8192 MiB host RAM cache (`common.h:633-638`) | Performance only. Recurrent-state checkpoints live in host RAM; my estimate is about 117 MB each (36 layers × 48×128×128 fp32), so 32 × 10 slots is about 37 GB worst case. Consider `--ctx-checkpoints 8`. |
| 32 | MTP speculative decoding and the FR-Spec map (cell14:307-320) | No harness dependency | The GGUF has no MTP tensors | None (it simply runs without speculation) |
| 33 | VRAM fit (SGLang offloads the per-layer embeddings with `--ple-offload-embedding`, cell14:295; uses an fp8 KV cache, cell14:27) | — | The 28.8 GB IQ4_NL `per_layer_token_embd.weight` sits in shard 2. KV is about 24 KB/token at fp16 (12 full-attention layers × 2 KV heads × 256), about 34 GB for 1.39M tokens. | Serving only: `-ot 'per_layer_token_embd\.weight=CPU'`, and `-ctk q8_0 -ctv q8_0` if needed |
| 34 | `kaggle/serve_report.py:33-37` parses SGLang's "Decode batch … accept len" log lines | Analysis after the run | Different log format | Optional: `--metrics`, or read the `timings` field in responses |
| 35 | Notebook plumbing: `_resolve_input` raises if the wheelhouse, model or drafter dataset isn't mounted (cell4:47-50). The precache thread reads `(MODEL_DIR, DRAFT_MODEL_DIR)` (cell8:72) and is started at cell14:323. | — | — | Keep those sources attached, or point the lines at the downloaded GGUF folder. Check disk space for the 59.3 GB download. Replace cell14:1-321 with the llama-server launch. |

## F001 (cell 6): nothing server-specific
- It makes no HTTP calls.
- Its only server-derived input is `_session_generated_tokens` and `_tokens_at_level_start` (cell6:55-56, 135-137, 178-190), which come from `usage.completion_tokens`. llama.cpp provides that field with the same tokenizer, so token counts mean the same thing.
- The pinned LEVEL LOG is a plain-text user/assistant pair (cell6:210-211), put back at the head of history after every trim (edits 4 and 5, cell6:263-278).
  - It works with any server.
  - On llama.cpp, a pin after a trim breaks the slot's cached prefix, so the server falls back to its nearest checkpoint, as on SGLang.
- Its edit anchors match the unchanged harness patch, so it applies as before.

## Minimal change set
1. **Server flags:** `--host 127.0.0.1 --port 8001 --alias flashnext --mmproj … -np 10 -c 1392640 --min-p 0`, plus the VRAM and checkpoint flags from rows 31 and 33.
2. **Harness:** add the llama.cpp overflow strings to `_is_context_length_error`. Optionally add JSON-form `<tool_call>` recovery.
3. **Environment:** unchanged. Keep `LOCAL_ANALYZER_PROVIDER=vllm` and `ARC3_REASONING_HISTORY_KEY=reasoning_content`. For the test only, raise the startup timeout, the HTTP grace and the analyzer timeout.

I only read files and wrote scratch copies of the notebook cells and the harness patch to `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/ls-contract/`. No commits, no pushes, and the user's checkout was not touched.
