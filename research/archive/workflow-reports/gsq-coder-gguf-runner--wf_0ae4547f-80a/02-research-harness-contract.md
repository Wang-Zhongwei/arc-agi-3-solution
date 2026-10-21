# gsq-coder-gguf-runner: research:harness-contract

Workflow run `wf_0ae4547f-80a`, phase Research. Serve the ISTA GSQ-RCO Coder GGUF the way its Hugging Face repo documents (llama.cpp or Strata, both shards, 256 experts, mmproj), validate the harness contract locally, build staging/engine/test notebooks and the 2-hour final-run notebook

---

I wrote no repo files, made no commits or pushes, and downloaded no model. The only things I created are evidence files in `gguf-plan/research/`; the spec is in this reply.

# Adapter spec: our harness on llama-server b11476, ISTA Coder GGUF

## Bottom line
- **No proxy is needed.** The harness only calls `POST /v1/chat/completions`, plus the notebook's `GET /health` poll.
- **What changes:**
  - Server flags.
  - One recommended 3-line harness patch so llama.cpp's "context too long" error is recognised.
  - Two timeout and env edits in the notebook.
  - A new cell 14.
- **Reasoning effort is already xhigh.** Both chat templates default to xhigh when `reasoning_effort` is not sent, and our harness never sends it by default. That matches the README's benchmark setting with no change.
- **Use the template embedded in the GGUF.** I rendered it and Intel's template with jinja2 on harness-shaped messages, for 6 combinations of thinking on/off, preserve on/off and effort xhigh/medium/low. The output was byte-identical every time. Tool-call parsing is also the same either way.
- **Shard 2 and the mmproj need almost no new staging.** I checked the Hugging Face LFS sha256 values:
  - Coder shard 2 has sha256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` and is 28,800,138,432 bytes. That is the same file as every unpruned release's shard 2, including `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`. That file is in the public Kaggle dataset `clothespin/strata-qfn-q2-0-assets`, so we can attach it instead of staging 28.8 GB.
  - The mmproj (`b1a82259…`) is also identical across the releases, but that dataset doesn't contain it. It still has to be staged (0.91 GB, one part).

## File layout llama.cpp needs (checked in `src/llama.cpp:570`, `llama-model-loader.cpp:596-672`)
- **Naming:** `-m` must end in `-00001-of-00002.gguf`. llama.cpp builds the second shard's path from that name, in the same directory.
  - `build_fable_gsq.py:142` currently writes `/tmp/gsq/shard1.gguf`, which fails with "invalid split file name".
  - Use `/tmp/gsq/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` instead.
  - Add a symlink `/tmp/gsq/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` pointing at the mounted clothespin `…Q2_0-00002-of-00002.gguf`.
  - The loader checks `split.no` and the tensor count, and the file is byte-identical, so this passes.
- **Warm shard 2 before serving.** `per_layer_token_embd` is an input-layer tensor, so it stays on the CPU, memory-mapped from the file. Reading it lazily from `/kaggle/input` would mean random page faults on a network mount. Use one pass with the cell 8 `precache()`, which both checks the sha256 and fills the page cache.
- **Retarget the precache thread.** Cell 14 starts `precache_model_thread` on the Intel model and drafter. Point it at shard 2, or it will read about 70 GB and push shard 2 out of the cache.

## Adaptation table
Harness line refs are to `wt/runner-gsq/ARC3-Inference/inference`; I checked it equals F001's patched harness (`git apply -R --check` passes).

| Dependency | Harness | llama-server b11476 | Handled by |
|---|---|---|---|
| `/model_info`, `/get_server_info`, `/flush_cache`, `/freeze_gc`, `/metrics`, `/tokenize` | Not on the run path (only `tools/chat.py` and the unused vLLM `framework/kaggle.py`). No notebook cell calls the server except the `/health` poll. | — | Nothing |
| `GET /health` | Cell 14 wait loop | 503 while loading, then 200 | Nothing; keep the loop |
| `priority` | `openai_compat.py:63` supports it, but `_chat_completion` (`tool_agent:5224`) never passes it | Ignored | Nothing |
| `separate_reasoning`, `max_completion_tokens`, `seed` | Never sent; `seed` only if `LOCAL_ANALYZER_SEED` ≥ 0 | `max_tokens` and `max_completion_tokens` both map to `n_predict` (`server-schema.cpp:44-47`) | Nothing |
| `top_k`, `chat_template_kwargs` | Sent only when provider normalises to `vllm` (`openai_compat:91-94`) | Request kwargs merge over CLI kwargs; `enable_thinking` overrides (`server-common.cpp:1396-1411`) | **Env: keep `LOCAL_ANALYZER_PROVIDER=vllm`** |
| `min_p` | Not sent; SGLang default is 0 | Default is 0.05 (`common.h:235`) | **Flag: `--min-p 0`** |
| `preserve_thinking` | Always sent as true (`tool_agent:1317`, `5200-5216`) | Template variable; set after llama.cpp's own `preserve_reasoning` default | Nothing |
| `reasoning_effort` | Only from `ARC3_REASONING_EFFORT_LADDER` after a truncated reply (`1301`, `6827-6841`); unset in F001 | Template default is xhigh. Embedded template maps "high" to xhigh; Intel's raises on "high"; both raise on "max" and "minimal" | Env: keep the ladder unset. If set, use only `medium,low`. |
| History reasoning | `ARC3_REASONING_HISTORY_KEY` (`1254`) | Parser reads only `reasoning_content` (`chat.cpp:453`) | **Env: keep `reasoning_content`** |
| Response reasoning | Reads `reasoning`, then `reasoning_content` (`3431`) | Returns `reasoning_content` (the default reasoning format is deepseek) | Nothing; never pass `--reasoning-format none` |
| `tools` and `tool_choice: "auto"` | `5176`, `2502` | Embedded template routes to the Qwen3-Coder PEG parser (`chat.cpp:1222-1233`), with thinking supported. `parallel_tool_calls` defaults to true from the template's capabilities. A lazy grammar starts at `<tool_call>` and is suppressed while thinking (`sampling.cpp:460-472`). | Nothing. Keep `tool_choice` auto. |
| Tool calls leaking into content (JSON form) | Harness only recovers the XML form (`85-171`) | Once triggered, the grammar forces `<function=python>\n<parameter=code>`, so a JSON-form call can't be produced. A `<tool_call>` written inside thinking ends the thinking and is parsed as a call. | Nothing |
| `finish_reason`, `usage` (`completion_tokens`, `prompt_tokens` including images, `total_tokens`, `cached_tokens`) | `3632-3675`, `6827` | Returns stop / tool_calls / length (`server-task.cpp:414-428`) and the usage fields (`365-372`) | Nothing |
| `image_url` data PNG parts | `vision_context`, `tool_agent:4660-4694` | Accepted only with an mmproj. A 640×640 image becomes 400 tokens plus vision_start/end (limits 8–4096, `clip.cpp:1670-1680`), same as SGLang. Text and image parts alternate, so llama.cpp's "\n" join between adjacent text parts never applies. | **Flag: `--mmproj`** |
| Trailing assistant message | The harness always ends a request on a user or tool message | Default `prefill_assistant` would continue a trailing assistant message (`server-common.cpp:1369`) | Flag: `--no-prefill-assistant` (matches SGLang; safety only) |
| Context overflow | `_is_context_length_error` (`3438`) matches SGLang and vLLM wording only | HTTP 400, "request (N tokens) exceeds the available context size (M tokens)…", type `exceed_context_size_error` (`server-context.cpp:3408-3424`). Unmatched, it is treated as a failure; 10 in a row and the game gives up (`solver:105`). | **Harness patch, below** |
| Parse failure on the final message | — | Throws, so HTTP 500 (`chat.cpp:1510`, `server.cpp:69-82`), which the harness retries 3 times | Nothing |
| 10 concurrent streams | `ARC3_MAX_ACTIVE_STREAMS=10` | Default is 4 slots with a shared (unified) cache | **Flags: `-np 10 --no-kv-unified -c 1392640`** (139,264 tokens per slot, same as SGLang's CTX) |
| Read timeout | `analyzer_timeout` 900 s (cell 18) | 10-stream decode is much slower than SGLang, so a 12,288-token reply can approach 900 s | **Cell 18: `analyzer_timeout = 1800`.** Server `-to` defaults to 3600. |
| Prefix cache | The harness keeps history byte-stable | Per-slot cache, with checkpoints at the last user-message start and near the prompt end (`server-context.cpp:3775-3830`); the previous reply is processed again each turn | Performance only. Use `-cram 8192` for 10 games; for the 25-game run use `-cram 49152`, since slots are shared and each idle game's state is about 3 GB. |
| `serve_report.py` | Parses SGLang log lines | Different log format | Use `--metrics` and the `timings` field |

## Harness patch (behaviour change: put it on its own branch, or apply it as an F001-style anchor edit)
In `tool_agent.py`, inside `_is_context_length_error`, after the line `        or "context_length_exceeded" in message`, add:
```python
        # llama-server wording (HTTP 400, type exceed_context_size_error)
        or "exceeds the available context size" in message
        or "exceed_context_size_error" in message
        or "is larger than the max context size" in message
```

## Server command
```
LD_LIBRARY_PATH=<cuda13 libs> llama-server \
  -m /tmp/gsq/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --mmproj /tmp/gsq/mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  --host 127.0.0.1 --port 8001 --alias flashnext \
  -ngl all -fit off -fa on -ctk f16 -ctv f16 \
  -np 10 --no-kv-unified -c 1392640 -b 2048 -ub 512 \
  --jinja --reasoning-format deepseek --min-p 0 --no-prefill-assistant \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  -cram 8192 --metrics -to 3600
```
- **Leave these out:**
  - `--chat-template-file`: the embedded template renders the same and also maps "high" to xhigh.
  - `--reasoning-budget`, `--image-min-tokens`, `--api-key`.
  - Any speculative decoding: the Coder has no MTP tensors.
- **f16 KV is required for speed.** The sparse attention gather in flash attention runs only with f16 K and V; q8_0 falls back to dense attention.

## Notebook changes (cells 4, 14 and 18)
- **Cell 4:**
  - Remove `_resolve_input(MODEL_DIR)` and `_resolve_input(DRAFT_MODEL_DIR)` and unmount those models. The Intel template is no longer needed.
  - Add the clothespin dataset and the staged mmproj and llama-server build.
  - Keep `pennyroyal-v253` for the CUDA 13 runtime wheels.
  - All other env stays as it is.
- **Cell 14:**
  1. Reassemble shard 1 under the split name.
  2. Create the shard 2 symlink.
  3. Precache shard 2 and check its sha256.
  4. Launch the server.
  5. Keep the existing `/health` wait loop.
  6. Run the smoke checks below before releasing the harness.
- **Cell 18:** `analyzer_timeout = 1800`.

The GPU notebook has no internet, so the llama-server build has to come in as a staged output. Building it in a CPU notebook works because it uses the same image (glibc 2.35).

**Smoke checks:**
1. Send one request with the python tool and `chat_template_kwargs` `{enable_thinking: true, preserve_thinking: true}`. Expect `finish_reason == "tool_calls"`, the name `python`, `json.loads(arguments)["code"]` to succeed, and a non-empty `reasoning_content`.
2. Send one 640×640 PNG. Expect `usage.prompt_tokens` to include about 402 image tokens.
3. Check `/props` reports the embedded template.

## Unverified risks (check in the first run)
- **Cut-off tool call.** I traced the parser code but didn't run it. A reply cut off at max tokens inside a tool call should come back as arguments missing their closing `}`. The harness then drops it as malformed, which is the right outcome. Confirm it in the transcripts.
- **Wrong tool name is forced to `python`.** If the model writes `<function=action>`, the grammar forces it to `python` / `code`. Under SGLang that call was dropped instead; either way the turn is wasted.
- **Tool-schema key order** may differ cosmetically in the system prompt (llama.cpp re-serialises the tool parameters). Not checked.

## Strata: not recommended (fallback only)
I read its README, `docs/BATCHING.md` and `docs/DETAILS.md`.
- **Concurrency doesn't fit.** It handles one request at a time by default, and batches at most 8 slots per window, so 2 of our 10 streams would wait. Batched slots also run without MTP drafts by default.
- **It's built for 12–24 GB cards** that keep experts in system RAM. On a 96 GB card that design gains nothing.
- **More to stage.** It needs its own prepared pack, an MTP layer and a Python server environment. clothespin's pack is built for the unpruned Q2_0 model, not the Coder.
- **The API would work.** It serves OpenAI chat completions with tools, `reasoning_content` and `image_url`, so the harness adapter would be the same as above. Note it caps `top_k` at 64 and its own effort levels are off/low/medium/high.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/gguf-plan/research/`:
- `render_templates.py`
- `rendered_gguf.txt`
- `strata_readme.md`
- `strata_docs_BATCHING.md`
- `strata_docs_DETAILS.md`
- `strata_docs_INSTALL.md`
- `strata_docs_MODELS.md`
- `strata_docs_HOW_IT_WORKS.md`
