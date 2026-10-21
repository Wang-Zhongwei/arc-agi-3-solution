# gsq-coder-gguf-runner: validate:local

Workflow run `wf_0ae4547f-80a`, phase Local validation. Serve the ISTA GSQ-RCO Coder GGUF the way its Hugging Face repo documents (llama.cpp or Strata, both shards, 256 experts, mmproj), validate the harness contract locally, build staging/engine/test notebooks and the 2-hour final-run notebook

---

Local validation: everything I could check passes, and I found nothing broken in `arm/runner-gguf` (HEAD `d64ab57`, tree clean, no new commits). I did not restart llama-server or rerun the harness because the laptop is at **1% battery** ("Battery Warning: Final"). A Metal run draws roughly 20–30 W, which would likely shut the laptop down and take this session with it. The table below uses the saved run in `gguf-local/` from 02:46–03:07 today, which I re-analysed. The only things I ran this time were `llama-server --version` and two unit-test files (17 passed in 2.2 s).

| Check | Result | Evidence |
|---|---|---|
| llama.cpp at b11476, Metal | PASS | `lcpp-mac/bin/llama-server --version` gives build 11476, commit 9881906, arm64; built in 55 s (`build-time.txt`) |
| Small Qwen-family model with a tool template and vision | PASS | `Qwen3.5-0.8B-Q4_K_M.gguf` (533 MB) plus `mmproj-F16.gguf` |
| Same flags and template handling | PASS | `validate_gguf_local.py` calls the Kaggle argv builder `gguf_serve.launch_args` directly. Only these differ: -m, --mmproj, `-np 2`, 16,384-token slots, `-cram 8192` and `--chat-template-file` (the Coder's own template) |
| `/props` and `/apply-template` | PASS | Pinned template hash, vision on, the prompt says "Reasoning effort is set to xhigh", ends in an open `<think>` block and has `<tools>` |
| `/health`, `/metrics`, `/slots` | PASS | Startup waits on `/health`; `/metrics` and `/props` were saved as test fixtures (`metrics-local.json`, `props-local.json`, `serve-local.log` in `kaggle/tests/fixtures/gguf/`); the validation script also captures `/slots` |
| `POST /v1/chat/completions` through the real harness (`openai_compat.build_chat_payload` and `tool_agent._chat_completion`), at least 5 turns | PASS | 6-minute run, sk48 + ft09, 222 requests, no 5xx |
| Tool calls parsed | PASS | sk48: 86 of 88 responses have `finish_reason=tool_calls` with exactly one parsed call. Both games: 0 responses with tool-call markup left in the text, 0 recovered from markup |
| Image input | PASS | sk48: 89 of 89 requests carry an image. A 640 px frame costs exactly 402 tokens, which matches what the harness assumes |
| `reasoning_content` | PASS | Every assistant message in the history the harness sends back has `reasoning_content` (sk48 2,344 of 2,344, ft09 3,120 of 3,120). Prefix-cache reuse of 96.6% (sk48) and 93.1% (ft09) shows the echoed reasoning renders the same way each time |
| `usage` fields | PASS | `completion_tokens`, `prompt_tokens`, `total_tokens` and `prompt_tokens_details.cached_tokens` all present. The harness counted them: note tokens=23,590 and 23,292 |
| `finish_reason` values | PASS | `tool_calls`, `stop` and `length` all seen. A cut-off call comes back with its partial arguments and the harness flags it as invalid JSON |
| Context-overflow 400 | PARTIAL | A real `exceed_context_size_error` 400 was captured and the patched matcher recognises it (`test_context_error`). The harness's trim-and-retry path was never triggered against a live server |
| Smoke suite | PASS | 9 of 9 local checks: 140 tok/s on one stream, 190 tok/s across 2 parallel streams |

Two things look like failures in the saved run but are not server problems:
- **ft09 never called the tool.** It made 5 attempts, all cut off at 2,048 tokens. That is the 0.8B model.
- **54 ft09 requests had no image.** They are the "You have not acted yet" nudges after the context was trimmed. That is the harness's own behaviour and does not depend on the server.

The summary line's `tokens=0` is how the framework counts (it only counts tokens per action), not lost usage data.

**Only checkable on the GPU:**
- The engine build for sm_120a, and whether the engine-build image works in a CPU session.
- The real shards loading: shard 2 found next to shard 1, the n-gram table read lazily (the prefetch log line), and all layers offloaded.
- 10 slots × 139,264 tokens with f16 KV staying stable and within VRAM.
- The template's sha256 inside the real GGUF matching the pin, and the real mmproj working.
- Real speed under 10 streams, the actual 116K-token prompts, and cache reuse at that size.
- Whether the Coder falls into repeat loops.
- The overflow trim-and-retry path end to end.

Files:
- `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/gguf-local/` (`smoke-local.json`, `l5/l5-patched.json`, `l5/runs-patched/20261008_024613_gguf-local-patched/`, `serve-run1.log`)
- `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gguf/serving/tools/validate_gguf_local.py`
