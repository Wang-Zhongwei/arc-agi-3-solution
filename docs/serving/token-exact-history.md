# Token-exact history

`ARC3_TOKEN_EXACT_HISTORY` plus `kaggle.vllm.token_exact_history` make each reply come back to the model as the ids it sampled.

Before this, a reply went ids → text → reasoning and tool parsers → stripped fields → chat template → re-tokenized. `ARC3_NORMALIZE_HISTORY=0` removes only the blank-line edit. A reply could still change at a blank line before `</think>`, a single newline before `<tool_call>` or after `</think>`, a missing newline after `<parameter=code>`, or text after a tool call, and any id sequence that does not re-encode to itself.

## How it works

1. The harness asks for `return_token_ids` and stores the ids on the assistant message, next to the parsed fields. Decisions, trims and estimates keep reading the fields.
2. At send time, an unchanged reply goes out as an assistant message whose reasoning is a marker holding its ids. A reply the harness has edited since (compared by a hash of its fields) goes as text.
3. The patched `HfRenderer` replaces marker through `<|im_end|>` with the ids and tokenizes the text around them. The prompt then goes on as ids, and vLLM's multimodal processor expands the image placeholders.

No request schema changes, because the ids travel in an existing field.

## What is not token-exact

- System prompt, tool results, and user messages are rendered text. Only the model's own replies come back as ids.
- A reply stops being sent as ids once the harness edits it. Stubbed replies have none.
- A reply cut off inside its thinking gets the template's `\n</think>\n\n` and `<|im_end|>`, which it never generated.
- A sampled image, video or audio pad token is dropped. Detokenizing drops it from text too.
- Requests with thinking off, such as summaries, keep text: their reply starts after a different generation prompt.

## Checks

- Each anchor in `token_exact/runtime_patches.json` must occur exactly once in the runtime's `renderers/hf.py`, or setup fails before the load.
- After the smoke test, a live probe sends a reply whose ids re-tokenizing would merge, next to an image. Setup fails if the ids are not in the prompt or the image was not expanded. The probe is skipped in submission mode.
- Every request checks that the last reply's ids are in the returned `prompt_token_ids`. If not, that game logs an error and sends text from then on.

## Cost

Responses carry the prompt ids and requests carry each reply's ids as decimal text. Prefix-cache behaviour does not improve while `prefix_cache_retention_interval` is 0: Mamba state is kept only at prefill ends, so replies are still re-prefilled.
