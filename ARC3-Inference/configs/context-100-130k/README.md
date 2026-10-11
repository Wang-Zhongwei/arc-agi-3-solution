# Context 100-130K at 10 streams

Retains more history per game at lower concurrency. History is trimmed when it reaches
130K tokens and drained back to 100K, instead of 118K to 58K, with 10 streams instead
of 15 game slots over 14 engine sequences. Notebook: `jonathanwang2022/context-100-130k`.

Changes from [../level-up-board/v1.json](../level-up-board/v1.json) (`allocation-vc-one-wave`
tip agent, lorddraft vLLM serving, 25 public games, 120-minute soft end):

| Setting | level-up-board v1 | v1.json |
|---|---:|---:|
| `shared.context_window` | 131,072 | 145,920 |
| Prompt budget (window - 12,288 output - 512 margin) | 118,272 | 133,120 (130K) |
| `ARC3_CONTEXT_DRAIN_TOKENS` | 59,392 | 30,720 |
| Drain target (budget - drain) | 58,880 | 102,400 (100K) |
| `kaggle.vllm.max_model_len` | 147,072 | 161,920 |
| `kaggle.vllm.max_num_seqs` | 14 | 10 |
| `ARC3_MAX_ACTIVE_STREAMS` | 15 | 10 |

`max_model_len` keeps the old 16,000-token margin over the context window, so a full
prompt plus the 12,288-token reply still fits.

The assistant-message cap (150, drain 30) stays as is. In the lorddraft forks every
drain fired on the token budget, at 27-46 assistant messages (about 3K tokens each),
so 130K is about 52 messages.

## KV cache

The lorddraft engine's pool is 1,558,044 tokens (level-up-board v1, 21.65 GiB fp8).
Ten games averaging about 120K tokens use about 1.2M (77%). If all ten reach the top of
the band at once, they use 1.37-1.46M (88-94%). level-up-board v1 peaked at 93% KV
usage (p50 75%, p99 88%). All ten prefixes fit at once, so an idle game's cache is not
evicted. Game slots equal engine sequences, because an eleventh cached game would
need about 97%.
