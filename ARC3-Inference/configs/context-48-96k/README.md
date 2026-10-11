# Context 48-96K at 16 streams

Keeps less history per game and runs more games at once, with the view tool and the
level-up board on. History is trimmed when it reaches 96K tokens and drained back to
48K, instead of 118K to 58K, with 16 game slots over 16 engine sequences instead of 15
over 14. Notebook: `jonathanwang2022/context-48-96k`.

Changes from [../view-tool/v1.json](../view-tool/v1.json) (`allocation-vc-one-wave`
agent at `d49a940` plus `ARC3_VIEW_TOOL`, lorddraft vLLM serving, 25 public games,
120-minute soft end):

| Setting | view-tool v1 | v1.json |
|---|---:|---:|
| `shared.context_window` | 131,072 | 111,104 |
| Prompt budget (window - 12,288 output - 512 margin) | 118,272 | 98,304 (96K) |
| `ARC3_CONTEXT_DRAIN_TOKENS` | 59,392 | 49,152 |
| Drain target (budget - drain) | 58,880 | 49,152 (48K) |
| `kaggle.vllm.max_model_len` | 147,072 | 127,104 |
| `kaggle.vllm.max_num_seqs` | 14 | 16 |
| `ARC3_MAX_ACTIVE_STREAMS` | 15 | 16 |
| `ARC3_LEVELUP_BOARD_IMAGE` | off | 1 |

`max_model_len` keeps the old 16,000-token margin over the context window. The CUDA
graph capture size follows `max_num_seqs` (16 x 4 MTP tokens = 64). The assistant-message
cap (150, drain 30) stays as is; at about 3K tokens per message, 96K is about 32 messages.
The view tool's trimmed-history note points the agent at `history` frames the shorter
window has dropped.

## KV cache

The lorddraft pool was 1,558,044 tokens at 14 sequences and 1,572,937 at 10
(context-100-130k v1), about 3.7K tokens per sequence, so 16 sequences leave about
1.55M. Sixteen games averaging about 74K tokens use about 1.18M (76%). If all sixteen
reach the top of the band at once, they use 1.57M (101%); the baseline's worst case
(14 x 118K = 1.66M, 106%) peaked at 93% in level-up-board v1, since drains are staggered.
Game slots equal engine sequences, so no game waits inside vLLM.

## References

| Run | Context | Streams | Score |
|---|---|---:|---:|
| lorddraft-baseline-no-io v01 | 58-118K | 15/14 | 50.17 |
| level-up-board v1 | 58-118K | 15/14 | 55.94 |
| view-tool v2 | 58-118K | 15/14 | 51.25 |
| context-100-130k v1 | 100-130K | 10/10 | 47.78 |

This run changes three things at once (context band, streams, level-up board) against
view-tool, so it tests the combination, not any one change.

[Version 1](https://www.kaggle.com/code/jonathanwang2022/context-48-96k) was pushed
2026-10-11 from commit `9bb7186`; the bundle carries the settings above and its harness
files match that commit. Status at push: `RUNNING`, not yet evaluated.
