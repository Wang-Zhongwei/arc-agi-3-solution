# Thinking vs actions per call

![Thinking tokens vs actions executed, per model call](figures/thinking_vs_actions_per_call.png)

(a) Daniel's base notebook, Kaggle `dangthuygiang/arc3-v207-danibase-base-dfranzen`
v05. (b) Our base, Kaggle `jonathanwang2022/adaptive-reasoning-ucb` v30. Each
point is one model call; the orange line is the mean actions per thinking bin.


| | (a) Daniel v207 | (b) Our v30 |
|---|---:|---:|
| Pearson r / Spearman ρ | 0.01 / 0.07 | 0.03 / 0.04 |
| Calls with 0 actions | 52.4% | 51.7% |
| Mean actions per call | 2.29 | 2.57 |

## Conclusion
- There is no correlation between thinking tokens and emited actions. 
- Daniel's solution has similar thinking tokens per action as ours. But increase actions per token could be equivalent to increase inference throughput if model can maintain the quality of actions. Once the understanding has consolidated, we could afford to take a longer train of calculated actions. This should be a promosing direction to explore. 

## Data

- **Daniel:** exact `usage.reasoning_tokens` from the request logs. Actions per
  call are the jump in the `action` counter between consecutive requests.
- **v30:** no request logs, so `scripts/extract_transcript_calls.py` rebuilds
  each call from the transcripts into `data/v30_calls.csv`. It re-tokenizes the
  `[THINKING]` text with the Qwen3.8-Flash-Next tokenizer and credits each
  turn's action-counter jump to the turn's last call. Checked on Daniel's run,
  where exact counts exist: thinking tokens total 99.9% of the logged count
  (r = 1.000), and actions match on 2,476 of 2,477 calls. On v30 the per-game
  action sums equal the run's totals in all 25 games.

Regenerate:

```bash
pip install tokenizers  # only for the extraction step
python3 docs/reasoning/scripts/extract_transcript_calls.py
python3 docs/reasoning/scripts/thinking_budget.py
```

