# v222f compact prompt port

Source: Kaggle notebook `dangthuygiang/arc3-v222f-advisor-12x96k-base-v222d-36k`, public 25 games, 1 pass:
mean 50.73, median 47.62. The patched harness lives in the private dataset `dangthuygiang/arc3-taaf-v222f-advisor`,
which returns 403 for jonathanwang2022 and ticbudja. The prompts below were recovered from that run's
`ft09-0d8bbf25_p0_requests.jsonl`, which holds the full system prompt, the tool schema and every user message.

## Their harness changes, and what this branch ports

| Change (their log tag) | What it does | Ported |
|---|---|---|
| SYSCOMPACT | System prompt 18.6k → 9.4k chars with the same guidance; tool description 1,052 → 230 chars; per-turn methodology lines removed; run-length executed actions | Yes, behind `ARC3_COMPACT_PROMPT=1` |
| NOTE-PAGE | `note()` in Python edits a 7-line notes page (World/Goal/Action model, Recent findings, Open questions) that is reprinted in every turn and survives context eviction; level-clear and pre/post-eviction nudges to rewrite it | No (user's choice) |
| Advisor slim queue | 12 lanes, 25k-token slices, tail requeue of games with no progress | No: allocation and serving stay ours |

They have no RESET action and no level-up board. Both stay on here: `EXPOSE_RESET=on` adds a compact RESET line to
the Actions section, and `ARC3_LEVELUP_BOARD_IMAGE=1` still splits the level-clear opener around the solved board.

## Files

- `theirs-system-prompt.txt`: their system prompt as sent.
- `compact-system-prompt.txt`: ours with `configs/compact-prompt/v1.json`. It differs from theirs only by the
  dropped `note(...)` line, a missing period after the vision sentence, the RESET line, and the trailing newline.
- `ours-system-prompt.txt`: main's prompt with `configs/reset-baseline/v1.json`. It is unchanged with the flag off.

Experiment: `configs/compact-prompt/v1.json` = `configs/reset-baseline/v1.json` + `ARC3_COMPACT_PROMPT=1`
(only the slug, title and dataset are renamed). Compare against reset-baseline v1 (58.13).
