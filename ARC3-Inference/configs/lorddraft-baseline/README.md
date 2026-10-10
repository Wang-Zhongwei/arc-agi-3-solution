# Lorddraft baseline

The 10-08 submission harness (public 30.78) on the 2026-10-10 engine: Daniel's Intel
W4A16 weights on Lord Hansolo's vLLM with Lord's NVFP4 MTP draft head (PR #27) and
the parallel startup prefetch (`52ad067`).

Two notebooks on `jonathanwang2022`, identical except for the agent code:

| Config | Notebook | Agent code |
|---|---|---|
| [v1.json](v1.json) | `lorddraft-baseline` | `inference/agent/` restored from `92dece7` (branch `lorddraft-baseline-agent-92dece7`), byte-identical to the 10-08 and 10-09 submissions |
| [../lorddraft-baseline-no-io/v1.json](../lorddraft-baseline-no-io/v1.json) | `lorddraft-baseline-no-io` | `allocation-vc-one-wave` tip |

The tip agent's system prompt is identical to `92dece7`'s. Its `python` tool description
adds "No I/O is allowed: do not read or write files. Only use the exposed packages; any
other import fails.", and its sandbox allows 12 more modules (`abc`, `array`, `cmath`,
`dataclasses`, `decimal`, `enum`, `hashlib`, `inspect`, `numbers`, `time`, `types`,
`typing`). Its other changes sit behind flags these configs leave off.

Changes from [priority-one-wave-lorddraft/v1.json](../priority-one-wave-lorddraft/v1.json):

- Public run: all 25 games, 120-minute soft end, no share version.
- `ARC3_MAX_ACTIVE_STREAMS = 15` game slots over `max_num_seqs = 14` engine sequences.
  The `lorddraft-15-games-14-seqs` fork measured +3% tok/s, at-cap 88% vs 58%, KV peak
  87.5%.
- `ARC3_ANIMATION_IMAGE = off` and `multimodal.intermediate_frames = false` are written
  out; both match the defaults.

The private rerun is unchanged from 10-09: one wave, 120 lanes over the 110 hidden games,
532 minutes.

Relative to the 10-09 submission (`ticbudja/priority-one-wave-lord-vllm` v2), the
harness settings differ only in the stream count (13 there).
