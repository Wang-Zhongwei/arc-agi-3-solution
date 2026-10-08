# Which change and object fields Lord Hansolo's agent reads

The same analysis as [`frame_diff` fields](frame-diff-fields.md), for Lord
Hansolo's harness. It has no `frame_diff`. Change information comes from the
action result instead:

- `last_action_result`, which is also the return value of `action(...)`, with
  `changed_pixels`, `largest_changed_regions`, `animation` and per-step lists.
- The compact `history[i].result`.

Objects come from segmentation nodes with `pixel_count`, `bbox` and `mask`.

## Method

Run: `lordhansolo/built-on-tufa-labs-duck-harness-milestone-2/v293`, under
`ARC3-Inference/runs/kaggle/`. The notebook calls this its Milestone 2 version.
Later versions up to v306 have no run output; v306 holds only the notebook
conversion log. It is the only run with Lord's harness that has transcripts.

The sample is thin. This is the offline debug run (2.49, 8m 56s, 472
actions): each game stopped at about 33–36k tokens, and 11 of 25 games took no
action. That leaves 354 python calls in 14 games, all from the first 7–22
turns of a game. Read it as early-game behaviour.

```bash
cd ARC3-Inference/runs/kaggle
python3 -I ../../../docs/harness/scripts/lord_fields.py \
  lordhansolo/built-on-tufa-labs-duck-harness-milestone-2/v293
```

[`lord_fields.py`](scripts/lord_fields.py) traces values through the code the
same way [`frame_diff_fields.py`](scripts/frame_diff_fields.py) does. A key
only counts on the right kind of object. For example, `bbox` on a node,
`bbox` on a changed region and `bbox` on an animation are counted apart.
Percentages are over all 354 python calls; all of them parsed.

The harness also shows change information without being asked:

- **Each user prompt** summarizes the previous turn: actions executed, "changed
  N of 4096 board cells", and each changed region's rows and columns. The
  summary is in 164 of 189 prompts; animation and per-step lines are in 27–28.
- **Each tool result of a call that calls `action()`** carries the compact
  result: `changed_pixels`, flags, `executed_actions`, plus `animation` and
  per-step lists when present. That covers all 171 such results.

Reads in code are therefore only part of what the model sees.

## Evidence

### Action result

Read via `last_action_result` in 102 calls, the `action()` return value in
45 and `history[i].result` in 3. 154 calls (44%) touch a result.

| Key | Calls | Share |
|---|---:|---:|
| `changed_pixels` | 120 | 34% |
| `level_completed` | 82 | 23% |
| `animation` | 39 | 11% |
| `per_step_changed_pixels` | 37 | 10% |
| `game_over` | 34 | 10% |
| `score` | 21 | 6% |
| `state` | 18 | 5% |
| `largest_changed_regions` | 16 | 5% |
| `stop_reason` (undocumented) | 8 | 2% |
| `executed`, `run_complete` | 2 each | 1% |
| `changed_region_count` | 1 | 0% |
| `total_pixels`, `per_step_frames`, `done`, `valid_actions` | 0 | 0% |

`history[i].result` keys are read 1–2 times each.

### Inside `animation` and `largest_changed_regions`

| Object | Calls touching it | Shown whole | Any subfield read by name |
|---|---:|---:|---:|
| `animation` | 39 | 39 | 0 of 11 (`frame_count`, `bbox`, `type`, `from`, `to`, `points`, `cycles`, `parts`, `hash`, `color`, `pixel_count`) |
| changed region | 16 | 16 | 0 of 2 (`bbox`, `pixel_count`) |

What the 171 attached results contained:

| Field | Present |
|---|---:|
| `animation` | 49 (29%) |
| `animation['type']` | `other` 39, `tween` 7, `path` 3 |
| `per_step_changed_pixels` | 43 (batches only) |
| `per_step_frames` | 24 |
| `stop_reason` | 17 |

### Segmentation nodes

158 calls (45%) touch nodes, and 8 show a node whole.

| Field | Calls | Share of node calls |
|---|---:|---:|
| `color` | 157 | 99% |
| `bbox` | 157 | 99% |
| `pixel_count` | 134 | 85% |
| `id` | 86 | 54% |
| `mask` | 19 | 12% |
| `hash` | 16 | 10% |
| `children` | 12 | 8% |
| `boundary` | 2 | 1% |

These agree with the Lord Hansolo row in
[segmentation node fields](segmentation-node-fields.md), which used a regex
over all calls containing `segmentation`.

### Frame views, history and game overs

| Object | Attribute reads (calls) |
|---|---|
| frame view | `segmentation` 158, `rows` 101, `crop` 49, `animation_frames` 36, `step` 34, `level` 17, `shape` 10, `ascii` 5, `frame_count` 3 |
| `history[i]` | `frame` 35, `action` 13, `result` 6, `plan` 0 |
| transition | `action`, `before_frame`, `after_frame` 2 each; `result` 1; `frame`, `plan` 0 |
| `game_overs[i]` | `frame` 2; `actions`, `action_count` 0 |

29 calls (8.2%, 8 games) write their own cell-by-cell diff between two frames.
This is a heuristic count: two frame references, `.rows` or `.ascii`, `!=`
and a loop. That is about the share of calls that use `frame_diff` in Daniel's
runs (8.7%).

## Observations

1. **Scalars are read; structures are printed.** `changed_pixels`,
   `level_completed`, `game_over` and `per_step_changed_pixels` are read by
   name. `animation` and `largest_changed_regions` are only ever printed
   whole, and the regions are already in the user prompt. This matches
   Daniel's `frame_diff`, where the agent dumps the structure and reads few
   fields.
2. **`animation` is mostly empty or minimal.** It appears on 29% of attached
   results, and 80% of those are `other`, which gives only a `bbox`. The
   `tween`/`path` readings, with `from`/`to`/`points`, hash, color and pixel
   count, come up 10 times. The prompt spends about ten lines on the readings.
3. **Prompt-documented fields nobody reads**: `total_pixels`,
   `per_step_frames`, `done`, `valid_actions` (also a global),
   `history[i].plan`, `transition.frame`, `game_overs[i].actions` and
   `.action_count`. The per-turn summary may make some redundant. `done`
   duplicates `run_complete`.
4. **Nodes are read through `bbox`, `color` and `pixel_count`.** `boundary`
   drops to 1% once nodes have a `bbox`, as in the node-field note. `mask`
   (12%) and `hash` (10%) are used occasionally.
5. **The `.rows` view and `crop` are used heavily** (101 and 49 calls). Lord
   gives the agent cell access and region printing, and the agent uses both.
   Hand-written diffs fill the gap left by having no `frame_diff`.

## Candidate changes for our harness

From a thin sample; each needs an A/B test per the
[A/B guide](../rules/t-test.md).

- **Changed-cell count and changed-region boxes in the turn prompt.** Lord
  puts these in every user prompt, and `changed_pixels` is his most-read
  result key. Our `ARC3_AUTO_FRAME_DIFF` covers similar ground but is off by
  default; compare the two summaries' sizes before turning either on.
- **A `rows`/`crop` view of the frame.** Ours exposes `.ascii` only. These two
  are among the most-read frame attributes here.
- **Don't copy the full `animation` reading list.** Its subfields are never
  read and the rich readings are rare.
