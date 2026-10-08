# How often the agent uses `frame_diff`, and which fields it reads

Companion to [segmentation node fields](segmentation-node-fields.md). This
note answers two questions for Daniel's baseline:

- How often does the agent call `frame_diff`?
- Which of the fields it returns does the agent read?

For comparison, it does the same for Lord Hansolo's harness, which has no
`frame_diff`.

## Runs

Daniel's baseline: two runs of the same code, pooled, under
`ARC3-Inference/runs/kaggle/`. That is 35 game runs over 25 distinct games.

- `dangthuygiang/arc3-v207-danibase-base-dfranzen/v05` (25 games, 48.36)
- `dfranzen/arc-agi-3-milestone-2-solution/v01` (10 games, 36.56)

Their `frame_diff` matches `main`'s except that `appeared`/`disappeared`
entries also carry `shape_hash` (removed in
[PR #9](https://github.com/Wang-Zhongwei/arc-agi-3-solution/pull/9)).

Lord Hansolo: `lordhansolo/built-on-tufa-labs-duck-harness-milestone-2/v293`,
see [below](#lord-hansolo-no-frame_diff).

```bash
cd ARC3-Inference/runs/kaggle
A=dangthuygiang/arc3-v207-danibase-base-dfranzen/v05
B=dfranzen/arc-agi-3-milestone-2-solution/v01
S=../../../docs/harness/scripts
python3 -I $S/frame_diff_frequency.py $A $B
python3 -I $S/frame_diff_fields.py $A/transcripts/*.txt $B/transcripts/*.txt
python3 -I $S/frame_diff_contents.py $A $B
python3 -I $S/lord_fields.py lordhansolo/built-on-tufa-labs-duck-harness-milestone-2/v293
```

## How often `frame_diff` is called

From [`frame_diff_frequency.py`](scripts/frame_diff_frequency.py). A call
counts when its code contains `frame_diff(`.

| Measure | Count | Share |
|---|---:|---:|
| Python calls using `frame_diff` | 263 of 3,017 | **8.7%** |
| Turns with at least one `frame_diff` call | 258 of 2,098 | 12.3% |
| Game runs that use it at least once | 33 of 35 | 94% |
| `action()` calls that also call `frame_diff` | 228 of 1,283 | 17.8% |

How it is called, over 276 invocations in those 263 calls:

| Pattern | Share |
|---|---:|
| No arguments, `frame_diff()`, i.e. previous vs current frame | 86% |
| Preceded by `action()` in the same code | 85% |
| Inside a loop or comprehension | 14% |

Most uses follow the pattern "act, then diff": `action([...]); d = frame_diff()`.
With explicit arguments, the frames are mostly saved local variables or
`history[...]` frames.

When it is called:

| Phase | Turns using `frame_diff` |
|---|---:|
| Level 1, first 3 turns | 65 of 103 (63%) |
| Level 1, later turns | 47 of 207 (23%) |
| Level 2+, first 3 turns | 36 of 390 (9%) |
| Level 2+, later turns | 110 of 1,398 (8%) |

Calls with `frame_diff` error less often than other calls: 3.4% (9 of 263)
against 9.5%. None of the 9 errors is a KeyError on a diff field.

## Which fields are read

[`frame_diff_fields.py`](scripts/frame_diff_fields.py) parses each call that
calls `frame_diff`. It traces values from `frame_diff(...)` to a category list
(`d['moved']`) to an entry (`for m in d['moved']`). Tracing is needed because
entry fields (`color`, `pixels`, `hash`, `at`, `bbox`) share names with node
fields, so a regex would mix them up. It counts two things:

- **Named**: the code reads the key or field by name, e.g. `d['moved']`,
  `m['from']`, or `d.get(k)` with `k` looping over a literal tuple of keys.
- **Shown**: the code prints or stores the value whole, e.g. `print(d)`,
  `for k, v in d.items(): print(k, v)`, or `[m for m in d['moved']]`. Every field
  is displayed, but the code never picks one.

Percentages are over the 263 `frame_diff` calls; all of them parsed. 100 calls
(38%) show the whole diff, and 112 (43%) iterate its keys generically.

### Top-level keys

| Key | Named | Named or shown |
|---|---:|---:|
| `changed_cell_count` | 149 (57%) | 211 (80%) |
| `moved` | 131 (50%) | 232 (88%) |
| `appeared` | 109 (41%) | 209 (79%) |
| `disappeared` | 109 (41%) | 209 (79%) |
| `resized` | 86 (33%) | 191 (73%) |
| `changed_color` | 75 (29%) | 175 (67%) |
| `rotated` | 28 (11%) | 139 (53%) |

### Entry fields

Calls that read the field by name. The last column is the share of calls that
show that category's entries whole, which displays every field.

| Category | Read by name | Never read by name | Entries shown whole |
|---|---|---|---:|
| `moved` | `color` 43, `to` 42, `from` 41, `rotated_by` 12, `pixels` 6 | `hash` | 64% |
| `appeared` | `color` 24, `bbox` 22, `pixels` 18, `at` 8 | `hash`, `shape_hash` | 61% |
| `disappeared` | `color` 24, `bbox` 22, `pixels` 16, `at` 8 | `hash`, `shape_hash` | 61% |
| `resized` | `color` 22, `before_pixels` 21, `after_pixels` 17, `at_before` 16, `at_after` 16 | `before_hash`, `after_hash` | 56% |
| `changed_color` | `at` 10, `pixels` 4, `before_color` 4, `after_color` 4 | `before_hash`, `after_hash` | 56% |
| `rotated` | `at` 4, `pixels` 2, `color` 2 | `hash`, `rotated_by` | 45% |

Caveats:

- A field read in a loop over several categories counts for each of them.
- Whole views are often truncated (`str(v)[:600]`), so "shown" is an upper
  bound on what the model saw.
- A reference in code is not proof the value changed a decision.

### What the diff contains

[`frame_diff_contents.py`](scripts/frame_diff_contents.py) recomputes
`frame_diff` with `main`'s code for every consecutive pair of action boards in
`artifacts/*_events.jsonl`. Over the 6,938 transitions that changed at least
one cell:

| Category | Non-empty | Entries when non-empty |
|---|---:|---:|
| `resized` | 6,689 (96%) | 3.1 |
| `moved` | 5,501 (79%) | 6.9 |
| `appeared` | 2,231 (32%) | 5.1 |
| `disappeared` | 2,061 (30%) | 4.7 |
| `changed_color` | 780 (11%) | 2.2 |
| `rotated` | 80 (1.2%) | 1.3 |

| Measure | Value |
|---|---:|
| `str(diff)` characters, median / p90 | 993 / 2,362 (v05), 1,041 / 4,363 (v01) |
| Hash key/value share of `str(diff)`, without `shape_hash` | 31% |
| `frame_diff` results the model saw that show a hash | 115 of 263 (44%) |
| Hash share of those results' characters | 17% |

The last share also counts node hashes printed in the same call. In v01, 24% of
`resized` entries are objects of 300 px or more (backgrounds). The most common
size change is exactly one pixel (1,453 of 3,983 entries).

## Lord Hansolo: no `frame_diff`

Lord's harness has no diff helper at all:

- The sandbox's globals are `current_frame`, `previous_frame`, `history`,
  `transitions`, `last_transition`, `game_overs`, `valid_actions`,
  `last_action_result` and `action()`.
- Frame views have only `.ascii`, `.rows`, `.segmentation` and `.crop()`.
- The prompt tells the agent to diff frames itself: "For before/after diffs
  use `.before_frame` and `.after_frame`".

Instead, each action returns a summary of what changed (`last_action_result`,
also the return value of `action()`). It has `changed_pixels`,
`largest_changed_regions` (boxes and cell counts), `per_step_changed_pixels`
and an `animation` reading. The harness also shows change information without
being asked:

- Each user prompt summarizes the last turn, e.g. "changed 12 of 4096 board
  cells … rows 4-9 cols 62-63". This is in 164 of 189 prompts.
- Each tool result of a call that calls `action()` carries the compact result.
  That is all 171 such results.

These say where and how much changed, never which object moved, appeared or
recolored.

The sample is thin. v293 is the only Lord run with transcripts: the offline
debug run, which scored 2.49 and took 8m 56s. Each game stopped at about 33–36k
tokens, and 11 of 25 games took no action. That leaves 354 python calls in 14
games, all from the first 7–22 turns, so read it as early-game behaviour.

### Frequency

| Measure | Count | Share of 354 calls |
|---|---:|---:|
| Calls touching the action result | 154 | 44% |
| Calls writing their own cell-by-cell diff of two frames | 29 | 8.2% |
| …of which define a function named `diff` | 4 | 1.1% |

The hand-written diff count is a heuristic: two frame references, `.rows` or
`.ascii`, `!=` and a loop. Its share is close to how often Daniel's agent calls
`frame_diff` (8.7%).

### Fields read

From [`lord_fields.py`](scripts/lord_fields.py), which traces values like
`frame_diff_fields.py`, so `bbox` on a node, a changed region and an animation
are counted apart. The result is read via `last_action_result` in 102 calls,
the `action()` return value in 45, and `history[i].result` in 3.

| Result key | Calls | Share |
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

| Nested object | Calls touching it | Shown whole | Subfields read by name |
|---|---:|---:|---:|
| `animation` | 39 | 39 | 0 of 11 |
| changed region | 16 | 16 | 0 of 2 (`bbox`, `pixel_count`) |

`animation` is present on 49 of the 171 attached results (29%). Its `type` is
`other` 39 times, which gives only a `bbox`; `tween` 7 and `path` 3.

Nodes (158 calls touch them) are read through `color` and `bbox` (99%),
`pixel_count` (85%), `id` (54%), `mask` (12%), `hash` (10%), `children` (8%)
and `boundary` (1%). Frame views are read through `segmentation` (158 calls),
`rows` (101), `crop` (49), `animation_frames` (36) and `step` (34).

## Observations

1. **`frame_diff` is a probe tool, used in 8.7% of calls.** It is used right
   after `action()`, without arguments, mostly at the start of a game: 63% of
   level 1's first three turns, against 8–9% on later levels. When a new level
   adds mechanics, the agent does not go back to it.
2. **The agent dumps the diff rather than picking fields.** 38% of calls
   print the whole diff. The fields read by name are few:
   - `changed_cell_count`, in 57% of calls
   - `moved`'s `color`/`from`/`to`, in 16%
   - `bbox`/`color`/`pixels` on `appeared`/`disappeared`, in 6–9%
   - `resized`'s pixel counts and positions, in 6–8%
3. **No hash field is ever read by name**: `hash`, `shape_hash`,
   `before_hash` and `after_hash` are read by name in 0 of 263 calls. Yet
   hashes are about 31% of a diff's text and 17% of what `frame_diff` calls
   displayed.
4. **Each category names position differently.** The keys are
   `moved.from/to`, `rotated.at`, `appeared.at/bbox`, `changed_color.at` and
   `resized.at_before/at_after`. Eight calls chain fallbacks such as
   `e.get('from') or e.get('at') or e.get('at_before')`. 20 calls read another
   category's position key on an entry that lacks it.
5. **`rotated` is nearly always empty (1.2% of transitions); `resized`
   nearly never is (96%).** `resized` is mostly backgrounds and one-pixel
   budget-bar changes.
6. **Without `frame_diff`, Lord's agent behaves alike.** It writes its own
   cell diffs in about the same share of calls (8.2%). It reads plain counts
   and flags (`changed_pixels`, `level_completed`) and prints the nested
   structures (`animation`, changed regions) whole, never reading one of
   their 13 subfields.

## Candidate changes

These are not decided; each needs an A/B test per the
[A/B guide](../rules/t-test.md).

- **Drop hash fields from `frame_diff` output.** Matching uses node hashes
  internally, so pairing does not change. Diffs shrink by about a third, and no
  code reads these fields. The token saving is about 0.15% of a run, too small
  to show in an A/B, so verify it with tests like PR #9 did.
- **Use one position layout for every category**, e.g. `bbox` before/after.
  This removes the fallback chains in observation 4.
- **Filter `resized` noise**, e.g. backgrounds above a size threshold. Keep
  budget bars visible, since the prompt tells the agent to watch them.
- **A/B removing `frame_diff` entirely.** This is the only ablation here that
  could move the score. Lord's agent shows the cost: it hand-writes cell diffs
  instead. The arm is `ARC3_FRAME_DIFF=0` in `harness_env` (default `1`, see
  [CONFIGURATION.md](../../ARC3-Inference/CONFIGURATION.md)). It removes the
  helper from the sandbox and every prompt mention of it: with `new-baseline`'s
  `harness_env`, that is 1,734 characters of system prompt.
- **Keep `rotated`**: an empty list is cheap and the entries are rare but
  informative.
