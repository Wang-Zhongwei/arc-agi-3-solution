# Which `frame_diff` fields the agent reads

Companion to [segmentation node fields](segmentation-node-fields.md). This
note counts how often the agent reads each top-level key and entry field that
`frame_diff()` returns, in Daniel's baseline, and how often each category is
non-empty. [Lord Hansolo's harness](lord-result-fields.md) has no
`frame_diff`; its change fields are counted separately.

## Method

Two runs of the same code, pooled, under `ARC3-Inference/runs/kaggle/`:

- `dangthuygiang/arc3-v207-danibase-base-dfranzen/v05` (25 games, 48.36)
- `dfranzen/arc-agi-3-milestone-2-solution/v01` (10 games, 36.56)

That makes 35 game runs over 25 distinct games. Their `frame_diff` matches
`main`'s except that `appeared`/`disappeared` entries also carry `shape_hash`
(removed in
[PR #9](https://github.com/Wang-Zhongwei/arc-agi-3-solution/pull/9)).

```bash
cd ARC3-Inference/runs/kaggle
A=dangthuygiang/arc3-v207-danibase-base-dfranzen/v05
B=dfranzen/arc-agi-3-milestone-2-solution/v01
python3 -I ../../../docs/harness/scripts/frame_diff_fields.py $A/transcripts/*.txt $B/transcripts/*.txt
python3 -I ../../../docs/harness/scripts/frame_diff_contents.py $A $B
```

Entry fields share names with node fields (`color`, `pixels`, `hash`, `at`,
`bbox`), so the regex from the node-field note would mix them up.
[`frame_diff_fields.py`](scripts/frame_diff_fields.py) parses each `python`
call that calls `frame_diff` and traces values from `frame_diff(...)` to a
category list (`d['moved']`) to an entry (`for m in d['moved']`). It counts two
things:

- **Named**: the code reads the key or field by name, e.g. `d['moved']`,
  `m['from']`, or `d.get(k)` with `k` looping over a literal tuple of keys.
- **Shown**: the code prints or stores a traced value whole, e.g. `print(d)`,
  `for k, v in d.items(): print(k, v)`, or `[m for m in d['moved']]`. Every field
  is displayed, but the code never picks one.

[`frame_diff_contents.py`](scripts/frame_diff_contents.py) recomputes
`frame_diff` with `main`'s code for every consecutive pair of action boards in
`artifacts/*_events.jsonl`. It also measures hash text in the tool results of
`frame_diff` calls.

Caveats:

- A field read in a loop over several categories counts for each of them.
- Whole views are often truncated (`str(v)[:600]`), so "shown" is an upper
  bound on what the model saw.
- A reference in code is not proof the value changed a decision.
- The displayed hash share also counts node hashes printed in the same call.

## Evidence

### Calls

| Python calls | `frame_diff` calls | Iterate keys generically | Show the whole diff |
|---:|---:|---:|---:|
| 3,009 | 263 (8.7%) | 112 (43%) | 100 (38%) |

All calls parsed. Percentages below are over the 263 `frame_diff` calls.

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

### Entry fields, read by name

Raw call counts. The last column is the share of calls that show that
category's entries whole.

| Category | Fields read by name | Hash fields | Entries shown whole |
|---|---|---|---:|
| `moved` | `color` 43, `to` 42, `from` 41, `rotated_by` 12, `pixels` 6 | `hash` 0 | 64% |
| `appeared` | `color` 24, `bbox` 22, `pixels` 18, `at` 8 | `hash` 0, `shape_hash` 0 | 61% |
| `disappeared` | `color` 24, `bbox` 22, `pixels` 16, `at` 8 | `hash` 0, `shape_hash` 0 | 61% |
| `resized` | `color` 22, `before_pixels` 21, `after_pixels` 17, `at_before` 16, `at_after` 16 | `before_hash` 0, `after_hash` 0 | 56% |
| `changed_color` | `at` 10, `pixels` 4, `before_color` 4, `after_color` 4 | `before_hash` 0, `after_hash` 0 | 56% |
| `rotated` | `at` 4, `pixels` 2, `color` 2, `rotated_by` 0 | `hash` 0 | 45% |

### What the diff contains

Recomputed over the 6,938 action transitions that changed at least one cell:

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
| `str(diff)` characters, median / p90 (per run) | 993 / 2,362 (v05), 1,041 / 4,363 (v01) |
| Hash key/value share of `str(diff)`, without `shape_hash` | 31% |
| `frame_diff` results showing a hash | 115 (44%) |
| Hash share of those results' displayed characters | 34,185 of 204,154 (17%) |

In v01, 24% of `resized` entries are objects of 300 px or more, i.e.
backgrounds. The most common size change is exactly one pixel (1,453 of 3,983
entries).

## Observations

1. **No hash field is ever read by name.** That covers `hash`, `shape_hash`,
   `before_hash` and `after_hash`, in 263 calls. Hashes still make up about 31%
   of a diff's text and 17% of what `frame_diff` calls displayed. Daniel's
   version also carried `shape_hash`, so its true share was higher.
2. **Code mostly dumps the diff rather than picking fields.** 38% of calls
   print the whole diff, and 45–64% show some category's entries whole. The
   fields read by name are a few: `changed_cell_count`, `moved`'s
   `color`/`from`/`to`, `bbox` and `pixels` on `appeared`/`disappeared`, and
   `resized`'s pixel counts and positions.
3. **Each category names position differently, and the agent works around
   it.** The keys are `moved.from/to`, `rotated.at`, `appeared.at/bbox`,
   `changed_color.at` and `resized.at_before/at_after`. Eight calls chain
   fallbacks such as `e.get('from') or e.get('at') or e.get('at_before')`. In
   20 calls the code reads a field that belongs to another category on an
   entry that lacks it (e.g. `moved.at`, `resized.from`). It does this mostly
   via `.get` in one accessor applied across categories. No call reads a key
   that no category has, and none of the errors in `frame_diff` calls is a
   KeyError on a diff field.
4. **`rotated` is nearly always empty**: 1.2% of transitions. Only 11% of
   calls read it by name. `moved.rotated_by` is read 12 times.
5. **`resized` is non-empty on 96% of changed transitions.** It is mostly
   backgrounds and one-pixel changes, which look like budget bars. This makes
   `resized` the noisiest category and one reason whole dumps get long.

## Candidate changes

These are not decided; each needs an A/B test per the
[A/B guide](../rules/t-test.md).

- **Drop hash fields from `frame_diff` output.** Matching uses node hashes
  internally, so the returned `hash`/`before_hash`/`after_hash` can go without
  changing which objects pair up. Expected effect: shorter diffs (about a
  third of `str(diff)`), with no code that reads them.
- **Use one position layout for every category**, e.g. `bbox` before/after.
  This removes the fallback chains in observation 3.
- **Filter `resized` noise**, e.g. backgrounds above a size threshold, or
  report one-pixel bar changes separately.
- **Keep `rotated`** as long as it is cheap when empty. Entries are rare, so
  removing it would save almost nothing.
