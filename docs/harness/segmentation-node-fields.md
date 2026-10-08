# Which segmentation node fields the agent reads

Segmentation nodes on `main` had ten fields. This note counts how often the
agent reads each one in its `python` tool code, and uses the counts to decide
which fields to keep. The change ships in
[PR #9](https://github.com/Wang-Zhongwei/arc-agi-3-solution/pull/9).

## Method

For every `python` tool call in a run's `transcripts/*.txt`, I took the code
from the `raw_tool_calls` arguments. A call counts as a segmentation call when
its code contains `segmentation`. A field counts as read when the code has
`n['field']`, `n.get('field')` or `'field')`. Percentages are over segmentation
calls.

```bash
cd ARC3-Inference/runs/kaggle
python3 -I ../../../docs/harness/scripts/node_fields.py \
  id,color,hash,rotation,rotational_symmetry,pose_hash,shape_hash,pixels,boundary,children,bbox \
  dangthuygiang/arc3-v207-danibase-base-dfranzen/v05/transcripts/*.txt
python3 -I ../../../docs/harness/scripts/hash_mentions.py dangthuygiang/arc3-v207-danibase-base-dfranzen/v05
```

Caveats:

- `color`, `pixels`, `id` and `hash` are also keys in `frame_diff` entries, so
  a call reading both can count for either.
- A reference in code is not proof the value changed a decision.
- The runs differ in model, prompt and scheduler, so compare patterns, not
  exact percentages.

## Evidence

### Runs with `main`'s field set (no node `bbox`)

Run paths are under `ARC3-Inference/runs/kaggle/`.

| Run | Games | Score | Python calls | Seg calls | `color` | `pixels` | `boundary` | `id` | `children` | `rotation` | `rotational_symmetry` | `hash` | `shape_hash` | `pose_hash` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dangthuygiang/arc3-v207-danibase-base-dfranzen/v05` (Daniel base) | 25 | 48.36 | 2,479 | 421 (17%) | 99% | 92% | 85% | 31% | 10% | 6% | 4% | 1% | 0% (2) | 0% |
| `dfranzen/arc-agi-3-milestone-2-solution/v01` | 10 | 36.56 | 530 | 147 (28%) | 99% | 95% | 97% | 40% | 12% | 7% | 5% | 1% | 0% | 0% |
| `jonathanwang2022/persistent-animations/v01` | 25 | 50.45 | 2,543 | 575 (23%) | 99% | 94% | 87% | 26% | 9% | 6% | 3% | 1% | 0% | 0% |
| `ticbudja/thinking-budget-and-user-prompt-trim/v03` | 25 | 52.92 | 2,551 | 530 (21%) | 99% | 93% | 92% | 26% | 10% | 4% | 2% | 1% | 1% (5) | 0% (1) |
| `ticbudja/yield-ucb-scheduling/v01` | 10 | 94.54 | 925 | 153 (17%) | 99% | 98% | 96% | 35% | 22% | 3% | 3% | 5% | 0% | 0% |
| `jonathanwang2022/bp35-lordhansolo/v12` | 25 | 46.18 | 3,008 | 992 (33%) | 99% | 83% | 96% | 17% | 5% | 3% | 0% (4) | 1% | 1% (6) | 0% (2) |

Brackets give raw call counts where the percentage rounds to 0–1%.

### Runs whose nodes have `bbox`

| Run | Games | Score | Python calls | Seg calls | `bbox` | `color` | size | `id` | `children` | `boundary` |
|---|---|---|---|---|---|---|---|---|---|---|
| [LohitSiriki/arc-agi-3-milestone2-solution](https://github.com/LohitSiriki/arc-agi-3-milestone2-solution) `run/transcripts`, our 11.64 harness from duck-harness@6420079 | 25 | public LB 22.53 | 663 | 349 (53%) | 99% | 99% | `area` 81% | 44% | 17% | 5% |
| `lordhansolo/built-on-tufa-labs-duck-harness-milestone-2/v293` | 25 | 2.49 | 354 | 186 (53%) | 100% | 99% | `pixel_count` 84% | 46% | 7% | 1% |

Only 14 of the 25 Lord Hansolo games read segmentation at all, so that row is
weak. Its nodes also carry `mask` (12%) and `hash` (9%).

### `pose_hash` and `shape_hash` anywhere in the code

Across all 12,036 python calls in the six `main`-field runs, counting any
mention, including on `frame_diff` entries:

| | Calls |
|---|---|
| mention `pose_hash` | 3 |
| mention `shape_hash` | 15 |
| mention `shape_hash` and call `frame_diff` | 2 |
| call `frame_diff` | 979 |

## Observations

1. **`boundary` is read as a stand-in for a bounding box.** Without a node
   `bbox`, 85–97% of segmentation calls read `boundary`. With one, 1–5% do.
   `persistent-animations/v01` alone has 514 snippets taking `min`/`max` over
   boundary points.
2. **`pose_hash` and `shape_hash` are close to unused**: at most 6 calls per
   run, 3 and 15 calls in total. The prompt still spent two clauses
   describing them.
3. **`rotation`, `rotational_symmetry` and `hash` are rarely read directly**
   (0–7%), but `frame_diff` uses all three to match objects and report
   rotations, so they cannot go without changing `frame_diff`.
4. **Segmentation gets read more when nodes have `bbox`.** Both `bbox` runs
   read it in 53% of python calls, against 17–33% for the rest. This is
   confounded by harness and prompt differences, so it is a hypothesis for the
   A/B test, not a result.
5. **The model does not confuse node `pixels` with a list of cells.** In
   Daniel's base run, 0 of 411 calls that read `pixels` iterate, index or
   `len()` it. The run has 10 int-as-list `TypeError`s, and none involve node
   `pixels`. At least 4 come from animation fields: `transient_pixels` (2 of
   15 calls) and timeline `changed` (2 of 86). Both are counts, but the prompt
   does not say so.

## Decision

- **Add `bbox`** to every node, `[min_row, min_col, max_row, max_col]`
  inclusive, the layout `frame_diff` already uses.
- **Remove `pose_hash` and `shape_hash`** from nodes and from `frame_diff`'s
  `appeared`/`disappeared` entries. `frame_diff` still pairs recolored objects
  by their color-free shape. The signature now sits on a `Node.shape_signature`
  attribute outside the dict, so printing a node shows only documented fields.
- **Order fields by use**, in the prompt and the node dict: `id`, `color`,
  `hash`, `bbox`, `pixels`, `rotation`, `rotational_symmetry`, `boundary`,
  `children`.
- **Keep** `hash`, `rotation` and `rotational_symmetry`, because `frame_diff`
  depends on them.
- **Not changed here:** stating that `transient_pixels` and `changed` are cell
  counts (observation 5). That is a separate prompt edit.

## Verification

- `ARC3-Inference/tests/test_segmentation.py`:
  - checks the bbox values;
  - checks nodes carry exactly the nine documented fields;
  - checks `frame_diff` still reports a recolor and leaks no hidden keys;
  - checks the same inside a real sandbox run.
- All 149 tests in `ARC3-Inference/tests` pass.
- Old and new `compute_frame_diff` give identical output on 400 random grid
  pairs (recolors, noise, shifts) once `shape_hash` is dropped from the old
  output. That covers 227 `changed_color` pairings. The old and new
  `shape_hash` values are identical too.
- Not yet A/B tested on Kaggle. The PR stays open until a comparison per the
  [A/B guide](../rules/t-test.md) is in.
