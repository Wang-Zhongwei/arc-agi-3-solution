[Game ls20 level-5](./images/lvl-5.png)

# Problems
## Forget and rediscover
There are three knobs
- Irregular gray knob controls the shape
- Cross gray knob controls the rotation
- Color square knob controls the color


- At step 291, agent realizes the cross gray knob controls the rotation so it heads up to the ireggualr knob. 
    - "Hmm, is that a rotation of the box "X.XXX..XX"?" 
- At step 326, agent get the correct shape but continue to toggle the shape one more time, why? 
    - model was forced to match the exact shape as the target


At the time of this ls20 run, the APIs (see [report](../../../comparison/comparison_report_zhongwei.md)) already exposed:
```python
class Node:
    id: int                        # list index, top-most then left-most
    hash: str                      # color + shape, rotation-INVARIANT
    pose_hash: str                 # color + shape, rotation-sensitive
    shape_hash: str                # shape only, color-free
    rotation: int | None           # 0/90/180/270 from canonical; None if fully symmetric
    rotational_symmetry: int       # 1, 2 or 4
    pixels: int

def frame_diff(before: Frame | None = None, after: Frame | None = None) -> dict: ...
    # defaults previous_frame -> current_frame; shape mismatch -> changed_cell_count = -1
    # {changed_cell_count, moved, rotated, appeared, disappeared, changed_color, resized}
    #
    # Real example from the ls20 transcript (level 2, action 37: DOWN onto the cross knob; the agent
    # printed it at step 41, transcripts/ls20-9607627b_p0.txt:4244). Coordinates are [row, col]; hashes truncated.
    # {
    #   'changed_cell_count': 62,
    #   'moved': [                      # player = O top half + b bottom half, moved 5 rows down
    #     {'color': 'O', 'pixels': 10, 'hash': 'eb47…', 'from': [40, 49], 'to': [45, 49]},
    #     {'color': 'b', 'pixels': 15, 'hash': '8249…', 'from': [42, 49], 'to': [47, 49]},
    #     {'color': 'b', 'pixels': 4,  'hash': 'e4ca…', 'from': [59, 3],  'to': [55, 3]},  # legend piece
    #   ],
    #   'rotated': [                    # <-- the cross knob's effect: legend box + shape turn 90° CW
    #     {'color': 'B', 'pixels': 76, 'hash': 'a7d3…', 'rotated_by': 90, 'at': [53, 1]},
    #     {'color': 'b', 'pixels': 20, 'hash': '0242…', 'rotated_by': 90, 'at': [55, 3]},
    #   ],
    #   'appeared': [],
    #   'disappeared': [                # the cross knob (W 3px + two w 1px) is covered by the player
    #     {'color': 'W', 'pixels': 3, 'hash': '2d4e…', 'shape_hash': '407b…', 'at': [46, 51], 'bbox': [46, 51, 47, 52]},
    #     {'color': 'w', 'pixels': 1, 'hash': 'ef8a…', 'shape_hash': 'da5e…', 'at': [47, 50], 'bbox': [47, 50, 47, 50]},
    #     {'color': 'w', 'pixels': 1, 'hash': 'ef8a…', 'shape_hash': 'da5e…', 'at': [48, 51], 'bbox': [48, 51, 48, 51]},
    #   ],
    #   'changed_color': [],
    #   'resized': [                    # floor reshapes around the player; Y fuel bar shrinks 20 -> 16
    #     {'color': 'G', 'at_before': [5, 9],   'at_after': [5, 9],   'before_pixels': 1354, 'after_pixels': 1359, 'before_hash': 'ce42…', 'after_hash': 'd46c…'},
    #     {'color': 'G', 'at_before': [61, 13], 'at_after': [61, 13], 'before_pixels': 64,   'after_pixels': 68,   'before_hash': '8efb…', 'after_hash': '97e0…'},
    #     {'color': 'Y', 'at_before': [61, 45], 'at_after': [61, 47], 'before_pixels': 20,   'after_pixels': 16,   ...},
    #   ],
    # }
```

To summarize, the agent learned the cross knob's function (rotate 90° CW) in level 2 via frame_diff's rotated_by, but in level 4 mis-attached the "rotates" label to the shape knob, and in level 5 treated the cross as a hazard that scrambles the legend; it only re-derived the rotation effect at step ~522, as the run ended. In level 5 it never called `frame_diff` and never read `pose_hash`/`shape_hash` for the legend across frames. Those would have shown the rotation, though no node hash can compare the legend to the box directly, because the two are drawn at different `scales`.


## Game over loops
The agent knows game reset but didn't trigger better replanning and reflection. call compare hud() of last frame and current frame might be useful. 


# Solutions
The current contract and remaining prompt work are defined in [Implementation](#implementation). The earlier, broader suggestions below are retained for reference; they are not the current scope.

1. Add `scale` and make `shape_hash` be scale invariant. Also `frame_diff` only shows a single transform. How to redesign it to make it compositional?
    - **Adjusted:** make *every* hash ignore scale (and position), add `scale`, `bbox`, `mirrored` and a logical-scale `grid`, and add `form_hash` ("same shape at all?"). A scale-invariant `shape_hash` alone isn't enough: it keeps orientation, so it still can't say "same shape, different rotation", which was the agent's exact question at step 326. (§1)
    - **Is `frame_diff` compositional?** Only partly. It already reports rotate + translate as one change, but anything involving color, mirroring, scale or a new shape falls apart (table in §4). Make it list every op per matched object. That's needed for ARC in general, but it would not have fixed ls20 by itself: each ls20 knob applies one op per step. The composition happens across steps and against the target, which needs `relate()` (§3) and a memory of which knob does what (§7).
2. frame[10:20][10:20].segmentation["nodes"], add bbox to node
    - **Adjusted:** keep, with two changes. Use a 2-D slice `frame[10:20, 10:20]` (the chained form slices rows twice), and give the region both `.nodes` ("what's here?") and `.shape` ("this area as ONE shape"). Also add `frame.contents(box)` and multi-color `frame.objects`. The ls20 legend and target are each two nodes, so only a whole-area shape can be compared. (§2)
3. Prompt agent to define hud() in system prompt.
    - **Adjusted:** the harness provides `hud()` and `hud_diff()`; the agent only adds widgets the harness missed. The ls20 agent wrote `yellowbar()` but never tracked lives, and an agent-defined `hud()` can't drive harness messages such as "life lost". `hud_diff()` is the "compare hud() of last frame and current frame" from *Game over loops*. (§5)
4. **New: `action(..., until=...)`.** Both overshoots were batches without a working stop: the step-314 loop waited for an exact pose, and the step-374 batch printed MATCH at move 2 and kept going. (§6)
5. **New: harness-recorded mechanics + goal watch in the user prompt.** Better APIs don't fix forgetting. The agent learned "cross = rotate" in level 2 and had relabelled it by level 5. A ledger built from frames can't drift, and a per-turn goal line keeps "remaining: rotate 90°" in view. (§7, §8)


# Implementation

## Node
Implemented node contract. Nodes remain same-color, 4-connected components with containment; compare explicitly selected parts when a pattern spans multiple nodes.

Nodes remain dictionaries, accessed as `node["children"]`, for example. The types below describe their fields.

```python
from typing import TypedDict

class Node(TypedDict):
    id: int                         # index in this frame's nodes, top-most then left-most
    color: str                      # this component's color
    pose_hash: str                  # exact colored mask, ignoring position
    hash: str                       # colored mask, ignoring position and D4 pose
    shape_hash: str                 # binary mask, ignoring position, color and D4 pose
    form_hash: str                  # shape_hash after exact integer scale reduction
    silhouette_hash: str            # shape_hash after filling enclosed holes
    scale: int                      # largest exact square-block factor; >= 1
    rotation: int | None            # clockwise angle AFTER reflection; None if 4-fold symmetric
    mirrored: bool | None           # reflect canonical mask left-right; None if ambiguous
    rotational_symmetry: int        # matching quarter-turn rotations: 1, 2 or 4
    pixels: int                     # own pixel count, excluding children
    boundary: list[list[int]]       # outer-contour corner cells [row, col]; holes omitted
    bbox: tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1), full-frame coordinates
    children: list[int]             # IDs of directly enclosed nodes
    parent: int | None              # ID of innermost enclosing node, or None
```

### Hash meanings

A mask is the node's tightly cropped binary occupancy: `1` for its own pixels, `0` elsewhere, including children. All hashes ignore position. D4 pose means the four quarter-turn rotations and the four rotations of a left-right reflection.

| Field | Differences ignored | Differences preserved |
|---|---|---|
| `pose_hash` | Position | Color, rotation, reflection, size, holes |
| `hash` | Position, rotation, reflection | Color, size, holes |
| `shape_hash` | Position, rotation, reflection, color | Size, holes |
| `form_hash` | Position, rotation, reflection, color, uniform integer scale | Holes and remaining geometry |
| `silhouette_hash` | Position, rotation, reflection, color, enclosed holes | Size and exterior geometry |

“Preserved” means no normalization removes that property; a symmetric mask can already equal its reflection or rotation. Hashes describe appearance equivalence, not persistent object identity.

Let `canonical(mask)` select the lexicographically smallest `(height, width, row-major bits)` encoding[^row-major] among its eight D4 transforms. Use the same digest and mask encoding for the three color-free hashes:

- `pose_hash`: digest of `(color, observed mask encoding)`.
- `hash`: digest of `(color, canonical(mask))`.
- `shape_hash`: digest of `canonical(mask)`.
- `form_hash`: digest of `canonical(reduced mask)`.
- `silhouette_hash`: digest of `canonical(hole-filled mask)`.

Equality follows `pose_hash → hash → shape_hash → form_hash`. `silhouette_hash` is a separate comparison: it equals `shape_hash` when there are no holes. With `scale=1`, `form_hash` also equals `shape_hash`.

[^row-major]: Read each row left to right, then move downward. For this L-shaped mask, `1` marks the node's own pixels and `0` marks empty cells:

    ```text
    1 0
    1 0
    1 1
    ```

    Concatenating the rows `10`, `10`, and `11` gives `101011`. The full `(height, width, row-major bits)` encoding is `(3, 2, (1, 0, 1, 0, 1, 1))`.

### Scale and silhouette

`scale` is the largest integer k dividing both mask dimensions such that every aligned k×k block is entirely `0` or entirely `1`, starting at the crop's top-left. Replace each block by one cell to obtain the reduced mask. If no k > 1 works, `scale=1`.

A primitive 3-cell L has scale 1; its uniform ×2 enlargement has scale 2 and 12 pixels. A filled 4×4 square has scale 4. This is a geometric reduction factor, not a claim about the game's intended cell size. Compute it from the node's own mask before filling holes.

For `silhouette_hash`, fill zero-cells unreachable from a padded exterior by four-neighbor steps. Fill enclosed holes only; preserve open gaps and concavities. A blue ring and a solid blue square of the same outer dimensions have equal silhouette hashes but different shape hashes. The fill includes enclosed child pixels and background holes without creating a composite node or changing `pixels`.

### D4 pose and symmetry

Use the canonical reduced, color-free mask as the reference for both pose fields. The reconstruction order is fixed:

```text
canonical → optional left-right reflection → clockwise rotation → enlargement
```

Enumerate the eight `(mirrored, rotation)` pairs that could map the canonical mask to the observed reduced mask:

- If only one reflection flag has valid matches, return that flag and its smallest valid clockwise angle.
- If both flags have valid matches, `mirrored=None`: reflection is indistinguishable from a rotation for this shape. Use the unmirrored matches to select the smallest angle.
- `rotational_symmetry` counts the rotations in `{0, 90, 180, 270}` that leave the mask unchanged. If it is 4, return `rotation=None`; otherwise return the angle selected above.

For reconstruction, interpret `mirrored=None` as no reflection and `rotation=None` as zero. The null values report ambiguity, not missing computation. `mirrored` describes a pose relative to the canonical reference; it does not establish that a reflection action occurred.

Mathematically, this field is reflection parity relative to the reference: `False` is orientation-preserving, `True` is orientation-reversing, and `None` means both are possible. Keep the agent-facing name `mirrored`. Chirality is a property of the shape (whether its mirror can be matched by rotation), not its current pose.

| Mask | `mirrored` | `rotation` | `rotational_symmetry` |
|---|---|---|---|
| Square | `None` | `None` | 4 |
| Non-square rectangle | `None` | 0 or 90, relative to canonical | 2 |
| Shape with no rotation or reflection symmetry | `False` or `True` | 0, 90, 180 or 270 | 1 |

The four rotations form C4; adding reflections gives D4. Scale reduction and hole filling are separate normalizations.

### Geometry and containment

`id` is a frame-local list index, not an identity across frames. `bbox` and `boundary` use full-frame row/column coordinates. `boundary` is the existing ordered outer-contour corner list; it omits inner hole contours. Hash masks must come from the component's cells, not just its outer boundary.

`children` lists directly enclosed node IDs; `parent` is the inverse relation, with `None` for an unenclosed node. Enclosure means every four-neighbor path from the enclosed component to the frame edge crosses the enclosing component. Choose the innermost encloser. Both fields describe containment, not object membership: an 8-pixel blue ring can contain a 1-pixel red child, and its own mask still has a hole.

### Frame slicing

`frame[r0:r1, c0:c1]` returns the same frame interface, including `.ascii`, `.shape`, and `.segmentation`, and works directly with `frame_diff`. Slicing is implemented in the sandbox `FrameView`.

```python
a = before[1:2, 2:3]                # half-open bounds: one row and one column
b = after[1:2, 2:3]
diff = frame_diff(a, b)              # same as frame_diff(before[1:2, 2:3], after[1:2, 2:3])
nodes = a.segmentation["nodes"]
```

- Slices use Python's half-open bounds and unit steps. The crop keeps the frame's step/level; nested slices accumulate their offset into the original frame.
- Segment the cropped pixels. IDs and containment refer to that crop's node list; bbox/boundary coordinates remain in the original frame. Components crossing a crop edge are clipped, so their hashes and relations describe only the visible fragment.
- `frame_diff` accepts full frames or crops with the same shape and original-frame bounds. Count changed cells only inside those bounds; compare the supplied views' nodes. Reject different origins; retain the existing `changed_cell_count=-1` result for different shapes.

### Compatibility with frame_diff

Daniel's original API had a rotation-invariant `hash` and a color-free, orientation-sensitive `shape_hash`. The implementation now uses the D4 meanings and canonical rotation reference defined above.

- Equal `hash` values can include reflected poses. Match classification verifies rotations and reflections against the masks instead of subtracting canonical rotation angles.
- Exact recoloring requires equal `shape_hash`, original color-free masks, and bboxes, with different colors. A changed pose cannot be classified as only a recolor. This needs no additional public hash.
- `resized` pairs remaining same-color nodes with different `form_hash` values using bounding-box overlap. It can include a 4-pixel shape changing into another 4-pixel shape. Exact uniform scaling is verified separately and reported as `scaled`.

These contracts are implemented and covered by focused tests. The [matching study](../../matching/README.md) records the assignment comparison and its limitations; it does not measure game-solving performance.

### System prompt edit

Use a short field guide alongside the schema:

> `pose_hash` compares exact colored masks. `hash` also allows rotation and reflection; `shape_hash` also ignores color; `form_hash` also ignores uniform scale. `silhouette_hash` compares exterior shapes at the same size, ignoring holes. `rotation` and `mirrored` describe pose relative to a canonical mask, not actions that occurred. Node masks exclude children.

**Pending behavioral prompt edit:** retain observed control effects; check each action when searching for a state and stop when the property being changed matches; if a budget or lives indicator is identified, read it with an agent-written `hud(frame)` helper and compare readings before retrying after a restart. Replace conflicting bar/batching guidance in place.

## Frame_diff

```python
# Implemented return schema: types stand in for values; lists may have zero or more entries.
# IDs belong to the supplied frames/crops; bboxes use original-frame coordinates.
# Bboxes are inclusive (r0, c0, r1, c1).
pair = {
    "before_id": int, "after_id": int,
    "before_bbox": tuple[int, int, int, int],
    "after_bbox": tuple[int, int, int, int],
}
node = {"id": int, "color": str, "hash": str,
        "bbox": tuple[int, int, int, int], "pixels": int}

result = {
    "changed_cell_count": int,
    "moved": [{**pair, "hash": str}],
    "rotated": [{**pair, "hash": str, "rotated_by": int}],  # 90 / 180 / 270 clockwise
    "mirrored": [{**pair, "hash": str, "axis": str}],
        # "-": horizontal axis -> reverse rows
        # "|": vertical axis -> reverse columns
        # "/": anti-diagonal -> transpose, then reverse both axes
        # "\\": main diagonal -> transpose (Python literal for one backslash)
    "scaled": [{**pair, "form_hash": str,
                "before_scale": int, "after_scale": int,
                "scale_factor": float}],  # after_scale / before_scale
    "changed_color": [{**pair, "shape_hash": str,
                       "before_color": str, "after_color": str}],
    "resized": [{**pair, "color": str,
                 "before_pixels": int, "after_pixels": int}],
    "appeared": [node],       # after-frame nodes
    "disappeared": [node],    # before-frame nodes
}
```

### Grouping rules

Use hashes to find candidates, then verify masks. Omit unchanged nodes first. Within each form group, Hungarian assignment maximizes the number of valid pairs, then minimizes total bbox-center Manhattan distance. Equal-cost assignments are resolved deterministically by solver order; they do not establish persistent identity. Each pair belongs to one bucket only. `resized` uses the separate fallback below.

| Bucket | Rule |
|---|---|
| `moved` | Equal `pose_hash`; only position changes. |
| `rotated` | Equal `hash`; one nonzero rotation maps the masks; bbox centers *differ by at most one pixel per axis*. Choose the smallest valid angle. |
| `mirrored` | Equal `hash`; one reflection maps the masks; same center tolerance. Prefer `rotated` when both work; otherwise choose the first valid axis in the order listed above. |
| `scaled` | Equal `form_hash`, color, and oriented reduced masks; different scales, bbox centers differ by at most one pixel per axis. Verify exact uniform scaling. |
| `changed_color` | Equal `shape_hash`, original binary masks, and bboxes; different colors. |
| `resized` | After exact matches: same color, different `form_hash`, bbox intersection area / smaller bbox area ≥ 0.5. Pair greedily by overlap, then bbox-area similarity; leave tied candidates sharing a node unmatched. This is a geometry-change heuristic, not exact scaling. |
| `appeared` / `disappeared` | Unmatched after/before nodes, including unsupported combinations. |

Transform fields describe the before-to-after change, not the nodes' canonical pose. No combined-change records.

## Minimal prompt changes for frame_diff

`prompts.py` now documents slicing in the existing frame description and the new return fields in `FRAME_DIFF_HINT_ADDENDUM`. The concise guidance is:

> `frame[r0:r1, c0:c1]` returns a cropped frame with half-open bounds. Crop node IDs are local to the crop; coordinates stay in the original frame. Edge-crossing components are clipped.
>
> `frame_diff(before=None, after=None)` defaults to `previous_frame` and `current_frame`. It also accepts matching crops: `frame_diff(before[1:2, 2:3], after[1:2, 2:3])`.
>
> Returns `changed_cell_count` and lists `moved`, `rotated`, `mirrored`, `scaled`, `changed_color`, `resized`, `appeared`, `disappeared`. Matched entries include `before_id`, `after_id`, `before_bbox`, `after_bbox`; appearance/disappearance entries include `id`, `bbox`, `color`, `hash`, `pixels`.
>
> `rotated_by` is clockwise degrees; `mirrored.axis` names the reflection; `scaled.scale_factor` is the verified uniform scale ratio. Recolor entries give `shape_hash`, `before_color`, `after_color`. `resized` is a same-color overlap heuristic with `before_pixels` and `after_pixels`, not proof of scaling. Unchanged nodes are omitted; unsupported combinations remain unmatched. Hashes are not unique tracking IDs.

For the user prompt, the existing auto-diff renderer in `tool_agent.py` (`_DIFF_BUCKETS`, `_diff_entry_line`, `_render_auto_frame_diff`) includes `mirrored` and `scaled` and uses the new IDs/bboxes. Discarded coordinate fields and combined move/rotation entries are removed. Enablement, token budget, and truncation are preserved, with no additional instruction block. Aggregation resolves color/pixel metadata from the paired nodes.

```text
- rotated: node 3->7, 90° CW, bbox (...) -> (...)
- mirrored: node 4->8, axis |, bbox (...) -> (...)
- scaled: node 5->9, scale 1->2 (×2), bbox (...) -> (...)
- changed_color: node 6->10, B->R, bbox (...)
- resized: node 11->12, color B, pixels 4->5, bbox (...) -> (...)
```
