"""Segmentation nodes carry an inclusive ``bbox`` and only the documented fields, on the
host and inside the Python tool sandbox; frame_diff still pairs recolored objects."""

import random

from inference.agent.prompts import STRUCTURED_RUNTIME_STATE_ADDENDUM
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.frame_diff import _node_bbox, compute_frame_diff
from inference.utils.grid_utils import ARC_COLOR_CHARS, format_grid_ascii
from inference.utils.segmentation import segment_layer

NODE_FIELDS = {
    "id", "color", "hash", "rotation", "rotational_symmetry",
    "pixels", "bbox", "boundary", "children",
}

# A 4x5 blue (9) ring around a red (8) cell, on a white (0) background.
RING = [
    [0, 0, 0, 0, 0, 0],
    [0, 9, 9, 9, 9, 9],
    [0, 9, 8, 0, 0, 9],
    [0, 9, 9, 9, 9, 9],
]
# The same ring, recolored dark gray (3).
RECOLORED = [[3 if v == 9 else v for v in row] for row in RING]


def _node(nodes, color):
    (node,) = [n for n in nodes if n["color"] == color]
    return node


def _nodes(grid):
    return segment_layer(grid, ARC_COLOR_CHARS)["nodes"]


def _frame(grid):
    return {
        "ascii": format_grid_ascii(grid),
        "step": 0,
        "level": 1,
        "shape": [len(grid), len(grid[0])],
        "grid": grid,
    }


def test_bbox_is_inclusive_and_spans_enclosed_children():
    nodes = _nodes(RING)
    ring = _node(nodes, ARC_COLOR_CHARS[9])
    red = _node(nodes, ARC_COLOR_CHARS[8])

    assert ring["bbox"] == [1, 1, 3, 5]
    assert ring["pixels"] == 12
    assert red["id"] in ring["children"]
    assert red["bbox"] == [2, 2, 2, 2]


def test_bbox_matches_the_cells_and_frame_diff_bbox():
    rng = random.Random(0)
    for _ in range(25):
        grid = [[rng.choice([0, 0, 0, 3, 9]) for _ in range(12)] for _ in range(10)]
        for node in _nodes(grid):
            r0, c0, r1, c1 = node["bbox"]
            assert 0 <= r0 <= r1 < 10 and 0 <= c0 <= c1 < 12
            assert (r1 - r0 + 1) * (c1 - c0 + 1) >= node["pixels"]
            # boundary corners are the component's own cells, so their extremes are the bbox
            assert node["bbox"] == list(_node_bbox(node))


def test_nodes_carry_only_documented_fields():
    for node in _nodes(RING):
        assert set(node) == NODE_FIELDS
    for field in ("pose_hash", "shape_hash"):
        assert field not in STRUCTURED_RUNTIME_STATE_ADDENDUM


def test_frame_diff_pairs_a_recolor_without_exposing_the_shape_signature():
    diff = compute_frame_diff(RING, RECOLORED, _nodes(RING), _nodes(RECOLORED))

    assert [(e["at"], e["before_color"], e["after_color"]) for e in diff["changed_color"]] == [
        ([1, 1], ARC_COLOR_CHARS[9], ARC_COLOR_CHARS[3])
    ]
    assert diff["appeared"] == [] and diff["disappeared"] == []


def test_frame_diff_entries_carry_no_hidden_keys():
    rng = random.Random(1)
    for _ in range(25):
        a = [[rng.choice([0, 0, 0, 3, 9]) for _ in range(12)] for _ in range(10)]
        b = [[rng.choice([0, 3, 9]) if rng.random() < 0.2 else v for v in row] for row in a]
        diff = compute_frame_diff(a, b, _nodes(a), _nodes(b))
        for bucket in ("moved", "rotated", "appeared", "disappeared", "changed_color", "resized"):
            for entry in diff[bucket]:
                assert not any(k.startswith("_") or k == "shape_hash" for k in entry), entry


def test_sandbox_nodes_and_frame_diff():
    out = run_sandboxed_python(
        code=(
            "ring = [n for n in current_frame.segmentation['nodes'] if n['pixels'] == 12][0]\n"
            "d = frame_diff()\n"
            "result = {'bbox': ring['bbox'], 'fields': sorted(ring),\n"
            "          'recolors': [[e['before_color'], e['after_color']] for e in d['changed_color']]}\n"
        ),
        timeout_seconds=20,
        initial_state={
            "current_frame": _frame(RECOLORED),
            "history": [
                {"action": "", "frame": _frame(RING)},
                {"action": "ACTION1", "frame": _frame(RECOLORED)},
            ],
        },
        action_handler=lambda *a, **k: {},
    )

    assert out.get("error") == "", out
    assert out["result"] == {
        "bbox": [1, 1, 3, 5],
        "fields": sorted(NODE_FIELDS),
        "recolors": [[ARC_COLOR_CHARS[9], ARC_COLOR_CHARS[3]]],
    }
