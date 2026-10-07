"""Segmentation nodes carry an inclusive ``bbox``, on the host and inside the Python tool sandbox."""

import random

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.frame_diff import _node_bbox
from inference.utils.grid_utils import ARC_COLOR_CHARS, format_grid_ascii
from inference.utils.segmentation import segment_layer

# A 4x5 blue (9) ring around a red (8) cell, on a white (0) background.
RING = [
    [0, 0, 0, 0, 0, 0],
    [0, 9, 9, 9, 9, 9],
    [0, 9, 8, 0, 0, 9],
    [0, 9, 9, 9, 9, 9],
]


def _node(nodes, color):
    (node,) = [n for n in nodes if n["color"] == color]
    return node


def test_bbox_is_inclusive_and_spans_enclosed_children():
    nodes = segment_layer(RING, ARC_COLOR_CHARS)["nodes"]
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
        for node in segment_layer(grid, ARC_COLOR_CHARS)["nodes"]:
            r0, c0, r1, c1 = node["bbox"]
            assert 0 <= r0 <= r1 < 10 and 0 <= c0 <= c1 < 12
            assert (r1 - r0 + 1) * (c1 - c0 + 1) >= node["pixels"]
            # boundary corners are the component's own cells, so their extremes are the bbox
            assert node["bbox"] == list(_node_bbox(node))


def test_sandbox_nodes_expose_bbox():
    frame = {
        "ascii": format_grid_ascii(RING),
        "step": 0,
        "level": 1,
        "shape": [len(RING), len(RING[0])],
        "grid": RING,
    }
    out = run_sandboxed_python(
        code=(
            "ring = [n for n in current_frame.segmentation['nodes'] if n['pixels'] == 12][0]\n"
            "result = ring['bbox']\n"
        ),
        timeout_seconds=20,
        initial_state={"current_frame": frame, "history": [{"action": None, "frame": frame}]},
        action_handler=lambda *a, **k: {},
    )

    assert out.get("error") == "", out
    assert out["result"] == [1, 1, 3, 5]
