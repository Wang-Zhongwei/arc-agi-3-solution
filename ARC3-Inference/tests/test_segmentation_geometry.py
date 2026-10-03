import pytest

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.segmentation import segment_layer


COLORS = ".BRG"


def test_bbox_uses_extrema_not_first_occupied_cell():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    node = next(n for n in segment_layer(grid, COLORS)["nodes"] if n["color"] == "B")

    assert node["bbox"] == (1, 1, 3, 3)
    assert grid[node["bbox"][0]][node["bbox"][1]] == 0
    assert node["pixels"] == 5


def test_parent_is_innermost_encloser_and_inverse_of_direct_children():
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2, 1, 0],
        [0, 1, 2, 3, 2, 1, 0],
        [0, 1, 2, 2, 2, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    nodes = segment_layer(grid, COLORS)["nodes"]

    assert [n["parent"] for n in nodes] == [None, 0, 1, 2]
    assert [n["children"] for n in nodes] == [[1], [2], [3], []]
    assert [n["bbox"] for n in nodes] == [(0, 0, 6, 6), (1, 1, 5, 5), (2, 2, 4, 4), (3, 3, 3, 3)]
    assert [n["pixels"] for n in nodes] == [24, 16, 8, 1]


def test_open_ring_does_not_enclose_node_inside_its_bbox():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    nodes = segment_layer(grid, COLORS)["nodes"]

    assert nodes[1]["bbox"] == (1, 1, 3, 3)
    assert nodes[1]["children"] == []
    assert nodes[2]["parent"] == 0
    assert nodes[0]["children"] == [1, 2]


@pytest.mark.parametrize("grid", [[], [[]]])
def test_empty_frame_has_no_nodes(grid):
    assert segment_layer(grid, COLORS) == {"nodes": [], "adjacency_list": []}


@pytest.mark.parametrize("grid, bbox", [
    ([[1]], (0, 0, 0, 0)),
    ([[1, 1, 1]], (0, 0, 0, 2)),
    ([[1], [1], [1]], (0, 0, 2, 0)),
])
def test_frame_spanning_component_has_no_parent(grid, bbox):
    node = segment_layer(grid, COLORS)["nodes"][0]

    assert node["bbox"] == bbox
    assert node["parent"] is None
    assert node["children"] == []


def test_geometry_fields_are_available_in_isolated_python_tool():
    def no_action(_):
        raise AssertionError("This read-only snippet must not call action")

    result = run_sandboxed_python(
        code="""
nodes = current_frame.segmentation['nodes']
assert isinstance(nodes[1]['bbox'], tuple)
assert nodes[nodes[1]['parent']]['children'] == [nodes[1]['id']]
result = {'bbox': nodes[1]['bbox'], 'parent': nodes[1]['parent']}
""",
        timeout_seconds=10,
        initial_state={"current_frame": {
            "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "ascii": "WWW\nWwW\nWWW",
            "shape": [3, 3], "step": 0, "level": 1,
        }},
        action_handler=no_action,
    )

    assert result["error"] == "", result
    assert result["result"] == {"bbox": [1, 1, 1, 1], "parent": 0}
