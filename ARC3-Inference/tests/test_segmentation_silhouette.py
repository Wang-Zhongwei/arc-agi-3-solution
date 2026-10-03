import pytest

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.segmentation import segment_layer


def node(mask, color=1, scale=1):
    grid = [[color * cell for cell in row for _ in range(scale)]
            for row in mask for _ in range(scale)]
    # Offset the object to exercise position invariance and exterior access.
    grid = [[0] * (len(grid[0]) + 4)] + [[0, 0] + row + [0, 0] for row in grid]
    return next(n for n in segment_layer(grid, '.BR')['nodes'] if n['color'] == '.BR'[color])


def test_ring_matches_solid_exterior_without_changing_own_geometry():
    ring = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    solid = [[1] * 3 for _ in range(3)]
    hollow, filled = node(ring), node(solid)
    assert hollow['silhouette_hash'] == filled['shape_hash'] == filled['silhouette_hash']
    assert hollow['shape_hash'] != filled['shape_hash']
    assert hollow['form_hash'] != filled['form_hash']
    assert hollow['pixels'] == 8 and hollow['scale'] == 1
    assert filled['pixels'] == 9 and filled['scale'] == 3


@pytest.mark.parametrize('mask', [
    [[1]],
    [[1, 0, 0], [1, 1, 1]],
    [[1, 0, 1], [1, 0, 1], [1, 1, 1]],  # open gap
    [[1, 1, 1], [1, 0, 1], [1, 0, 1]],  # opening at another edge
])
def test_hole_free_masks_keep_shape_hash(mask):
    result = node(mask)
    assert result['silhouette_hash'] == result['shape_hash']


def test_diagonal_escape_does_not_open_a_four_neighbor_hole():
    mask = [[0, 1, 1], [1, 0, 1], [1, 1, 1]]
    expected = [[0, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert node(mask)['silhouette_hash'] == node(expected)['shape_hash']
    assert node(mask)['silhouette_hash'] != node([[1]*3 for _ in range(3)])['shape_hash']


def test_silhouette_ignores_color_and_d4_pose_but_preserves_size():
    mask = [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]]
    reference = node(mask)['silhouette_hash']
    for reflected in (False, True):
        pose = [row[::-1] for row in mask] if reflected else mask
        for _ in range(4):
            assert node(pose, color=2)['silhouette_hash'] == reference
            assert node(pose, scale=2)['silhouette_hash'] != reference
            pose = [list(row) for row in zip(*pose[::-1])]


def test_child_is_filled_only_for_signature_and_crop_can_open_hole():
    response = run_sandboxed_python(
        code="""
nodes = current_frame.segmentation['nodes']
ring, child = nodes
assert ring['children'] == [1] and child['parent'] == 0
assert ring['pixels'] == 8 and child['pixels'] == 1
assert ring['silhouette_hash'] != ring['shape_hash']
clipped = current_frame[0:2, :].segmentation['nodes'][0]
assert clipped['silhouette_hash'] == clipped['shape_hash']
assert clipped['children'] == []
result = True
""", timeout_seconds=10,
        initial_state={'current_frame': {'grid': [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                                        'shape': [3, 3], 'ascii': '', 'step': 1, 'level': 1}},
        action_handler=lambda _: None,
    )
    assert response['error'] == '', response
    assert response['result'] is True
