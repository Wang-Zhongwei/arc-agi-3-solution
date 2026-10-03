import pytest

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.segmentation import segment_layer


def node(mask, k=1, color=1):
    enlarged = [[color * cell for cell in row for _ in range(k)]
                for row in mask for _ in range(k)]
    grid = [[0] * (len(enlarged[0]) + 4) for _ in range(2)]
    grid += [[0, 0] + row + [0, 0] for row in enlarged]
    return next(n for n in segment_layer(grid, '.BR')['nodes'] if n['color'] == '.BR'[color])


@pytest.mark.parametrize('mask', [
    [[1]],
    [[1, 1, 1], [1, 1, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
])
def test_form_ignores_color_d4_pose_and_integer_scale(mask):
    base = node(mask)
    assert base['scale'] == 1
    assert base['form_hash'] == base['shape_hash']
    for reflected in (False, True):
        pose = [row[::-1] for row in mask] if reflected else mask
        for _ in range(4):
            for k in (1, 2, 3):
                for color in (1, 2):
                    observed = node(pose, k, color)
                    assert observed['form_hash'] == base['form_hash']
                    assert observed['scale'] == k
                    if k > 1:
                        assert observed['shape_hash'] != base['shape_hash']
            pose = [list(row) for row in zip(*pose[::-1])]


def test_form_preserves_holes_and_aspect_ratio():
    ring = node([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    solid = node([[1]*3 for _ in range(3)])
    assert ring['form_hash'] != solid['form_hash']
    assert node([[1]*6 for _ in range(4)])['form_hash'] == node([[1]*3 for _ in range(2)])['form_hash']
    assert node([[1]*5 for _ in range(4)])['form_hash'] != node([[1]*3 for _ in range(2)])['form_hash']


def test_form_available_in_sandbox_and_recomputed_after_clipping():
    response = run_sandboxed_python(
        code="""
full = current_frame.segmentation['nodes'][0]
similar = current_frame[0:2, 0:3].segmentation['nodes'][0]
clipped = current_frame[0:3, 0:3].segmentation['nodes'][0]
assert full['scale'] == 2 and similar['scale'] == 1
assert full['form_hash'] == similar['form_hash'] == similar['shape_hash']
assert full['form_hash'] != clipped['form_hash']
result = True
""", timeout_seconds=10,
        initial_state={'current_frame': {'grid': [[1]*6 for _ in range(4)],
                                        'shape': [4, 6], 'ascii': '', 'step': 1, 'level': 1}},
        action_handler=lambda _: None,
    )
    assert response['error'] == '', response
    assert response['result'] is True
