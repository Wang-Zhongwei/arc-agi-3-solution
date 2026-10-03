import random

import pytest

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.segmentation import _object_scale, segment_layer


def enlarge(mask, k):
    return [[cell for cell in row for _ in range(k)] for row in mask for _ in range(k)]


@pytest.mark.parametrize('mask,k,expected', [
    ([[1]], 1, 1),
    ([[1]], 4, 4),
    ([[1, 1, 1]], 2, 2),
    ([[1, 0], [1, 1]], 3, 3),
    ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], 2, 2),
    ([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], 1, 1),
])
def test_scale_uses_own_pixels_including_holes(mask, k, expected):
    grid = enlarge(mask, k)
    grid = [[0] * (len(grid[0]) + 4)] + [[0, 0] + row + [0, 0] for row in grid]
    node = next(n for n in segment_layer(grid, '.B')['nodes'] if n['color'] == 'B')
    assert node['scale'] == expected


def test_scale_matches_dense_block_definition_for_random_masks():
    rng = random.Random(42)
    for _ in range(150):
        h, w = rng.randint(1, 8), rng.randint(1, 8)
        mask = [[rng.randrange(2) for _ in range(w)] for _ in range(h)]
        mask[0][0] = mask[-1][-1] = 1  # tight bounds
        mask = enlarge(mask, rng.randint(1, 4))
        h, w = len(mask), len(mask[0])
        factors = [k for k in range(1, min(h, w) + 1) if h % k == w % k == 0
                   and all(len({mask[r + i][c + j] for i in range(k) for j in range(k)}) == 1
                           for r in range(0, h, k) for c in range(0, w, k))]
        cells = {(r + 5, c + 7) for r in range(h) for c in range(w) if mask[r][c]}
        assert _object_scale(cells) == max(factors)
        assert _object_scale({(c, -r) for r, c in cells}) == max(factors)


def test_scale_available_in_sandbox_and_recomputed_for_crop():
    response = run_sandboxed_python(
        code="""
assert current_frame.segmentation['nodes'][0]['scale'] == 4
assert current_frame[1:3, 1:3].segmentation['nodes'][0]['scale'] == 2
assert current_frame[0:3, 0:4].segmentation['nodes'][0]['scale'] == 1
result = True
""",
        timeout_seconds=10,
        initial_state={'current_frame': {'grid': [[1] * 4 for _ in range(4)],
                                        'shape': [4, 4], 'ascii': '', 'step': 0, 'level': 0}},
        action_handler=lambda _: None,
    )
    assert response['error'] == '', response
    assert response['result'] is True
