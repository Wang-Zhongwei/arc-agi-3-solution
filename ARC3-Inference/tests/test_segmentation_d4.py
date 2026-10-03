import itertools

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.segmentation import _object_signature, segment_layer
from inference.utils.frame_diff import compute_frame_diff


def rotate(mask):
    return [list(row) for row in zip(*mask[::-1])]


def variants(mask):
    for reflected in (False, True):
        current = [row[::-1] for row in mask] if reflected else mask
        for angle in (0, 90, 180, 270):
            yield reflected, angle, current
            current = rotate(current)


def encoding(mask):
    return len(mask), len(mask[0]), tuple(itertools.chain.from_iterable(mask))


def cells(mask, k=1):
    return {(r*k+i+3, c*k+j+5) for r, row in enumerate(mask) for c, v in enumerate(row)
            if v for i in range(k) for j in range(k)}


def test_exhaustive_small_masks_pose_reconstruction_and_d4_invariance():
    for bits in itertools.product((0, 1), repeat=9):
        mask = [list(bits[r:r+3]) for r in (0, 3, 6)]
        if not (any(mask[0]) and any(mask[-1]) and any(r[0] for r in mask)
                and any(r[-1] for r in mask)):
            continue
        canonical = min((m for _, _, m in variants(mask)), key=encoding)
        expected_hashes = set()
        for _, _, observed in variants(mask):
            signature = _object_signature(cells(observed), 'B')
            colored, angle, symmetry, pose, shape, mirror, form = signature
            expected_hashes.add((colored, shape))
            recovered = [row[::-1] for row in canonical] if mirror else canonical
            for _ in range((angle or 0) // 90):
                recovered = rotate(recovered)
            assert recovered == observed
            assert symmetry == sum(m == observed for f, _, m in variants(observed) if not f)
            assert (angle is None) == (symmetry == 4)
            assert (mirror is None) == any(m == observed for f, _, m in variants(observed) if f)
            recolored = _object_signature(cells(observed), 'R')
            assert recolored[4] == shape and recolored[0] != colored and recolored[3] != pose
            assert recolored[1:3] == signature[1:3] and recolored[5] == mirror
        assert len(expected_hashes) == 1


def test_pose_reference_does_not_change_with_scale():
    primitive = [[1, 0, 0], [1, 1, 1]]
    for _, _, mask in variants(primitive):
        a = _object_signature(cells(mask), 'B')
        b = _object_signature(cells(mask, 3), 'B')
        assert (a[1], a[2], a[5]) == (b[1], b[2], b[5])
        assert a[0] != b[0] and a[3] != b[3] and a[4] != b[4]


def grid(mask, color=1):
    result = [[0]*10 for _ in range(10)]
    for r, row in enumerate(mask, 3):
        for c, value in enumerate(row, 3):
            result[r][c] = color if value else 0
    return result


def diff(a, b):
    return compute_frame_diff(a, b, segment_layer(a, '.BR')['nodes'], segment_layer(b, '.BR')['nodes'])


def test_chiral_reflection_is_not_dropped_or_called_rotation_or_resize():
    mask = [[1, 0, 0], [1, 1, 1]]
    result = diff(grid(mask), grid([row[::-1] for row in mask]))
    for bucket in ('rotated', 'moved', 'resized'):
        assert not any(n['color'] == 'B' for n in result[bucket])
    assert any(n['color'] == 'B' for n in result['appeared'])
    assert any(n['color'] == 'B' for n in result['disappeared'])
    assert all('_mask' not in n for bucket in result.values() if isinstance(bucket, list) for n in bucket)


def test_recolor_requires_exact_pose_even_when_shape_hash_matches():
    mask = [[1, 0, 0], [1, 1, 1]]
    assert len(diff(grid(mask), grid(mask, 2))['changed_color']) == 1
    assert diff(grid(mask), grid(rotate(mask), 2))['changed_color'] == []
    assert diff(grid(mask), grid([row[::-1] for row in mask], 2))['changed_color'] == []


def test_rotations_work_in_both_reflection_classes():
    mask = [[1, 0, 0], [1, 1, 1]]
    for reflected in (mask, [row[::-1] for row in mask]):
        result = diff(grid(reflected), grid(rotate(reflected)))
        entry = next(n for n in result['rotated'] if n['color'] == 'B')
        assert entry['rotated_by'] == 90


def test_d4_fields_available_in_sandbox_crops():
    response = run_sandboxed_python(
        code="""
a = current_frame.segmentation['nodes'][1]
b = current_frame[2:7, 2:8].segmentation['nodes'][1]
assert isinstance(a['mirrored'], bool)
assert a['mirrored'] == b['mirrored']
assert a['rotation'] == b['rotation']
assert a['hash'] == b['hash']
assert a['shape_hash'] == b['shape_hash']
result = True
""", timeout_seconds=10,
        initial_state={'current_frame': {'grid': grid([[1, 0, 0], [1, 1, 1]]),
                                        'shape': [10, 10], 'ascii': '', 'step': 1, 'level': 1}},
        action_handler=lambda _: None,
    )
    assert response['error'] == '', response
    assert response['result'] is True
