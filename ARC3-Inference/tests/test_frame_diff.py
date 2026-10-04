import pytest

from inference.agent.tool_agent import _diff_entry_line, _render_auto_frame_diff
from inference.utils.frame_diff import compute_frame_diff
from inference.utils.segmentation import segment_layer


L = [[1, 0, 0], [1, 1, 1]]
BUCKETS = {'moved', 'rotated', 'mirrored', 'scaled', 'changed_color', 'resized', 'appeared', 'disappeared'}
PAIR = {'before_id', 'after_id', 'before_bbox', 'after_bbox'}
FIELDS = {'moved': {'hash'}, 'rotated': {'hash', 'rotated_by'}, 'mirrored': {'hash', 'axis'},
          'scaled': {'form_hash', 'before_scale', 'after_scale', 'scale_factor'},
          'changed_color': {'shape_hash', 'before_color', 'after_color'},
          'resized': {'color', 'before_pixels', 'after_pixels'}}


def rotate(mask):
    return [list(row) for row in zip(*mask[::-1])]


def enlarge(mask, k):
    return [[v for v in row for _ in range(k)] for row in mask for _ in range(k)]


def board(mask, *, color=1, at=None):
    grid = [[0]*20 for _ in range(20)]
    r0, c0 = at or ((20-len(mask))//2, (20-len(mask[0]))//2)
    for r, row in enumerate(mask, r0):
        for c, v in enumerate(row, c0):
            grid[r][c] = color*v
    return grid


def run(a, b, **kwargs):
    na, nb = segment_layer(a, '.BR')['nodes'], segment_layer(b, '.BR')['nodes']
    result = compute_frame_diff(a, b, na, nb, **kwargs)
    assert set(result) == BUCKETS | {'changed_cell_count'}
    used_a, used_b = [], []
    for bucket in BUCKETS:
        for entry in result[bucket]:
            if bucket in FIELDS:
                assert set(entry) == PAIR | FIELDS[bucket]
                assert entry['before_bbox'] == na[entry['before_id']]['bbox']
                assert entry['after_bbox'] == nb[entry['after_id']]['bbox']
                used_a.append(entry['before_id'])
                used_b.append(entry['after_id'])
            else:
                assert set(entry) == {'id', 'color', 'hash', 'bbox', 'pixels'}
                (used_a if bucket == 'disappeared' else used_b).append(entry['id'])
    assert len(set(used_a)) == len(used_a)
    assert len(set(used_b)) == len(used_b)
    return result, na, nb


def target(result, bucket):
    return next(e for e in result[bucket] if e.get('before_id', e.get('id')) == 1)


@pytest.mark.parametrize('axis', ['-', '|', '/', '\\'])
def test_all_reflection_axes(axis):
    transformed = {'-': L[::-1], '|': [row[::-1] for row in L],
                   '/': [list(row) for row in zip(*L)][::-1],
                   '\\': [list(row) for row in zip(*L)]}
    transformed['/'] = [row[::-1] for row in transformed['/']]
    result, _, _ = run(board(L), board(transformed[axis]))
    assert target(result, 'mirrored')['axis'] == axis
    assert not any(e['before_id'] == 1 for e in result['rotated'])


@pytest.mark.parametrize('angle', [90, 180, 270])
def test_clockwise_rotation(angle):
    mask = L
    for _ in range(angle//90):
        mask = rotate(mask)
    result, _, _ = run(board(L), board(mask))
    assert target(result, 'rotated')['rotated_by'] == angle


def test_rotation_precedes_reflection_for_symmetric_shapes():
    mask = [[1, 0], [1, 1]]
    result, _, _ = run(board(mask), board([row[::-1] for row in mask]))
    assert target(result, 'rotated')['rotated_by'] == 270
    assert not any(e['before_id'] == 1 for e in result['mirrored'])


@pytest.mark.parametrize('before_scale,after_scale', [(1, 2), (3, 2), (2, 1)])
def test_scaling_in_both_directions(before_scale, after_scale):
    result, _, _ = run(board(enlarge(L, before_scale)), board(enlarge(L, after_scale)))
    entry = target(result, 'scaled')
    assert entry['before_scale'] == before_scale
    assert entry['after_scale'] == after_scale
    assert entry['scale_factor'] == after_scale/before_scale
    assert not any(e['before_id'] == 1 for e in result['resized'])


@pytest.mark.parametrize('mask,color,at', [
    (rotate(L), 2, None), (L, 2, (2, 2)),
    (enlarge(rotate(L), 2), 1, None), (enlarge(L, 2), 2, None),
    (rotate(L), 1, (2, 2)), ([row[::-1] for row in L], 1, (2, 2)),
    (enlarge(L, 2), 1, (2, 2)),
])
def test_unsupported_combinations_remain_unmatched(mask, color, at):
    result, _, _ = run(board(L), board(mask, color=color, at=at))
    for bucket in FIELDS:
        assert not any(e['before_id'] == 1 for e in result[bucket])
    assert target(result, 'disappeared')['color'] == 'B'
    assert target(result, 'appeared')['color'] == '.BR'[color]


def test_equal_pixel_geometry_change_is_resized():
    result, _, _ = run(board([[1, 1, 1], [0, 1, 0]]), board(L))
    entry = target(result, 'resized')
    assert entry['before_pixels'] == entry['after_pixels'] == 4


def test_recolor_and_translation_have_separate_payloads():
    recolor, _, _ = run(board(L), board(L, color=2))
    assert target(recolor, 'changed_color')['before_color'] == 'B'
    assert target(recolor, 'changed_color')['after_color'] == 'R'
    moved, _, _ = run(board(L), board(L, at=(2, 2)))
    assert target(moved, 'moved')['after_bbox'] == (2, 2, 3, 4)


def points(coords):
    grid = [[0]*20 for _ in range(20)]
    for r, c in coords:
        grid[r][c] = 1
    return grid


def test_unchanged_duplicate_is_removed_before_matching():
    result, _, _ = run(points([(5, 3), (5, 7)]), points([(5, 3), (5, 9)]))
    assert len(result['moved']) == 1
    assert result['moved'][0]['before_bbox'] == (5, 7, 5, 7)
    assert result['moved'][0]['after_bbox'] == (5, 9, 5, 9)


def test_equidistant_duplicates_get_one_deterministic_assignment():
    result, _, _ = run(points([(5, 3), (5, 7)]), points([(5, 5)]))
    assert len(result['moved']) == 1
    assert len([e for e in result['disappeared'] if e['color'] == 'B']) == 1
    assert not any(e['color'] == 'B' for e in result['appeared'])
    assert run(points([(5, 3), (5, 7)]), points([(5, 5)]))[0] == result


def test_ambiguous_resize_stays_unmatched():
    result, _, _ = run(points([(5, 3), (5, 7)]), points([(5, c) for c in range(3, 8)]))
    assert not any(e['color'] == 'B' for e in result['resized'])
    assert len([e for e in result['disappeared'] if e['color'] == 'B']) == 2


def test_automatic_large_group_limit_and_manual_matching():
    a = points([(3, 3), (3, 8), (8, 3), (8, 8)])
    b = points([(3, 4), (3, 9), (8, 4), (8, 9)])
    assert len(run(a, b)[0]['moved']) == 4
    limited, _, _ = run(a, b, max_group_match=2)
    assert limited['moved'] == []
    assert len([e for e in limited['appeared'] if e['color'] == 'B']) == 4


def test_hash_candidates_are_verified_against_masks():
    a, b = board(L), board([[1, 1, 1], [0, 1, 0]])
    na, nb = segment_layer(a, '.BR')['nodes'], segment_layer(b, '.BR')['nodes']
    for field in ('hash', 'pose_hash', 'shape_hash', 'form_hash'):
        nb[1][field] = na[1][field]
    result = compute_frame_diff(a, b, na, nb)
    for bucket in FIELDS:
        assert not any(e['before_id'] == 1 for e in result[bucket])


def test_renderer_new_fields_and_budget_fallbacks():
    result, na, nb = run(board(L), board(enlarge(L, 2)))
    lines = _render_auto_frame_diff(result, 2000, total_cells=400, nodes_before=na, nodes_after=nb)
    assert any('scaled: node 1->1, scale 1->2 (×2)' in line for line in lines)
    assert all('None' not in line for line in lines)
    entry = target(result, 'scaled')
    many = {'changed_cell_count': 100, 'scaled': [dict(entry) for _ in range(30)]}
    compact = _render_auto_frame_diff(many, 200, nodes_before=na, nodes_after=nb)
    assert any('30x color B 16px' in line for line in compact)
    counts = _render_auto_frame_diff(many, 1, nodes_before=na, nodes_after=nb)
    assert len(counts) == 1 and '30 scaled' in counts[0]
    assert 'before_id' in entry  # rendering does not mutate the diff
    for axis in ('-', '|', '/', '\\'):
        assert f'axis {axis}' in _diff_entry_line('mirrored', {**entry, 'axis': axis})


def test_zero_change_and_shape_mismatch_renderer():
    result, _, _ = run(board(L), board(L))
    assert result['changed_cell_count'] == 0
    assert all(result[bucket] == [] for bucket in BUCKETS)
    lines = _render_auto_frame_diff({'changed_cell_count': -1, 'before_shape': (2, 2), 'after_shape': (3, 3)}, 100)
    assert 'grid shape changed' in lines[0]


def test_exact_assignment_does_not_fall_through_to_resize():
    a = points([(5, 5)])
    b = points([(5, 3), (5, 7), (4, 5), (5, 5), (6, 5)])
    result, _, _ = run(a, b)
    assert len(result['moved']) == 1
    assert not any(e['color'] == 'B' for e in result['resized'])
    assert not any(e['color'] == 'B' for e in result['disappeared'])


@pytest.mark.parametrize('destinations', [(11, 16), (10, 16)])
def test_hungarian_preserves_globally_unique_assignment(destinations):
    result, _, _ = run(points([(5, 8), (5, 12)]), points([(5, c) for c in destinations]))
    pairs = {(e['before_bbox'][1], e['after_bbox'][1]) for e in result['moved']}
    assert pairs == {(8, destinations[0]), (12, destinations[1])}


def test_hungarian_handles_sparse_candidates_and_unmatched_objects():
    from inference.utils.frame_diff import _hungarian_candidates
    candidates = [(1, 0, 0, 'move'), (3, 0, 1, 'move'), (2, 1, 0, 'rotate'),
                  (1, 2, 0, 'move')]
    matched = _hungarian_candidates(candidates)
    assert len(matched) == 2
    assert (0, 1, 'move') in matched
    assert len({i for i, _, _ in matched}) == len({j for _, j, _ in matched}) == 2
    assert all((i, j, change) in [(a, b, c) for _, a, b, c in candidates]
               for i, j, change in matched)
    assert _hungarian_candidates([]) == []
