import pytest

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.animation import build_animation_frames
from inference.utils.grid_utils import ARC_COLOR_CHARS, format_grid_ascii


def _run(code, before, after=None, *, animation_record=None):
    def payload(grid, step):
        return {"grid": grid, "ascii": format_grid_ascii(grid),
                "shape": [len(grid), len(grid[0]) if grid else 0],
                "step": step, "level": 2}

    def no_action(_):
        raise AssertionError("This read-only snippet must not call action")

    return run_sandboxed_python(
        code=code, timeout_seconds=10,
        initial_state={
            "history": [
                {"action": "", "frame": payload(before, 4)},
                {"action": "ACTION1", "frame": payload(before if after is None else after, 5)},
            ],
            "current_frame": payload(before if after is None else after, 5),
        },
        action_handler=no_action,
        animation_handler=(lambda _: build_animation_frames(animation_record))
        if animation_record is not None else None,
    )


def _result(code, before, after=None):
    response = _run(code, before, after)
    assert response["error"] == "", response
    return response["result"]


def test_animation_frames_support_crops_segmentation_and_frame_diff():
    before = [[0] * 7 for _ in range(7)]
    after = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2, 1, 0],
        [0, 1, 2, 3, 2, 1, 0],
        [0, 1, 2, 2, 2, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    intermediate = [row[:] for row in after]
    intermediate[3][3] = 4
    record = {"chain": [before, intermediate, after], "step": 5, "level": 2}
    response = _run("""
assert len(last_animation_frames) == 2
frame = last_animation_frames[1]
original = frame.segmentation
for r0, r1, c0, c1 in [(1, 6, 1, 6), (1, 4, 1, 6), (2, 5, 2, 5)]:
    crop = last_animation_frames[1][r0:r1, c0:c1]
    current_crop = current_frame[r0:r1, c0:c1]
    assert crop.segmentation == current_crop.segmentation
    assert (crop.shape, crop.step, crop.level) == (current_crop.shape, current_crop.step, current_crop.level)
assert frame.segmentation is original
assert last_animation_frames[1][0:7, 0:7][1:6, 1:6].segmentation == frame[1:6, 1:6].segmentation
result = {
    'closed': frame[1:6, 1:6].segmentation['nodes'],
    'clipped': frame[1:4, 1:6].segmentation['nodes'],
    'diff': frame_diff(last_animation_frames[0][1:6, 1:6], last_animation_frames[1][1:6, 1:6]),
}
""", before, after, animation_record=record)
    assert response["error"] == "", response
    result = response["result"]
    assert [n["id"] for n in result["closed"]] == [0, 1, 2]
    assert [n["parent"] for n in result["closed"]] == [None, 0, 1]
    assert [n["bbox"] for n in result["closed"]] == [[1, 1, 5, 5], [2, 2, 4, 4], [3, 3, 3, 3]]
    assert [n["parent"] for n in result["clipped"]] == [None, None, None]
    assert result["diff"]["changed_cell_count"] == 1
    assert result["diff"]["changed_color"][0]["before_bbox"] == [3, 3, 3, 3]


def test_nested_crops_preserve_coordinates_metadata_and_parent_frame():
    grid = [[0] * 9 for _ in range(8)]
    grid[4][5] = 1
    result = _result("""
frame = current_frame
original = frame.segmentation
crop = frame[2:7, 3:8][1:4, 1:4]
nodes = crop.segmentation['nodes']
assert isinstance(nodes[1]['bbox'], tuple)
assert frame.segmentation is original
assert original['nodes'][0]['bbox'] == (0, 0, 7, 8)
assert crop.segmentation is crop.segmentation
result = {'shape': crop.shape, 'ascii': crop.ascii,
          'step': crop.step, 'level': crop.level, 'nodes': nodes}
""", grid)

    assert result["shape"] == [3, 3]
    assert result["ascii"] == "WWW\nWwW\nWWW"
    assert (result["step"], result["level"]) == (5, 2)
    background, pixel = result["nodes"]
    assert background["bbox"] == [3, 4, 5, 6]
    assert background["children"] == [1]
    assert pixel["id"] == 1
    assert pixel["bbox"] == [4, 5, 4, 5]
    assert pixel["boundary"] == [[4, 5]]
    assert pixel["parent"] == 0


def test_clipped_components_get_crop_local_ids_and_containment():
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2, 1, 0],
        [0, 1, 2, 3, 2, 1, 0],
        [0, 1, 2, 2, 2, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    result = _result("""
original = current_frame.segmentation['nodes']
closed = current_frame[1:6, 1:6].segmentation['nodes']
clipped = current_frame[1:4, 1:6].segmentation['nodes']
result = {'original': original, 'closed': closed, 'clipped': clipped}
""", grid)

    assert [n["parent"] for n in result["original"]] == [None, 0, 1, 2]
    assert [n["parent"] for n in result["closed"]] == [None, 0, 1]
    assert [n["id"] for n in result["clipped"]] == [0, 1, 2]
    assert [n["parent"] for n in result["clipped"]] == [None, None, None]
    assert [n["pixels"] for n in result["clipped"]] == [9, 5, 1]
    assert result["clipped"][0]["bbox"] == [1, 1, 3, 5]
    assert result["clipped"][0]["hash"] != result["original"][1]["hash"]


def test_python_slice_bounds_full_and_empty_views():
    grid = [[0] * 7 for _ in range(7)]
    assert _result("""
frame = current_frame
assert frame[:, :].segmentation == frame.segmentation
assert frame[-6:-1, 1:999].segmentation == frame[1:6, 1:7].segmentation
assert frame[4:2, :].shape == (0, 7)
assert frame[1:3, 3:1].shape == (2, 0)
assert frame[99:, :].shape == (0, 7)
assert frame[4:2, :].segmentation == {'nodes': [], 'adjacency_list': []}
assert frame[1:3, 3:1].segmentation == {'nodes': [], 'adjacency_list': []}
assert frame_diff(previous_frame[4:2, :], frame[4:2, :])['changed_cell_count'] == 0
assert frame_diff(previous_frame[1:3, 3:1], frame[1:3, 3:1])['changed_cell_count'] == 0
result = True
""", grid) is True


@pytest.mark.parametrize("expression, error", [
    ("current_frame[1]", "TypeError"),
    ("current_frame[1:2]", "TypeError"),
    ("current_frame[1, 2]", "TypeError"),
    ("current_frame[:, :, :]", "TypeError"),
    ("current_frame[::2, :]", "ValueError"),
    ("current_frame[:, ::-1]", "ValueError"),
    ("current_frame[::0, :]", "ValueError"),
])
def test_invalid_slices_have_explicit_errors(expression, error):
    response = _run(expression, [[0] * 3 for _ in range(3)])
    assert error in response["error"], response
    assert "slice" in response["error"].lower()


def test_single_cell_crop_recolor_uses_original_coordinates():
    before = [[0] * 4 for _ in range(3)]
    after = [row[:] for row in before]
    after[1][2] = 1
    diff = _result("result = frame_diff(previous_frame[1:2, 2:3], current_frame[1:2, 2:3])", before, after)

    assert diff["changed_cell_count"] == 1
    assert len(diff["changed_color"]) == 1
    assert diff["changed_color"][0]["before_bbox"] == [1, 2, 1, 2]
    assert diff["changed_color"][0]["before_color"] == ARC_COLOR_CHARS[0]
    assert diff["changed_color"][0]["after_color"] == ARC_COLOR_CHARS[1]


def test_cropped_diff_filters_changes_outside_crop_and_reports_absolute_movement():
    before = [[0] * 14 for _ in range(12)]
    before[6][8] = 1
    after = [[0] * 14 for _ in range(12)]
    after[7][9] = 1
    after[0][0] = 2
    result = _result("""
result = {'crop': frame_diff(previous_frame[5:10, 7:12], current_frame[5:10, 7:12]),
          'full': frame_diff()}
""", before, after)

    assert result["crop"]["changed_cell_count"] == 2
    assert result["full"]["changed_cell_count"] == 3
    moved = next(n for n in result["crop"]["moved"] if n["before_bbox"] == [6, 8, 6, 8])
    assert moved["before_bbox"] == [6, 8, 6, 8]
    assert moved["after_bbox"] == [7, 9, 7, 9]


@pytest.mark.parametrize("bucket, before_mask, after_mask", [
    ("appeared", [[0]], [[1]]),
    ("disappeared", [[1]], [[0]]),
    ("rotated", [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]]),
    ("resized", [[1, 1, 1], [0, 1, 0]], [[1, 0, 0], [1, 1, 1]]),
])
def test_existing_diff_buckets_keep_original_coordinates(bucket, before_mask, after_mask):
    def embed(mask):
        grid = [[0] * 12 for _ in range(10)]
        for r, row in enumerate(mask, 4):
            for c, value in enumerate(row, 5):
                grid[r][c] = value
        return grid

    diff = _result("result = frame_diff(previous_frame[3:8, 4:10], current_frame[3:8, 4:10])",
                   embed(before_mask), embed(after_mask))
    entry = next(n for n in diff[bucket] if n.get("before_id", n.get("id")) == 1)
    if bucket == "resized":
        assert entry["before_bbox"] == entry["after_bbox"] == [4, 5, 5, 7]
        assert entry["before_pixels"] == entry["after_pixels"] == 4
    else:
        if bucket == "rotated":
            assert entry["rotated_by"] == 90
            assert entry["before_bbox"] == [4, 5, 6, 6]
            assert entry["after_bbox"] == [4, 5, 5, 7]
        else:
            assert entry["bbox"] == [4, 5, 4, 5]


def test_nested_and_direct_crops_have_identical_diffs():
    before = [[0] * 8 for _ in range(8)]
    after = [row[:] for row in before]
    after[4][4] = 1
    assert _result("""
direct = frame_diff(previous_frame[3:6, 3:6], current_frame[3:6, 3:6])
nested = frame_diff(previous_frame[1:7, 1:7][2:5, 2:5], current_frame[3:6, 3:6])
assert direct == nested
result = True
""", before, after) is True


def test_different_origins_are_rejected_but_shape_mismatch_is_preserved():
    grid = [[0] * 6 for _ in range(6)]
    response = _run("frame_diff(previous_frame[1:3, 1:3], current_frame[2:4, 1:3])", grid)
    assert "ValueError" in response["error"]
    assert "same original-frame bounds" in response["error"]

    diff = _result("result = frame_diff(previous_frame[1:3, 1:3], current_frame[1:4, 1:3])", grid)
    assert diff["changed_cell_count"] == -1
    assert diff["before_shape"] == [2, 2]
    assert diff["after_shape"] == [3, 2]


@pytest.mark.parametrize('bucket', ['mirrored', 'scaled'])
def test_new_transform_buckets_work_in_cropped_sandbox(bucket):
    before = [[0]*12 for _ in range(10)]
    after = [[0]*12 for _ in range(10)]
    mask = [[1, 0, 0], [1, 1, 1]]
    if bucket == 'mirrored':
        other = [row[::-1] for row in mask]
        start = (5, 5)
    else:
        other = [[v for v in row for _ in range(2)] for row in mask for _ in range(2)]
        start = (4, 4)
    for r, row in enumerate(mask, 5):
        for c, value in enumerate(row, 5):
            before[r][c] = value
    for r, row in enumerate(other, start[0]):
        for c, value in enumerate(row, start[1]):
            after[r][c] = value
    result = _result("""
result = frame_diff(previous_frame[3:9, 3:11], current_frame[3:9, 3:11])
""", before, after)
    entry = next(e for e in result[bucket] if e['before_bbox'] == [5, 5, 6, 7])
    if bucket == 'mirrored':
        assert entry['axis'] == '|'
        assert entry['after_bbox'] == [5, 5, 6, 7]
    else:
        assert entry['scale_factor'] == 2
        assert entry['after_bbox'] == [4, 4, 7, 9]
