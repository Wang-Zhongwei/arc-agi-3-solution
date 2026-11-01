"""The no-op guard ignores HUD bars in the border but not pieces moving there."""

from inference.framework.solver import _interior_gameplay_changed


def _blank():
    return [[0] * 64 for _ in range(64)]


def test_interior_change_counts():
    before, after = _blank(), _blank()
    after[30][30] = 5
    assert _interior_gameplay_changed(before, after)


def test_thin_bar_in_border_is_hud():
    before, after = _blank(), _blank()
    before[63] = [9] * 64
    after[63] = [9] * 63 + [3]  # timer bar loses one cell
    assert not _interior_gameplay_changed(before, after)
    before[62] = [9] * 64
    after[62] = [9] * 60 + [3] * 4  # two-row bar loses four columns
    assert not _interior_gameplay_changed(before, after)


def test_piece_moving_inside_border_counts():
    # wa30: a 4x4 player walking down the right edge (cols 60-63)
    before, after = _blank(), _blank()
    for r in range(32, 36):
        for c in range(60, 64):
            before[r][c] = 7
    for r in range(36, 40):
        for c in range(60, 64):
            after[r][c] = 7
    assert _interior_gameplay_changed(before, after)
    assert not _interior_gameplay_changed(before, after, solid_side=0)


def test_border_disabled_is_plain_inequality():
    before, after = _blank(), _blank()
    after[0][0] = 1
    assert _interior_gameplay_changed(before, after, border=0)
