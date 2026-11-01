"""The no-op guard's border (ARC3_NOOP_GUARD_BORDER, default 3) hides HUD bars
but not pieces moving along the edge."""

from inference.framework.solver import NOOP_GUARD_BORDER_WIDTH, _interior_gameplay_changed


def _blank():
    return [[0] * 64 for _ in range(64)]


def test_default_border_is_three():
    assert NOOP_GUARD_BORDER_WIDTH == 3


def test_ls20_step_bar_is_hud():
    # ls20: a 2-row step bar in rows 61-62 (depth 1-2) loses two columns
    before, after = _blank(), _blank()
    for r in (61, 62):
        before[r][15:40] = [4] * 25
        after[r][15:40] = [5] * 2 + [4] * 23
    assert not _interior_gameplay_changed(before, after)


def test_wa30_player_on_the_edge_counts():
    # wa30: a 4x4 player walking down the right edge (cols 60-63, depth 0-3)
    before, after = _blank(), _blank()
    for r in range(32, 36):
        before[r][60:64] = [7] * 4
    for r in range(36, 40):
        after[r][60:64] = [7] * 4
    assert _interior_gameplay_changed(before, after)
    assert not _interior_gameplay_changed(before, after, border=4)
