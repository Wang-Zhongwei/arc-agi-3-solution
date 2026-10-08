import pytest

from inference.agent.prompts import ANIMATION_ADDENDUM
from inference.utils.animation import _transient_cells, describe_animation


@pytest.mark.parametrize("values", [(0, 1, 0), (0, 1, 2)])
@pytest.mark.parametrize("game_over", [False, True])
@pytest.mark.parametrize("timeline", [False, True])
@pytest.mark.parametrize("gameplay_changed", [False, True])
def test_multiple_changes_need_not_return_to_original_color(values, game_over, timeline, gameplay_changed):
    chain = [((value,),) for value in values]
    assert _transient_cells(chain) == [(0, 0)]
    description = describe_animation(
        {"frames": 2, "transient_pixels": 1, "transient_bbox": (0, 0, 0, 0)},
        action_display="RIGHT", gameplay_changed=gameplay_changed,
        game_over=game_over, timeline=timeline,
    )
    assert "changed more than once" in description
    assert "changed back" not in description
    assert "last_animation_frames" in description
    assert ("last_animation_timeline" in description) is timeline


def test_single_change_is_visible_in_endpoints():
    assert _transient_cells([((0,),), ((1,),), ((1,),)]) == []
    assert describe_animation(
        {"frames": 2, "transient_pixels": 0},
        action_display="RIGHT", gameplay_changed=True,
    ) == ""
    assert "changed more than once" in ANIMATION_ADDENDUM
