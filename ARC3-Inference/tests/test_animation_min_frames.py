import pytest

from inference.framework.solver import _counted_animation_summary


def _chain(frames: int):
    # pre-action board, then `frames` distinct boards; the cell flashes 0 -> 1 -> 0 ...
    return [((index % 2,),) for index in range(frames + 1)]


@pytest.mark.parametrize("frames", [2, 3, 4, 7])
def test_default_counts_every_animation(frames):
    summary = _counted_animation_summary(_chain(frames), min_frames=2)
    assert summary is not None and summary["frames"] == frames


@pytest.mark.parametrize("frames,counted", [(2, False), (3, False), (4, True), (9, True)])
def test_min_frames_four_drops_animations_of_three_or_fewer_frames(frames, counted):
    assert (_counted_animation_summary(_chain(frames), min_frames=4) is not None) is counted


def test_single_frame_action_is_never_an_animation():
    assert _counted_animation_summary(_chain(1), min_frames=2) is None
