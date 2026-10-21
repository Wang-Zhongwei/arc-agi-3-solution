"""Unit tests for the animation digest helpers in utils/animation.py."""
from __future__ import annotations

from inference.utils.animation import (
    ANIMATION_RING_MAX_FRAMES,
    _transient_cell_set,
    _transient_cells,
    action_name_without_coords,
    animation_signature,
    build_animation_digest,
    cap_chain,
    collapse_frames,
    decode_chain,
    decode_grid_hex,
    encode_grid_hex,
    even_subsample_indices,
    overlay_trails,
    select_keyframes,
    summarize_animation,
)

SIZE = 64


def _board(background: int = 0) -> list[list[int]]:
    return [[background] * SIZE for _ in range(SIZE)]


def _frozen(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in grid)


def _paint(grid, top, left, height, width, color):
    for row in range(top, top + height):
        for col in range(left, left + width):
            grid[row][col] = color


def _sweep_chain(token_cols=(20, 26, 32, 38, 44), top=30):
    """tn36-like: a 3x3 highlight steps over five 3x3 tokens, then reverts."""
    base = _board()
    for left in token_cols:
        _paint(base, top, left, 3, 3, 3)
    before = _frozen(base)
    frames = []
    for left in token_cols:
        grid = [row[:] for row in base]
        _paint(grid, top, left, 3, 3, 9)
        frames.append(_frozen(grid))
    frames.append(before)
    return collapse_frames(before, frames)


def _record(chain, action_num, display, *, gameplay_changed=False, level=1):
    return {
        "action_num": action_num,
        "action_display": display,
        "summary": summarize_animation(chain),
        "signature": animation_signature(chain, display),
        "level": level,
        "frames": [encode_grid_hex(grid) for grid in cap_chain(chain)],
        "gameplay_changed": gameplay_changed,
    }


def _slide_chain():
    """A 2x2 block sliding right three cells, plus a static wall."""
    grids = []
    for left in (10, 11, 12, 13):
        grid = _board()
        _paint(grid, 20, left, 2, 2, 11)
        _paint(grid, 10, 40, 3, 3, 3)
        grids.append(_frozen(grid))
    return collapse_frames(grids[0], grids[1:])


# --- subsampling and storage -------------------------------------------------


def test_even_subsample_keeps_first_and_last_and_exact_count():
    for count in range(2, 120):
        for limit in (2, 3, 4, 8, 32):
            picks = even_subsample_indices(count, limit)
            assert picks == sorted(set(picks))
            assert len(picks) == min(count, limit)
            assert picks[0] == 0 and picks[-1] == count - 1


def test_even_subsample_edge_cases():
    assert even_subsample_indices(0, 4) == []
    assert even_subsample_indices(5, 0) == []
    assert even_subsample_indices(5, 1) == [4]
    assert even_subsample_indices(3, 8) == [0, 1, 2]


def test_dedupe_happens_before_the_cap():
    """A held pose collapses first, so it cannot eat the frame budget."""
    before = _frozen(_board())
    moving = []
    for step in range(10):
        grid = _board()
        grid[30][step + 5] = 9
        moving.append(_frozen(grid))
    held = [moving[-1]] * 60
    chain = collapse_frames(before, moving + held)
    assert len(chain) == 11
    assert cap_chain(chain) == chain


def test_cap_preserves_first_and_last_state():
    before = _frozen(_board())
    frames = []
    for step in range(50):
        grid = _board()
        grid[step % 60][step] = 9
        frames.append(_frozen(grid))
    chain = collapse_frames(before, frames)
    capped = cap_chain(chain)
    assert len(capped) == ANIMATION_RING_MAX_FRAMES
    assert capped[0] == chain[0] and capped[-1] == chain[-1]


def test_hex_roundtrip():
    grid = tuple(tuple((row * 7 + col) % 16 for col in range(64)) for row in range(64))
    assert decode_grid_hex(encode_grid_hex(grid)) == grid
    assert decode_chain([encode_grid_hex(grid), grid]) == [grid, grid]


def test_fast_transient_matches_reference():
    chain = _sweep_chain()
    assert sorted(_transient_cell_set(chain)) == _transient_cells(chain)


# --- signature ---------------------------------------------------------------


def test_signature_ignores_coordinates():
    assert action_name_without_coords("MOUSE(row=3, col=4)") == "MOUSE"
    assert action_name_without_coords("UP") == "UP"
    here = _sweep_chain(top=30)
    there = _sweep_chain(top=12)
    assert animation_signature(here, "MOUSE(row=31, col=21)") == animation_signature(
        there, "MOUSE(row=13, col=45)"
    )


def test_signature_separates_actions_and_mechanics():
    sweep = _sweep_chain()
    assert animation_signature(sweep, "MOUSE(row=1, col=1)") != animation_signature(
        sweep, "SPACE"
    )
    shorter = _sweep_chain(token_cols=(20, 26, 32))
    assert animation_signature(sweep, "SPACE") != animation_signature(shorter, "SPACE")


# --- keyframes ---------------------------------------------------------------


def test_keyframes_are_intermediate_and_capped():
    chain = _sweep_chain()  # S0..S6, intermediates S1..S5
    assert select_keyframes(chain, 0) == []
    assert select_keyframes(chain, 10) == [1, 2, 3, 4, 5]
    picks = select_keyframes(chain, 3)
    assert len(picks) == 3
    assert picks[0] == 1 and picks[-1] == 5
    assert all(0 < index < len(chain) - 1 for index in picks)
    assert select_keyframes(chain[:2], 4) == []


def test_keyframes_include_the_peak_transient_state():
    before = _frozen(_board())
    frames = []
    for step in range(12):
        grid = _board()
        grid[5][step + 5] = 9  # a dot walking right
        if step == 4:
            _paint(grid, 30, 30, 4, 4, 8)  # a flash only at S5
        frames.append(_frozen(grid))
    frames.append(before)
    chain = collapse_frames(before, frames)
    picks = select_keyframes(chain, 4)
    assert 5 in picks
    assert picks[0] == 1 and picks[-1] == len(chain) - 2
    assert len(picks) == 4


# --- lens A trails -----------------------------------------------------------


def test_overlay_trails_on_sweep():
    trails = overlay_trails(_sweep_chain(), border=4)
    assert len(trails) == 1
    trail = trails[0]
    assert (trail["color"], trail["height"], trail["width"]) == (9, 3, 3)
    assert trail["points"] == [(1, 30, 20), (2, 30, 26), (3, 30, 32), (4, 30, 38), (5, 30, 44)]


def test_overlay_trails_mask_the_border():
    before = _board()
    _paint(before, 0, 10, 2, 30, 12)  # a HUD bar inside the 4-cell border
    after = [row[:] for row in before]
    middle = [row[:] for row in before]
    _paint(middle, 0, 10, 2, 30, 8)  # the bar flashes
    chain = collapse_frames(_frozen(before), [_frozen(middle), _frozen(after)])
    assert overlay_trails(chain, border=4) == []
    assert overlay_trails(chain, border=0)


# --- digest ------------------------------------------------------------------


def test_digest_matches_worked_example_one():
    chain = _sweep_chain()
    record = _record(chain, 212, "MOUSE(row=31, col=21)", gameplay_changed=False)
    text, shown = build_animation_digest([record], 900, 4, {}, last_action_num=212)
    lines = text.splitlines()
    assert lines[0] == (
        "Animation digest (novel): action 212 `MOUSE(row=31, col=21)`: "
        "path b 3x3 (30,20)>(30,26)>(30,32)>(30,38)>(30,44); 6 states "
        "(S0 = board before; S6 = current board). Board area unchanged. "
        "45 cells changed and changed back."
    )
    assert lines[1] == (
        "- b 3x3 seen at (30,20) > (30,26) > (30,32) > (30,38) > (30,44); steps S1-S5"
    )
    assert "last_animation_timeline" in lines[-1]
    assert "step k = last_animation_frames[k-1]" in lines[-1]
    assert shown == {record["signature"]: 212}


def test_digest_repeat_becomes_one_line_pointer():
    first = _record(_sweep_chain(), 212, "MOUSE(row=31, col=21)")
    again = _record(_sweep_chain(top=12), 215, "MOUSE(row=13, col=33)")
    text, shown = build_animation_digest(
        [again], 900, 4, {first["signature"]: 212}, last_action_num=215
    )
    assert text == (
        "Animation digest: action 215 `MOUSE(row=13, col=33)` repeats the animation of "
        "action 212 (same signature); `animation_for(215)`."
    )
    assert "\n" not in text
    assert shown == {}


def test_digest_repeat_within_one_sequence():
    first = _record(_sweep_chain(), 10, "MOUSE(row=31, col=21)")
    again = _record(_sweep_chain(top=12), 11, "MOUSE(row=13, col=33)")
    text, shown = build_animation_digest([first, again], 900, 4, {}, last_action_num=11)
    assert "action 10 `MOUSE(row=31, col=21)`: path b 3x3" in text and "; 6 states" in text
    assert "action 11 `MOUSE(row=13, col=33)` repeats the animation of action 10" in text
    assert shown == {first["signature"]: 10}


def test_digest_pointer_never_names_an_unshown_original():
    # 101 has its own signature but is neither first, peak nor last, so it is
    # only counted; its repeat 102 must not point at it
    records = [
        _record(_sweep_chain(), 100, "SPACE"),
        _record(_sweep_chain(token_cols=(8, 14)), 101, "MOUSE(row=1, col=1)"),
        _record(_sweep_chain(token_cols=(8, 14), top=12), 102, "MOUSE(row=2, col=2)"),
        _record(_sweep_chain(token_cols=tuple(range(8, 56, 6))), 103, "UP"),
        _record(_sweep_chain(token_cols=(8, 14, 20)), 104, "DOWN"),
    ]
    assert records[1]["signature"] == records[2]["signature"]
    text, shown = build_animation_digest(records, 5000, 4, {}, last_action_num=104)
    assert "action 101" not in text and "action 102" not in text
    assert "(+2 more animated action(s)" in text
    assert records[1]["signature"] not in shown
    # once the original was shown on an earlier turn, the repeat points at it
    text, _ = build_animation_digest(
        records, 5000, 4, {records[1]["signature"]: 90}, last_action_num=104
    )
    assert "action 102 `MOUSE(row=2, col=2)` repeats the animation of action 90" in text


def test_digest_names_animation_for_only_for_readable_actions():
    first = _record(_sweep_chain(), 10, "SPACE")
    last = _record(_sweep_chain(token_cols=(8, 14, 20)), 12, "UP")
    text, _ = build_animation_digest([first, last], 5000, 4, {}, last_action_num=12,
                                     readable={12})
    assert "animation_for(10)" not in text and "animation_for(n)" not in text
    assert text.splitlines()[-1].endswith(
        "Exact: `animation_for(12)['timeline']`; S<k> = `animation_for(12)['frames'][k-1]`. "
        "Action 10's frames are no longer kept."
    )
    # a repeat whose own record was evicted gets no animation_for pointer
    again = _record(_sweep_chain(top=12), 11, "SPACE")
    text, _ = build_animation_digest([again], 900, 4, {first["signature"]: 10},
                                     last_action_num=11, readable=set())
    assert text == (
        "Animation digest: action 11 `SPACE` repeats the animation of action 10 "
        "(same signature)."
    )
    # nothing readable at all: no Exact pointer
    text, _ = build_animation_digest([first, last], 5000, 4, {}, last_action_num=12,
                                     readable=set(), game_over=True)
    assert "animation_for" not in text
    assert text.endswith("Frames of actions 10, 12 are no longer kept.")


def test_subsampled_last_action_points_at_the_stored_states():
    record = _record(_sweep_chain(), 7, "SPACE")
    record["summary"] = {**record["summary"], "frames": 61}  # stored 6 of 61
    text, _ = build_animation_digest([record], 900, 4, {}, last_action_num=7)
    assert "6 states (S0 = board before; S6 = current board), subsampled from 61" in text
    assert "last_animation" not in text
    assert text.splitlines()[-1].endswith(
        "Exact: `animation_for(7)['timeline']`; S<k> = `animation_for(7)['frames'][k-1]`."
    )


def test_digest_gates():
    slide = _slide_chain()
    # pure motion of a 1-wide object would be silent; force a no-transient record
    record = _record(slide, 5, "RIGHT", gameplay_changed=True)
    record["summary"] = {**record["summary"], "transient_pixels": 0}
    for gate in ("novel", "transient"):
        assert build_animation_digest([record], 900, 4, {}, gate=gate) == ("", {})
    text, _ = build_animation_digest([record], 900, 4, {}, gate="all", last_action_num=5)
    assert text.startswith("Animation digest (all): action 5 `RIGHT`")
    # transient gate shows a repeat in full; novel turns it into a pointer
    sweep = _record(_sweep_chain(), 7, "SPACE")
    seen = {sweep["signature"]: 3}
    assert build_animation_digest([sweep], 900, 4, seen)[0].startswith(
        "Animation digest: action 7"
    )
    assert build_animation_digest([sweep], 900, 4, seen, gate="transient")[0].startswith(
        "Animation digest (transient): action 7"
    )
    # an animation that left the board area unchanged qualifies with no transients
    quiet = dict(record, gameplay_changed=False)
    assert build_animation_digest([quiet], 900, 4, {})[0]


def test_digest_lists_at_most_three_actions():
    records = []
    for index, tokens in enumerate((2, 3, 8, 4, 5, 6)):
        cols = tuple(range(8, 8 + 6 * tokens, 6))
        records.append(_record(_sweep_chain(token_cols=cols), 100 + index, "SPACE"))
    text, shown = build_animation_digest(records, 5000, 4, {}, last_action_num=105)
    assert text.startswith("Animation digest (novel), 3 of 6 animated actions in the sequence:")
    # the first, the one with the most transient cells (8 tokens), and the last
    for number in (100, 102, 105):
        assert f"action {number} `SPACE`" in text
    for number in (101, 103, 104):
        assert f"action {number} `SPACE`" not in text
    assert "(+3 more animated action(s); `animations` lists the readable ones.)" in text
    assert set(shown.values()) == {100, 102, 105}


def test_digest_never_exceeds_budget():
    records = [
        _record(_sweep_chain(), 1, "MOUSE(row=31, col=21)"),
        _record(_slide_chain(), 2, "RIGHT", gameplay_changed=True),
        _record(_sweep_chain(token_cols=(8, 14, 20)), 3, "SPACE"),
    ]
    for budget in (40, 80, 150, 300, 450, 600, 900, 1400):
        text, shown = build_animation_digest(records, budget, 4, {}, last_action_num=3)
        assert len(text) <= budget
        for signature in shown:
            assert signature in {record["signature"] for record in records}
    full_text, _ = build_animation_digest(records, 10000, 4, {}, last_action_num=3)
    short_text, _ = build_animation_digest(records, 400, 4, {}, last_action_num=3)
    assert len(short_text) < len(full_text)
    assert short_text.splitlines()[0] == full_text.splitlines()[0]


def test_digest_game_over_names_death_frame():
    record = _record(_sweep_chain(), 9, "SPACE")
    text, _ = build_animation_digest([record], 900, 4, {}, last_action_num=9, game_over=True)
    assert "S6 = death frame" in text
    assert "animation_for(9)" in text


def test_digest_empty_inputs():
    assert build_animation_digest([], 900, 4, {}) == ("", {})
    record = _record(_sweep_chain(), 1, "SPACE")
    assert build_animation_digest([record], 0, 4, {}) == ("", {})


def test_overlay_trails_put_scatter_after_coherent_trails():
    before = _board()
    frames = []
    for step in range(4):
        grid = [row[:] for row in before]
        for index in range(10):  # ten scattered single cells per state
            grid[10 + 2 * index][10 + step] = 8
        _paint(grid, 40, 10 + 5 * step, 2, 2, 9)  # one coherent 2x2 sweep
        frames.append(_frozen(grid))
    frames.append(_frozen(before))
    trails = overlay_trails(collapse_frames(_frozen(before), frames), border=4)
    assert (trails[0]["color"], trails[0]["height"], trails[0]["width"]) == (9, 2, 2)
    assert trails[-1]["color"] == 8
