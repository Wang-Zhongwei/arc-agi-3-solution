import base64
import io
import re

import pytest
from PIL import Image

from inference.agent import tool_agent, vision_context as vision
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.utils.animation import build_animation_frames, build_animation_view, summarize_animation


def frame(step, value=0, level=1):
    return Frame(grid=tuple((value,) * 64 for _ in range(64)), step=step, level=level)


def image(part):
    return Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',', 1)[1])))


@pytest.fixture(autouse=True)
def scales(monkeypatch):
    monkeypatch.setenv('MULTIMODAL_CONTEXT', 'current_grid')
    monkeypatch.setenv('MULTIMODAL_UPSCALE', '8')
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_UPSCALE', '2')
    monkeypatch.setenv('MULTIMODAL_ANIMATION_UPSCALE', '2')
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', 'false')


@pytest.mark.parametrize('enabled,count', [(True, 3), (True, 1), (False, 3)])
def test_python_call_attaches_only_new_intermediate_boards(monkeypatch, tmp_path, enabled, count):
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', str(enabled).lower())
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    state = tmp_path / 'state.json'
    start = frame(5)
    entries = [HistoryEntry('UP', frame(4)), HistoryEntry('UP', start)]
    write_runtime_state(state, current_frame=start, history=entries)

    # Simulate the sandbox completing a call: the solver writes each executed
    # action to history, while the sandbox returns compact execution metadata.
    def sandbox(**kwargs):
        for step in range(6, 6 + count):
            entries.append(HistoryEntry('DOWN', frame(step, step)))
        write_runtime_state(state, current_frame=entries[-1].frame, history=entries)
        return {'action_results': [{'executed': True, 'gameplay_changed': True}]}

    monkeypatch.setattr(tool_agent, 'run_sandboxed_python', sandbox)
    result = agent._run_python_tool(state, {'code': 'pass'})
    assert result.step_executed
    message = agent._build_user_message('Continue', entries[-1].frame)
    images = [part for part in message['content'] if part['type'] == 'image_url']
    expected = count - 1 if enabled else 0
    assert len(images) == 1 + expected
    assert image(images[-1]).size == (512, 512)
    assert [image(part).size for part in images[:-1]] == [(128, 128)] * expected
    labels = [part['text'] for part in message['content'] if part['type'] == 'text'
              and re.fullmatch(r'history\[\d+\]\.frame \(after DOWN\):', part['text'])]
    assert labels == [f'history[{2 + index}].frame (after DOWN):' for index in range(expected)]
    # The history index is absolute, not the action step or the batch position.
    payload = tool_agent._ascii_history_view_payload(entries)
    for index, part in enumerate(images[:-1]):
        assert image(part).getpixel((0, 0)) == vision.ARC_COLOR_MAP[payload[2 + index]['frame']['grid'][0][0]]
    again = agent._build_user_message('Continue', entries[-1].frame)
    assert sum(part['type'] == 'image_url' for part in again['content']) == 1


def test_fatal_board_and_level_transition_labels():
    entries = [HistoryEntry('UP', frame(2, level=2)),
               HistoryEntry('DOWN', frame(3, level=2), {'state': 'GAME_OVER'}),
               HistoryEntry('RESET', frame(4, level=2))]
    parts = vision.intermediate_frame_parts(entries, after_step=1, before_step=4)
    assert len(parts) == 2
    assert parts[0][0] == 'history[0].frame (after UP):'
    assert parts[1][0] == 'history[1].frame (after DOWN):'
    assert vision.intermediate_frame_parts(entries, after_step=4, before_step=4) == []


@pytest.mark.parametrize('count', [0, 1, 8, 9, 29])
def test_intermediate_keeps_every_eligible_history_index(count):
    entries = [HistoryEntry('INIT', frame(0))]
    entries.extend(HistoryEntry('RIGHT', frame(i, i % 16)) for i in range(1, count + 2))
    parts = vision.intermediate_frame_parts(entries, after_step=0, before_step=count + 1)
    indices = [int(re.fullmatch(r'history\[(\d+)\]\.frame \(after RIGHT\):', label).group(1))
               for label, _ in parts]
    assert indices == list(range(1, count + 1))
    if count:
        for index, (_, part) in zip(indices, parts):
            assert image(part).size == (128, 128)
            assert image(part).getpixel((0, 0)) == vision.ARC_COLOR_MAP[index % 16]
    assert count + 1 not in indices  # the current board is attached separately


@pytest.mark.parametrize('inherit', [False, True])
def test_animation_scale_and_existing_board_scale(monkeypatch, inherit):
    if inherit:
        monkeypatch.delenv('MULTIMODAL_ANIMATION_UPSCALE')
    chain = [frame(0).grid, frame(0, 8).grid, frame(0).grid]
    peak, index = vision.animation_peak_part(chain, [(1, 1)])
    composite = vision.animation_composite_part(chain, [(1, 1)], border=0)
    size = (512, 512) if inherit else (128, 128)
    assert index == 0
    assert image(peak).size == image(composite).size == size
    assert image(peak).getpixel((0, 0)) == vision.ARC_COLOR_MAP[8]
    assert image(vision.current_grid_image_part(frame(0))).size == (512, 512)


def test_intermediate_inherits_scale_and_respects_text_only(monkeypatch):
    monkeypatch.delenv('MULTIMODAL_INTERMEDIATE_UPSCALE')
    entries = [HistoryEntry('UP', frame(1))]
    parts = vision.intermediate_frame_parts(entries, after_step=0, before_step=2)
    assert image(parts[0][1]).size == (512, 512)
    monkeypatch.setenv('MULTIMODAL_CONTEXT', '')
    assert vision.intermediate_frame_parts(entries, after_step=0, before_step=2) == []


def test_sampled_animation_excludes_pre_action_and_final_boards_and_consecutive_duplicates():
    chain = [frame(0, v).grid for v in [0, 0, 1, 1, 0, 2, 3, 3]]
    parts = vision.animation_sampled_parts(chain)
    assert [image(part).getpixel((0, 0)) for _, part in parts] == [
        vision.ARC_COLOR_MAP[1], vision.ARC_COLOR_MAP[0],
        vision.ARC_COLOR_MAP[2], vision.ARC_COLOR_MAP[3],
    ]
    assert all(image(part).size == (128, 128) for _, part in parts)
    assert vision.animation_sampled_parts(chain[:2]) == []


def test_sampled_animation_caps_and_spreads_frames():
    chain = [frame(0, v).grid for v in range(16)]
    parts = vision.animation_sampled_parts(chain)
    assert len(parts) == 8
    assert [image(part).getpixel((0, 0)) for _, part in parts] == [
        vision.ARC_COLOR_MAP[v] for v in [1, 2, 4, 6, 8, 10, 12, 14]
    ]


def test_animation_samples_can_show_a_return_to_the_pre_action_board():
    chain = [frame(0, v).grid for v in [0, 1, 0, 2, 3]]
    parts = vision.animation_sampled_parts(chain)
    assert [label for label, _ in parts] == [f'last_animation_frames[{i}]:' for i in range(3)]
    assert [image(part).getpixel((0, 0)) for _, part in parts] == [
        vision.ARC_COLOR_MAP[v] for v in [1, 0, 2]
    ]


def test_executed_action_runs_are_grouped_before_middle_is_shortened():
    assert tool_agent._format_executed_actions(['RIGHT'] * 4 + ['LEFT'] * 3) == 'RIGHT x 4, LEFT x 3'
    actions = ['UP', 'DOWN'] * 7
    assert tool_agent._format_executed_actions(actions) == (
        'UP, DOWN, UP, DOWN, UP, DOWN, ..., UP, DOWN, UP, DOWN, UP, DOWN'
    )
    assert tool_agent._format_executed_actions(['RIGHT'] * 100) == 'RIGHT x 100'


@pytest.mark.parametrize('values', [list(range(16)), [0, 0, 1, 1, 0, 2, 3, 3]])
def test_sampled_captions_index_the_exposed_animation_frames(values):
    import re
    chain = [frame(0, value).grid for value in values]
    record = {'chain': chain, 'step': 27, 'level': 2}
    exposed = build_animation_frames(record)
    timeline = build_animation_view(record)
    for label, part in vision.animation_sampled_parts(chain):
        index = int(re.fullmatch(r'last_animation_frames\[(\d+)\]:', label).group(1))
        expected = frame(27, exposed[index]['grid'][0][0], level=2)
        expected_image = vision.frame_to_png_data_url(expected, upscale=2)
        assert part['image_url']['url'] == expected_image
        assert any(step['step'] == index + 1 for step in timeline['steps'])


@pytest.mark.parametrize('fatal', [False, True])
def test_sampled_mode_bypasses_old_gates_and_keeps_diffs(monkeypatch, fatal):
    monkeypatch.setenv('ARC3_ANIMATION', '1')
    monkeypatch.setenv('ARC3_ANIMATION_IMAGE', 'sampled')
    monkeypatch.setenv('ARC3_ANIMATION_DEATH_IMAGE', 'sampled')
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._last_animation_chain = [frame(0, v).grid for v in [0, 1, 2, 3]]
    agent._last_action_gameplay_changed = True
    summary = {'animation': {'transient_pixels': 0}}
    if fatal:
        agent._append_fatal_animation_image(summary)
    else:
        agent._prepare_animation_image(summary)
    agent._pending_diff_image = vision.current_grid_image_part(frame(0, 4))
    agent._pending_diff_image_label = 'Existing diff'
    message = agent._build_user_message('Continue', frame(1, 3))
    parts = [p for p in message['content'] if p['type'] == 'image_url']
    assert len(parts) == 4  # current board, existing diff, two interior frames
    assert image(parts[-1]).getpixel((0, 0)) == vision.ARC_COLOR_MAP[4]
    again = agent._build_user_message('Continue', frame(1, 3))
    assert sum(p['type'] == 'image_url' for p in again['content']) == 1


@pytest.mark.parametrize('kind', ['actions', 'animation', 'death'])
def test_images_follow_related_prompt_text(monkeypatch, tmp_path, kind):
    monkeypatch.setenv('ARC3_ANIMATION', '1')
    monkeypatch.setenv('ARC3_ANIMATION_IMAGE', 'sampled')
    monkeypatch.setenv('ARC3_ANIMATION_DEATH_IMAGE', 'sampled')
    monkeypatch.setenv('ARC3_GAMEOVER_DIFF_IMAGE', '1')
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._ensure_session(tmp_path / 'state.json')
    agent._last_animation_chain = [frame(0, v).grid for v in [0, 1, 2, 3]]
    agent._last_action_gameplay_changed = True
    entries = [HistoryEntry('UP', frame(1)), HistoryEntry('RIGHT', frame(2, 3), {'state': 'GAME_OVER'}), HistoryEntry('RESET', frame(3))]
    if kind == 'actions':
        monkeypatch.setenv('ARC3_ANIMATION_IMAGE', 'off')
        agent._pending_intermediate_images = vision.intermediate_frame_parts(entries, after_step=0, before_step=2)
    summary = {'executed_count': 2, 'executed_actions': ['UP', 'RIGHT'], 'animation': {'frames': 3, 'transient_pixels': 0}, 'game_over': kind == 'death'}
    prompt = agent._build_user_prompt(3, valid_actions=['UP'], current_frame=frame(3), history_entries=entries, previous_step_summary=summary)
    message = agent._build_user_message(prompt, frame(3))
    parts = message['content']
    anchor = agent._prompt_image_anchors[kind]
    index = next(i for i, part in enumerate(parts) if part['type'] == 'text' and anchor in part['text'])
    assert parts[index + 1]['type'] == 'text'
    assert parts[index + 2]['type'] == 'image_url'
    current = next(i for i, part in enumerate(parts) if part['type'] == 'text'
                   and ('CURRENT board' in part['text'] or part['text'].startswith('Current board')))
    assert current > index
    assert parts[current + 1]['type'] == 'image_url'
    assert any('Valid actions right now:' in part.get('text', '') for part in parts[current + 2:])
    assert 'shown above' not in str(parts)


def test_long_sequence_keeps_absolute_indices_and_current_before_diff(monkeypatch, tmp_path):
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', 'true')
    monkeypatch.setenv('ARC3_MEMORY_SECTIONS', 'off')
    monkeypatch.setenv('ARC3_ANIMATION_IMAGE', 'off')
    monkeypatch.setattr(tool_agent, '_AUTO_FRAME_DIFF', True)
    monkeypatch.setattr(tool_agent, '_DIFF_IMAGE', True)
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._ensure_session(tmp_path / 'state.json')
    actions = ['RIGHT', 'UP', 'UP', 'UP', 'UP', 'UP', 'RIGHT', 'RIGHT', 'DOWN',
               'RIGHT', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN', 'LEFT', 'DOWN',
               'DOWN', 'LEFT', 'RIGHT', 'UP', 'UP', 'RIGHT', 'UP', 'UP', 'UP', 'UP', 'UP', 'RIGHT']
    entries = [HistoryEntry('UP', frame(index, index % 16, level=2)) for index in range(253)]
    entries.extend(HistoryEntry(action, frame(index, index % 16, level=2))
                   for index, action in enumerate(actions, 253))
    current_frame = entries[-1].frame
    agent._auto_diff_prev_frame = entries[252].frame
    agent._pending_intermediate_images = vision.intermediate_frame_parts(entries, after_step=252, before_step=281)
    prompt = agent._build_user_prompt(281, valid_actions=['UP', 'DOWN', 'LEFT', 'RIGHT'],
                                    current_frame=current_frame, history_entries=entries,
                                    previous_step_summary={'executed_count': 29, 'executed_actions': actions})
    parts = agent._build_user_message(prompt, current_frame)['content']
    texts = [part['text'] for part in parts if part['type'] == 'text']
    assert texts[0].startswith('The code executed 29 actions.\nExecuted actions: ')
    assert texts[1] == 'history[253].frame (after RIGHT):'
    assert 'history[256].frame (after UP):' in texts
    assert 'history[280].frame (after UP):' in texts
    current_index = next(i for i, part in enumerate(parts) if part.get('text') == 'history[281].frame (after RIGHT): this is the CURRENT board to act on.')
    assert parts[current_index + 1]['image_url'] == vision.current_grid_image_part(current_frame)['image_url']
    assert parts[current_index + 2]['text'] == ('Diff image (history[252].frame -> history[281].frame): cells that changed keep their true color; '
                                              'unchanged cells dimmed to dark navy.')
    assert parts[current_index + 3]['type'] == 'image_url'
    assert parts[current_index + 4]['text'].index('Current state: step 282, level 2.') < parts[current_index + 4]['text'].index('You are still on the same level.')
    images = [part for part in parts if part['type'] == 'image_url']
    assert len(images) == 30  # every intermediate board, one current board, one diff
    assert [image(part).size for part in images] == [(128, 128)] * 28 + [(512, 512)] * 2
    for index, part in zip(range(253, 281), images[:-2]):
        assert image(part).getpixel((0, 0)) == vision.ARC_COLOR_MAP[entries[index].frame.grid[0][0]]
    assert 'Executed actions: RIGHT, UP x 5, RIGHT x 2' in ''.join(texts)
    assert ', ..., LEFT, RIGHT, UP x 2' in ''.join(texts)
    assert 'Auto frame diff (history[252].frame -> history[281].frame;' in ''.join(texts)
    assert '[image attached]' not in ''.join(texts)
    assert 'Use Python to inspect' not in ''.join(texts)
    assert 'Focus on what changed most recently' not in ''.join(texts)
    assert 'Inspect the evidence from the newest history' in ''.join(texts)


@pytest.mark.parametrize('kind', ['single', 'reset', 'initial'])
def test_current_caption_maps_to_history_for_single_action_reset_and_initial(monkeypatch, tmp_path, kind):
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', 'true')
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._ensure_session(tmp_path / 'state.json')
    entries = [HistoryEntry('INIT', frame(5)), HistoryEntry('RIGHT', frame(6, 8))]
    summary = {'executed_count': 1, 'executed_actions': ['RIGHT']}
    if kind == 'initial':
        entries = entries[:1]
        summary = None
    elif kind == 'reset':
        entries.append(HistoryEntry('RESET', frame(7)))
        summary.update(game_over=True, fatal_action='RIGHT')
        agent._pending_intermediate_images = vision.intermediate_frame_parts(entries, after_step=5, before_step=7)
    current_frame = entries[-1].frame
    prompt = agent._build_user_prompt(current_frame.step, valid_actions=['UP'], current_frame=current_frame,
                                    history_entries=entries, previous_step_summary=summary)
    parts = agent._build_user_message(prompt, current_frame)['content']
    current = next(i for i, part in enumerate(parts) if 'CURRENT board' in part.get('text', ''))
    expected = f'history[{len(entries) - 1}].frame (after {entries[-1].action})'
    assert parts[current]['text'] == f'{expected}: this is the CURRENT board to act on.'
    assert parts[current + 1] == vision.current_grid_image_part(current_frame)
    assert any('Current state:' in part.get('text', '') for part in parts[current + 2:])
    if kind == 'reset':
        assert 'automatically RESET the current level after GAME OVER' in prompt


@pytest.mark.parametrize('text_only', [False, True])
def test_baseline_and_text_only_keep_action_echo(monkeypatch, tmp_path, text_only):
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', str(text_only).lower())
    if text_only:
        monkeypatch.setenv('MULTIMODAL_CONTEXT', '')
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._ensure_session(tmp_path / 'state.json')
    prompt = agent._build_user_prompt(1, valid_actions=['RIGHT'], current_frame=frame(1),
                                    previous_step_summary={'executed_count': 1, 'executed_actions': ['RIGHT']})
    assert 'The code executed 1 action.' in prompt
    assert 'Executed actions: RIGHT.' in prompt


@pytest.mark.parametrize('auto_diff', [False, True])
def test_shorter_sequence_preserves_animation_and_optional_change_summaries(monkeypatch, tmp_path, auto_diff):
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', 'true')
    monkeypatch.setenv('ARC3_ANIMATION', '1')
    monkeypatch.setenv('ARC3_ANIMATION_IMAGE', 'off')
    monkeypatch.setattr(tool_agent, '_AUTO_FRAME_DIFF', auto_diff)
    monkeypatch.setattr(tool_agent, '_DIFF_IMAGE', True)
    initial = frame(0)
    first_grid = [list(row) for row in initial.grid]
    for row in range(37, 42):
        for col in range(13, 18):
            if (row, col) not in {(41, 16), (41, 17)}:
                first_grid[row][col] = 14
    first = Frame(grid=tuple(map(tuple, first_grid)), step=1, level=1)
    final_grid = [list(row) for row in first.grid]
    for row in range(37, 42):
        for col in range(19, 24):
            if (row, col) not in {(41, 22), (41, 23)}:
                final_grid[row][col] = 14
    final_grid[63][1] = 9
    final = Frame(grid=tuple(map(tuple, final_grid)), step=2, level=1)
    chain = [first.grid]
    for color in (8, 9):
        grid = [list(row) for row in first.grid]
        for row in range(13, 18):
            for col in range(49, 54):
                grid[row][col] = color
        chain.append(tuple(map(tuple, grid)))
    chain.extend([first.grid, final.grid])
    action = 'MOUSE(row=15, col=51)'
    entries = [HistoryEntry('INIT', initial), HistoryEntry('RIGHT', first), HistoryEntry(action, final)]
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._ensure_session(tmp_path / 'state.json')
    agent._auto_diff_prev_frame = initial
    agent._last_animation_chain = chain
    agent._last_action_gameplay_changed = True
    agent._pending_intermediate_images = vision.intermediate_frame_parts(entries, after_step=0, before_step=2)
    summary = {'executed_count': 2, 'executed_actions': ['RIGHT', action],
               'animation': summarize_animation(chain), 'animation_action': action}
    prompt = agent._build_user_prompt(2, valid_actions=['UP'], current_frame=final,
                                    history_entries=entries, previous_step_summary=summary)
    parts = agent._build_user_message(prompt, final)['content']
    text = '\n'.join(part.get('text', '') for part in parts)
    assert 'animated over 4 frames.' in text
    assert '25 cells changed more than once around rows 13-17, cols 49-53.' in text
    assert '`last_animation_frames` or `last_animation_timeline`' in text
    assert ('47 cells changed.' in text) is auto_diff
    assert f'Executed actions: RIGHT, {action}.' in text
    assert sum(part['type'] == 'image_url' for part in parts) == 3


@pytest.mark.parametrize('mode', ['animation', 'intermediate', 'both'])
@pytest.mark.parametrize('event', ['level', 'death', 'noop_one', 'noop_many', 'stale'])
def test_shared_event_prompts_and_evidence_order(monkeypatch, tmp_path, mode, event):
    intermediate = mode != 'animation'
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', str(intermediate).lower())
    monkeypatch.setenv('ARC3_MEMORY_SECTIONS', 'off')
    monkeypatch.setenv('ARC3_ANIMATION', '1')
    monkeypatch.setenv('ARC3_ANIMATION_IMAGE', 'off' if mode == 'intermediate' else 'sampled')
    monkeypatch.setenv('ARC3_ANIMATION_DEATH_IMAGE', 'off' if mode == 'intermediate' else 'sampled')
    monkeypatch.setenv('ARC3_GAMEOVER_DIFF_IMAGE', '1')
    monkeypatch.setenv('ARC3_LEVEL_TRANSFER_GUIDANCE', '1')
    monkeypatch.setenv('ARC3_PERSISTENT_FUNCTIONS', '1')
    monkeypatch.setenv('ARC3_PERSISTENT_FUNCTIONS_SCOPE', 'game')
    monkeypatch.setattr(tool_agent, '_AUTO_FRAME_DIFF', False)
    monkeypatch.setattr(tool_agent, '_DIFF_IMAGE', True)
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._ensure_session(tmp_path / 'state.json')
    entries = [HistoryEntry('INIT', frame(100))]
    entries.extend(HistoryEntry('RIGHT', frame(100 + i, i)) for i in range(1, 5))
    summary = {'executed_count': 4, 'executed_actions': ['RIGHT'] * 4}
    if event == 'level':
        entries[-1] = HistoryEntry('RIGHT', frame(104, 4, level=2))
        summary['level_transition'] = True
    elif event == 'death':
        summary.update(game_over=True, fatal_action='RIGHT',
                       animation={'frames': 4, 'transient_pixels': 25, 'transient_bbox': (13, 49, 17, 53)})
        entries.append(HistoryEntry('RESET', frame(105)))
        agent._last_animation_chain = [frame(0, v).grid for v in [3, 8, 9, 4]]
    elif event.startswith('noop'):
        skipped = ['MOUSE(row=51, col=39)'] if event == 'noop_one' else ['UP', 'DOWN', 'LEFT']
        summary['no_op_stops'] = [{'index': 4, 'action': 'RIGHT', 'skipped': skipped}]
    else:
        summary['guard_stops'] = [{'kind': 'stale_state', 'action': 'DOWN', 'skipped': ['DOWN', 'LEFT']}]
    agent._last_step_summary = summary
    agent._kept_functions = {'pos': 'def pos(color):\n    return color\n'}
    agent._auto_diff_prev_frame = entries[0].frame
    current = entries[-1].frame
    if intermediate:
        agent._pending_intermediate_images = vision.intermediate_frame_parts(entries, after_step=100, before_step=current.step)
    prompt = agent._build_user_prompt(current.step, valid_actions=['RIGHT'], current_frame=current,
                                    history_entries=entries, previous_step_summary=summary)
    parts = agent._build_user_message(prompt, current)['content']
    text = '\n'.join(part.get('text', '') for part in parts)
    current_index = next(i for i, part in enumerate(parts) if 'CURRENT board' in part.get('text', ''))
    state_index = next(i for i, part in enumerate(parts) if 'Current state:' in part.get('text', ''))
    assert current_index < state_index
    assert image(parts[current_index + 1]).size == (512, 512)
    assert 'Inspect the evidence from the newest history' not in text
    assert '[image attached]' not in text
    assert 'Retained functions in this game: pos(color).' in text
    if event == 'level':
        assert 'Build a plan for this layout.' in text
        assert text.index('Current state:') < text.index('You have completed the previous level.')
        assert 'Carry forward established mechanics.' in text
        assert 'Check level-specific assumptions against the new board before reusing them.' in text
        assert 'Diff image (' not in text
    elif event == 'death':
        death_index = next(i for i, part in enumerate(parts) if 'DEATH board' in part.get('text', ''))
        fatal_animation_index = next(i for i, part in enumerate(parts) if 'The fatal action RIGHT' in part.get('text', ''))
        fatal_diff_index = next(i for i, part in enumerate(parts) if 'Fatal step diff image' in part.get('text', ''))
        assert fatal_animation_index < death_index < current_index < fatal_diff_index < state_index
        assert 'history[3].frame -> history[4].frame' in text
        assert text.count('DEATH board') == 1
        assert '25 cells changed more than once' in text
        assert 'before treating this as a no-op' not in text
        assert 'BAR RULE:' in text and "that belief is exactly what this death falsified." in text
        labels = [part.get('text', '') for part in parts]
        assert '4 (RIGHT):' not in labels  # death is not duplicated as an intermediate
        assert image(parts[death_index + 1]).size == (512, 512)
    elif event.startswith('noop'):
        assert 'Action 4, RIGHT, executed but changed nothing in the board interior.' in text
        if event == 'noop_one':
            assert 'before action 5, MOUSE(row=51, col=39), which was never executed.' in text
        else:
            assert 'remaining 3 actions, starting at action 5; none were executed.' in text
        assert 'Diff image (history[0].frame -> history[4].frame)' in text
    else:
        assert 'The next action, DOWN, was not executed:' in text
        assert 'All 2 remaining actions were skipped and never executed.' in text
    animation_labels = [part['text'] for part in parts if part.get('text', '').startswith('last_animation_frames[')]
    if event == 'death' and mode != 'intermediate':
        assert animation_labels == ['last_animation_frames[0]:', 'last_animation_frames[1]:']
    else:
        assert animation_labels == []


@pytest.mark.parametrize('mode', ['animation', 'intermediate', 'both'])
@pytest.mark.parametrize('changed', [False, True])
def test_animation_guidance_is_shared_and_samples_follow_it(monkeypatch, tmp_path, mode, changed):
    monkeypatch.setenv('ARC3_MEMORY_SECTIONS', 'off')
    monkeypatch.setenv('ARC3_ANIMATION', '1')
    monkeypatch.setenv('ARC3_ANIMATION_IMAGE', 'off' if mode == 'intermediate' else 'sampled')
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', str(mode != 'animation').lower())
    monkeypatch.setattr(tool_agent, '_AUTO_FRAME_DIFF', False)
    monkeypatch.setattr(tool_agent, '_DIFF_IMAGE', True)
    entries = [HistoryEntry('INIT', frame(100)), HistoryEntry('UP', frame(101, 3)),
               HistoryEntry('SPACE', frame(102, 4 if changed else 3))]
    chain = [entries[1].frame.grid, frame(0, 8).grid, frame(0, 9).grid, entries[2].frame.grid]
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._ensure_session(tmp_path / 'state.json')
    agent._auto_diff_prev_frame = entries[0].frame
    agent._last_action_gameplay_changed = changed
    agent._last_animation_chain = chain
    if mode != 'animation':
        agent._pending_intermediate_images = vision.intermediate_frame_parts(entries, after_step=100, before_step=102)
    prompt = agent._build_user_prompt(102, valid_actions=['UP'], current_frame=entries[-1].frame,
                                    history_entries=entries, previous_step_summary={
                                        'executed_count': 2, 'executed_actions': ['UP', 'SPACE'],
                                        'animation': summarize_animation(chain), 'animation_action': 'SPACE'})
    parts = agent._build_user_message(prompt, entries[-1].frame)['content']
    text = '\n'.join(part.get('text', '') for part in parts)
    current_caption = next(part['text'] for part in parts if 'CURRENT board' in part.get('text', ''))
    assert current_caption == (
        'history[2].frame: this is the CURRENT board to act on.' if mode == 'animation'
        else 'history[2].frame (after SPACE): this is the CURRENT board to act on.'
    )
    assert 'The last action SPACE animated over 3 frames.' in text
    assert 'changed more than once' in text
    assert ('board interior is unchanged after the animation' in text) is (not changed)
    assert 'Inspect the evidence from the newest history' not in text
    if mode != 'intermediate':
        index = next(i for i, part in enumerate(parts) if 'The last action SPACE animated' in part.get('text', ''))
        assert parts[index + 1]['text'] == 'last_animation_frames[0]:'
        assert image(parts[index + 2]).getpixel((0, 0)) == vision.ARC_COLOR_MAP[8]
        assert parts[index + 3]['text'] == 'last_animation_frames[1]:'
        assert image(parts[index + 4]).getpixel((0, 0)) == vision.ARC_COLOR_MAP[9]
        assert 'CURRENT board' in parts[index + 5]['text']
        assert parts[index + 6] == vision.current_grid_image_part(entries[-1].frame)
        assert parts[index + 7]['text'].startswith('Diff image (history[0].frame -> history[2].frame):')
        assert 'Current state:' in parts[index + 9]['text']
    assert sum(part['type'] == 'image_url' for part in parts) == {
        'animation': 4, 'intermediate': 3, 'both': 5}[mode]


@pytest.mark.parametrize('event', ['ordinary', 'level', 'death'])
@pytest.mark.parametrize('mode', ['off', 'sampled', 'peak', 'composite'])
def test_chronological_order_across_animation_modes(monkeypatch, tmp_path, event, mode):
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', 'true')
    monkeypatch.setenv('ARC3_ANIMATION', '1')
    monkeypatch.setenv('ARC3_ANIMATION_IMAGE', mode)
    monkeypatch.setenv('ARC3_ANIMATION_DEATH_IMAGE', mode)
    monkeypatch.setenv('ARC3_GAMEOVER_DIFF_IMAGE', '1')
    monkeypatch.setenv('ARC3_ANIMATION_TIMELINE', '0')
    monkeypatch.setattr(tool_agent, '_AUTO_FRAME_DIFF', False)
    monkeypatch.setattr(tool_agent, '_DIFF_IMAGE', True)
    result = frame(2, 3 if event == 'ordinary' else 4, level=2 if event == 'level' else 1)
    entries = [HistoryEntry('INIT', frame(0)), HistoryEntry('RIGHT', frame(1, 3)),
               HistoryEntry('SPACE', result)]
    chain = [entries[1].frame.grid, frame(0, 8).grid, frame(0, 9).grid, result.grid]
    summary = {'executed_count': 2, 'executed_actions': ['RIGHT', 'SPACE'],
               'animation': summarize_animation(chain), 'animation_action': 'SPACE'}
    if event == 'level':
        summary['level_transition'] = True
    elif event == 'death':
        summary.update(game_over=True, fatal_action='SPACE')
        entries.append(HistoryEntry('RESET', frame(3)))
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._ensure_session(tmp_path / 'state.json')
    agent._auto_diff_prev_frame = entries[0].frame
    agent._last_action_gameplay_changed = event != 'ordinary'
    agent._last_animation_chain = chain
    current = entries[-1].frame
    agent._pending_intermediate_images = vision.intermediate_frame_parts(entries, after_step=0, before_step=current.step)
    prompt = agent._build_user_prompt(current.step, valid_actions=['UP'], current_frame=current,
                                    history_entries=entries, previous_step_summary=summary)
    parts = agent._build_user_message(prompt, current)['content']
    labels = [part.get('text', '') for part in parts]
    earlier = labels.index('history[1].frame (after RIGHT):')
    animation = next(i for i, text in enumerate(labels) if 'animated over 3 frames.' in text)
    outcome = next(i for i, text in enumerate(labels) if ('DEATH board' if event == 'death' else 'CURRENT board') in text)
    current_index = next(i for i, text in enumerate(labels) if 'CURRENT board' in text)
    state = next(i for i, text in enumerate(labels) if 'Current state:' in text)
    assert earlier < animation < outcome <= current_index < state
    assert 'last_animation_timeline' not in labels[animation]
    if mode == 'sampled' or (mode in ('peak', 'composite') and event != 'level'):
        assert any(part['type'] == 'image_url' for part in parts[animation + 1:outcome])
        for label in labels[animation + 1:outcome]:
            if label:
                assert label.startswith(('last_animation_frames[', 'Animation image', 'Fatal action animation image'))
    else:
        assert outcome == animation + 1
    if event == 'death':
        assert labels.count('history[2].frame (after SPACE): this is the DEATH board.') == 1
        assert parts[outcome + 1] == vision.current_grid_image_part(result)
        assert current_index == outcome + 2


@pytest.mark.parametrize('death_images', [False, True])
@pytest.mark.parametrize('animated', [False, True])
@pytest.mark.parametrize('intermediate', [False, True])
def test_single_fatal_action_with_optional_images_and_animation(monkeypatch, tmp_path, death_images, animated, intermediate):
    monkeypatch.setenv('MULTIMODAL_INTERMEDIATE_FRAMES', str(intermediate).lower())
    monkeypatch.setenv('ARC3_ANIMATION', str(int(animated)))
    monkeypatch.setenv('ARC3_ANIMATION_DEATH_IMAGE', 'sampled')
    monkeypatch.setenv('ARC3_GAMEOVER_DIFF_IMAGE', str(int(death_images)))
    monkeypatch.setattr(tool_agent, '_AUTO_FRAME_DIFF', False)
    monkeypatch.setattr(tool_agent, '_DIFF_IMAGE', True)
    entries = [HistoryEntry('INIT', frame(0)), HistoryEntry('RIGHT', frame(1, 3)),
               HistoryEntry('RESET', frame(2))]
    agent = tool_agent.ToolAgent(model='test-model', base_url='http://127.0.0.1:8001/v1', provider='vllm')
    agent._ensure_session(tmp_path / 'state.json')
    agent._last_animation_chain = [frame(0, value).grid for value in (0, 8, 9, 3)]
    agent._auto_diff_prev_frame = entries[0].frame
    summary = {'executed_count': 1, 'executed_actions': ['RIGHT'], 'game_over': True, 'fatal_action': 'RIGHT'}
    if animated:
        summary['animation'] = summarize_animation(agent._last_animation_chain)
    if intermediate:
        agent._pending_intermediate_images = vision.intermediate_frame_parts(entries, after_step=0, before_step=2)
    prompt = agent._build_user_prompt(2, valid_actions=['UP'], current_frame=entries[-1].frame,
                                    history_entries=entries, previous_step_summary=summary)
    parts = agent._build_user_message(prompt, entries[-1].frame)['content']
    labels = [part.get('text', '') for part in parts]
    current = next(i for i, text in enumerate(labels) if 'CURRENT board' in text)
    death_indices = [i for i, text in enumerate(labels) if 'DEATH board' in text]
    assert len(death_indices) == int(death_images or intermediate)
    if death_indices:
        death = death_indices[0]
        assert labels[death] == 'history[1].frame (after RIGHT): this is the DEATH board.'
        assert death + 2 == current
        assert image(parts[death + 1]).size == ((512, 512) if death_images else (128, 128))
    if animated:
        animation = next(i for i, text in enumerate(labels) if 'The fatal action RIGHT animated' in text)
        assert animation < (death_indices[0] if death_indices else current)
    assert sum(text.startswith('Fatal step diff image') for text in labels) == int(death_images)
    assert all(not text.startswith('Diff image') for text in labels)
