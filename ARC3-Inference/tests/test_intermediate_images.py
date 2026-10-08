import base64
import io

import pytest
from PIL import Image

from inference.agent import tool_agent, vision_context as vision
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.utils.animation import build_animation_frames, build_animation_view


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
    labels = [part['text'] for part in message['content'] if part['type'] == 'text' and part['text'].startswith('history[')]
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


def test_sampled_animation_excludes_endpoints_and_consecutive_duplicates():
    chain = [frame(0, v).grid for v in [0, 0, 1, 1, 0, 2, 3, 3]]
    parts = vision.animation_sampled_parts(chain)
    assert [image(part).getpixel((0, 0)) for _, part in parts] == [
        vision.ARC_COLOR_MAP[1], vision.ARC_COLOR_MAP[2],
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
    assert image(parts[-2]).getpixel((0, 0)) == vision.ARC_COLOR_MAP[4]
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
    current = next(i for i, part in enumerate(parts) if part['type'] == 'text' and part['text'].startswith('Current board'))
    assert current > index
    assert parts[current + 1]['type'] == 'image_url'
    assert any('Valid actions right now:' in part.get('text', '') for part in parts[current + 2:])
    assert 'shown above' not in str(parts)
