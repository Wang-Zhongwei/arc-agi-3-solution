import base64
import io
import json

import pytest
import requests
from PIL import Image

from inference.agent import tool_agent, vision_context
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.transition_reminders import EXCHANGE_KEY, OMISSION_MARKER, REMINDER_KIND


def frame(step, value, level=1):
    return Frame(((value, value), (value, value)), step, level)


def text(message):
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    return '\n'.join(p.get('text', '') for p in content if p.get('type') == 'text')


def colors(message):
    return [Image.open(io.BytesIO(base64.b64decode(p['image_url']['url'].split(',', 1)[1]))).getpixel((0, 0))
            for p in message['content'] if p['type'] == 'image_url']


@pytest.fixture
def agent(monkeypatch):
    for name in ('ARC3_RETAIN_LAST_LEVEL_UP', 'ARC3_RETAIN_LAST_GAME_OVER', 'ARC3_CONTEXT_DRAIN_TOKENS',
                 'ARC3_HISTORY_TURN_DRAIN', 'ARC3_HISTORY_DRAIN_COALESCE', 'ARC3_SUMMARY_INTERVAL_TOKENS'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('MULTIMODAL_UPSCALE', '1')
    return tool_agent.ToolAgent(model='test', provider='vllm', base_url='http://server/v1')


@pytest.fixture
def tokenizer(monkeypatch):
    calls = []

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    # A deterministic tokenizer with one Unicode code point per token, which
    # makes the 1,024 / 1,024 boundary assertions exact.
    def post(url, *, json, **kwargs):
        calls.append((url, json))
        if url.endswith('/tokenize'):
            assert json['add_special_tokens'] is False
            return Response({'tokens': [ord(c) for c in json['prompt']]})
        assert url.endswith('/detokenize')
        return Response({'prompt': ''.join(chr(t) for t in json['tokens'])})

    monkeypatch.setattr(tool_agent.requests, 'post', post)
    return calls


def record(agent, tmp_path, entries, *, start=1, exchange=1, reasoning='I expect this move to work.'):
    state = tmp_path / 'state.json'
    write_runtime_state(state, current_frame=entries[-1].frame, history=entries)
    agent._record_transition_reminders(state, sequence_start=start, reasoning=reasoning,
                                       source_exchange=exchange, source_tool_call='call')
    return state


def source(exchange=1):
    return [
        {'role': 'user', 'content': 'old board'},
        {'role': 'assistant', 'tool_calls': [{'id': 'call'}], EXCHANGE_KEY: exchange},
        {'role': 'tool', 'tool_call_id': 'call', 'content': 'event', EXCHANGE_KEY: exchange},
    ]


@pytest.mark.parametrize('size', [0, 17, 2047, 2048, 2049, 5000])
def test_original_reasoning_head_and_tail(agent, tokenizer, size):
    original = ''.join(chr(0x400 + i % 100) for i in range(size))
    excerpt = agent._transition_reasoning_excerpt(original)
    if size <= 2048:
        assert excerpt == original
        assert len(tokenizer) == bool(size)
    else:
        assert excerpt == original[:1024] + '\n' + OMISSION_MARKER + '\n' + original[-1024:]
        assert len(tokenizer) == 3
        assert len(tokenizer[1][1]['tokens']) == len(tokenizer[2][1]['tokens']) == 1024


def test_tokenizer_failure_does_not_use_character_approximation(agent, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise requests.ConnectionError('offline')
    monkeypatch.setattr(tool_agent.requests, 'post', fail)
    assert agent._transition_reasoning_excerpt('original reasoning') == ''
    assert 'pre-action reasoning omitted' in caplog.text


@pytest.mark.parametrize('level_up,game_over', [(False, False), (True, False), (False, True), (True, True)])
def test_flags_latest_events_order_and_images(agent, monkeypatch, tokenizer, tmp_path, level_up, game_over):
    monkeypatch.setenv('ARC3_RETAIN_LAST_LEVEL_UP', str(level_up))
    monkeypatch.setenv('ARC3_RETAIN_LAST_GAME_OVER', str(game_over))
    entries = [HistoryEntry('', frame(0, 1)),
               HistoryEntry('RIGHT', frame(1, 2, level=2), {'level_completed': True, 'action_num': 1}),
               HistoryEntry('UP', frame(2, 3, level=2), {'game_over': True, 'action_num': 2}),
               HistoryEntry('RESET', frame(3, 4, level=2), {'automatic': True})]
    record(agent, tmp_path, entries)
    prefix = agent._transition_reminder_prefix([{'role': 'user', 'content': 'recent'}])
    if not level_up and not game_over:
        assert prefix == []
        assert tokenizer == []
        return
    assert len(prefix) == 1
    expected = ([vision_context.ARC_COLOR_MAP[1]] if level_up else [])
    expected += [vision_context.ARC_COLOR_MAP[2], vision_context.ARC_COLOR_MAP[3]] if game_over else []
    assert colors(prefix[0]) == expected  # RESET's color 4 is never an image.
    rendered = text(prefix[0])
    assert ('Last level-up 1 → 2' in rendered) == level_up
    assert ('Last GAME OVER' in rendered) == game_over
    if game_over:
        assert 'may contain mistaken assumptions' in rendered
        assert 'automatically RESET at history[3]' in rendered
    if level_up and game_over:
        assert rendered.index('Last level-up') < rendered.index('Last GAME OVER')
    assert agent._transition_reminder_prefix(source()) == []
    # A reused tool-call ID in a different exchange cannot hide this reminder.
    assert agent._transition_reminder_prefix(source(exchange=2))


def test_newer_event_replaces_previous_even_while_retained(agent, monkeypatch, tokenizer, tmp_path):
    monkeypatch.setenv('ARC3_RETAIN_LAST_LEVEL_UP', 'true')
    entries = [HistoryEntry('', frame(0, 1)),
               HistoryEntry('RIGHT', frame(1, 2, 2), {'level_completed': True})]
    record(agent, tmp_path, entries)
    entries.append(HistoryEntry('UP', frame(2, 3, 3), {'level_completed': True}))
    record(agent, tmp_path, entries, start=2, exchange=2, reasoning='new plan')
    assert len(agent._transition_reminders) == 1
    assert agent._transition_reminder_prefix(source(exchange=2)) == []
    rendered = text(agent._transition_reminder_prefix([])[0])
    assert 'Last level-up 2 → 3' in rendered and 'Last level-up 1 → 2' not in rendered
    assert 'new plan' in rendered


def test_final_win_and_session_reset(agent, monkeypatch, tokenizer, tmp_path):
    monkeypatch.setenv('ARC3_RETAIN_LAST_LEVEL_UP', 'true')
    entries = [HistoryEntry('', frame(0, 1)), HistoryEntry('UP', frame(1, 2), {'run_complete': True})]
    record(agent, tmp_path, entries)
    assert 'Game completed' in text(agent._transition_reminder_prefix([])[0])
    agent._ensure_session(tmp_path / 'new-game' / 'state.json')
    assert agent._transition_reminder_prefix([]) == []


def test_reminder_budget_duplicates_and_original_tail(agent, monkeypatch, tokenizer, tmp_path):
    monkeypatch.setenv('ARC3_RETAIN_LAST_GAME_OVER', 'true')
    entries = [HistoryEntry('', frame(0, 1)), HistoryEntry('DOWN', frame(1, 2), {'game_over': True})]
    record(agent, tmp_path, entries)
    system = {'role': 'system', 'content': 'system'}
    latest = {'role': 'user', 'content': 'unchanged recent prompt'}
    prefix = agent._transition_reminder_prefix([latest])
    agent._context_budget_tokens = agent._estimate_request_input_tokens([system, *prefix, latest]) + 10
    old = source()
    old[0]['content'] = 'old ' * 1000
    messages = [system, *old, latest]
    trimmed = agent._trim_messages_for_context(messages)
    assert trimmed[-1] is latest
    assert len(trimmed) == 3
    assert agent._estimate_request_input_tokens(trimmed) <= agent._context_budget_tokens
    again = agent._trim_messages_for_context(trimmed)
    assert again == trimmed
    assert sum(m.get(tool_agent._CONTROL_MESSAGE_KEY) == REMINDER_KIND for m in again) == 1
    wire = tool_agent._strip_control_keys(again + source()[1:])
    assert all(EXCHANGE_KEY not in m and tool_agent._CONTROL_MESSAGE_KEY not in m for m in wire)


def test_turn_cap_and_image_window_keep_reminder_boards(agent, monkeypatch, tokenizer, tmp_path):
    monkeypatch.setenv('ARC3_RETAIN_LAST_GAME_OVER', 'true')
    monkeypatch.setenv('ARC3_HISTORY_ASSISTANT_TURNS', '1')
    monkeypatch.setenv('ARC3_HISTORY_IMAGE_KEEP', '0')
    entries = [HistoryEntry('', frame(0, 1)), HistoryEntry('DOWN', frame(1, 2), {'game_over': True})]
    record(agent, tmp_path, entries)
    tail = [{'role': 'user', 'content': 'recent'}, {'role': 'assistant', 'content': 'original tail'}]
    messages = [{'role': 'system', 'content': 'system'}, *source(), *tail]
    kept = agent._persistent_history_messages(messages)
    assert kept[1:] == tail
    assert len(colors(kept[0])) == 2
    agent._context_was_trimmed = False
    agent._pending_trim_reasons = []
    again = agent._persistent_history_messages([messages[0], *kept])
    assert again == kept
    assert not agent._context_was_trimmed
    assert agent._pending_trim_reasons == []


def test_incomplete_multi_call_group_is_not_split(agent, monkeypatch):
    monkeypatch.setenv('ARC3_RETAIN_LAST_LEVEL_UP', 'true')
    block = [{'role': 'user', 'content': 'old'},
             {'role': 'assistant', 'tool_calls': [{'id': 'a'}, {'id': 'b'}]},
             {'role': 'tool', 'tool_call_id': 'a', 'content': 'result'},
             {'role': 'user', 'content': 'recent'}]
    original = list(block)
    assert not agent._drop_oldest_history_block(block, preserve_recent=1)
    assert block == original
    block.insert(3, {'role': 'tool', 'tool_call_id': 'b', 'content': 'result'})
    assert agent._drop_oldest_history_block(block, preserve_recent=1)
    assert block == [original[-1]]


def test_overflow_recovery_keeps_one_reminder_and_complete_tail(agent, monkeypatch, tokenizer, tmp_path):
    monkeypatch.setenv('ARC3_RETAIN_LAST_GAME_OVER', 'true')
    entries = [HistoryEntry('', frame(0, 1)), HistoryEntry('DOWN', frame(1, 2), {'game_over': True})]
    record(agent, tmp_path, entries)
    system = {'role': 'system', 'content': 'system'}
    old = source()
    recent = [{'role': 'user', 'content': 'latest'},
              {'role': 'assistant', 'tool_calls': [{'id': 'b'}]},
              {'role': 'tool', 'tool_call_id': 'b', 'content': 'unchanged result'}]
    reduced = agent._force_reduce_messages([system, *old, *recent])
    assert reduced[2:] == recent
    assert len(colors(reduced[1])) == 2
    assert agent._trim_messages_for_context(reduced) == reduced


def test_analyze_captures_action_producing_reasoning(agent, monkeypatch, tokenizer, tmp_path):
    monkeypatch.setenv('ARC3_RETAIN_LAST_LEVEL_UP', 'true')
    monkeypatch.setenv('MULTIMODAL_CONTEXT', 'current_grid')
    entries = [HistoryEntry('', frame(0, 1))]
    state = tmp_path / 'state.json'
    write_runtime_state(state, current_frame=entries[0].frame, history=entries)
    agent._yield_seconds = agent._yield_tokens = None
    replies = iter([
        tool_agent._ChatCompletionResult(message={'reasoning_content': 'Earlier unrelated investigation.', 'content': ''}),
        tool_agent._ChatCompletionResult(message={
            'reasoning_content': 'I expect RIGHT to complete this level.',
            'tool_calls': [{'id': 'call', 'type': 'function', 'function': {
                'name': 'python', 'arguments': json.dumps({'code': 'action("RIGHT")'})}}],
        }),
    ])
    monkeypatch.setattr(agent, '_chat_completion', lambda *a, **kw: next(replies))

    def dispatch(*args):
        entries.append(HistoryEntry('RIGHT', frame(1, 2, 2), {'level_completed': True}))
        write_runtime_state(state, current_frame=entries[-1].frame, history=entries)
        return tool_agent._ToolDispatchResult('LEVEL UP', step_executed=True, made_progress=True)

    monkeypatch.setattr(agent, '_dispatch_tool', dispatch)
    agent.analyze(state, 0, ['RIGHT'])
    rendered = text(agent._transition_reminder_prefix([])[0])
    assert 'I expect RIGHT to complete this level.' in rendered
    assert 'Earlier unrelated investigation.' not in rendered
    assert agent._transition_reminder_prefix(agent._history_messages) == []
