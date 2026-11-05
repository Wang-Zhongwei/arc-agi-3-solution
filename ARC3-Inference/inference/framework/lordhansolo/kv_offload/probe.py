"""Does a prefix reloaded from the CPU KV tier give the same output as the GPU hit it replaces?

Run once on the live server in test runs, before the games start, and needs VLLM_SERVER_DEV_MODE=1 for the reset route.
Serves one prompt from a GPU hit twice, clears the GPU prefix cache only, then serves it again from the tier.

Exit 0: same output at the expected depth. Exit 1: the reload failed or differs; the launcher drops the tier.
Exit 3: could not judge, because the setup before the reload failed; the tier stays and the log says so.
"""
import json
import random
import sys
import time
import urllib.parse
import urllib.request

PASSED, FAILED, INCONCLUSIVE = 0, 1, 3
FIRST_TURN_LINES = 800
SECOND_TURN_LINES = 160
LINE_WORDS = 8
MAX_TOKENS = 64
REQUEST_TIMEOUT_SECONDS = 180
RESET_RETRY_SECONDS = 30
# A hit stops at the last Mamba checkpoint: between 128 and 255 tokens before the prompt's end.
MAX_HIT_GAP_TOKENS = 300
LOGPROB_FLOOR = 0.02
WORDS = 'red green blue amber slate north south east west river stone cloud field ridge lantern copper'.split()


class Inconclusive(Exception):
    pass


def filler(lines: int, seed: int) -> str:
    rng = random.Random(seed)
    return '\n'.join(f'{i}: ' + ' '.join(rng.choice(WORDS) for _ in range(LINE_WORDS)) for i in range(lines))


def post_json(url: str, payload: dict | None = None) -> dict:
    data = b'' if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode('utf-8'))


def serve(base_url: str, model: str, messages: list) -> dict:
    response = post_json(f'{base_url}/chat/completions', {
        'model': model, 'messages': messages, 'temperature': 0, 'seed': 0, 'max_tokens': MAX_TOKENS,
        'logprobs': True, 'top_logprobs': 5, 'return_token_ids': True,
    })
    choice = response['choices'][0]
    content = (choice.get('logprobs') or {}).get('content') or []
    first = {item['token']: item['logprob'] for item in content[0]['top_logprobs']} if content else {}
    usage = response.get('usage') or {}
    return {
        'ids': choice.get('token_ids') or [item['token'] for item in content],
        'first': first,
        'prompt_tokens': usage.get('prompt_tokens', 0),
        'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0,
    }


def logprob_gap(left: dict, right: dict) -> float:
    if set(left) != set(right):
        return float('inf')
    return max((abs(left[token] - right[token]) for token in left), default=0.0)


def reset_gpu_prefix_cache(root: str) -> None:
    query = urllib.parse.urlencode({'reset_running_requests': 'false', 'reset_external': 'false'})
    deadline = time.monotonic() + RESET_RETRY_SECONDS
    while time.monotonic() < deadline:
        if post_json(f'{root}/reset_prefix_cache?{query}').get('success'):
            return
        time.sleep(1)
    raise Inconclusive(f'the GPU prefix cache did not reset in {RESET_RETRY_SECONDS} s')


def prepare(base_url: str, model: str, image_url: str) -> tuple[list, dict, dict]:
    '''Fill the tier with one prompt and return two GPU hits on it, with the GPU prefix cache then cleared.'''
    messages = [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': image_url}}, {'type': 'text', 'text': filler(FIRST_TURN_LINES, 1)},
    ]}]
    serve(base_url, model, messages)
    messages += [
        {'role': 'assistant', 'content': 'Noted.'},
        {'role': 'user', 'content': filler(SECOND_TURN_LINES, 2) + '\nContinue the list in the same format.'},
    ]
    serve(base_url, model, messages)
    first_hit, second_hit = serve(base_url, model, messages), serve(base_url, model, messages)
    if not first_hit['ids']:
        raise Inconclusive('the server returned no token ids')
    if first_hit['cached_tokens'] < first_hit['prompt_tokens'] - MAX_HIT_GAP_TOKENS:
        raise Inconclusive(f'GPU hit is too shallow: {first_hit["cached_tokens"]} of {first_hit["prompt_tokens"]}')
    reset_gpu_prefix_cache(base_url.removesuffix('/v1'))
    return messages, first_hit, second_hit


def judge(reload: dict, first_hit: dict, second_hit: dict) -> str | None:
    '''Why the reload is wrong, or None. Two GPU hits that differ set how much difference means nothing.'''
    depth = reload['prompt_tokens'] - MAX_HIT_GAP_TOKENS
    if reload['cached_tokens'] < depth:
        return f'tier hit too shallow: {reload["cached_tokens"]} cached, needed {depth}'
    if first_hit['ids'] == second_hit['ids'] and reload['ids'] != first_hit['ids']:
        return f'greedy ids differ: {reload["ids"]} vs {first_hit["ids"]}'
    allowed = max(LOGPROB_FLOOR, 3 * logprob_gap(first_hit['first'], second_hit['first']))
    if logprob_gap(reload['first'], first_hit['first']) > allowed:
        return f'first-token logprobs differ: {reload["first"]} vs {first_hit["first"]} (allowed {allowed:.3f})'
    return None


def main() -> int:
    base_url, model, image_url = sys.argv[1:4]
    try:
        messages, first_hit, second_hit = prepare(base_url, model, image_url)
    except Exception as error:
        print(f'KV_OFFLOAD_PROBE_INCONCLUSIVE {error!r}', flush=True)
        return INCONCLUSIVE
    started = time.monotonic()
    try:
        reload = serve(base_url, model, messages)
    except Exception as error:
        print(f'KV_OFFLOAD_PROBE_FAILED the reload request failed: {error!r}', flush=True)
        return FAILED
    print(f'KV_OFFLOAD_PROBE prompt={reload["prompt_tokens"]} gpu_cached={first_hit["cached_tokens"]} '
          f'tier_cached={reload["cached_tokens"]} seconds={time.monotonic() - started:.2f} '
          f'gpu_hits_identical={first_hit["ids"] == second_hit["ids"]}', flush=True)
    reason = judge(reload, first_hit, second_hit)
    if reason:
        print(f'KV_OFFLOAD_PROBE_FAILED {reason}', flush=True)
        return FAILED
    print('KV_OFFLOAD_PROBE_PASSED', flush=True)
    return PASSED


if __name__ == '__main__':
    sys.exit(main())
