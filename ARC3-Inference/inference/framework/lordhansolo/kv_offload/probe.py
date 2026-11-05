"""Does a prefix reloaded from the CPU KV tier give the same output as the GPU hit it replaces?

Run once on the live server in test runs, before the games start, and needs VLLM_SERVER_DEV_MODE=1 for the reset route.
Serves one prompt from a GPU hit twice, clears the GPU prefix cache only, then serves it again from the tier. Two GPU hits
are not identical on this runtime (int4-kv v4: up to 1.3 nats on a low-ranked token, different greedy ids), so ids cannot
gate. The test is the distance between next-token distributions, against the distance between the two GPU hits.

Exit 0: same output at the expected depth. Exit 1: the reload failed or differs; the launcher drops the tier.
Exit 3: could not judge, because the server refused a request before the reload; the tier stays and the log says so.
A crash, hang or server error before the reload counts as a failure: those prompts are the first the tier stores.
"""
import http.client
import json
import math
import random
import sys
import time
import urllib.error
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
# Total variation, on the top-5 probabilities, averaged over the positions where the replies agree so far.
DISTANCE_FLOOR = 0.02
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
    usage = response.get('usage') or {}
    return {
        'ids': choice.get('token_ids') or [item['token'] for item in content],
        'tokens': [item['token'] for item in content],
        'steps': [{top['token']: math.exp(top['logprob']) for top in item['top_logprobs']} for item in content],
        'prompt_tokens': usage.get('prompt_tokens', 0),
        'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0,
    }


def distance(left: dict, right: dict) -> float:
    '''Mean total variation over the positions with the same context. A token missing from a top-5 counts as 0.'''
    gaps = []
    for position, (a, b) in enumerate(zip(left['steps'], right['steps'])):
        gaps.append(0.5 * sum(abs(a.get(token, 0.0) - b.get(token, 0.0)) for token in {*a, *b}))
        if left['tokens'][position] != right['tokens'][position]:
            break
    return sum(gaps) / len(gaps) if gaps else 0.0


def common_prefix(left: dict, right: dict) -> int:
    count = 0
    for a, b in zip(left['ids'], right['ids']):
        if a != b:
            break
        count += 1
    return count


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
    if not first_hit['steps']:
        raise Inconclusive('the server returned no logprobs')
    if first_hit['cached_tokens'] < first_hit['prompt_tokens'] - MAX_HIT_GAP_TOKENS:
        raise Inconclusive(f'GPU hit is too shallow: {first_hit["cached_tokens"]} of {first_hit["prompt_tokens"]}')
    reset_gpu_prefix_cache(base_url.removesuffix('/v1'))
    return messages, first_hit, second_hit


def judge(reload: dict, first_hit: dict, second_hit: dict) -> str | None:
    '''Why the reload is wrong, or None. The difference between two GPU hits sets how much means nothing.'''
    depth = reload['prompt_tokens'] - MAX_HIT_GAP_TOKENS
    if reload['cached_tokens'] < depth:
        return f'tier hit too shallow: {reload["cached_tokens"]} cached, needed {depth}'
    if not reload['steps']:
        return 'the reload returned no logprobs'
    allowed = max(DISTANCE_FLOOR, 3 * distance(first_hit, second_hit))
    gap = distance(reload, first_hit)
    if gap > allowed:
        return f'next-token distributions differ: distance {gap:.4f}, allowed {allowed:.4f}, first {reload["steps"][0]} vs {first_hit["steps"][0]}'
    return None


def main() -> int:
    base_url, model, image_url = sys.argv[1:4]
    try:
        messages, first_hit, second_hit = prepare(base_url, model, image_url)
    except urllib.error.HTTPError as error:
        if error.code >= 500:
            print(f'KV_OFFLOAD_PROBE_FAILED the server failed before the reload: {error!r}', flush=True)
            return FAILED
        print(f'KV_OFFLOAD_PROBE_INCONCLUSIVE {error!r}', flush=True)
        return INCONCLUSIVE
    except (OSError, http.client.HTTPException) as error:  # refused, reset, disconnected or timed out
        print(f'KV_OFFLOAD_PROBE_FAILED the server failed before the reload: {error!r}', flush=True)
        return FAILED
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
          f'distance_gpu_gpu={distance(first_hit, second_hit):.4f} distance_tier_gpu={distance(reload, first_hit):.4f} '
          f'same_ids_gpu_gpu={common_prefix(first_hit, second_hit)} same_ids_tier_gpu={common_prefix(reload, first_hit)}', flush=True)
    reason = judge(reload, first_hit, second_hit)
    if reason:
        print(f'KV_OFFLOAD_PROBE_FAILED {reason}', flush=True)
        return FAILED
    print('KV_OFFLOAD_PROBE_PASSED', flush=True)
    return PASSED


if __name__ == '__main__':
    sys.exit(main())
