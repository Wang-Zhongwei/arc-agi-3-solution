"""CPU check of the replay-tail hold in Lord's SlidingWindowManager (run with PYTHONPATH=<runtime tree>).

Scaled down: 16-token blocks, a 32-token window, EAGLE drop (3 contiguous blocks per hit, as 1664/2048 on
the server). Request A: 200-token prompt prefilled in 32-token chunks, decoded to 300. Its replay boundary
(get_replay_boundaries, EAGLE, no fine-grained hits) is 176, so blocks 9-11 serve a later hit. After A
finishes, other allocations evict all but the 8 most recently freed blocks (request Y frees 8 cached blocks
during A's decode); request B extends A and looks
up its hit. Prints the outcome; with HOLD=1 it also asserts the patched behaviour.
"""
import hashlib
import os
from types import SimpleNamespace

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec

BLOCK, WINDOW, PROMPT, TOTAL, POOL = 16, 32, 200, 300, 64
BOUNDARY = 176
HOLD = os.environ.get('HOLD') == '1'

tokens = list(range(1000, 1000 + TOTAL + 64))


def hashes(n_tokens):
    return [hashlib.sha256(repr(tokens[: (i + 1) * BLOCK]).encode()).digest() for i in range(n_tokens // BLOCK)]


spec = SlidingWindowSpec(block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, sliding_window=WINDOW)
pool = BlockPool(num_gpu_blocks=POOL, enable_caching=True, hash_block_size=BLOCK)
manager = SlidingWindowManager(spec, block_pool=pool, enable_caching=True, kv_cache_group_id=0, scheduler_block_size=BLOCK)
manager.use_eagle = True
request = SimpleNamespace(request_id='A', block_hashes=hashes(TOTAL), shared_prefix_boundary=0, num_prompt_tokens=PROMPT)

computed, tail_blocks = 0, None
while computed < TOTAL:
    manager.remove_skipped_blocks('A', computed)
    target = min(computed + (32 if computed < PROMPT else 1), TOTAL)
    manager.allocate_new_blocks('A', target, target)
    computed = target
    manager.cache_blocks(request, computed, 0, replay_boundaries=(BOUNDARY,))
    if tail_blocks is None and len(manager.req_to_blocks['A']) >= 12:
        tail_blocks = manager.req_to_blocks['A'][9:12]
    if computed == 260:
        # Meanwhile another request finishes and frees 8 cached blocks (on the server: other
        # requests' full-attention and GDN blocks), after the window has passed A's tail.
        other = pool.get_new_blocks(8)
        other_hashes = [hashlib.sha256(b'Y%d' % i).digest() for i in range(8)]
        pool.cache_full_blocks(
            request=SimpleNamespace(request_id='Y', block_hashes=other_hashes),
            blocks=other, num_cached_blocks=0, num_full_blocks=8, block_size=BLOCK, kv_cache_group_id=0,
        )
        pool.free_blocks(other)
manager.remove_skipped_blocks('A', computed)

table = manager.req_to_blocks['A']
print('table null at 9-11:', [table[i].is_null for i in (9, 10, 11)], 'ref_cnt of 9-11:', [b.ref_cnt for b in tail_blocks])
print('tail cached (hash set):', [b.block_hash is not None for b in tail_blocks])
if HOLD:
    assert all(table[i].is_null for i in (9, 10, 11)), 'held blocks must leave the block table'
    assert all(b.ref_cnt == 1 for b in tail_blocks), 'held blocks must stay referenced while A runs'
    assert manager._held_replay_tail['A'].keys() == {9, 10, 11}
before_free = pool.get_num_free_blocks()
manager.free('A')
print('free blocks before/after freeing A:', before_free, pool.get_num_free_blocks())
assert all(b.ref_cnt == 0 for b in tail_blocks), 'every block of A must be released once'
assert 'A' not in manager._held_replay_tail if HOLD else True

# Churn: other requests take every free block except the 8 freed last.
manager.allocate_new_blocks('X', (pool.get_num_free_blocks() - 8) * BLOCK, 0)
hit = SlidingWindowManager.find_longest_cache_hit(
    block_hashes=hashes(TOTAL + 32),
    max_length=TOTAL + 31,
    kv_cache_group_ids=[0],
    block_pool=pool,
    kv_cache_spec=spec,
    drop_eagle_block=True,
    alignment_tokens=BLOCK,
)
blocks = hit[0]
while blocks and isinstance(blocks[0], (list, tuple)):
    blocks = blocks[0]
hit_tokens = len(blocks) * BLOCK  # leading null blocks are part of a sliding-window hit
print('B hit tokens:', hit_tokens)
if HOLD:
    assert hit_tokens == BOUNDARY, hit_tokens
print('OK' if HOLD else 'stock run done')
