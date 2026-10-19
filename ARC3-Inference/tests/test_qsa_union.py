"""qsa_kernel.union_tiles (plain PyTorch, runs on CPU): per tile of neighbouring queries, the union of
their selected blocks and which query selected which block, checked against a brute-force loop."""

import pytest

torch = pytest.importorskip("torch")

from inference.rl import qsa_kernel  # noqa: E402
from test_qsa_kernel_common import random_blocks  # noqa: E402


@pytest.mark.parametrize("n,topk,qt", [(203, 8, 16), (64, 32, 4), (5, 4, 16)])
def test_union_tiles_match_brute_force(n, topk, qt):
    blocks = random_blocks(n, topk, torch.Generator().manual_seed(n))
    union, bits = qsa_kernel.union_tiles(blocks, qt)
    tiles = -(-n // qt)
    assert union.shape[0] == tiles
    for t in range(tiles):
        expected: dict[int, int] = {}
        for j in range(qt):
            i = t * qt + j
            if i >= n:
                continue
            for b in blocks[i].tolist():
                if b >= 0:
                    expected[b] = expected.get(b, 0) | (1 << j)
        got = {int(b): int(m) for b, m in zip(union[t].tolist(), bits[t].tolist()) if b >= 0}
        assert got == expected, t
        count = len(expected)
        assert (union[t, :count] >= 0).all() and (union[t, count:] == -1).all()  # union first, then padding
