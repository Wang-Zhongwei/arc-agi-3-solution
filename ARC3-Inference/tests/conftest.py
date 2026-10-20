"""Skip main's tests that assert prompts and tools this branch does not use.

The list is in branch_divergent_tests.txt, one `file::test[params]` per line.
"""
from pathlib import Path

import pytest

_LIST = Path(__file__).with_name("branch_divergent_tests.txt")
_DIVERGENT = {
    line.strip() for line in _LIST.read_text().splitlines()
    if line.strip() and not line.startswith("#")
}
_REASON = (
    "This branch keeps its pre-merge prompts and tools; this test asserts main's "
    "wording or segmentation fields (branch_divergent_tests.txt)."
)


def pytest_collection_modifyitems(items):
    for item in items:
        if f"{Path(str(item.fspath)).name}::{item.name}" in _DIVERGENT:
            item.add_marker(pytest.mark.skip(reason=_REASON))


def _bf16_grouped_mm_only():
    """transformers sends FP32 experts to torch._grouped_mm on Hopper (torch 2.8: BF16 only), so the tests'
    FP32 reference models failed there with 'Expected mat_a to be BFloat16'. Its fallback loop for them."""
    try:
        import torch
        from transformers.integrations import moe
    except ImportError:
        return
    can_use = moe._can_use_grouped_mm
    moe._can_use_grouped_mm = lambda input, weight, offs: (weight.device.type != "cuda" or weight.dtype == torch.bfloat16) \
        and can_use(input, weight, offs)


_bf16_grouped_mm_only()
