"""inference/rft/grouped_gemm.py (Triton, CUDA only): the grouped GEMM equals per-group matmuls, forward
and input gradient (with the weight transposed), including empty and uneven groups, at the experts' shapes.
The reference is FP32 matmul on the same BF16 inputs; the kernel may be at most 1.5x as far from it as
cuBLAS in BF16."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the grouped GEMM needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

from inference.rft import grouped_gemm  # noqa: E402


def per_group(x, w, offsets, transpose=False, dtype=None):
    """Per-group matmuls, optionally in `dtype` (cast one group at a time: the experts in FP32 are 6.7 GB)."""
    out, start = [], 0
    for g, end in enumerate(offsets.tolist()):
        a, b = x[start:end], (w[g].t() if transpose else w[g])
        if dtype is not None:
            a, b = a.to(dtype), b.to(dtype)
        out.append(a @ b)
        start = end
    return torch.cat(out)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("groups,k,n", [(512, 2560, 1280), (512, 640, 2560), (7, 96, 200)])
def test_grouped_matmul_matches_per_group(groups, k, n):
    g = torch.Generator().manual_seed(groups + k)
    counts = torch.randint(0, 400, (groups,), generator=g)
    counts[::5] = 0  # empty groups
    offsets = counts.cumsum(0).cuda()
    m = int(counts.sum())
    x = torch.randn(m, k, generator=g).cuda().bfloat16()
    w = (torch.randn(groups, k, n, generator=g) / k ** 0.5).cuda().bfloat16()
    dy = torch.randn(m, n, generator=g).cuda().bfloat16()
    ref = per_group(x, w, offsets, dtype=torch.float32)
    ours = grouped_gemm.grouped_matmul(x, w, offsets)
    cublas = per_group(x, w, offsets)
    assert rel(ours, ref) <= 1.5 * rel(cublas, ref) + 1e-4, (rel(ours, ref), rel(cublas, ref))
    ref_t = per_group(dy, w, offsets, transpose=True, dtype=torch.float32)
    ours_t = grouped_gemm.grouped_matmul(dy, w, offsets, transpose=True)
    cublas_t = per_group(dy, w, offsets, transpose=True)
    assert rel(ours_t, ref_t) <= 1.5 * rel(cublas_t, ref_t) + 1e-4, (rel(ours_t, ref_t), rel(cublas_t, ref_t))


def test_grouped_mm_autograd_input_gradient():
    g = torch.Generator().manual_seed(0)
    offsets = torch.tensor([5, 5, 40, 64]).cuda()
    x = torch.randn(64, 32, generator=g).cuda().bfloat16().requires_grad_(True)
    w = torch.randn(4, 32, 48, generator=g).cuda().bfloat16()
    dy = torch.randn(64, 48, generator=g).cuda().bfloat16()
    grouped_gemm.GroupedMM.apply(x, w, offsets).backward(dy)
    xr = x.detach().float().requires_grad_(True)
    per_group(xr, w.float(), offsets).backward(dy.float())
    assert rel(x.grad, xr.grad) < 1e-2
