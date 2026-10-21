"""Bench patch (plan L2, issue #12): serve M <= 64 with the SM120 low-M BF16 GEMM.

Verify and draft-extend run at M = 4 * bs, i.e. 36 and 40 at bs 9 and 10. That
is above _MAX_M = 32 in Gabriel's 0004 kernel, so those shapes fall back to
cuBLAS, which streams at 60-75% of DRAM bandwidth in a CUDA graph. BLOCK_M
becomes 64 for them (next power of two of M).

Run as `python m64.py <site-packages>`, like serving/runtime_patches/*.py.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "sglang" / "kernels" / "ops" / "gemm" / "sm120_lowm_bf16_gemm.py"
text = path.read_text()
old, new = "\n_MAX_M = 32\n", "\n_MAX_M = 64\n"
if text.count(old) != 1:
    raise SystemExit(f"{path}: expected one '_MAX_M = 32', found {text.count(old)}")
path.write_text(text.replace(old, new))
print(f"patched {path}: _MAX_M 32 -> 64")
