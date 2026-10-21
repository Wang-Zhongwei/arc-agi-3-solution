"""Bench patch (plan L2): drop split-K from the SM120 low-M BF16 GEMM's generic tactic.

Gabriel's 0005 measured split-K slower than the split-free kernel on every
real shape. Without it every untuned shape runs (1, 16, 128, 4, 4): one CTA
per 16 output columns and a bf16 store instead of fp32 atomics.

Run as `python nosplitk.py <site-packages>`, like serving/runtime_patches/*.py.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "sglang" / "kernels" / "ops" / "gemm" / "sm120_lowm_bf16_gemm.py"
text = path.read_text()
old = (
    "    while split_k < 16 and n_ctas * split_k < 128 and (k // (split_k * 2)) >= 256:\n"
    "        split_k *= 2\n"
)
if text.count(old) != 1:
    raise SystemExit(f"{path}: expected one split-K loop in _generic_tactic, found {text.count(old)}")
path.write_text(text.replace(old, ""))
print(f"patched {path}: generic tactic without split-K")
