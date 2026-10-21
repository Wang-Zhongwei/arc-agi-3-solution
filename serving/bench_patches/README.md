# Bench runtime patches

Patches that only a serving-benchmark config applies. A config in `kaggle/bench.json` names them in `"runtime_patches"`, and `kaggle/build_notebook.py --bench` embeds the ones its enabled configs use.

They use the format of `serving/runtime_patches/` (`*.py` run as `python <file> <site-packages>`, `*.patch` applied with `patch -p1`). They are applied after the shared runtime patches, to a fresh copy of the installed SGLang for every config. A patch must therefore edit files under `site-packages/sglang/` only. A patch that does not match fails that config's launch, and the bench moves on to the next config.

- `m64.py`: the SM120 low-M BF16 GEMM serves M up to 64, so verify and draft-extend at bs 9/10 (M 36/40) leave cuBLAS (plan L2, issue #12).
- `nosplitk.py`: the same kernel's generic tactic drops split-K.

A patch that proves itself here reaches the game runs in one of two ways:
- An arm lists it in `kaggle/arms.json` `"runtime_patches"`, so only that arm runs it.
- Once it is adopted for every arm, it moves to `serving/runtime_patches/`.
