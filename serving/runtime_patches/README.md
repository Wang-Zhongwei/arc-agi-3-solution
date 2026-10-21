# Runtime patches to the installed SGLang

Files here are embedded into the notebook's serving cell by `kaggle/build_notebook.py` and applied, in name order, to the installed SGLang 2.5.3 before the server starts. Only `*.patch` and `*.py` files are embedded; this README is ignored.

- `NN-name.patch`: a unified diff against site-packages, applied with `patch -p1`, so paths look like `a/sglang/srt/managers/scheduler.py`. Make one with `diff -u` against a copy of the installed file.
- `NN-name.py`: a script run as `python NN-name.py <site-packages>`, for edits that a diff expresses poorly.

Each file is applied once per install, keyed by its content hash, and a failure stops the launch. An empty folder leaves serving exactly as Franzen shipped it. See `serving/CUSTOM.md` for what is worth patching.

## Patches here
- `20-moe-routing.py`: MoE routing variants (`SGLANG_ARC_MOE_ROUTING=base_norm|nucleus|temp`). Inert unless that variable is set. See `serving/CUSTOM.md`.
