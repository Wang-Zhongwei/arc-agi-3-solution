# Lord Hansolo serving source

`kaggle.py` and `cache_diagnostics.py` are unchanged copies of the serving source
linked by [Kaggle discussion 744792](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792).
Credit belongs to Lord Hansolo and the upstream authors.

Source repository: `tonghuikang/daniel-franzen-arc-agi-3`.
Snapshot commit: `f472820b24f85e4bfe7790c66bce7d6463b5d568`.
Source directory: `kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/framework/`.
The matching `configs/draft_vocab_32k.json` is copied from the same snapshot.

Runtime dataset: `lordhansolo/vllm-main-e975732-arc3`.
vLLM commit: `e9757321527ca1ecd514c07c1418dd2c53da3d19`.
Model: `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1`.

The adapter in the parent `kaggle.py` changes the server's model alias and port,
uses this checkout's template defaults and complete harness settings, and
passes the private-wave settings to the notebook. It also fixes the vendored
module's project root so the draft vocabulary resolves inside `ARC3-Inference/`.
It does not import Lord's agent, prompt, context-trimming, or scheduler code.
