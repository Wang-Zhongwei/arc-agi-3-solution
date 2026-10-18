# Agent notes for `inference/rl`

Pitfalls hit while building the RFT pipeline (2026-10-17/18). Read before changing the rebuild, the smoke test or anything that runs on a pod. Decisions and architecture: [README](README.md).

## Context rebuild

- Transcripts hold every message but not the context window. [context_rebuild.py](context_rebuild.py) rebuilds it from the per-turn `history_messages` count; check any change with [validate_rebuild.py](validate_rebuild.py) against runs that saved `*_requests.jsonl`.
- Pair logged and rebuilt requests by the order of `response` rows, never by `(analysis_step, request_index_within_turn)`: a turn retried after a timeout repeats the key. A `response` row has `usage`, but its `messages` are the request, not the reply.
- A turn that ends with the history count it started with was rolled back (yield or failure); its messages never enter history.
- `vc_sampling: replaced reply` swaps the reply and its tool results for another candidate, logged without a meta section.
- Runs with features not on main (view tool, transition reminders, trimmed-history notes) add messages the transcript never logs, so their windows rebuild badly. Do not train on them.
- Trims inside a turn are not logged. Requests close to the budget are flagged `uncertain`; drop them from training.
- Tool results are logged as display text, so the rebuilt JSON payload lacks `function_retention` notes (~11% of tool-result chars).

## Chat template and tokenization

- The template reads `reasoning_content`, not `reasoning`; tool-call `arguments` must be a dict; past thinking is kept when `preserve_thinking` is unset or true. A reply's training target starts after `<|im_start|>assistant\n<think>\n` and ends with `<|im_end|>`.
- Pass the processor's `mm_token_type_ids` with images, or the model refuses them (M-RoPE). Tokenizing piecewise at reply boundaries gives the same ids as tokenizing the whole text.
- A 640×640 board image is 400 tokens plus 2 markers.
- Never compute full logits (vocab 248,320): pass `logits_to_keep` with the trained positions.

## Loading the model with transformers

- `qwen4_exp` needs transformers ≥ 5.16 (5.20.1 tested).
- Do not use `device_map="auto"` with the AutoRound W4A16 checkpoint. It sizes the routed experts as BF16 (~242 GB) before AutoRound swaps in INT4, and offloads all of them to CPU (log: "requires 62757274136 bytes of buffer for offloaded layers"). Use `gpu_device_map` in [smoke_lora.py](smoke_lora.py): everything on `cuda:0` except `_no_placement_params`, i.e. the single 102.5 GB PLE n-gram table, which stays in host RAM and is looked up there.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, or loading OOMs from fragmentation (66.6 GiB allocated, 72.7 GiB reserved but unused).
- AutoRound unfuses the 48×512 experts (~3 min) and picks `auto_round:tritonv2_zp`. A full load takes ~5 min and holds 68.5 GiB on the GPU. The gptqmodel backend it suggests is inference-only.
- Without `causal_conv1d`, GDN's short convolution falls back to the PyTorch reference (a warning). fla 0.5.2 covers the delta rule.
- GDN backward on Hopper: torch 2.8 ships Triton 3.4, whose gated `chunk_bwd_dqkwg` is wrong on sm90, so fla raises (fla #640). Install `tilelang` (0.1.15 keeps torch 2.8) and export `CUDA_HOME=/usr/local/cuda` so it finds nvcc. fla checks for tilelang at import, so a running process must be restarted. The first backward JIT-compiles for ~1 min.

## QSA sparse attention

- transformers' QSA is a reference implementation: a Python loop over every query plus a dense [N, N] mask. It OOMs at 32K tokens (the failing 2.13 GiB allocation is 33,811² bf16) and would be far too slow anyway. Train through [qsa.py](qsa.py) (`patch_qsa_attention`), which chunks queries, gathers each query's ~2,051 selected keys and checkpoints each chunk.
- [tests/test_qsa.py](../../tests/test_qsa.py) checks it against the reference on a tiny model: identical selections, attention output, logits with an image, and gradients. At the real budget (512 blocks) selections match up to blocks tied at the cut-off: several blocks often score exactly 0 after the relu, and topk breaks such ties arbitrarily (as any kernel does).
- Use the model's own `apply_rotary_pos_emb`: RoPE covers only the first 25% of each head (interleaved M-RoPE).
- The indexer stays frozen (no LoRA; topk has no gradient), as in NeMo AutoModel.

## Activation memory

- Call `model.train()`: `from_pretrained` returns an eval-mode model, and transformers' `GradientCheckpointingLayer` checkpoints only in training mode. Otherwise `gradient_checkpointing_enable()` silently does nothing: 32K OOMed on the GPU at 137.6 GiB, and with offload all activations went to pinned host memory and the cgroup OOM-killed the process (`/sys/fs/cgroup/memory.events`: `oom_kill`). `smoke_lora` reports `checkpointed_layers`.

- Hyper-connections widen the residual to 4 × 2,560, so each layer's checkpointed input is N × 10,240: ~31.5 GB over 48 layers at 32K, ~98 GB at 100K, next to 68.5 GiB of weights. Offload them to host memory (`step(offload=True)`, the `o` suffix in `smoke_lora --lengths`); the backward recompute runs outside `save_on_cpu`, so only the checkpoints move.
- Never build [K, vocab] logits at once (10K trained positions × 248,320 = 10 GB in fp32): `chunked_cross_entropy` checkpoints the lm_head + loss per 2,048 positions. Both changes are exact (loss and gradients equal to 1e-9 on the tiny model).

## Pods (RunPod)

- One script per job ([pod/smoke_pod.sh](pod/smoke_pod.sh)): install, download, run, results, `DONE`/`FAILED` marker. Watch the first run of a new script step by step, and fold every surprise back into the script and this file.
- Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`: the system Python is PEP 668 managed. Use `python3 -m venv --system-site-packages /workspace/venv` (keeps torch 2.8.0+cu128).
- Start background jobs as `ssh ... 'cd DIR; nohup CMD > log 2>&1 < /dev/null &' < /dev/null`. With `cd DIR && nohup CMD &` the whole list runs in a subshell that holds ssh's stdout, and ssh never returns.
- Stop jobs by PID (`run.pid`). `pkill -f NAME` over ssh also matches and kills the ssh command itself.
- `free` inside a pod shows the host. The pod's RAM is RunPod's `gpu.memory` (251 GB on 1×H200; the PLE table takes ~102 GB of it).
- Direct SSH (needed for scp) shows up in the REST `publicIp` and `portMappings` about a minute after creation. The 181 GB checkpoint download speed depends on the data center: 52 s on US-NC-1, 665 s on US-CO-1 (both `hf_transfer`). Pin `dataCenterIds` to a fast one, or keep the checkpoint on a network volume.
- A failed run does not stop the pod. Delete pods you created as soon as they sit idle.
- Before uploading code, run its path on a tiny random `qwen4_exp` on CPU: every pod retry costs a 5-minute reload.
