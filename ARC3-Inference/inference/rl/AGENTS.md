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

## QSA sparse attention

- transformers' QSA is a reference implementation: a Python loop over every query plus a dense [N, N] mask. It OOMs at 32K tokens (the failing 2.13 GiB allocation is 33,811² bf16) and would be far too slow anyway. Train through the chunked replacement instead (`qsa.py`, being validated).
- The indexer stays frozen (no LoRA; topk has no gradient), as in NeMo AutoModel.

## Pods (RunPod)

- One script per job ([pod/smoke_pod.sh](pod/smoke_pod.sh)): install, download, run, results, `DONE`/`FAILED` marker. Watch the first run of a new script step by step, and fold every surprise back into the script and this file.
- Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`: the system Python is PEP 668 managed. Use `python3 -m venv --system-site-packages /workspace/venv` (keeps torch 2.8.0+cu128).
- Start background jobs as `ssh ... 'cd DIR; nohup CMD > log 2>&1 < /dev/null &' < /dev/null`. With `cd DIR && nohup CMD &` the whole list runs in a subshell that holds ssh's stdout, and ssh never returns.
- Stop jobs by PID (`run.pid`). `pkill -f NAME` over ssh also matches and kills the ssh command itself.
- `free` inside a pod shows the host. The pod's RAM is RunPod's `gpu.memory` (251 GB on 1×H200; the PLE table takes ~102 GB of it).
- Direct SSH (needed for scp) shows up in the REST `publicIp` and `portMappings` about a minute after creation. The 181 GB checkpoint downloads in ~1 min with `hf_transfer`.
- A failed run does not stop the pod. Delete pods you created as soon as they sit idle.
- Before uploading code, run its path on a tiny random `qwen4_exp` on CPU: every pod retry costs a 5-minute reload.
