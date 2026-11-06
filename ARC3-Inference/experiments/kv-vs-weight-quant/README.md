# KV cache vs projection-weight quantization (toy)

Question: does low precision on the KV cache hurt less than low precision on the attention projection
weights, and does the output projection hurt more than the input projections? Motivated by competitor
reports after ARC Prize 2026 (see `docs/serving/fp4-kv.md`).

## Run

```bash
uv venv .venv -p 3.11 && uv pip install -p .venv/bin/python torch transformers numpy
.venv/bin/python sens.py [model] [seq_len] [n_seq]      # default HuggingFaceTB/SmolLM2-135M 1024 3
```

Model weights are downloaded from Hugging Face on the first run. Results land in `results_<model>.json`
and the report goes to `docs/serving/kv-vs-weight-quant.md`.

## What it measures

Teacher-forced KL(FP32 ‖ perturbed) over real text (the repo's markdown docs), plus top-1 disagreement
and error growth ‖Δh_L‖/‖Δh_ℓ‖ from the injected block to the last block.

| Block | Perturbation | Claim tested |
|---|---|---|
| A | one tensor of one layer: W_q/W_k/W_v/W_o in FP8 (per-channel) or int4 (g128); K or V cache in FP8 (per-layer scale), int4 per (token, head), int4 + Hadamard | projection vs KV; input vs output; which layers |
| B | fixed ΔW vs fresh per-token noise at the same relative size | coherent vs independent error (vs accumulation) |
| C | whole-model layouts: all W_o FP8, all W_qkv FP8, all KV int4, middle KV int4 + edge FP8, edge int4 + middle FP8 | zhangzibo54's layout |

`checks.py` supersedes `sens.py`: it matches errors at the projection output and quantizes K after RoPE. Run
`checks.py <model> <seq> <nseq>`, with `REP=1` for new text and seeds, then `first_layer.py <model> <seq> <nseq>`,
then `compare.py`.

Caveats for `sens.py`: K is quantized before RoPE (KVQuant style; vLLM caches it after RoPE), FP8 KV uses the best-case
per-layer scale calibrated on the same text, and the toy has no MoE or GDN layers.
