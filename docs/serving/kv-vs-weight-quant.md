# KV cache vs projection-weight quantization: toy sensitivity study

2026-11-04. Code and raw results: `ARC3-Inference/experiments/kv-vs-weight-quant/` (`checks.py`, `first_layer.py`,
`compare.py`). Motivated by the competitor reports in [fp4-kv.md](fp4-kv.md). These are observations from small
models, not results on Flash-Next.

## Setup

- Models: SmolLM2-135M, Qwen2.5-0.5B, Qwen3-0.6B, Qwen2.5-1.5B, FP32 reference.
- Data: this repo's markdown docs, teacher-forced, 512–1,024 tokens per chunk. The repeat used new text slices and
  noise seeds. Long-context passes: SmolLM2 at 4,096 tokens, Qwen2.5-0.5B at 2,048.
- Metric: mean per-token KL(FP32 ‖ quantized) and top-1 flips. Repeat-run noise floor is 0.
- Weights: FP8 per output channel; int4 symmetric, group 128 (or the largest power of 2 dividing the input size).
- KV: FP8 with one scale per layer; int4 asymmetric per (token, head), optionally with a Hadamard rotation (RHT).
  K is quantized after RoPE unless stated.
- Not covered: MoE, GDN, MLP weights, QAT, per-channel KV scales, long context, task accuracy, Flash-Next.

## Results

Whole-model KL, first run / repeat. Across all layouts, log-KL correlation between runs is ≥ 0.999, and the largest
ratio between runs is 1.64 (Qwen3-0.6B, K int4+RHT).

| Layout | SmolLM2-135M | Qwen2.5-0.5B | Qwen3-0.6B | Qwen2.5-1.5B |
|---|---|---|---|---|
| W_o FP8 | 0.0007 / 0.0008 | 0.0006 / 0.0006 | 0.0016 / 0.0013 | 0.0005 / 0.0005 |
| W_v FP8 | 0.0009 / 0.0010 | 0.0007 / 0.0007 | 0.0014 / 0.0011 | 0.0004 / 0.0003 |
| W_qkvo FP8 | 0.0040 / 0.0045 | 0.0025 / 0.0027 | 0.0056 / 0.0046 | 0.0017 / 0.0013 |
| V FP8 | 0.0004 / 0.0005 | 0.0005 / 0.0005 | 0.0011 / 0.0009 | 0.0004 / 0.0004 |
| KV FP8 | 0.0064 / 0.0082 | 0.0303 / 0.0285 | 0.0139 / 0.0121 | 0.3892 / 0.3537 |
| W_o int4 | 0.0133 / 0.0147 | 0.0261 / 0.0284 | 0.0388 / 0.0317 | 0.0186 / 0.0145 |
| W_v int4 | 0.0641 / 0.0728 | 0.0353 / 0.0378 | 0.0936 / 0.0743 | 0.0203 / 0.0185 |
| W_qkvo int4 | 0.1163 / 0.1325 | 0.0867 / 0.0910 | 0.2122 / 0.1687 | 0.0524 / 0.0409 |
| V int4+RHT | 0.0036 / 0.0037 | 0.0046 / 0.0049 | 0.0146 / 0.0114 | 0.0053 / 0.0043 |
| K int4+RHT, before RoPE | 0.0308 / 0.0349 | 1.50 / 1.40 | 0.0296 / 0.0232 | 5.35 / 5.17 |
| K int4+RHT, after RoPE | 0.0294 / 0.0345 | 1.82 / 1.96 | 1.20 / 0.82 | 5.80 / 5.36 |
| KV int4 | 0.0791 / 0.0967 | 2.36 / 2.48 | 6.32 / 5.52 | 6.69 / 6.92 |
| KV int4+RHT | 0.0323 / 0.0382 | 1.81 / 2.02 | 1.37 / 0.84 | 5.75 / 5.36 |
| First 3 FP8, rest KV int4+RHT | 0.0327 / 0.0365 | 0.117 / 0.122 | 0.214 / 0.177 | 0.434 / 0.407 |
| Last 3 FP8, rest KV int4+RHT | 0.0314 / 0.0366 | 1.81 / 2.02 | 1.36 / 0.83 | 5.75 / 5.35 |

Layer 0 K left exact, three chunks of the first run's text:

| Layout | SmolLM2-135M | Qwen2.5-0.5B | Qwen3-0.6B | Qwen2.5-1.5B |
|---|---|---|---|---|
| KV int4+RHT | 0.034 | 2.95 | 1.26 | 5.87 |
| L0 K exact, rest KV int4+RHT | 0.034 | 0.24 | 0.28 | 0.065 |
| KV FP8 | 0.0072 | 0.041 | 0.014 | 0.32 |
| L0 K exact, rest KV FP8 | 0.0077 | 0.021 | 0.013 | 0.0078 |

Why Qwen2.5's layer 0 K is special (K before RoPE, chunk 0):

| Layer | Energy common to all tokens | FP8 error ÷ token-varying signal | K FP8 KL |
|---|---|---|---|
| 1.5B, L0 | 99.98% | 1.86 | 0.458 |
| 1.5B, median layer | 65.5% | 0.045 | |
| 0.5B, L0 | 99.96% | 1.01 | 0.015 |
| 0.5B, L1 | 99.68% | 0.50 | 0.009 |
| 0.5B, L2 | 98.96% | 0.26 | 0.004 |
| 0.5B, median layer | 73.0% | 0.052 | |

Matched output error (5% error at the projection output), fixed ΔW vs fresh per-token noise, both runs:

| Tensor | KL ratio, fixed / per-token | Squared attention-output error ratio |
|---|---|---|
| v_proj | 2.1–9.1 | 2.1–11.2 |
| k_proj | 0.8–1.6 | 0.8–1.8 |
| q_proj | 1.1–1.4 | 0.9–1.0 |
| o_proj | 1.3–2.1 | n/a |

## Observations

1. **V tolerates low precision; K sets the cost.** Excluding layer 0, one layer's K at int4+RHT costs 2.5–33× its V.
2. **Layer 0 K can dominate.** On Qwen2.5 it is 89–99% of the summed single-layer int4+RHT K cost. Its energy is
   almost all shared across tokens, so a 2–3% error is as large as the token-varying signal. Qwen3-0.6B is not
   explained this way: its common share is 70%, and its K cost appears mainly after RoPE. Its cause is unmeasured.
3. **Protect early layers, not late ones.** First 3 FP8 cuts KL 5–17× on the three Qwen models. Last 3 FP8 changes
   almost nothing on any model. SmolLM2 needs neither.
4. **FP8 KV is not always cheap.** It costs 1.6–270× FP8 on all four attention projections. With L0 K exact, the
   1.5B gap falls to about 5×.
5. **Projections vs KV depends on bits and layout.** At int4, attention weights cost 3–4× more than unprotected KV on
   SmolLM2. On the Qwen models, they cost much less than unprotected KV and are within about 3× of protected KV.
6. **Output projection is not the most sensitive attention weight.** W_o and W_v are similar at FP8. At int4, W_v costs
   1.1–5× W_o. W_o costs 0.8–3.5× W_q. xz may mean other layers.
7. **Errors add across layers without much compounding.** Whole-model KL is 0.91–1.20× the single-layer sum.
8. **Averaging explains V, but not the other tensors.** Fixed and per-token errors at equal size cost about the same on
   q, k and o. Only V shows a clear gap, and squared attention-output error tracks that gap.
9. **RoPE order can matter.** On Qwen3-0.6B, K int4+RHT is 35–41× worse after RoPE than before it.

## Consistency with the literature

Sources were read through summarizing fetches, so their numbers are not verified against the originals.

| Observation | Sources | Fit |
|---|---|---|
| K more sensitive than V | [KVTuner](https://ar5iv.labs.arxiv.org/html/2502.04420), [KIVI](https://arxiv.org/abs/2402.02750), [More for Keys](https://arxiv.org/html/2502.15075v3) | Agrees |
| Layer 0 singled out; Qwen2.5-14B/32B layers 0–4 sensitive | KVTuner | Agrees |
| Keys have large fixed channels | KIVI, [SmoothQuant](https://arxiv.org/html/2211.10438v7) (activations) | Compatible; none reports a common-energy share |
| V survives because attention is sparse | KIVI | Agrees. More for Keys instead argues from norms |
| Weight sensitivity v ≈ o > k > q | [FP4 layer study](https://arxiv.org/html/2603.08747v1) | Agrees; its top tier, MLP up/down, was not tested here |
| Sensitive depth is non-monotonic and family-dependent | FP4 layer study, [MXSens](https://arxiv.org/html/2607.17733) | Agrees |
| Early-layer errors compound | MXSens (Llama, Mistral); [Inventive HQ](https://inventivehq.com/blog/kv-cache-quantization-quality-benchmark) asserts it | Differs; KL here is close to additive |
| Rotation rescues int4 on Qwen3-4B (73.11 vs 75.64, plain int4 0) | [SAW-INT4](https://arxiv.org/html/2604.19157) | Partly; KL falls 4.6–6.6× on Qwen3-0.6B but flips stay near 50%. Size and metric differ |
| Naive 4-bit KV has a cliff | Inventive HQ (q4_0 8.3% similarity vs q8_0 81.6%), KVTuner (Qwen2.5-7B) | Agrees |
| FP8 KV near lossless | KVTuner (asserted; KV8 measured with KIVI-style keys), [vLLM docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) (no numbers) | Partly; quantizers differ, and task accuracy was not measured here |
| Larger models tolerate more | KVTuner: Qwen2.5-14B/32B tolerate KV4, 3B/7B do not | Not shown; 1.5B is worse than 0.5B here |
| Weights dominate KV | Only the [AI Weekly item](https://aiweekly.co/alerts/qwen3-27b-tests-show-weights-dominate-kv-cache-quality), which returned 403 | Unverified; no source read compares them |
