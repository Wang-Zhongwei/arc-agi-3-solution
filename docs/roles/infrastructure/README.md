# Infrastructure Engineer

## Objective

Maximize useful inference completed within the hardware and runtime budget. Find optimal configs for different concurrency and context length.

## Responsibilities

Develop a deep understanding of existing infrastructure recipes: know what each configuration and patch accomplishes, why they are effective, and how to systematically search for optimal settings. Be prepared to adjust configurations and patches as serving conditions change, or when switching to a different model (e.g., Qwen4).

- Target and draft quantization, PLE precision/offloading, and compatibility with the selected kernels.
- Speculative decoding configuration, draft vocabulary, acceptance rate, and draft memory cost.
- Attention KV and Mamba/recurrent-state capacity, cache dtype, CUDA graphs, and allocation headroom.
- Prefix-state retention across tool calls, partial prompt tails, re-tokenization, and game switching.
- Model loading, checkpoint layout, shard prefetching, and host-memory transfers.

