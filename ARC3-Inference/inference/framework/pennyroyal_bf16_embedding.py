"""Optional ordinary input-table offload for Pennyroyal's BF16 Flash-Next.

Installed next to qwen4_exp.py. Reuses its pinned allocator and exact BF16
gather; PLE and the GPU lm_head are unchanged. One target/draft pair, TP=1.
"""

import os
import weakref

import torch

_target_embedding = None


def enabled():
    return os.environ.get("ARC3_OFFLOAD_EMBED_TOKENS", "0") == "1"


def build_embedding(config, *, is_draft, use_attn_tp_group):
    from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_size != 1 or getattr(config, "tie_word_embeddings", False):
        raise ValueError("BF16 embed_tokens offload currently requires TP=1 and an untied lm_head")
    global _target_embedding
    if is_draft:
        target = _target_embedding() if _target_embedding is not None else None
        if target is None:
            raise RuntimeError("BF16 embedding offload requires target and MTP draft in the same process")
        if target.org_vocab_size != config.vocab_size or target.embedding_dim != config.hidden_size:
            raise ValueError("Target and draft embedding dimensions must match")
        # Share the module before loading the draft; no second host allocation.
        # Pennyroyal's inherited set_embed_and_head later reuses this parameter.
        return target

    with torch.device("meta"):
        template = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            org_num_embeddings=config.vocab_size,
            params_dtype=torch.bfloat16,
            use_attn_tp_group=use_attn_tp_group,
        )
    # The PLE wrapper expects this buffer; ordinary lookup does not multiply it.
    template.register_buffer("weight_scale", torch.ones(1, dtype=torch.bfloat16, device="cpu"))
    result = Qwen4ExpPinnedHostEmbedding(template)
    _target_embedding = weakref.ref(result)
    return result
