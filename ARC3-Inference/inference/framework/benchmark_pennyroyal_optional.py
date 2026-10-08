"""GPU checks for optional Pennyroyal patches; run with its Linux venv Python.

This validates small pinned BF16 lookups and times candidate GEMMs, not a
complete model or ARC score. Prints JSON lines; redirect to a run artifact.
"""

import argparse
import json
import math
from types import SimpleNamespace
from unittest.mock import patch


def check_embedding():
    import torch
    from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    from sglang.srt.models import arc3_bf16_embedding as offload

    rows, width = 4096, 2560

    def template(*args, **kwargs):
        # Exercise the production allocator/gather on a small table without
        # initializing distributed model execution or allocating the full PLE.
        obj = torch.nn.Module()
        for name in Qwen4ExpPinnedHostEmbedding._COPIED_ATTRIBUTES:
            setattr(obj, name, None)
        obj.quant_method = UnquantizedEmbeddingMethod()
        obj.tp_size = 1
        obj.use_attn_tp_group = False
        obj.num_embeddings = obj.org_vocab_size = obj.org_vocab_size_padded = rows
        obj.num_embeddings_padded = obj.num_embeddings_per_partition = rows
        obj.num_org_embeddings_per_partition = rows
        obj.num_added_embeddings = obj.num_added_embeddings_per_partition = 0
        obj.embedding_dim = width
        obj.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=rows)
        obj.weight = torch.nn.Parameter(torch.empty((rows, width), dtype=torch.bfloat16, device="meta"), requires_grad=False)
        return obj

    config = SimpleNamespace(vocab_size=rows, hidden_size=width, tie_word_embeddings=False)
    with patch("sglang.srt.runtime_context.get_parallel", return_value=SimpleNamespace(tp_size=1)), \
         patch("sglang.srt.layers.vocab_parallel_embedding.VocabParallelEmbedding", side_effect=template):
        target = offload.build_embedding(config, is_draft=False, use_attn_tp_group=False)
        draft = offload.build_embedding(config, is_draft=True, use_attn_tp_group=False)
    assert target is draft and target.weight is draft.weight
    reference = torch.randn(rows, width, dtype=torch.bfloat16, device="cpu")
    target.weight.data.copy_(reference)
    ids_cpu = torch.tensor([0, rows - 1, 23, 23, 1024], dtype=torch.long)
    ids = ids_cpu.cuda()
    expected = reference[ids_cpu].cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            actual = target(ids)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(actual, expected), "BF16 pinned lookup changed values"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = draft(ids)
    graph.replay()
    assert torch.equal(captured, expected), "Captured BF16 lookup changed values"
    # Replay with different IDs in the same allocation, as serving does.
    ids.copy_(ids.flip(0))
    graph.replay()
    assert torch.equal(captured, expected.flip(0)), "Captured lookup retained stale IDs"
    print(json.dumps({"check": "bf16_embedding", "exact": True, "shared_target_draft": True,
                      "cuda_graph": True, "host_bytes": target.weight.numel() * 2}), flush=True)


def benchmark_gemms(rows):
    import torch
    from triton.testing import do_bench_cudagraph
    from sglang.kernels.ops.gemm.arc3_wide_bf16_gemm import PLANS, ROWS, gemm

    torch.backends.cuda.matmul.allow_tf32 = False
    for m in rows or ROWS:
        if m not in ROWS:
            raise ValueError(f"Unsupported candidate M={m}; choose from {ROWS}")
        for n, k in PLANS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
            def baseline():
                return torch.nn.functional.linear(x, w)

            def candidate():
                return gemm(x, w)
            expected = baseline()
            actual = candidate()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            # Check graph output too; timing alone does not validate capture.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = candidate()
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0.02, atol=0.02)
            baseline_ms = do_bench_cudagraph(baseline, rep=200)
            candidate_ms = do_bench_cudagraph(candidate, rep=200)
            print(json.dumps({"check": "bf16_gemm", "m": m, "n": n, "k": k,
                              "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
                              "torch_us": baseline_ms * 1000, "candidate_us": candidate_ms * 1000,
                              "speedup": baseline_ms / candidate_ms}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--gemm", action="store_true")
    parser.add_argument("--rows", type=int, nargs="+")
    args = parser.parse_args()
    if not args.embedding and not args.gemm:
        parser.error("Select --embedding and/or --gemm")
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Run on the Kaggle SM120 GPU with the Pennyroyal venv")
    torch.manual_seed(0)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}), flush=True)
    if args.embedding:
        check_embedding()
    if args.gemm:
        benchmark_gemms(args.rows)
