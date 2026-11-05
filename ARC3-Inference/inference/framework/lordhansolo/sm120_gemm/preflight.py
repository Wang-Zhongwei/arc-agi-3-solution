"""GPU check of the SM120 GEMM plans above M=64, run once before the vLLM server starts.

Lord tuned the tiles for M up to 64. For each candidate row count, runs every projection's plan and the original linear
method inside a captured CUDA graph, as the server does, on random BF16 tensors. A row is accepted only if every output
row stays close, so a wrong masked tail row cannot hide in a whole-matrix norm, and its replay is faster in total.
M=64 is the control: if the plan does not win there, the timing is not trusted and nothing is accepted.
Prints SM120_GEMM_ACCEPTED [...]; the launcher keeps the runtime's rows otherwise.
"""
import json
import sys
import time

import torch

from vllm.models.qwen4_exp.nvidia.arc3_sm120_gemm import PROJECTIONS, run_gemm

CONTROL_ROWS = 64
WARMUP_REPLAYS = 20
TIMED_ROUNDS = 100
MAX_ROW_ERROR = 1e-2
MAX_TIME_RATIO = 0.98
# Each row compiles its own kernels. Stop starting rows after this; the launcher's limit is 600 s.
BUDGET_SECONDS = 420


def capture(fn, *args):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(*args)
    return graph, output


def original(x, weight):
    return torch.nn.functional.linear(x, weight)


def worst_row_error(actual, expected):
    return ((actual - expected).norm(dim=1) / expected.norm(dim=1)).max().item()


@torch.no_grad()
def measure(rows, weights):
    plans, originals, errors = [], [], {}
    for name, ((columns, inner), plan) in PROJECTIONS.items():
        x = torch.randn((rows, inner), device='cuda', dtype=torch.bfloat16)
        weight = weights[name]
        errors[name] = worst_row_error(run_gemm(x, weight, plan).float(), original(x, weight).float())
        plans.append(capture(run_gemm, x, weight, plan)[0])
        originals.append(capture(original, x, weight)[0])
    for graphs in (plans, originals):
        for graph in graphs:
            for _ in range(WARMUP_REPLAYS):
                graph.replay()
    torch.cuda.synchronize()
    totals = [0.0, 0.0]
    for _ in range(TIMED_ROUNDS):
        for which, graphs in enumerate((plans, originals)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            totals[which] += time.perf_counter() - start
    return max(errors.values()), totals[0] / totals[1], totals[1] / TIMED_ROUNDS


def main():
    started = time.monotonic()
    candidates = json.loads(sys.argv[1])
    torch.manual_seed(0)
    weights = {name: torch.randn(shape, device='cuda', dtype=torch.bfloat16) for name, (shape, _) in PROJECTIONS.items()}
    control_error, control_ratio, _ = measure(CONTROL_ROWS, weights)
    print(f'SM120_GEMM_PREFLIGHT rows={CONTROL_ROWS} control error={control_error:.2e} time_ratio={control_ratio:.3f}', flush=True)
    accepted = []
    if control_ratio <= 1.0:
        for rows in candidates:
            if time.monotonic() - started > BUDGET_SECONDS:
                print(f'SM120_GEMM_PREFLIGHT rows={rows} skipped, past {BUDGET_SECONDS} s', flush=True)
                continue
            error, ratio, original_ms = measure(rows, weights)
            passed = error <= max(MAX_ROW_ERROR, 2 * control_error) and ratio <= MAX_TIME_RATIO
            print(f'SM120_GEMM_PREFLIGHT rows={rows} error={error:.2e} time_ratio={ratio:.3f} '
                  f'original_ms={1000 * original_ms:.3f} accepted={passed}', flush=True)
            if passed:
                accepted.append(rows)
    print(f'SM120_GEMM_ACCEPTED {json.dumps(accepted)}', flush=True)


if __name__ == '__main__':
    main()
