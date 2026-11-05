"""GPU self-test of the int4 KV path, run once before the vLLM server starts.

Writes random K/V through the real int4 writer (randomized Hadamard rotation included), reads them back and runs the
real sparse attention kernel at decode and prefill row counts. Exits non-zero on a mismatch, so a wrong kernel costs a
few minutes of a notebook instead of a whole run of bad attention.
"""
import torch
from types import SimpleNamespace

from vllm import taaf_qsa_kv8 as kv8

HEAD_DIM = 256
KV_HEADS = 2
QUERY_HEADS = 40
PAGE = 64
BLOCKS = 256
REQUESTS = 4
WIDTH = 2048
WRITE_CHUNK = 2048
SAMPLED_ROWS = 16
# (rows, use_prefill_config): 14, 27 and 30 requests of a k = 3 decode step, then prefill chunks.
CASES = ((56, False), (108, False), (120, False), (1024, True), (2048, True))
# Random Gaussian rows lose about 11% to 16 levels, so a healthy read-back sits near cosine 0.99.
MIN_ROUND_TRIP_COSINE = 0.95
# The kernel and the same-codes Python reference differ only by BF16 rounding.
MAX_REFERENCE_ERROR = 5e-2
# Quantized against unquantized attention: a rotation mismatch would land near 0.
MIN_EXACT_COSINE = 0.8


def cosine(left, right):
    return torch.nn.functional.cosine_similarity(left.flatten(1).float(), right.flatten(1).float(), dim=1).mean().item()


def exact_attention(rows, query, keys, values, indices, table, owners, gate):
    group = query.shape[1] // KV_HEADS
    out = []
    for row in rows:
        picked = indices[row, :WIDTH].long()
        slots = table[int(owners[row])].long()[picked // PAGE] * PAGE + picked % PAGE
        grouped = query[row].float().view(KV_HEADS, group, HEAD_DIM)
        weights = torch.softmax(torch.einsum("hgd,nhd->hgn", grouped, keys[slots].float()) * HEAD_DIM ** -0.5, dim=-1)
        attended = torch.einsum("hgn,nhd->hgd", weights, values[slots].float()).reshape(-1, HEAD_DIM)
        out.append(attended * torch.sigmoid(gate[row].float()))
    return torch.stack(out)


@torch.no_grad()
def main():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    content = kv8.content_bytes(*kv8.widths("int4", HEAD_DIM))
    physical = torch.zeros((BLOCKS, PAGE, KV_HEADS, content), dtype=torch.uint8, device=device)
    cache = physical.permute(0, 2, 1, 3)
    holder = SimpleNamespace(kv_cache_dtype=kv8.INT4_DTYPE, head_size=HEAD_DIM)
    tokens = BLOCKS * PAGE
    keys = torch.randn((tokens, KV_HEADS, HEAD_DIM), generator=generator, device=device).to(torch.bfloat16)
    values = torch.randn((tokens, KV_HEADS, HEAD_DIM), generator=generator, device=device).to(torch.bfloat16)
    slots = torch.arange(tokens, device=device)
    for start in range(0, tokens, WRITE_CHUNK):
        chunk = slots[start:start + WRITE_CHUNK]
        kv8.write(holder, keys[chunk], values[chunk], cache, chunk)

    key_codes, value_codes, key_scales, value_scales = kv8.views(holder, cache, HEAD_DIM, "int4")
    block, offset = slots[:WRITE_CHUNK] // PAGE, slots[:WRITE_CHUNK] % PAGE
    key_round_trip = cosine(kv8.dequantized("int4", key_codes[block, offset], key_scales[block, offset], HEAD_DIM),
                            keys[:WRITE_CHUNK])
    value_round_trip = cosine(kv8.dequantized("int4", value_codes[block, offset], value_scales[block, offset], HEAD_DIM),
                              values[:WRITE_CHUNK])
    print(f"INT4_KV_PREFLIGHT round_trip key_cos={key_round_trip:.4f} value_cos={value_round_trip:.4f}", flush=True)
    if min(key_round_trip, value_round_trip) < MIN_ROUND_TRIP_COSINE:
        raise SystemExit(f"int4 KV read-back below cosine {MIN_ROUND_TRIP_COSINE}")

    table = torch.randperm(BLOCKS, generator=generator, device=device).view(REQUESTS, -1).to(torch.int32)
    span = table.shape[1] * PAGE
    for rows, use_prefill in CASES:
        picks = torch.stack([torch.randperm(span, generator=generator, device=device)[:WIDTH] for _ in range(rows)])
        indices = torch.cat([picks, torch.full((rows, 1), WIDTH, device=device)], dim=1).to(torch.int32).contiguous()
        owners = torch.randint(0, REQUESTS, (rows,), generator=generator, device=device, dtype=torch.int32)
        query = torch.randn((rows, QUERY_HEADS, HEAD_DIM), generator=generator, device=device).to(torch.bfloat16)
        gate = torch.randn((rows, QUERY_HEADS, HEAD_DIM), generator=generator, device=device).to(torch.bfloat16)
        out = torch.empty_like(query)
        kv8.attention(holder, query, cache, indices, table, owners, use_prefill, out, gate)
        sampled = list(range(0, rows, max(rows // SAMPLED_ROWS, 1)))[:SAMPLED_ROWS]
        reference = kv8.int4_reference(sampled, query, key_codes, value_codes, key_scales, value_scales, indices, table,
                                       owners, gate)
        kernel = out[sampled].float()
        error = ((kernel - reference).norm() / reference.norm()).item()
        exact = cosine(kernel, exact_attention(sampled, query, keys, values, indices, table, owners, gate))
        print(f"INT4_KV_PREFLIGHT rows={rows} prefill={use_prefill} reference_error={error:.2e} exact_cos={exact:.4f}",
              flush=True)
        if error > MAX_REFERENCE_ERROR or exact < MIN_EXACT_COSINE:
            raise SystemExit(f"int4 attention failed at {rows} rows: error {error:.2e}, exact cosine {exact:.4f}")
    print("INT4_KV_PREFLIGHT passed", flush=True)


if __name__ == "__main__":
    main()
