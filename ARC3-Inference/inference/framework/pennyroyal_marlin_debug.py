"""Bisect the NVFP4 Marlin MoE self-check failure (lord-nvfp4-pennyroyal-tuned v1).

Run by pennyroyal_launch.py with the Pennyroyal venv's Python on the GPU before
any server starts (kaggle.sglang.MARLIN_DEBUG_ONLY). Usage:

    python pennyroyal_marlin_debug.py <model dir> <output json>

Every case compares a Marlin result with a float32 reference dequantized the
compressed-tensors way (E2M1 nibble, low nibble first, x fp8 block scale per 16
values, / weight_global_scale). Cases, in bisect order:

1. dense-synthetic: the fork's tested dense NVFP4 Marlin GEMM on random weights.
2. dense-real: the same kernel on layer 0 expert 0's gate_proj from the checkpoint.
3. moe-prep-in-dense: experts prepared by prepare_moe_nvfp4_layer_for_marlin, then
   one expert's gate/up run through the dense kernel. Passing means the MoE
   preparation is right and the fault is in the MoE kernel or its call.
4. moe-synthetic / moe-real: the full fused_marlin_moe expert, several M.
5. server-path: the patched scheme's _process_nvfp4_moe_for_marlin on the real
   layer, exactly as the server ran it.

A case that raises is recorded with its traceback; later cases still run.
"""
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DEVICE = "cuda"
DTYPE = torch.bfloat16
RESULTS: list[dict] = []


def dequant(packed, block_scale, global_scale):
    """compressed-tensors NVFP4 [N, K/2] -> float32 [N, K]."""
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=packed.device)
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    return lut[nibbles.long()] * block_scale.float().repeat_interleave(16, dim=-1) / global_scale.float()


def compare(name, got, want, **extra):
    got, want = got.float(), want.float()
    err = (got - want).abs().max().item()
    peak = want.abs().max().item()
    row = {
        "case": name,
        "ok": bool(err / max(peak, 1e-6) <= 5e-2),
        "rel_err": err / max(peak, 1e-6),
        "max_abs_ref": peak,
        "max_abs_got": got.abs().max().item(),
        "got_finite": bool(torch.isfinite(got).all().item()),
        "zero_rows": int((got.abs().amax(dim=-1) == 0).sum().item()),
        "rows": got.shape[0],
        # Best scalar fit got ~ a * want: a far from 1 means a scale error.
        "fit_scale": ((got * want).sum() / (want * want).sum().clamp_min(1e-30)).item(),
        **extra,
    }
    RESULTS.append(row)
    print(json.dumps(row), flush=True)


def run_case(name, fn):
    start = time.time()
    try:
        fn()
    except Exception:
        RESULTS.append({"case": name, "ok": False, "error": traceback.format_exc()[-3000:]})
        print(f"{name}: EXCEPTION\n{traceback.format_exc()}", flush=True)
    print(f"{name}: {time.time() - start:.1f}s", flush=True)


def synthetic(n, k, experts=None, seed=0):
    """Random packed weights, fp8 block scales in [1, 448], one global scale."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    shape = (n, k) if experts is None else (experts, n, k)
    packed = torch.randint(0, 256, (*shape[:-1], k // 2), generator=gen, device=DEVICE, dtype=torch.uint8)
    scales = (torch.rand((*shape[:-1], k // 16), generator=gen, device=DEVICE) * 447 + 1).to(torch.float8_e4m3fn)
    lead = () if experts is None else (experts,)
    global_scale = torch.full(lead or (1,), 2.0e4, device=DEVICE)
    return packed, scales, global_scale


def dense_case(name, packed, scales, global_scale, m=8):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_nvfp4_layer_for_marlin,
    )

    n, k = packed.shape[0], packed.shape[1] * 2
    ref_w = dequant(packed, scales, global_scale)
    layer = torch.nn.Module()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.params_dtype = DTYPE
    layer.weight = torch.nn.Parameter(packed.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales.clone(), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        (1 / global_scale.float()).reshape(1), requires_grad=False
    )
    prepare_nvfp4_layer_for_marlin(layer)
    x = torch.randn((m, k), device=DEVICE).to(DTYPE)
    got = apply_fp4_marlin_linear(
        x, layer.weight, layer.weight_scale, layer.weight_global_scale, layer.workspace, n, k
    )
    compare(name, got, x.float() @ ref_w.T, n=n, k=k, m=m)


def moe_layer(w13, s13, g13, w2, s2, g2):
    """A FusedMoE stand-in holding compressed-tensors tensors (gate before up)."""
    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=16)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.intermediate_size_per_partition = w13.shape[1] // 2
    layer.params_dtype = DTYPE
    for name, value in (
        ("w13_weight", w13), ("w13_weight_scale", s13), ("w13_weight_global_scale", g13),
        ("w2_weight", w2), ("w2_weight_scale", s2), ("w2_weight_global_scale", g2),
    ):
        setattr(layer, name, torch.nn.Parameter(value.clone(), requires_grad=False))
    return layer


def expert_reference(layer_tensors, e, x):
    w13, s13, g13, w2, s2, g2 = layer_tensors
    inter = w13.shape[1] // 2
    gate = dequant(w13[e, :inter], s13[e, :inter], g13[e, 0])
    up = dequant(w13[e, inter:], s13[e, inter:], g13[e, 1])
    down = dequant(w2[e], s2[e], g2[e])
    xf = x.float()
    return (torch.nn.functional.silu(xf @ gate.T) * (xf @ up.T)) @ down.T, torch.cat([gate, up])


def prepare_moe(layer):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

    layer.w13_weight_scale_2 = torch.nn.Parameter(
        (1 / layer.w13_weight_global_scale.data[:, 0]).contiguous(), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        (1 / layer.w2_weight_global_scale.data).contiguous(), requires_grad=False
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)


def moe_cases(prefix, tensors, experts, ms=(1, 8, 56)):
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils_fp4 import apply_fp4_marlin_linear

    layer = moe_layer(*tensors)
    prepare_moe(layer)
    num_experts, hidden = tensors[0].shape[0], tensors[3].shape[1]
    inter = tensors[0].shape[1] // 2
    RESULTS.append({
        "case": f"{prefix}-prepared",
        "w13_shape": list(layer.w13_weight.shape), "w2_shape": list(layer.w2_weight.shape),
        "w13_scale_shape": list(layer.w13_weight_scale.shape), "w13_scale_dtype": str(layer.w13_weight_scale.dtype),
        "w13_scale_2": layer.w13_weight_scale_2[:4].float().tolist(),
        "w2_scale_2": layer.w2_weight_scale_2[:4].float().tolist(),
        "w13_scale_2_finite": bool(torch.isfinite(layer.w13_weight_scale_2.float()).all().item()),
    })
    for e in experts:
        for m in ms:
            x = torch.randn((m, hidden), device=DEVICE).to(DTYPE)
            want, gate_up_ref = expert_reference(tensors, e, x)

            def dense_on_moe_prep():
                got = apply_fp4_marlin_linear(
                    x, layer.w13_weight[e], layer.w13_weight_scale[e],
                    layer.w13_weight_scale_2[e : e + 1], layer.workspace, 2 * inter, hidden,
                )
                compare(f"{prefix}-prep-in-dense", got, x.float() @ gate_up_ref.T, expert=e, m=m)

            def fused():
                ids = torch.full((m, 1), e, dtype=torch.int32, device=DEVICE)
                got = fused_marlin_moe(
                    hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight,
                    w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
                    gating_output=torch.zeros((m, num_experts), device=DEVICE),
                    topk_weights=torch.ones((m, 1), device=DEVICE), topk_ids=ids,
                    w1_global_scale=layer.w13_weight_scale_2, w2_global_scale=layer.w2_weight_scale_2,
                    num_bits=4,
                )
                compare(f"{prefix}-fused", got, want, expert=e, m=m)

            if m == ms[1]:
                run_case(f"{prefix}-prep-in-dense e{e} m{m}", dense_on_moe_prep)
            run_case(f"{prefix}-fused e{e} m{m}", fused)


def real_layer(model_dir):
    """Layer 0's routed experts from the checkpoint, stacked as the loader would."""
    from safetensors import safe_open

    index = json.loads(Path(model_dir, "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.0.mlp.experts."
    names = [n for n in index if n.startswith(prefix)]
    num_experts = 1 + max(int(n[len(prefix):].split(".")[0]) for n in names)
    by_file = {}
    for name in names:
        by_file.setdefault(index[name], []).append(name)
    raw = {}
    for file_name, file_names in by_file.items():
        with safe_open(str(Path(model_dir, file_name)), framework="pt", device="cpu") as f:
            for name in file_names:
                raw[name[len(prefix):]] = f.get_tensor(name)

    def stack(proj, field):
        return torch.stack([raw[f"{e}.{proj}.{field}"] for e in range(num_experts)]).to(DEVICE)

    w13 = torch.cat([stack("gate_proj", "weight_packed"), stack("up_proj", "weight_packed")], dim=1)
    s13 = torch.cat([stack("gate_proj", "weight_scale"), stack("up_proj", "weight_scale")], dim=1)
    g13 = torch.cat([stack("gate_proj", "weight_global_scale").reshape(-1, 1),
                     stack("up_proj", "weight_global_scale").reshape(-1, 1)], dim=1).float()
    w2 = stack("down_proj", "weight_packed")
    s2 = stack("down_proj", "weight_scale")
    g2 = stack("down_proj", "weight_global_scale").reshape(-1).float()
    sample = raw["0.gate_proj.weight_packed"], raw["0.gate_proj.weight_scale"], raw["0.gate_proj.weight_global_scale"]
    RESULTS.append({
        "case": "real-layer-stats",
        "num_experts": num_experts, "files": sorted(by_file),
        "dtypes": [str(t.dtype) for t in sample], "shapes": [list(t.shape) for t in sample],
        "w13_shape": list(w13.shape), "w2_shape": list(w2.shape),
        "gate_global_min_max": [g13[:, 0].min().item(), g13[:, 0].max().item()],
        "down_global_min_max": [g2.min().item(), g2.max().item()],
        "gate_up_global_equal": bool(torch.equal(g13[:, 0], g13[:, 1])),
        "block_scale_min_max": [s13.float().min().item(), s13.float().max().item()],
    })
    return (w13, s13, g13, w2, s2, g2)


def server_path(tensors):
    """The patched scheme's own conversion + self-check, as the server ran it."""
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a4_nvfp4_moe as scheme,
    )

    scheme._marlin_checked = False
    scheme._process_nvfp4_moe_for_marlin(moe_layer(*tensors))
    RESULTS.append({"case": "server-path", "ok": True})


def main():
    model_dir, out = sys.argv[1], Path(sys.argv[2])
    torch.manual_seed(0)
    RESULTS.append({
        "case": "env", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    })
    try:
        run_case("dense-synthetic", lambda: dense_case("dense-synthetic", *synthetic(640, 2560)))
        tensors = {}
        run_case("load-real", lambda: tensors.update(real=real_layer(model_dir)))
        if "real" in tensors:
            w13, s13, g13 = tensors["real"][:3]
            inter = w13.shape[1] // 2
            run_case("dense-real", lambda: dense_case("dense-real", w13[0, :inter], s13[0, :inter], g13[0, 0]))
        gen = synthetic(2 * 640, 2560, experts=8, seed=1)
        down = synthetic(2560, 640, experts=8, seed=2)
        synthetic_tensors = (gen[0], gen[1], torch.stack([gen[2], gen[2]], dim=1), down[0], down[1], down[2])
        run_case("moe-synthetic", lambda: moe_cases("moe-synthetic", synthetic_tensors, experts=(0, 7)))
        if "real" in tensors:
            last = tensors["real"][0].shape[0] - 1
            run_case("moe-real", lambda: moe_cases("moe-real", tensors["real"], experts=(0, 1, last)))
            run_case("server-path", lambda: server_path(tensors["real"]))
    finally:
        out.write_text(json.dumps(RESULTS, indent=1))
        print(f"wrote {out}: {sum(1 for r in RESULTS if r.get('ok'))} ok, "
              f"{sum(1 for r in RESULTS if r.get('ok') is False)} failed", flush=True)


if __name__ == "__main__":
    main()
