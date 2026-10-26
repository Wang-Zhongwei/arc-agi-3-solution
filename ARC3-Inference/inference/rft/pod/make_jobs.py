"""Write the jobs file of a full run from the CUDA tests' outcome: a feature whose test failed on this GPU is
switched off in every job, so the run needs no decision after it starts.

    python make_jobs.py RESULTS_DIR JOBS_FILE [--rehearsal] [--ple-fp8] [--grpo]

--ple-fp8: only the PLE precision measurement (one sample, the training configuration): LoRA gradients with the
BF16 table twice (repeat noise), nvidia's FP8 table and a per-row FP8 rounding, then their agreement.

--grpo: GRPO's training shape. Every reply of an untrimmed trajectory is trained, so there is no prefix to skip:
the full backward at 64K (against 10-20's 22.6 s) and at 110K, the median trajectory under the 130K budget, in
layer mode (parts mode's ~2.0 MB of checkpoints per token do not fit the host arena there), plus a layer profile.

RESULTS_DIR holds pytest_<name>.exit files (0 = passed) for the test files run by full_run.sh."""

import json
import sys
from pathlib import Path

results, out = Path(sys.argv[1]), Path(sys.argv[2])
rehearsal = "--rehearsal" in sys.argv
ple_fp8 = "--ple-fp8" in sys.argv
grpo = "--grpo" in sys.argv


def passed(name: str) -> bool:
    f = results / f"pytest_{name}.exit"
    return f.exists() and f.read_text().strip() == "0"


flags = dict(qsa_fused=True, compile_hc=True, compile_moe=True, fast_conv=True, cache_experts=True, fast_lora=True,
             fast_gdn=True, compile_qsa=True)
off = []
if not passed("qsa_kernel"):
    off += ["qsa_fused", "compile_qsa"]
if not passed("compile_layer"):
    off += ["compile_hc", "compile_moe", "fast_conv", "fast_gdn"]
if not passed("fast_gdn"):  # fla with grouped value heads (tilelang backward on Hopper)
    off += ["fast_gdn"]
if not passed("compile_lora"):
    off += ["fast_lora"]
if not passed("fused_model"):
    off += ["compile_qsa"]
psg_ok = passed("prefix_cuda") and passed("prefix_cpu")
if psg_ok and not passed("prefix_cuda_fast") and not grpo:  # prefix stop-grad is worth more than fast_gdn
    off += ["fast_gdn"]
for name in set(off):
    flags[name] = False
base = dict(recompute="parts", async_copies=True, gpu_keep_gb="auto")
old = dict(base, qsa_fused=False, cache_experts=False)
opt = dict(base, **flags)
psg = dict(opt, prefix_stop_grad=True)
long_len, mid_len = ("32768o", "16384o") if rehearsal else ("65536o", "32768o")
layer_flags = {k: v for k, v in opt.items() if k not in base} | dict(token_chunk=2048)

if ple_fp8:
    one = dict(psg if psg_ok else opt, lengths=long_len, gpu_keep_gb=0)  # speed is not measured: nothing kept
    s = long_len
    jobs = [("01_bf16", dict(one, repeats=2, keep_grads="bf16")),
            ("02_nvidia", dict(one, repeats=1, keep_grads="nvidia", ple_fp8="nvidia")),
            ("03_row", dict(one, repeats=1, keep_grads="row", ple_fp8="row")),
            ("04_cmp", {"compare_grads": [[f"bf16/{s}/1", f"bf16/{s}/0"], [f"nvidia/{s}/0", f"bf16/{s}/1"],
                                          [f"row/{s}/0", f"bf16/{s}/1"], [f"nvidia/{s}/0", f"row/{s}/0"]]})]
    out.write_text("".join(f"{name} {json.dumps(job)}\n" for name, job in jobs))
    (results / "flags.json").write_text(json.dumps(dict(off=sorted(set(off)), prefix_stop_grad=psg_ok, opt=opt), indent=1))
    print("PLE FP8 measurement | features off:", sorted(set(off)) or "none", "| prefix stop-grad:", psg_ok)
    sys.exit(0)

check = ("01_check", {"layer_check": {"length": 8192, "layers": [0, 1, 3],
                                      "variants": {"reference": {"reference": True}, "opt": layer_flags}}})
if grpo:
    long = dict(opt, recompute="layer", transcript="game_ft09.txt")
    jobs = [check,
            ("02_opt64", dict(opt, lengths="65536o", repeats=3, timeline=True)),
            ("03_opt110", dict(long, lengths="110000o", repeats=3, timeline=True)),
            ("04_profile110", dict({"layer_profile": {"length": 110000, "layers": [0, 1, 3]}}, **long))]
    out.write_text("".join(f"{name} {json.dumps(job)}\n" for name, job in jobs))
    (results / "flags.json").write_text(json.dumps(dict(off=sorted(set(off)), opt=opt), indent=1))
    print("GRPO shape | features off:", sorted(set(off)) or "none", "|", len(jobs), "jobs")
    sys.exit(0)

jobs = [
    check,
    ("02_old", dict(old, lengths=long_len, repeats=2, timeline=True)),
    ("03_opt", dict(opt, lengths=long_len, repeats=3, timeline=True, keep_grads="opt")),
]
if psg_ok:
    jobs += [("04_psg", dict(psg, lengths=long_len, repeats=2, timeline=True, keep_grads="psg")),
             ("05_cmp", {"compare_grads": [[f"opt/{long_len}/2", f"opt/{long_len}/1"],
                                           [f"psg/{long_len}/1", f"opt/{long_len}/2"],
                                           [f"psg/{long_len}/1", f"psg/{long_len}/0"]]})]
jobs += [("06_mid", dict(opt, lengths=mid_len, repeats=2, timeline=True))]
if not rehearsal:
    jobs += [("07_long_opt", dict(opt, lengths="130000o", repeats=1, transcript="game_ft09.txt"))]
    if psg_ok:
        jobs += [("08_long_psg", dict(psg, lengths="130000o", repeats=1, transcript="game_ft09.txt"))]
jobs += [("09_profile", dict({"layer_profile": {"length": int(long_len[:-1]), "layers": [0, 1, 3]}}, **opt))]

out.write_text("".join(f"{name} {json.dumps(job)}\n" for name, job in jobs))
(results / "flags.json").write_text(json.dumps(dict(off=sorted(set(off)), prefix_stop_grad=psg_ok, opt=opt), indent=1))
print("features off:", sorted(set(off)) or "none", "| prefix stop-grad:", psg_ok, "|", len(jobs), "jobs")
