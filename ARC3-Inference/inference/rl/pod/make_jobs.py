"""Write the jobs file of a full run from the CUDA tests' outcome: a feature whose test failed on this GPU is
switched off in every job, so the run needs no decision after it starts.

    python make_jobs.py RESULTS_DIR JOBS_FILE [--rehearsal]

RESULTS_DIR holds pytest_<name>.exit files (0 = passed) for the test files run by full_run.sh."""

import json
import sys
from pathlib import Path

results, out = Path(sys.argv[1]), Path(sys.argv[2])
rehearsal = "--rehearsal" in sys.argv


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
if psg_ok and not passed("prefix_cuda_fast"):  # prefix stop-grad is worth more than fast_gdn
    off += ["fast_gdn"]
for name in set(off):
    flags[name] = False
base = dict(recompute="parts", async_copies=True, gpu_keep_gb="auto")
old = dict(base, qsa_fused=False, cache_experts=False)
opt = dict(base, **flags)
psg = dict(opt, prefix_stop_grad=True)
long_len, mid_len = ("32768o", "16384o") if rehearsal else ("65536o", "32768o")
layer_flags = {k: v for k, v in opt.items() if k not in base} | dict(token_chunk=2048)

jobs = [
    ("01_check", {"layer_check": {"length": 8192, "layers": [0, 1, 3],
                                  "variants": {"reference": {"reference": True}, "opt": layer_flags}}}),
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
