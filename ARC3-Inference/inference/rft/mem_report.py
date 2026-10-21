"""Live CUDA blocks of a torch.cuda.memory snapshot, grouped by the allocating code (our code, transformers, fla).

    python -m inference.rft.mem_report results/mem_psg16_16384o_0_fwd.pickle [more.pickle ...]

The snapshots come from a smoke_lora job with "memory_snapshot": "results/mem_<label>" (dumped at the end of
the forward and at an out-of-memory error). Blocks without a Python frame of ours are the model's weights."""
import pickle, sys
from collections import defaultdict

def key(frames):
    picked = []
    for f in frames:
        fn = f["filename"]
        if any(s in fn for s in ("inference/rft/", "transformers/models/", "/fla/", "smoke_lora")):
            picked.append(f"{fn.split('/')[-1]}:{f['line']} {f['name']}")
        if len(picked) == 3:
            break
    return " <- ".join(picked) or (frames[0]["filename"].split("/")[-1] + ":" + str(frames[0]["line"]) if frames else "?")

def live(path):
    snap = pickle.load(open(path, "rb"))
    groups = defaultdict(lambda: [0, 0])
    total = 0
    for seg in snap["segments"]:
        for b in seg["blocks"]:
            if b["state"] != "active_allocated":
                continue
            k = key(b.get("frames", []))
            groups[k][0] += b["size"]; groups[k][1] += 1
            total += b["size"]
    return total, groups

for path in sys.argv[1:]:
    total, groups = live(path)
    print(f"== {path}: {total / 2**30:.2f} GiB live")
    for k, (size, n) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:18]:
        print(f"  {size / 2**30:7.2f} GiB  {n:5d}  {k}")
