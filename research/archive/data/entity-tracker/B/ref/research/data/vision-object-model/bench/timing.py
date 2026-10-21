"""Per-frame CPU latency of each candidate on the same 50 frames (2 per game), 1 thread unless THREADS set.
Usage: THREADS=1 python timing.py baseline lexicon tiny tiny_cnn mobilesam"""
import importlib, os, sys, time, resource
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
th = os.environ.get("THREADS", "1")
os.environ["SAM_THREADS"] = th; os.environ["TINY_THREADS"] = th; os.environ["LEX_MODE"] = "cc"
DATA = HERE.parent / "gt" / "data"
frames = []
for p in sorted(DATA.glob("*.npz")):
    z = np.load(p)["frames"]; frames += [(p.stem, z[5].astype(np.int64)), (p.stem, z[30].astype(np.int64))]
for cand in sys.argv[1:]:
    if cand == "tiny_cnn": os.environ["TINY_CNN"] = "1"; mod = importlib.import_module("cand_tiny")
    else: mod = importlib.import_module("cand_" + cand)
    ts = []; st = None
    for i, (g, fr) in enumerate(frames):
        if hasattr(mod, "setup"): mod.setup(g)
        t0 = time.perf_counter(); _, _, _, st = mod.segment(fr, state=None if i % 2 == 0 else st); dt = time.perf_counter() - t0
        ts.append(mod.TIMED if getattr(mod, "TIMED", None) is not None else dt)
    ts = np.array(ts[3:]) * 1000
    print(f"{cand} threads={th}: median {np.median(ts):.1f} ms, p95 {np.percentile(ts,95):.1f} ms, max {ts.max():.1f} ms; maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.0f} MB; load {os.getloadavg()[0]:.0f}", flush=True)
