"""Run the tracker over games outside the role benchmark (neither dev nor held-out): crash check + latency."""
import sys, json, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tracker as TR
import eval_tracker as E
games = [g for g in "ar25 bp35 cd82 ft09 lf52 lp85 r11l s5i5 sb26 su15 tn36 tr87 vc33".split()]
assert not set(games) & set(E.SPLIT["dev"] + E.SPLIT["heldout"])
allms = []
for g in games:
    z = np.load(E.FULL / f"{g}.npz"); meta = json.load(open(E.FULL / f"{g}.meta.json"))
    F, L = z["frames"].astype(np.int64), z["level"]
    tr = TR.EntityTracker(); ms = []; claims = 0; facts = 0
    try:
        for i in range(len(F)):
            t0 = time.perf_counter()
            tr.update(F[i], meta[i]["act"], meta[i]["click"], new_level=(i == 0 or L[i] != L[i - 1]))
            tr.tail_line(); m, c = tr.claim(); tr.selection()
            ms.append((time.perf_counter() - t0) * 1000)
            claims += c == "committed"; facts += len(tr.events)
        print(f"{g}: ok steps {len(F)} committed-claim steps {claims} events {facts} ms p50 {np.percentile(ms,50):.2f} p95 {np.percentile(ms,95):.2f} max {max(ms):.1f}")
    except Exception:
        print(g, "CRASH at step", i); traceback.print_exc()
    allms += ms
print(f"all: p50 {np.percentile(allms,50):.2f} p95 {np.percentile(allms,95):.2f} max {max(allms):.1f}")
