"""Size the prefill-v2 levers in sim2 at matched wall-clock, with paired replicates.

Every setting runs the same replicate seeds (same level draws, arrivals and request
streams) as the control, 25 games on 10 slots for 121 min under the Franzen scheduler, so
the differences are paired. Settings are prefill_levers.csv levers (`--lever`), a drain
change, or both.

Prefill credit. sim2's decode curve is the rate requests actually got, so the stall that
other games' prefill chunks put on every decoding stream is baked in at the calibration
runs' prefill load. A lever that prefills fewer tokens frees that GPU time, which the
calibrated curve cannot see. Each setting is therefore run twice:
- `none`: the curve as calibrated (no credit, the lower bound);
- `full`: the curve scaled by 1 + dP / (T - P), where P is the control's prefill seconds
  (uncached tokens / 11.6k tok/s) and dP the setting's saving (negative when it prefills
  more). This assumes a prefill chunk stalls all decoding, the upper bound.

Usage (from analysis/; stdlib only):
    nice -n 15 python3 prefill_sizing.py --reps 300 --ledger off --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import sim2
from sim2 import SimConfig

PREFILL_TPS = 11_600.0
# name -> (lever column or None, drain tokens)
SETTINGS = {
    "control": (None, sim2.DEFAULT_DRAIN),
    "static": ("static", sim2.DEFAULT_DRAIN),
    "image": ("image", sim2.DEFAULT_DRAIN),
    "dedup": ("dedup", sim2.DEFAULT_DRAIN),
    "yield-4096": ("y4096", sim2.DEFAULT_DRAIN),
    "yield-6144": ("y6144", sim2.DEFAULT_DRAIN),
    "yield-8192": ("y8192", sim2.DEFAULT_DRAIN),
    "yield-12288": ("y12288", sim2.DEFAULT_DRAIN),
    "yield-120s": ("t120", sim2.DEFAULT_DRAIN),
    "yield-150s": ("t150", sim2.DEFAULT_DRAIN),
    "yield-6144+dedup": ("y6144+dedup", sim2.DEFAULT_DRAIN),
    "yield-120s+dedup": ("t120+dedup", sim2.DEFAULT_DRAIN),
    "drain-70k": (None, 48 * 1024),          # keeps 69,120 after a trim
    "drain-80k": (None, 38 * 1024),          # keeps 79,360 (Lord Han Solo: 81.5k)
    "drain-80k+dedup": ("dedup", 38 * 1024),
}
METRICS = ("score", "total_tokens", "levels", "uncached_tokens", "trims", "switch_outs",
           "requests", "lost_tokens")


def run_setting(calib, games, lever, drain, reps, seed, ledger, speed=1.0):
    out = []
    for rep in range(reps):
        cfg = SimConfig(games=games, policy="franzen", ledger=ledger, drain_tokens=drain,
                        lever=lever, speed_scale=speed)
        r = sim2.simulate(cfg, calib, seed * 100_003 + rep)
        out.append(r.summary())
    return out


def paired(rows, base, key):
    d = [r[key] - b[key] for r, b in zip(rows, base)]
    m = st.fmean(d)
    se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    return m, se


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ledger", choices=["on", "off", "all"], default="off",
                    help="level costs: off = no-ledger runs (the template control has no ledger)")
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS))
    ap.add_argument("--pooled-gaps", action="store_true",
                    help="one turn-start client-gap term (the phase-3 fit); default separates "
                         "fresh turns from yield resumptions (sim2 --resume-gaps)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    group = {"on": "ledger", "off": "noledger", "all": "all"}[a.ledger]
    calib = sim2.Calibration(resume_gaps=not a.pooled_gaps)
    games = calib.meta["runs"]["fz-arms-v4-control"]["dispatch_order"]
    T = 121 * 60.0
    results = {}
    base = run_setting(calib, games, None, sim2.DEFAULT_DRAIN, a.reps, a.seed, group)
    p_ctrl = st.fmean(r["uncached_tokens"] for r in base) / PREFILL_TPS
    for name in a.settings:
        lever, drain = SETTINGS[name]
        rows0 = base if name == "control" else run_setting(calib, games, lever, drain, a.reps, a.seed, group)
        dp_s = p_ctrl - st.fmean(r["uncached_tokens"] for r in rows0) / PREFILL_TPS
        speed = 1.0 + dp_s / (T - p_ctrl)
        rows1 = rows0 if name == "control" else run_setting(calib, games, lever, drain, a.reps, a.seed,
                                                            group, speed)
        res = {"lever": lever, "drain": drain, "credit_speed": speed,
               "mean": {k: st.fmean(r[k] for r in rows0) for k in METRICS}}
        for tag, rows in (("none", rows0), ("full", rows1)):
            res[tag] = {k: paired(rows, base, k) for k in METRICS}
        results[name] = res
    hdr = ("| setting | prefill M (diff) | trims (diff) | switch-outs | tokens M (diff, none / full credit) | "
           "tok/s (none / full) | levels (none / full) | score diff, no credit | score diff, full credit |")
    print(f"sim2, 25 games x 10 slots x 121 min, Franzen, {group} level costs, {a.reps} paired reps; "
          f"control prefill {st.fmean(r['uncached_tokens'] for r in base)/1e6:.2f}M "
          f"({p_ctrl:.0f} s at {PREFILL_TPS/1e3:.1f}k tok/s), tokens "
          f"{st.fmean(r['total_tokens'] for r in base)/1e6:.3f}M, score {st.fmean(r['score'] for r in base):.2f}, "
          f"{'pooled' if a.pooled_gaps else 'resume-aware'} client gaps; diff (se)")
    print(hdr)
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        n, f = r["none"], r["full"]
        print(f"| {name} | {r['mean']['uncached_tokens']/1e6:.2f} ({n['uncached_tokens'][0]/1e6:+.2f}) "
              f"| {r['mean']['trims']:.0f} ({n['trims'][0]:+.1f}) | {n['switch_outs'][0]:+.1f} "
              f"| {r['mean']['total_tokens']/1e6:.3f} ({n['total_tokens'][0]/1e3:+.0f}k / {f['total_tokens'][0]/1e3:+.0f}k) "
              f"| {n['total_tokens'][0]/T:+.1f} / {f['total_tokens'][0]/T:+.1f} "
              f"| {n['levels'][0]:+.2f} / {f['levels'][0]:+.2f} "
              f"| {n['score'][0]:+.2f} ({n['score'][1]:.2f}) | {f['score'][0]:+.2f} ({f['score'][1]:.2f}) |")
    if a.json:
        Path(a.json).write_text(json.dumps({"reps": a.reps, "ledger": group, "pooled_gaps": a.pooled_gaps,
                                                "results": results}, indent=1))


if __name__ == "__main__":
    main()
