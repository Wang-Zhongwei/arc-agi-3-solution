"""Loaders for the 5,717-step ground-truth replay dump (vision/bench/full + vision/gt/data)."""
import json
from pathlib import Path
import numpy as np

SCR = Path("/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad")
FULL = SCR / "vision/bench/full"
GT = SCR / "vision/gt"
ROLES = json.load(open(GT / "roles_behaviour.json"))
SPLIT = json.load(open(Path(__file__).resolve().parent / "split.json"))
DIRS = {"ACTION1": (0, -1), "ACTION2": (0, 1), "ACTION3": (-1, 0), "ACTION4": (1, 0)}  # (dx, dy)
HEX = "0123456789abcdef"


def load(g):
    z = np.load(FULL / f"{g}.npz")
    meta = json.load(open(FULL / f"{g}.meta.json"))
    traj = json.load(open(GT / "data" / f"{g}.traj.json"))
    return dict(frames=z["frames"].astype(np.int64), inst=z["inst"].astype(np.int64), ui=z["ui"], level=z["level"], meta=meta, traj=traj,
                ctrl={n for n, r, _, _ in ROLES[g] if "controlled(dir)" in r})


def show(f, r0=0, r1=64, c0=0, c1=64):
    return "\n".join("".join(HEX[v] if v >= 0 else "." for v in row[c0:c1]) for row in f[r0:r1])
