"""The pf-keep-10x golden for memory v2 (arm/mem-v2): what the base branch sends.

Memory v2's flags are all off by default, and with them off every request must
be byte-identical to arm/pf-mem @ 1882bcb3 under pf-keep-10x's env. This file
runs tests/anim_loop_sim.py (the real game loop through the Qwen3.8 chat
template) under that env in two windows and fingerprints every request: the
sha256 of the system prompt, of the tool schema and of each request's
messages and rendered prompt. The fingerprints were written once, on the base
commit, before any memory-v2 code existed:

    python tests/memory_v2_golden.py tests/fixtures/pf_keep_golden.json

``test_memory_v2_loop.py`` compares this branch's runs against them.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_anim_loop import TIGHT, run_sim  # noqa: E402

REPO = HERE.parents[1]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text())["arms"]
BASE_ARM = "pf-keep-10x"
# the sim's model RESET, memory tags in three replies and the asked recap of level 1
SCRIPT = {"ANIM_SIM_RESET": "1", "ANIM_SIM_MEMORY_TAGS": "1", "ANIM_SIM_RECAP_ASKED": "1"}
CONFIGS = {
    # a 30k window: history is trimmed several times, so trims redraw <previous_levels>
    "tight": {**ARMS[BASE_ARM]["env"], **TIGHT, **SCRIPT},
    # the arm's own 98k window: no trim, so the clear is committed at a turn start
    "wide": {**ARMS[BASE_ARM]["env"], **SCRIPT},
}


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fingerprint(run: dict) -> dict:
    """Hashes of everything a request sends, in request order."""
    reqs = run["requests"]
    return {
        "system": sorted({_sha(r["messages"][0]) for r in reqs}),
        "tools": sorted({_sha(r["tools"]) for r in reqs}),
        "requests": [_sha({"messages": r["messages"], "prompt": r["prompt"]}) for r in reqs],
        "evictions": [r["evictions"] for r in reqs],
        "levels_completed": run["levels_completed"],
    }


def main(out: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pf-keep-golden-"))
    result = {"base": "arm/pf-mem @ 1882bcb3", "arm": BASE_ARM, "script": SCRIPT,
              "configs": {name: fingerprint(run_sim(tmp, name, env)) for name, env in CONFIGS.items()}}
    Path(out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
