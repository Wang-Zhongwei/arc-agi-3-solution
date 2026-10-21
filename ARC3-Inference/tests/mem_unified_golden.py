"""The pf-keep-v2 golden for arm/mem-unified: what the base branch sends.

arm/mem-unified adds ARC3_MEM_RULES, ARC3_MEM_MOVED and ARC3_TOUCHING_SPLIT,
each off by default. With them off (unset or 0) every request must be
byte-identical to arm/pf-keep-v2 @ d624b0f9. This file runs
tests/anim_loop_sim.py (the real game loop through the Qwen3.8 chat template)
with the object and rule scene (ANIM_SIM_OBJECTS=1, so the facts the new flags
would add are there to add) under three envs and fingerprints every request
with memory_v2_golden.fingerprint:

- ``pf-keep``: pf-keep-10x, in the 30k and the 98k window;
- ``v2-all``: pf-keep-v2-all-10x, in the 30k, 45k and 98k windows;
- ``base``: pf-keep-10x + ARC3_MEM_APPEND_COMMIT=1 + ARC3_MEM_RECAP_CHARS=800,
  mem-unified-10x's env without its new flags, in the same three windows.

The fingerprints were written once, on the base commit, before any
mem-unified harness code existed:

    python tests/mem_unified_golden.py tests/fixtures/pf_keep_v2_golden.json

``test_mem_unified_loop.py`` compares this branch's runs against them.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from memory_v2_golden import fingerprint  # noqa: E402
from test_anim_loop import TIGHT, run_sim  # noqa: E402

REPO = HERE.parents[1]
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text())["arms"]
SCRIPT = {"ANIM_SIM_RESET": "1", "ANIM_SIM_MEMORY_TAGS": "1", "ANIM_SIM_RECAP_ASKED": "1",
          "ANIM_SIM_OBJECTS": "1"}
MID = {"LOCAL_ANALYZER_CONTEXT_WINDOW": "45000"}
# mem-unified-10x's env without ARC3_MEM_RULES, ARC3_MEM_MOVED and ARC3_TOUCHING_SPLIT
BASE_ENV = {**ARMS["pf-keep-10x"]["env"], "ARC3_MEM_APPEND_COMMIT": "1", "ARC3_MEM_RECAP_CHARS": "800"}
ENVS = {"pf-keep": ARMS["pf-keep-10x"]["env"], "v2-all": ARMS["pf-keep-v2-all-10x"]["env"], "base": BASE_ENV}
WINDOWS = {"tight": TIGHT, "mid": MID, "wide": {}}
CONFIGS = {
    f"{env}-{window}": {**ENVS[env], **SCRIPT, **WINDOWS[window]}
    for env in ENVS for window in WINDOWS
    if not (env == "pf-keep" and window == "mid")
}


def main(out: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pf-keep-v2-golden-"))
    result = {"base": "arm/pf-keep-v2 @ d624b0f9", "script": SCRIPT,
              "configs": {name: fingerprint(run_sim(tmp, name, env)) for name, env in CONFIGS.items()}}
    Path(out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
