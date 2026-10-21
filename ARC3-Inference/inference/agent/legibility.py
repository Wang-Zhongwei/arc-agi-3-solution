"""ARC3_LEGIBILITY: harness text the model misread, restated as facts.

Each fix is its own default-off flag, so each can be tested and judged alone
(memstruct/readability/readability.md, pass 2, ranks them by measured harm).
The umbrella ARC3_LEGIBILITY=1 turns on BUNDLE: the six fixes whose harm was
measured as wasted tool rounds or wrong beliefs in every run (report items 3,
4, 6, 7, 8 and 9). The four smaller fixes (items 10, 14, 16 and 19: one to a
few short detours per run, no traced wrong action), the runs helper (not
verified in pass 2; optional structure Flash-Next tends to ignore) and the
held-out case pairs (item 17, unproven) are flag-only. A sub-flag set
explicitly wins over the umbrella in both directions: ARC3_LEGIBILITY=1
ARC3_RETAINED_SOURCE=0 runs the bundle without retained_source, and
ARC3_LEGIBILITY=1 ARC3_DIFF_SAME_CELLS=1 adds one flag-only fix.

Read at call time, not import time, so the solver, the agent and the sandbox
host all see the same value for the whole run. Standard library only.
"""
from __future__ import annotations

import os

UMBRELLA = "ARC3_LEGIBILITY"

SEG_BBOX = "ARC3_SEG_BBOX"
STALE_TRUTH = "ARC3_STALE_TRUTH"
FRAME_RUNS = "ARC3_FRAME_RUNS"
ANIM_SCHEMA = "ARC3_ANIM_SCHEMA"
TURN_BASED_FACT = "ARC3_TURN_BASED_FACT"
DIGEST_COORD_FACT = "ARC3_DIGEST_COORD_FACT"
DIFF_SAME_CELLS = "ARC3_DIFF_SAME_CELLS"
EDGE_NOOP_FACT = "ARC3_EDGE_NOOP_FACT"
ACTION_INDEX_FACT = "ARC3_ACTION_INDEX_FACT"
LEGEND_CASE_PAIRS = "ARC3_LEGEND_CASE_PAIRS"
RETAINED_SOURCE = "ARC3_RETAINED_SOURCE"
RESET_BAR_FACT = "ARC3_RESET_BAR_FACT"

# What ARC3_LEGIBILITY=1 turns on, in the report's rank order.
BUNDLE = (
    SEG_BBOX,          # 3: bbox key on segmentation nodes
    ANIM_SCHEMA,       # 4: a not-kept animation_for(n) tests false with every key
    RETAINED_SOURCE,   # 6: retained functions' source in the sandbox
    STALE_TRUTH,       # 7: a caught stale-state refusal still refuses later calls
    RESET_BAR_FACT,    # 8: RESET restores the step bar
    TURN_BASED_FACT,   # 9: the game is turn-based; no timer wording
)
# Flag-only, in rank order.
FLAG_ONLY = (
    DIFF_SAME_CELLS,    # 10: frame_diff entries whose cells did not change
    DIGEST_COORD_FACT,  # 14: what digest positions and sizes are
    EDGE_NOOP_FACT,     # 16: which border cells an edge-only action changed
    LEGEND_CASE_PAIRS,  # 17: held out (unproven)
    ACTION_INDEX_FACT,  # 19: game-wide action numbers in each turn message
    FRAME_RUNS,         # not verified in pass 2: frame.runs row helper
)
FLAGS = BUNDLE + FLAG_ONLY

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _explicit(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


def legibility_on(name: str) -> bool:
    """Whether one legibility fix is on: its own flag if set, else the umbrella."""
    explicit = _explicit(name)
    if explicit is not None:
        return explicit
    return name in BUNDLE and bool(_explicit(UMBRELLA))


def enabled_flags() -> frozenset[str]:
    """The legibility fixes that are on."""
    return frozenset(name for name in FLAGS if legibility_on(name))
