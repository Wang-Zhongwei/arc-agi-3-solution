"""arm/stall-fix-death through the real game loop.

tests/anim_loop_sim.py (the real solver session, ToolAgent.analyze(), the
python sandbox; a fake SGLang rendering every request through the Qwen3.8
chat template) under mem-unified-10x's env with the object and rule scene.
Its game has no budget bar, its clicks blink only the clicked cell (the game
marking the click, not an undone move) and DOWN dies on contact, so with
ARC3_DEATH_VERDICT and ARC3_UNDONE_FACT on there is nothing to add: every
request must be byte-identical to the flags-off run, the GAME OVER message
keeping its BAR RULE sentence. (Flags off against the base branch is
test_mem_unified_loop.py's golden check.)
"""
from __future__ import annotations

import json
from pathlib import Path

from mem_unified_golden import SCRIPT
from memory_v2_golden import fingerprint
from test_anim_loop import TIGHT, run_sim, texts

HERE = Path(__file__).resolve().parent
ARMS = json.loads((HERE.parents[1] / "kaggle" / "arms.json").read_text())["arms"]
ENV = {**ARMS["mem-unified-10x"]["env"], **SCRIPT, **TIGHT}


def test_flags_on_with_nothing_to_say_send_the_same_requests(tmp_path):
    off = run_sim(tmp_path, "off", ENV)
    on = run_sim(tmp_path, "on", {**ENV, "ARC3_DEATH_VERDICT": "1", "ARC3_UNDONE_FACT": "1"})
    assert fingerprint(on) == fingerprint(off)
    game_over = [t for r in on["requests"] for t in texts(r["messages"][-1]) if "GAME OVER occurred" in t]
    assert game_over and all("Apply the BAR RULE above to decide" in t for t in game_over)
