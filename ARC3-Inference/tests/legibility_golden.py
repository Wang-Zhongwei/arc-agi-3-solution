"""What the model is sent under the pf-keep-10x env, as hashes, for the legibility arm.

tests/fixtures/legibility_pf_keep_golden.json was written by this script on
arm/legibility's first commit, whose ARC3-Inference tree is arm/pf-mem @ 1882bcb3
byte for byte (the commit only adds this script and the fixture):

    python tests/legibility_golden.py tests/fixtures/legibility_pf_keep_golden.json

It holds two scenarios, each under the Kaggle notebook's setup_env plus
pf-keep-10x's env from kaggle/arms.json:
- ``snapshot``: tests/flags_off_snapshot.py - the system prompt, three full turn
  openers (an animated no-op, a game over, an ordinary step) and the sandbox's
  globals;
- ``loop``: tests/anim_loop_sim.py through the real game loop (the solver
  session, ToolAgent.analyze(), the python sandbox, a fake SGLang that renders
  the chat template) with a 30k window, a model RESET, memory tags and a level
  recap, so it trims, dies, clears a level and draws ``<previous_levels>``.
  Every request's messages (tool outputs included), its tool schema and the
  rendered prompt are hashed.

tests/test_legibility.py compares the flags-off runs of later commits with it.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from notebook_env import notebook_env  # noqa: E402

ROOT = HERE.parent
ARMS = json.loads((ROOT.parent / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
BASE_ARM = "pf-keep-10x"
# as tests/test_memory_layout_loop.py runs the memory arms: trims, a model
# RESET, memory tags in three replies and a recap of level 1
LOOP = {"LOCAL_ANALYZER_CONTEXT_WINDOW": "30000", "ANIM_SIM_RESET": "1",
        "ANIM_SIM_MEMORY_TAGS": "1", "ANIM_SIM_DEBRIEF": "1"}
FIXTURE = HERE / "fixtures" / "legibility_pf_keep_golden.json"


def arm_env(**extra: str) -> dict[str, str]:
    """setup_env with pf-keep-10x's overrides, as build_notebook applies them."""
    return {**notebook_env(), **{k: str(v) for k, v in ARMS[BASE_ARM]["env"].items()}, **extra}


def loop_env(**extra: str) -> dict[str, str]:
    """pf-keep-10x's env plus the loop knobs (run_sim adds setup_env itself)."""
    return {**{k: str(v) for k, v in ARMS[BASE_ARM]["env"].items()}, **LOOP, **extra}


def sha(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot_hashes(snapshot: dict) -> dict:
    return {
        "system_prompt": sha(snapshot["system_prompt"]),
        "openers": {name: sha(message) for name, message in sorted(snapshot["openers"].items())},
        "sandbox": sha(snapshot["sandbox"]),
    }


def loop_hashes(run: dict) -> dict:
    requests = run["requests"]
    return {
        "request_count": len(requests),
        "evictions": [r["evictions"] for r in requests],
        "system": sorted({sha(r["messages"][0]) for r in requests}),
        "tools": sorted({sha(r["tools"]) for r in requests}),
        "messages": [sha(r["messages"]) for r in requests],
        "prompts": [sha(r["prompt"]) for r in requests],
    }


def collect(tmp: Path, root: Path = ROOT, **extra: str) -> dict:
    """Both scenarios' hashes for this tree, with ``extra`` added to the env."""
    from test_anim_loop import run_sim
    from test_flags_off_identity import _snapshot

    return {
        "snapshot": snapshot_hashes(_snapshot(arm_env(**extra), root)),
        "loop": loop_hashes(run_sim(tmp, "golden", loop_env(**extra), root)),
    }


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE
    with tempfile.TemporaryDirectory() as tmp:
        data = {"base_arm": BASE_ARM, "loop_env": LOOP, **collect(Path(tmp))}
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}: {data['loop']['request_count']} loop requests")
