"""With every animation-digest flag unset, the model sees exactly what it did.

tests/fixtures/flags_off_snapshot.json was produced by tests/flags_off_snapshot.py
against the tree before the digest existed (arm/kaggle-builder, 4d63980):

    python tests/test_flags_off_identity.py <that tree's ARC3-Inference> > tests/fixtures/flags_off_snapshot.json

Each scenario runs in a fresh interpreter because some settings are read at
import time. The comparison covers the system prompt, three full turn
messages (images hashed) and the python sandbox's globals.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().parent / "flags_off_snapshot.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flags_off_snapshot.json"

_COMPETITION = {
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072",
    "ARC3_ACTION_ECHO": "1",
    "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1",
    "ARC3_NEW_CHANGED_PROMPTS": "1",
    "ARC3_LEVEL_TRANSFER_GUIDANCE": "1",
    "ARC3_ACTION_INFO": "1",
    "ARC3_DEDUPE_MULTICALL_LINE": "1",
    "ARC3_FRAME_DIFF_HINT": "1",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "10",
    "ARC3_DIFF_IMAGE": "1",
    "ARC3_GAMEOVER_DIFF_IMAGE": "1",
    "ARC3_ANIMATION": "1",
    "ARC3_GUARDS_FROM_LEVEL": "2",
    "ARC3_BATCH_NOOP_BLOCK": "1",
    "ARC3_STALE_STATE_BLOCK": "1",
    "ARC3_MEMORY_SECTIONS": "off",
    "ARC3_PERSISTENT_FUNCTIONS": "1",
    "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
    "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1",
    "EXPOSE_UNDO": "on",
}

SCENARIOS: dict[str, dict[str, str]] = {
    "default": {},
    "animation": {"ARC3_ANIMATION": "1"},
    "animation_images": {
        "ARC3_ANIMATION": "1",
        "MULTIMODAL_CONTEXT": "current_grid",
        "MULTIMODAL_UPSCALE": "4",
        "ARC3_ANIMATION_IMAGE": "peak",
        "ARC3_ANIMATION_DEATH_IMAGE": "composite",
        "ARC3_GAMEOVER_DIFF_IMAGE": "1",
        "ARC3_ANIMATION_TIMELINE": "0",
    },
    "competition": _COMPETITION,
}

# Knobs that only mean something under the animation layer flags: with
# every layer explicitly off (ARC3_ANIM_RING=0 is off) they must change
# nothing either.
DIGEST_OFF_EXTRAS = {
    "ARC3_ANIM_DIGEST": "0",
    "ARC3_ANIM_RING": "0",
    "ARC3_ANIM_DIGEST_GATE": "all",
    "ARC3_ANIM_DIGEST_CHARS": "5000",
    "ARC3_ANIM_KEYFRAMES": "4",
    "ARC3_ANIM_KEYFRAMES_UPSCALE": "2",
}


def _clean_env(extra: dict[str, str], root: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
    }
    env["PYTHONPATH"] = str(root)
    env["LOCAL_ANALYZER_MODEL_ID"] = "snapshot-model"
    env.update(extra)
    return env


def _snapshot(extra: dict[str, str], root: Path = ROOT) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(root),
        env=_clean_env(extra, root),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    imported_from = completed.stderr.strip().splitlines()[-1]
    assert imported_from.startswith(str(root)), imported_from
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_flags_off_matches_pre_digest_snapshot(name, golden):
    assert _snapshot(SCENARIOS[name]) == golden[name]


@pytest.mark.parametrize("name", ["animation", "competition"])
def test_digest_knobs_without_master_flag_change_nothing(name, golden):
    assert _snapshot({**SCENARIOS[name], **DIGEST_OFF_EXTRAS}) == golden[name]


def test_digest_without_animation_changes_nothing(golden):
    """ARC3_ANIM_DIGEST needs ARC3_ANIMATION; alone it is inert."""
    assert _snapshot({"ARC3_ANIM_DIGEST": "1", "ARC3_ANIM_RING": "8"}) == golden["default"]


def test_digest_flag_changes_the_system_prompt(golden):
    """Guards against the identity tests passing vacuously."""
    snapshot = _snapshot({**SCENARIOS["animation"], "ARC3_ANIM_DIGEST": "1", "ARC3_ANIM_RING": "8"})
    assert snapshot["system_prompt"] != golden["animation"]["system_prompt"]
    assert snapshot["system_prompt"].startswith(golden["animation"]["system_prompt"][:200])
    assert "Animation digest" in snapshot["system_prompt"]


# The prompt-facts flags (the prompt-facts-10x arm), explicitly off.
PROMPT_FACTS_OFF = {
    "ARC3_BUDGET_TRUTH": "0",
    "ARC3_PROMPT_HYGIENE": "0",
    "ARC3_IMAGE_GEOMETRY": "0",
    "ARC3_PREFER_TOOL_CALLS": "0",
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_prompt_facts_flags_off_change_nothing(name, golden):
    assert _snapshot({**SCENARIOS[name], **PROMPT_FACTS_OFF}) == golden[name]


def test_prompt_facts_flags_change_the_competition_snapshot(golden):
    """Guards against the identity test passing vacuously: on, the system
    prompt and the game-over opener change; the other openers do not."""
    snapshot = _snapshot({**SCENARIOS["competition"], **{k: "1" for k in PROMPT_FACTS_OFF}})
    off = golden["competition"]
    assert snapshot["system_prompt"] != off["system_prompt"]
    assert "The board image is 640x640 px" in snapshot["system_prompt"]  # MULTIMODAL_UPSCALE=10
    assert snapshot["openers"]["game_over"] != off["openers"]["game_over"]
    assert snapshot["openers"]["plain"] == off["openers"]["plain"]
    assert snapshot["sandbox"] == off["sandbox"]


if __name__ == "__main__":
    # regenerate the fixture from another tree: python this_file.py <ARC3-Inference root>
    source_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
    json.dump(
        {name: _snapshot(env, source_root) for name, env in SCENARIOS.items()},
        sys.stdout, indent=1, sort_keys=True, ensure_ascii=False,
    )
    sys.stdout.write("\n")
