"""Shared action-name mapping between model-facing labels and engine actions."""
from __future__ import annotations

import os
from typing import Iterable


ENGINE_TO_MODEL_ACTION = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
    "RESET": "RESET",
}

MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}

# ARC3_ACTION_ALIASES: the single letters a search alphabet or the memory's
# route notation ("L3 U4") uses for the four arrows. Not S: one run used "S"
# for "select squares". Accepted as input only; nothing advertises them.
ARROW_LETTER_ALIASES = {"U": "ACTION1", "D": "ACTION2", "L": "ACTION3", "R": "ACTION4"}


def action_aliases_enabled() -> bool:
    """ARC3_ACTION_ALIASES, read at call time; off keeps every name as it was."""
    return os.environ.get("ARC3_ACTION_ALIASES", "").strip().lower() in {
        "on", "1", "true", "yes",
    }


def reset_exposed() -> bool:
    """Opt-in RESET advertising and snippet guards; off preserves legacy use."""
    return os.environ.get("EXPOSE_RESET", "off").strip().lower() in {
        "on", "1", "true", "yes",
    }


def undo_exposure_mode() -> str:
    """Read at call time; unset/invalid values preserve the original wiring.

    tufa: advertise ACTION7 verbatim but reject execution (legacy behavior).
    off: hide ACTION7/UNDO and reject execution.
    on: advertise UNDO and accept either name for engine ACTION7.
    """
    mode = os.environ.get("EXPOSE_UNDO", "tufa").strip().lower()
    return mode if mode in {"tufa", "off", "on"} else "tufa"


def to_model_action(name: str | None) -> str:
    raw = str(name or "").strip().upper()
    if raw in {"ACTION7", "UNDO"}:
        mode = undo_exposure_mode()
        if mode == "off":
            return ""
        if mode == "on":
            return "UNDO"
    return ENGINE_TO_MODEL_ACTION.get(raw, raw)


def to_engine_action(name: str | None) -> str | None:
    raw = str(name or "").strip().upper()
    if not raw:
        return None
    if raw in {"ACTION7", "UNDO"}:
        return "ACTION7" if undo_exposure_mode() == "on" else None
    if raw in ENGINE_TO_MODEL_ACTION:
        return raw
    if raw in ARROW_LETTER_ALIASES and action_aliases_enabled():
        return ARROW_LETTER_ALIASES[raw]
    return MODEL_TO_ENGINE_ACTION.get(raw)


def to_model_actions(names: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for name in names:
        label = to_model_action(name)
        if label and label not in resolved:
            resolved.append(label)
    return resolved
