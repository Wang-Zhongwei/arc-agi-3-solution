"""Deterministic prediction check: compare the model's stated Prediction with the measured diff.

Before every ``action(...)`` call the model is asked to write a short labelled note
(``Hypothesis:`` / ``Test:`` / ``Prediction:`` / ``Next:``). After the actions execute,
the harness measures the actual board change and checks the prediction for the
contradictions it can detect without a model call:

* the prediction said "no change" but the board changed (or vice versa);
* the prediction named row/col ranges that do not overlap the changed region.

Everything else is reported as CONSISTENT (unverified details). Consecutive
contradictions are counted so the prompt can demand a revision of the action model.
"""
from __future__ import annotations

import re
from typing import Any

from inference.agent.exploration import describe_diff_summary, diff_summary

NOTE_LABELS = ["Hypothesis", "Test", "Prediction", "Next"]

_NO_CHANGE_RE = re.compile(
    r"\b(no (visible |board )?change|nothing (will |should )?(change|happen|move)s?|unchanged|no-?op|stays? the same|remains? (the same|unchanged))\b",
    re.IGNORECASE,
)
_ROWS_RE = re.compile(r"\brows?\s*(\d+)\s*(?:-|–|to|through)\s*(\d+)", re.IGNORECASE)
_COLS_RE = re.compile(r"\b(?:cols?|columns?)\s*(\d+)\s*(?:-|–|to|through)\s*(\d+)", re.IGNORECASE)


def _ranges(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in pattern.findall(text):
        lo, hi = sorted((int(a), int(b)))
        out.append((lo, hi))
    return out


def _overlaps(predicted: list[tuple[int, int]], actual: tuple[int, int], *, tolerance: int = 1) -> bool:
    lo, hi = actual
    return any(p_hi + tolerance >= lo and p_lo - tolerance <= hi for p_lo, p_hi in predicted)


def check_prediction(
    prediction: str,
    before_grid: tuple[tuple[int, ...], ...],
    after_grid: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    diff = diff_summary(before_grid, after_grid)
    changed = int(diff.get("changed_cells", 0) or 0) > 0
    text = " ".join(str(prediction or "").split())
    expects_no_change = bool(_NO_CHANGE_RE.search(text))
    reasons: list[str] = []
    if expects_no_change and changed:
        reasons.append(f"you predicted no change but {diff['changed_cells']} cells changed")
    elif not expects_no_change and not changed and text:
        reasons.append("you predicted a change but the board did not change")
    elif changed:
        rows = _ranges(_ROWS_RE, text)
        cols = _ranges(_COLS_RE, text)
        if rows and not _overlaps(rows, diff["row_range"]):
            reasons.append(
                f"you named rows {', '.join(f'{a}-{b}' for a, b in rows)} but the change was within rows {diff['row_range'][0]}-{diff['row_range'][1]}"
            )
        if cols and not _overlaps(cols, diff["col_range"]):
            reasons.append(
                f"you named cols {', '.join(f'{a}-{b}' for a, b in cols)} but the change was within cols {diff['col_range'][0]}-{diff['col_range'][1]}"
            )
    verdict = "CONTRADICTED" if reasons else "CONSISTENT"
    return {"verdict": verdict, "reasons": reasons, "diff": diff, "measured": describe_diff_summary(diff)}


def prediction_prompt_lines(
    *,
    prediction: str,
    hypothesis: str,
    executed_actions: list[str],
    check: dict[str, Any],
    stats: dict[str, int],
) -> list[str]:
    lines = [f"Prediction check (deterministic, measured after your last actions {', '.join(executed_actions) or '?'}):"]
    if hypothesis:
        lines.append(f"- Hypothesis under test: {hypothesis}")
    lines.append(f"- Your prediction: {prediction}")
    lines.append(f"- Measured: {check['measured']}.")
    if check["verdict"] == "CONTRADICTED":
        lines.append(f"- Verdict: CONTRADICTED ({'; '.join(check['reasons'])}).")
    else:
        lines.append("- Verdict: CONSISTENT with the measurement (details unverified; confirm the specifics in Python).")
    made = int(stats.get("made", 0)); contradicted = int(stats.get("contradicted", 0)); streak = int(stats.get("streak", 0))
    lines.append(f"- Predictions on this level: {made} made, {contradicted} contradicted" + (f" ({streak} in a row)." if streak else "."))
    if streak >= 2:
        lines.append(
            "- Two or more predictions in a row failed: some rule in your action model is wrong. Before acting again, "
            "state which rule failed, replace it, and write a new Prediction for the corrected rule."
        )
    return lines
