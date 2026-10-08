"""Recover missing comparison metrics from downloaded serving logs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from inference.tools.serving_logs import recover_server_block


def serving_metrics(run_dir: Path, recorded: dict[str, Any] | None) -> dict[str, Any]:
    """Keep populated evaluation metrics; recover absent metrics from local logs."""
    log_dirs = [run_dir, *sorted(run_dir.glob("passes/*")), *sorted(run_dir.glob("seeds/*"))]
    return recover_server_block(log_dirs, recorded)
