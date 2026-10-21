"""The launcher's startup deadline: a competition rerun releases the benchmark at
SERVER_STARTUP_TIMEOUT; an offline test run waits for the server up to
OFFLINE_STARTUP_TIMEOUT, so a slow weight load no longer eats into every game's clock."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "serving" / "kaggle" / "launch_sglang.py"


def _timeout(**names) -> float:
    """Run the launcher's two deadline assignments with the given notebook globals."""
    text = LAUNCHER.read_text(encoding="utf-8")
    block = re.search(r"^OFFLINE_STARTUP_TIMEOUT = .*?^\)\n", text, re.M | re.S)
    assert block, "deadline block not found in launch_sglang.py"
    namespace = {"SERVER_STARTUP_TIMEOUT": 12 * 60, **names}
    exec(block.group(0), namespace)
    return namespace["_startup_timeout"]


def test_rerun_keeps_the_12_minute_release():
    assert _timeout(TRUE_SUBMISSION=True) == 12 * 60


def test_offline_run_waits_up_to_45_minutes():
    assert _timeout(TRUE_SUBMISSION=False) == 45 * 60


def test_missing_flag_defaults_to_the_rerun_rule():
    assert _timeout() == 12 * 60


def test_offline_never_waits_less_than_the_rerun_deadline():
    assert _timeout(TRUE_SUBMISSION=False, SERVER_STARTUP_TIMEOUT=60 * 60) == 60 * 60


@pytest.mark.parametrize("needle", ["while time.time() - NOTEBOOK_START_TIME < _startup_timeout:"])
def test_the_wait_loop_uses_the_new_deadline(needle):
    assert needle in LAUNCHER.read_text(encoding="utf-8")
