"""The run rules of kaggle/build_notebook.py: repeats (--replica) and gated arms (--gate-passed).

Dry builds only: nothing is pushed and kaggle/runs.json is not read.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_builder_rules.py
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bn = _load("build_notebook", REPO / "kaggle" / "build_notebook.py")
TEMPLATE = json.loads((REPO / "kaggle" / "template.ipynb").read_text(encoding="utf-8"))
PATCH = "diff --git a/ARC3-Inference/x b/ARC3-Inference/x\n"  # stands in for the harness diff
ARMS = json.loads((REPO / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]


def _notebook(arm, replica=1):
    """The notebook main() fingerprints for run ``replica`` of ``arm``."""
    config = bn._arm_config(arm)
    return bn.build_notebook(
        TEMPLATE, PATCH, env={**config["env"], **bn._replica_env(replica)}, serve=config["serve"],
        serve_env=config["serve_env"], runtime_patches=bn._bench_patches(config["runtime_patches"], arm),
        demo_games=config["demo_games"], demo_minutes=config["demo_minutes"],
    )


def _run(arm, fingerprint):
    return {"arm": arm, "version": 1, "fingerprint": fingerprint, "code": "c", "commit": "abc"}


def test_first_run_builds_as_before():
    assert bn._replica_env(1) == {}
    config = bn._arm_config("stack-v2")
    plain = bn.build_notebook(
        TEMPLATE, PATCH, env=config["env"], serve=config["serve"], serve_env=config["serve_env"],
        demo_games=config["demo_games"], demo_minutes=config["demo_minutes"],
    )
    assert bn._fingerprint(_notebook("stack-v2")) == bn._fingerprint(plain)


def test_a_replica_passes_the_duplicate_check_once():
    first, second, third = (bn._fingerprint(_notebook("stack-v2", n)) for n in (1, 2, 3))
    assert len({first, second, third}) == 3
    runs = [_run("stack-v2", first)]
    with pytest.raises(SystemExit, match="not building a duplicate"):
        bn._check_unique("stack-v2", first, "c", runs, new_batch=False)
    bn._check_unique("stack-v2", second, "c", runs, new_batch=False)
    runs.append(_run("stack-v2", second))
    with pytest.raises(SystemExit, match="not building a duplicate"):
        bn._check_unique("stack-v2", second, "c", runs, new_batch=False)
    bn._check_unique("stack-v2", third, "c", runs, new_batch=False)


def test_a_replica_differs_only_by_the_inert_env_var():
    first, second = _notebook("stack-v2"), _notebook("stack-v2", 2)
    changed = [(a, b) for a, b in zip(first["cells"], second["cells"]) if a != b]
    assert len(changed) == 1
    before, after = (bn._cell_text(cell).splitlines(keepends=True) for cell in changed[0])
    assert [line for line in after if line not in before] == [f"setup_env[{bn.REPLICA_ENV!r}] = '2'\n"]
    assert [line for line in before if line not in after] == []
    # nothing in the harness or the framework reads it
    for root in ("ARC3-Inference/inference", "tufa-arc-agi-framework"):
        for path in (REPO / root).rglob("*.py"):
            assert bn.REPLICA_ENV not in path.read_text(encoding="utf-8", errors="replace"), path


@pytest.mark.parametrize("replica", [0, -1])
def test_replica_must_be_positive(replica):
    with pytest.raises(SystemExit, match="--replica"):
        bn._replica_env(replica)


# ---------------------------------------------------------------- gated arms

def test_the_bench_gated_arms_carry_their_gate():
    gates = {arm: bn._arm_config(arm)["gate"] for arm in ARMS}
    assert {arm for arm, gate in gates.items() if gate} == {"tput-safe-lowm", "tput-accept05"}
    assert "C3" in gates["tput-safe-lowm"] and "C1" in gates["tput-accept05"]
    assert all(gates[arm].startswith("bench tput-gates:") for arm in ("tput-safe-lowm", "tput-accept05"))


def test_a_gated_arm_builds_freely_and_pushes_only_with_gate_passed():
    gate = bn._arm_config("tput-accept05")["gate"]
    bn._check_gate("tput-accept05", gate, push=False, gate_passed=False)
    with pytest.raises(SystemExit, match="--gate-passed"):
        bn._check_gate("tput-accept05", gate, push=True, gate_passed=False)
    bn._check_gate("tput-accept05", gate, push=True, gate_passed=True)


def test_gate_passed_on_an_ungated_arm_is_refused():
    bn._check_gate("tput-safe", None, push=True, gate_passed=False)
    with pytest.raises(SystemExit, match="has no gate"):
        bn._check_gate("tput-safe", None, push=True, gate_passed=True)
