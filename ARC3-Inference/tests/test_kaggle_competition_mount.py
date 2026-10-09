"""Run the rendered setup cell against both Kaggle competition mount layouts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import taaf.deploy_kaggle


def setup_cell(monkeypatch, tmp_path):
    wheelhouse = tmp_path / "input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
    monkeypatch.setattr(taaf.deploy_kaggle, "COMPETITION_WHEELHOUSE", str(wheelhouse))
    notebook = json.loads(taaf.deploy_kaggle._render_kaggle_notebook(run_as_submission=False))
    environment_line = next(
        line.strip() for cell in notebook["cells"] for line in cell.get("source", [])
        if "competition_env_files =" in line
    )
    return "".join(notebook["cells"][2]["source"]) + "\n" + environment_line, wheelhouse


@pytest.mark.parametrize("layout", ["namespaced", "flat"])
def test_competition_install_uses_existing_mount(monkeypatch, tmp_path, layout):
    code, namespaced = setup_cell(monkeypatch, tmp_path)
    mounted = namespaced if layout == "namespaced" else namespaced.parents[2] / namespaced.parent.name / namespaced.name
    mounted.mkdir(parents=True)
    calls = []
    namespace = {
        "Path": Path,
        "subprocess": SimpleNamespace(check_call=calls.append),
        "sys": SimpleNamespace(executable="python"),
        "os": SimpleNamespace(getenv=lambda name: None),
    }
    exec(code, namespace)
    assert namespace["wheelhouse"] == mounted
    assert calls[0][calls[0].index("--find-links") + 1] == str(mounted)
    assert namespace["competition_env_files"] == str(mounted.parent / "environment_files")


def test_missing_competition_mount_reports_both_checked_paths(monkeypatch, tmp_path):
    code, namespaced = setup_cell(monkeypatch, tmp_path)
    namespace = {"Path": Path, "os": SimpleNamespace(getenv=lambda name: None)}
    with pytest.raises(RuntimeError, match="Competition wheelhouse not found") as error:
        exec(code, namespace)
    assert str(namespaced) in str(error.value)
    assert str(namespaced.parents[2] / namespaced.parent.name / namespaced.name) in str(error.value)
