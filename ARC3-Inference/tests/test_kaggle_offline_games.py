"""Public games use the competition's environment files, not the auto sentinel."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import arc_agi
import pytest
import taaf.game_api


NOTEBOOK = Path(__file__).resolve().parents[2] / "tufa-arc-agi-framework/src/taaf/kaggle/taaf_kaggle_run.ipynb"


def notebook_helpers():
    notebook = json.loads(NOTEBOOK.read_text())
    tree = ast.parse("".join(notebook["cells"][6]["source"]))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_offline_games", "_select_public_games"}]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_public_games_use_explicit_environment_path_and_requested_order(monkeypatch, tmp_path):
    env_dir = str(tmp_path / "environment_files")
    calls = []

    def arcade(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(available_environments=[
            SimpleNamespace(game_id="ar25-test"), SimpleNamespace(game_id="bp35-test"),
        ])

    monkeypatch.setattr(arc_agi, "Arcade", arcade)
    monkeypatch.setattr(taaf.game_api, "GameAPI", lambda **kwargs: SimpleNamespace(**kwargs))
    helpers = notebook_helpers()
    games = helpers["_offline_games"](env_dir)
    assert calls == [{"operation_mode": arc_agi.OperationMode.OFFLINE, "environments_dir": env_dir}]
    assert all(game.arcade_spec.environments_dir == env_dir for game in games)
    selected = helpers["_select_public_games"](games, ["bp35", "ar25"])
    assert [game.env_name for game in selected] == ["bp35-test", "ar25-test"]
    # An omitted public selection can retain just the games packaged by the launcher.
    selected = helpers["_select_public_games"](games, ["bp35-test"])
    assert [game.env_name for game in selected] == ["bp35-test"]


def test_missing_offline_environments_fail_before_play(monkeypatch, tmp_path):
    monkeypatch.setattr(arc_agi, "Arcade", lambda **kwargs: SimpleNamespace(available_environments=[]))
    with pytest.raises(RuntimeError, match="No offline environments found"):
        notebook_helpers()["_offline_games"](str(tmp_path))


def test_unknown_public_game_fails_instead_of_running_another_game():
    with pytest.raises(RuntimeError, match="matched no offline games"):
        notebook_helpers()["_select_public_games"]([SimpleNamespace(env_name="bp35-test")], ["ar25"])
