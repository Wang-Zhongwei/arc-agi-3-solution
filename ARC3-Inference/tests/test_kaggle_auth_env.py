"""Deploy subprocesses must preserve the CLI's access-token selection."""

import os
from types import SimpleNamespace

import pytest
import taaf.deploy_kaggle as deploy


@pytest.mark.parametrize("access_token", [None, "explicit-access-token"])
def test_auth_check_keeps_legacy_key_separate(monkeypatch, tmp_path, access_token):
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "previous-owner")
    monkeypatch.setenv("KAGGLE_KEY", "previous-key")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    if access_token is not None:
        monkeypatch.setenv("KAGGLE_API_TOKEN", access_token)
    # When no explicit token is set, the CLI must remain free to use this file.
    cached_token = tmp_path / "access_token"
    cached_token.write_text("cached-access-token")
    original_env = os.environ.copy()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        env = kwargs["env"]
        assert env["KAGGLE_USERNAME"] == "configured-owner"
        assert env["KAGGLE_KEY"] == "legacy-key"
        assert env["KAGGLE_CONFIG_DIR"] == str(tmp_path)
        assert env.get("KAGGLE_API_TOKEN") == access_token
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(deploy.subprocess, "run", run)
    deploy._check_kaggle_auth("configured-owner", "legacy-key")

    assert calls == [["kaggle", "kernels", "list", "--mine", "--page-size", "1"]]
    assert os.environ == original_env
    assert cached_token.read_text() == "cached-access-token"
