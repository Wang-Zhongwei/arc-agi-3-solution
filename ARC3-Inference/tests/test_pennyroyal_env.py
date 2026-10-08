"""Pennyroyal runs receive the config's analyzer and multimodal settings."""

import json
import shlex

from inference.framework import kaggle


def spec_env(monkeypatch, harness_env, **exported):
    monkeypatch.setenv("ARC3_KAGGLE_JSON", json.dumps({
        "serving": "pennyroyal",
        "wheelhouse_dataset_source": "owner/wheels",
        "model_source": "owner/model",
    }))
    monkeypatch.setenv("ARC3_HARNESS_ENV_JSON", json.dumps(harness_env))
    for key in kaggle._CONFIG_ANALYZER_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in exported.items():
        monkeypatch.setenv(key, value)
    command = kaggle.pennyroyal_setup_command()
    assignment = shlex.split(command)[0]
    return json.loads(assignment.removeprefix("ARC3_PENNYROYAL_SPEC="))["env"]


def test_analyzer_and_multimodal_sections_reach_the_harness(monkeypatch):
    env = spec_env(
        monkeypatch,
        {"ARC3_SCHEDULER": "priority"},
        MULTIMODAL_CONTEXT="current_grid",
        MULTIMODAL_UPSCALE="8",
        LOCAL_ANALYZER_CONTEXT_WINDOW="131072",
        LOCAL_ANALYZER_MAX_OUTPUT="12288",
        LOCAL_ANALYZER_TOOL_STEPS="0",
    )
    assert env["MULTIMODAL_CONTEXT"] == "current_grid"
    assert env["MULTIMODAL_UPSCALE"] == "8"
    assert env["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "131072"
    assert env["LOCAL_ANALYZER_MAX_OUTPUT"] == "12288"
    assert env["LOCAL_ANALYZER_TOOL_STEPS"] == "0"
    assert env["ARC3_SCHEDULER"] == "priority"


def test_harness_env_overrides_sections_and_empty_values_are_dropped(monkeypatch):
    env = spec_env(
        monkeypatch,
        {"MULTIMODAL_UPSCALE": 4},
        MULTIMODAL_CONTEXT="",
        MULTIMODAL_UPSCALE="16",
    )
    assert env["MULTIMODAL_UPSCALE"] == "4"
    assert "MULTIMODAL_CONTEXT" not in env
