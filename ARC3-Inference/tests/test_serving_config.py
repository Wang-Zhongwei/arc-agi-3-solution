"""A config's settings reach the harness the same way whatever `kaggle.serving` is.

new-baseline v1 lost its analyzer and multimodal sections because Pennyroyal
forwarded only harness_env: no board images and a 32K context window. These
tests run each serving option's real handoff (tests/serving_handoff.py) and
check that what a config says is what the harness receives.
"""

import glob
import json
import re
from pathlib import Path

import pytest

from inference.framework import kaggle
from serving_handoff import ROOT, handoff, make_exports

KAGGLE_CONFIGS = sorted(
    path for path in glob.glob(str(ROOT / "configs/**/*.json"), recursive=True)
    if "kaggle" in (json.loads(Path(path).read_text()) or {})
)
ALL_CONFIGS = sorted(
    path for path in glob.glob(str(ROOT / "configs/**/*.json"), recursive=True)
    if isinstance(json.loads(Path(path).read_text()), dict)
)
HANDED = [key for key in kaggle.CONFIG_FIELDS if key != "LOCAL_ANALYZER_TIMEOUT"]
SERVINGS = {
    "vllm": {},
    "pennyroyal": {
        "serving": "pennyroyal", "wheelhouse_dataset_source": "owner/wheels",
        "model_source": "owner/model/Transformers/default/1",
        "sglang": {"CTX": 155648, "MAXREQ": 16},
    },
    "lordhansolo": json.loads((ROOT / "configs/lordhansolo/v11.json").read_text())["kaggle"],
}
# Distinct from every default (Make's, the agent's, and each launcher's own).
TYPED = {
    "LOCAL_ANALYZER_APP_NAME": "Test Harness",
    "LOCAL_ANALYZER_SITE_URL": "https://example.test",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "100000",
    "LOCAL_ANALYZER_MAX_OUTPUT": "9000",
    "LOCAL_ANALYZER_THINKING_TOKEN_BUDGET": "4096",
    "LOCAL_ANALYZER_TIMEOUT": "77",
    "LOCAL_ANALYZER_TEMPERATURE": "0.33",
    "LOCAL_ANALYZER_TOP_P": "0.81",
    "LOCAL_ANALYZER_TOP_K": "17",
    "LOCAL_ANALYZER_TOOL_STEPS": "5",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "21",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "2500",
    "LOCAL_ANALYZER_YIELD_SECONDS": "7",
    "LOCAL_ANALYZER_ENABLE_THINKING": "false",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "7",
}
HARNESS = {"ARC3_SCHEDULER": "priority", "LOCAL_ANALYZER_YIELD_TOKENS": 2048, "ARC3_ACTION_ECHO": True}


def config_value(data, field):
    """A config field as Make's config_value.py prints it, or None if unset."""
    for part in field.split("."):
        if not isinstance(data, dict) or data.get(part) is None:
            return None
        data = data[part]
    if isinstance(data, bool):
        return "true" if data else "false"
    return str(data)


@pytest.fixture
def configure(monkeypatch):
    def apply(serving, typed=TYPED, harness=HARNESS):
        for key in kaggle.CONFIG_FIELDS:
            monkeypatch.delenv(key, raising=False)
        for key, value in typed.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "vllm")
        monkeypatch.setenv("ARC3_KAGGLE_JSON", json.dumps(SERVINGS[serving]))
        monkeypatch.setenv("ARC3_HARNESS_ENV_JSON", json.dumps(harness))
    return apply


def test_config_fields_match_the_makefile():
    makefile = (ROOT / "Makefile").read_text()
    read = dict(re.findall(r"^([A-Z_]+) :?= \$\(shell \$\(CONFIG_VALUE\) ([a-z_.]+)", makefile, re.M))
    exported = set(re.findall(r"^export ([A-Z_]+)$", makefile, re.M))
    endpoint = {"LOCAL_ANALYZER_BASE_URL", "LOCAL_ANALYZER_PROVIDER", "LOCAL_ANALYZER_MODEL_ID"}
    from_config = {
        name for name in read
        if name.startswith(("LOCAL_ANALYZER_", "MULTIMODAL_")) and name not in endpoint
    }
    assert from_config == set(kaggle.CONFIG_FIELDS)
    for name, field in kaggle.CONFIG_FIELDS.items():
        assert name in exported, name
        if name == "LOCAL_ANALYZER_CONTEXT_WINDOW":
            # analyzer.context_window, falling back to shared.context_window
            assert read[name] == "analyzer.context_window"
            assert read["SHARED_CONTEXT_WINDOW"] == field
        else:
            assert read[name] == field


@pytest.mark.parametrize("key", sorted(kaggle.CONFIG_FIELDS))
def test_harness_env_rejects_a_setting_that_has_a_config_field(monkeypatch, key):
    monkeypatch.setenv("ARC3_HARNESS_ENV_JSON", json.dumps({key: "1", "ARC3_SCHEDULER": "priority"}))
    with pytest.raises(ValueError, match=re.escape(f"{key} -> {kaggle.CONFIG_FIELDS[key]}")):
        kaggle.harness_env()
    with pytest.raises(ValueError):
        kaggle.apply_harness_env()


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: str(Path(p).relative_to(ROOT)))
def test_no_config_repeats_a_typed_setting_in_harness_env(path):
    env = json.loads(Path(path).read_text()).get("harness_env") or {}
    assert not set(env) & set(kaggle.CONFIG_FIELDS)


@pytest.mark.parametrize("serving", SERVINGS)
def test_every_serving_hands_the_harness_the_config_settings(configure, tmp_path, serving):
    configure(serving)
    namespace = {}
    handed = handoff(tmp_path, namespace=namespace)
    for key in HANDED:
        assert handed[key] == TYPED[key], key
    # kaggle-duck sets the request timeout through the solver instead.
    assert "LOCAL_ANALYZER_TIMEOUT" not in handed
    assert handed["ARC3_SCHEDULER"] == "priority"
    assert handed["LOCAL_ANALYZER_YIELD_TOKENS"] == "2048"
    assert handed["ARC3_ACTION_ECHO"] == "1"
    assert handed["LOCAL_ANALYZER_PROVIDER"] == "vllm"
    assert handed["LOCAL_ANALYZER_BASE_URL"].startswith("http://127.0.0.1:")
    assert handed["LOCAL_ANALYZER_MODEL_ID"] == handed["INFERENCE_ANALYZER_MODEL"]
    if serving == "lordhansolo":
        # the server's image cap follows the same upscale the harness renders at
        assert namespace["VLLM_IMAGE_MAX_PIXELS"] == (64 * 7) ** 2
        assert "LOCAL_ANALYZER_REASONING_EFFORT" not in handed
        assert handed["ARC3_KAGGLE_PRIVATE_WAVES"] == "4"


def test_settings_do_not_depend_on_the_serving(configure, tmp_path):
    handed = {}
    for serving in SERVINGS:
        configure(serving)
        result = handoff(tmp_path / serving)
        handed[serving] = {key: result.get(key) for key in [*kaggle.CONFIG_FIELDS, *HARNESS]}
    assert handed["vllm"] == handed["pennyroyal"] == handed["lordhansolo"]


def test_missing_settings_fall_back_to_make_defaults_not_the_launchers(configure, tmp_path):
    # Under Make every typed setting is exported, so a launcher's own fallback
    # (Lord: temperature 1.0, yield 120 s) never reaches the harness.
    configure("lordhansolo", typed={**TYPED, "LOCAL_ANALYZER_TEMPERATURE": "0.6",
                                    "LOCAL_ANALYZER_YIELD_SECONDS": "60"})
    handed = handoff(tmp_path)
    assert handed["LOCAL_ANALYZER_TEMPERATURE"] == "0.6"
    assert handed["LOCAL_ANALYZER_YIELD_SECONDS"] == "60"


@pytest.mark.parametrize("budget, strict", [("4096", True), ("0", False)])
def test_pennyroyal_strict_thinking_follows_the_typed_budget(configure, tmp_path, budget, strict):
    configure("pennyroyal", typed={**TYPED, "LOCAL_ANALYZER_THINKING_TOKEN_BUDGET": budget})
    namespace = {}
    handoff(tmp_path, namespace=namespace)
    launcher = (ROOT / "inference/framework/pennyroyal_launch.py").read_text()
    namespace["DRAFT_MODEL_DIR"] = "draft"
    exec(launcher[launcher.index("CFG = dict("):launcher.index("\nVENV = ")], namespace)
    assert namespace["CFG"]["STRICT_THINKING"] is strict
    assert namespace["CFG"]["CTX"] == 155648 and namespace["CFG"]["MAXREQ"] == 16


@pytest.mark.parametrize("path", KAGGLE_CONFIGS, ids=lambda p: str(Path(p).relative_to(ROOT)))
def test_each_kaggle_config_reaches_the_harness_as_written(tmp_path, path):
    """Real Make exports, real setup command for the config's serving."""
    data = json.loads(Path(path).read_text())
    keys = [*kaggle.CONFIG_FIELDS, "ARC3_KAGGLE_JSON", "ARC3_HARNESS_ENV_JSON", "LOCAL_ANALYZER_PROVIDER"]
    exports = make_exports(Path(path), keys)
    with pytest.MonkeyPatch.context() as monkeypatch:
        for key in keys:
            monkeypatch.delenv(key, raising=False)
        for key, value in exports.items():
            monkeypatch.setenv(key, value)
        handed = handoff(tmp_path)
    for key, field in kaggle.CONFIG_FIELDS.items():
        written = config_value(data, "analyzer.context_window") if key == "LOCAL_ANALYZER_CONTEXT_WINDOW" else None
        written = written or config_value(data, field)
        if written is not None:
            assert exports[key] == written, f"Make dropped {field}"
        if key in HANDED and exports.get(key, "").strip():
            assert handed[key] == exports[key], f"{data['kaggle'].get('serving', 'vllm')} dropped {field}"
    for key, value in (data.get("harness_env") or {}).items():
        expected = "1" if value is True else "0" if value is False else str(value)
        assert handed[key] == expected, key
