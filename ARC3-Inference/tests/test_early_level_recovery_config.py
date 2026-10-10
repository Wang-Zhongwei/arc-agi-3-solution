"""The A/B treatment survives the real Make and serving handoff."""

import json
from pathlib import Path

import pytest

from inference.framework.kaggle import CONFIG_FIELDS
from serving_handoff import ROOT, handoff, make_exports


CONFIGS = [ROOT / "configs" / name / "v1.json" for name in (
    "early-level-recovery-baseline", "early-level-recovery",
)]


def test_pair_has_only_one_operational_difference():
    baseline, treatment = [json.loads(path.read_text()) for path in CONFIGS]
    for key in ("kernel_slug", "kernel_title", "dataset_slug"):
        baseline["kaggle"].pop(key)
        treatment["kaggle"].pop(key)
    assert baseline["harness_env"].pop("ARC3_EARLY_LEVEL_RECOVERY_TOKENS") == "0"
    assert treatment["harness_env"].pop("ARC3_EARLY_LEVEL_RECOVERY_TOKENS") == "32768"
    assert baseline == treatment


@pytest.mark.parametrize("config_path,threshold", zip(CONFIGS, ["0", "32768"]))
def test_actual_serving_handoff(tmp_path, monkeypatch, config_path: Path, threshold):
    exports = make_exports(
        config_path, ["ARC3_KAGGLE_JSON", "ARC3_HARNESS_ENV_JSON", *CONFIG_FIELDS],
    )
    for key, value in exports.items():
        monkeypatch.setenv(key, value)
    namespace = {}
    effective = handoff(tmp_path, namespace=namespace)
    assert effective["ARC3_EARLY_LEVEL_RECOVERY_TOKENS"] == threshold
    assert str(effective["ARC3_MAX_ACTIVE_STREAMS"]) == "14"
    assert effective["ARC3_ANIMATION_IMAGE"] == "off"
    assert str(effective["MULTIMODAL_INTERMEDIATE_FRAMES"]).lower() == "false"
    assert str(effective["MULTIMODAL_UPSCALE"]) == "10"
    assert str(effective["LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS"]) == "3072"
    assert str(effective["LOCAL_ANALYZER_YIELD_TOKENS"]) == "2048"
    assert str(effective["LOCAL_ANALYZER_THINKING_TOKEN_BUDGET"]) == "8192"
    assert str(effective["LOCAL_ANALYZER_MAX_OUTPUT"]) == "12288"
    assert str(effective["LOCAL_ANALYZER_CONTEXT_WINDOW"]) == "131072"
    config = json.loads(config_path.read_text())
    assert config["kaggle"]["vllm"]["max_num_seqs"] == 14
    assert len(set(config["kaggle"]["public_games"])) == 15
    assert config["kaggle"]["draft_model_source"].startswith("lordhansolo/")
    assert config["kaggle"]["make_share_version"] is False
