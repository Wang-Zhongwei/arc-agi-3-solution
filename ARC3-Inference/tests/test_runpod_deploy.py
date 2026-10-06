import json
import subprocess
from pathlib import Path

import pytest

import taaf.deploy_kaggle
import taaf.deploy_runpod
import taaf.runpod_pod as pod
from inference.framework import run


def test_output_dataset_ref_fits_kaggle_slug_limit():
    ref = taaf.deploy_runpod.output_dataset_ref("me", "a-very-long-kernel-slug-" * 4, "ABCDEF1234567X")
    owner, slug = ref.split("/")
    assert owner == "me"
    assert len(slug) <= 50
    assert slug.endswith("-rp-abcdef1234567x")
    assert "--" not in slug


def test_model_mount_path_matches_pennyroyal_lookup():
    ref = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
    assert pod.model_mount_path(ref) == Path(
        "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Intel/Model", ("Intel/Model", None, "")),
        ("Intel/Model@abc123", ("Intel/Model", "abc123", "")),
        ("org/repo@rev:runtime/mtp/", ("org/repo", "rev", "runtime/mtp")),
        ("org/repo:sub", ("org/repo", None, "sub")),
    ],
)
def test_parse_hf_source(source, expected):
    assert pod.parse_hf_source(source) == expected


def test_bundle_requirements_drop_bundled_projects_and_competition_wheels(tmp_path):
    for name, deps in {
        "ARC3-Inference": ['"arcengine==0.9.3"', '"tufa-arc-agi-framework"', '"requests==2.32.5"'],
        "tufa-arc-agi-framework": ['"arc_agi>=0.9.8"', '"arcengine>=0.9.3"', '"numpy>=2.2"'],
    }.items():
        project = tmp_path / "src" / name
        project.mkdir(parents=True)
        dist = "arc3-inference" if name == "ARC3-Inference" else name
        (project / "pyproject.toml").write_text(
            f'[project]\nname = "{dist}"\ndependencies = [{", ".join(deps)}]\n', encoding="utf-8"
        )
    assert pod.bundle_requirements(tmp_path) == [
        "arcengine==0.9.3",
        "requests==2.32.5",
        "arcengine>=0.9.3",
        "numpy>=2.2",
        "pandas",
        "pyarrow",
    ]


@pytest.mark.parametrize("share", [False, True])
def test_rendered_notebooks_run_as_plain_python(tmp_path, share):
    template = taaf.deploy_kaggle._SHARE_NOTEBOOK_TEMPLATE if share else taaf.deploy_kaggle._NOTEBOOK_TEMPLATE
    notebook = tmp_path / "nb.ipynb"
    notebook.write_text(
        taaf.deploy_kaggle._render_kaggle_notebook(
            run_as_submission=False, dataset_sources=["me/src"], enable_gpu=True, template=template
        ),
        encoding="utf-8",
    )
    code = pod.notebook_code(notebook)
    assert "await bm.run(" in code
    assert "__TAAF_" not in code


def test_pod_payload_requests_rtx_pro_6000_with_cuda13_and_bootstrap():
    target = taaf.deploy_runpod.RunPodTarget(min_download_mbps=1000)
    payload = target.pod_payload(name="taaf-smoke", env={"A": "1"})
    assert payload["gpuTypeIds"][0] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert payload["allowedCudaVersions"] == ["13.0"]
    assert payload["minRAMPerGPU"] == 160
    assert payload["minDownloadMbps"] == 1000.0
    assert payload["dockerEntrypoint"] == ["bash", "-c"]
    assert "runpod_pod.py" in payload["dockerStartCmd"][0]
    assert payload["env"] == {"A": "1"}
    assert "dataCenterIds" not in payload


def test_rest_requests_send_a_user_agent(monkeypatch):
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"id": "pod1"}'

    def fake_urlopen(request, timeout):
        seen.update(request.header_items())
        return _Response()

    monkeypatch.setattr(taaf.deploy_runpod.urllib.request, "urlopen", fake_urlopen)
    assert taaf.deploy_runpod.create_pod({}, api_key="k") == {"id": "pod1"}
    assert seen["User-agent"].startswith("taaf-runpod")
    assert seen["Authorization"] == "Bearer k"


def test_bootstrap_is_valid_bash():
    subprocess.run(["bash", "-n", "-c", taaf.deploy_runpod._BOOTSTRAP], check=True)


def test_hard_limit_defaults_to_budget_plus_margin():
    assert taaf.deploy_runpod.RunPodTarget(max_runtime_s=600)._hard_limit_s() == 600 + 90 * 60
    assert taaf.deploy_runpod.RunPodTarget(hard_limit_s=123)._hard_limit_s() == 123


def test_runpod_settings_reads_config_section(monkeypatch):
    monkeypatch.setenv("ARC3_RUNPOD_JSON", json.dumps({"hf_models": {"a/b/c/d/1": "x/y"}, "cloud_type": "COMMUNITY"}))
    settings = run._runpod_settings()
    target = taaf.deploy_runpod.RunPodTarget(**settings)
    assert target.cloud_type == "COMMUNITY"
    assert target.hf_models == {"a/b/c/d/1": "x/y"}


@pytest.mark.parametrize("key", ["gpu_typ_ids", "kernel_slug"])
def test_runpod_settings_rejects_unknown_and_kaggle_keys(monkeypatch, key):
    monkeypatch.setenv("ARC3_RUNPOD_JSON", json.dumps({key: "x"}))
    with pytest.raises(ValueError, match="unknown runpod config keys"):
        run._runpod_settings()


def test_smoke_config_maps_both_kaggle_models_to_hugging_face():
    config = json.loads((Path(__file__).parents[1] / "configs/runpod-smoke/v1.json").read_text(encoding="utf-8"))
    kaggle_models = {config["kaggle"]["model_source"], config["kaggle"]["draft_model_source"]}
    assert set(config["runpod"]["hf_models"]) == kaggle_models
