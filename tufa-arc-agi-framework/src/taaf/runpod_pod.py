"""Pod-side runner for ``taaf.deploy_runpod``: stands in for Kaggle's notebook runtime.

The pod's start command downloads the TAAF source dataset to its Kaggle mount
path and runs this file from it with the system Python. It uses the standard
library only, because it runs before the bundle's dependencies are installed.

1. Rebuild ``/kaggle/input`` from the bundled ``kernel/kernel-metadata.json``:
   competition files and datasets through the Kaggle API; models from Hugging
   Face when ``hf_models`` maps them, otherwise from Kaggle. The Kaggle inputs
   and the models download at the same time.
2. Install the bundled projects' PyPI dependencies.
3. Execute the rendered notebook's code cells, as Kaggle would.
4. Upload ``/kaggle/working`` as a private Kaggle dataset (``working.tar.gz``).

The start command deletes the pod once this exits.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
HF_CACHE = Path("/root/hf-models")
OUTPUT_DIR = Path("/root/taaf-output")
OUTPUT_DATASET_SUFFIX = "-rp-"
# arc-agi comes from the competition wheelhouse, which the notebook installs itself.
_SKIP_REQUIREMENTS = {"arc-agi"}
_EXTRA_REQUIREMENTS = ["pandas", "pyarrow"]
_NOTEBOOK_RUNNER = (
    "import ast, asyncio, inspect, os\n"
    "code = open(os.environ['TAAF_RUNPOD_NOTEBOOK_CODE']).read()\n"
    "compiled = compile(code, 'taaf_kaggle_run.ipynb', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)\n"
    "result = eval(compiled, {'__name__': '__main__'})\n"
    "if inspect.isawaitable(result):\n"
    "    asyncio.run(result)\n"
)


def output_dataset_ref(owner: str, kernel_slug: str, pod_id: str) -> str:
    """One private output dataset per pod; Kaggle slugs are at most 50 chars."""
    suffix = f"{OUTPUT_DATASET_SUFFIX}{pod_id.lower()}"
    return f"{owner}/{kernel_slug[: 50 - len(suffix)].rstrip('-')}{suffix}"


def model_mount_path(ref: str) -> Path:
    """``owner/model/framework/variation/version`` -> Kaggle's model mount path."""
    owner, model, framework, variation, version = ref.split("/")
    return INPUT / "models" / owner / model / framework.lower() / variation / version


def parse_hf_source(source: str) -> tuple[str, str | None, str]:
    """``repo[@revision][:subdir]`` -> (repo, revision, subdir)."""
    repo, _, subdir = source.partition(":")
    repo, _, revision = repo.partition("@")
    return repo.strip(), revision.strip() or None, subdir.strip().strip("/")


def requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[@ ]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def bundle_requirements(bundle: Path) -> list[str]:
    """PyPI dependencies of the bundled projects, minus the bundled projects themselves."""
    projects = [
        tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
        for path in sorted(bundle.glob("src/*/pyproject.toml"))
    ]
    local = {requirement_name(project.get("name", "")) for project in projects}
    requirements: list[str] = []
    for project in projects:
        for requirement in project.get("dependencies", []):
            name = requirement_name(requirement)
            if name not in local and name not in _SKIP_REQUIREMENTS and requirement not in requirements:
                requirements.append(requirement)
    return requirements + [r for r in _EXTRA_REQUIREMENTS if r not in requirements]


def notebook_code(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    code = "\n\n".join("".join(cell.get("source", [])) for cell in cells)
    compile(code, notebook_path.name, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return code


def log(message: str) -> None:
    print(f"taaf.runpod: {message}", flush=True)


def run(cmd: list[str], **kwargs: object) -> None:
    log("$ " + shlex.join(cmd))
    started = time.monotonic()
    subprocess.run(cmd, check=True, **kwargs)  # type: ignore[call-overload]
    log(f"  done in {time.monotonic() - started:.0f}s")


def stage_competition(slug: str) -> None:
    dest = INPUT / "competitions" / slug
    dest.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "competitions", "download", "-c", slug, "-p", str(dest), "-q"])
    for archive in dest.glob("*.zip"):
        shutil.unpack_archive(archive, dest)
        archive.unlink()


def stage_dataset(ref: str) -> None:
    dest = INPUT / "datasets" / ref
    if dest.is_dir() and any(dest.iterdir()):
        return  # the source bundle, already downloaded by the start command
    dest.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", ref, "-p", str(dest), "--unzip", "-q"])


def stage_model(ref: str, hf_models: dict[str, str]) -> None:
    dest = model_mount_path(ref)
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = {key.lower(): value for key, value in hf_models.items()}.get(ref.lower())
    if source is None:
        dest.mkdir(exist_ok=True)
        run(["kaggle", "models", "instances", "versions", "download", ref, "-p", str(dest), "--untar", "-q"])
        return
    repo, revision, subdir = parse_hf_source(source)
    local_dir = HF_CACHE / repo.replace("/", "--")
    cmd = ["hf", "download", repo, "--local-dir", str(local_dir)]
    if revision:
        cmd += ["--revision", revision]
    if subdir:
        cmd += ["--include", f"{subdir}/*"]
    run(cmd, env={**os.environ, "HF_XET_HIGH_PERFORMANCE": "1"})
    dest.symlink_to(local_dir / subdir if subdir else local_dir, target_is_directory=True)


def competition_arc_agi_requirement(wheels: Path) -> str | None:
    """``arc-agi==<version>`` of the competition's bundled wheel, if there is one."""
    found = sorted(wheels.glob("arc_agi-*.whl"))
    return f"arc-agi=={found[-1].name.split('-')[1]}" if found else None


def install_requirements(bundle: Path) -> None:
    if not shutil.which("g++"):
        # Pennyroyal JIT-compiles kernels with the host C++ compiler.
        run(["apt-get", "update", "-q"])
        run(["apt-get", "install", "-y", "-q", "build-essential"])
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run([*pip, *bundle_requirements(bundle)])
    # The notebook installs arc-agi offline from the competition wheelhouse. Over
    # the image's apt-managed Python packages that fails (pip cannot uninstall
    # them: uninstall-no-record-file), so install the same wheel here with
    # --ignore-installed; the notebook's install then finds it satisfied.
    for wheels in sorted(INPUT.glob("competitions/*/arc_agi_3_wheels")):
        requirement = competition_arc_agi_requirement(wheels)
        if requirement:
            run([*pip, "--ignore-installed", "--find-links", str(wheels), requirement])


def stage_inputs(bundle: Path, metadata: dict, hf_models: dict[str, str]) -> None:
    """Fetch Kaggle inputs and models at the same time.

    The two lanes hit different servers: Kaggle serves datasets at ~40 MB/s,
    Hugging Face at ~1.6 GB/s, so neither saturates the pod's link and staging
    takes the longer lane instead of their sum. pip needs the competition
    wheels, so it stays in the Kaggle lane.
    """

    def kaggle_lane() -> None:
        for slug in metadata.get("competition_sources", []):
            stage_competition(slug)
        for ref in metadata.get("dataset_sources", []):
            stage_dataset(ref)
        install_requirements(bundle)

    def model_lane() -> None:
        for ref in metadata.get("model_sources", []):
            stage_model(ref, hf_models)

    with ThreadPoolExecutor(max_workers=2) as pool:
        lanes = [pool.submit(kaggle_lane), pool.submit(model_lane)]
        for lane in lanes:
            lane.result()  # re-raise the first lane failure


def run_notebook(bundle: Path, notebook_path: Path) -> int:
    code_path = WORKING / "runpod_notebook.py"
    code_path.write_text(notebook_code(notebook_path), encoding="utf-8")
    env = {
        **os.environ,
        "TAAF_KAGGLE_BUNDLE_DIR": str(bundle),
        "TAAF_KAGGLE_WORKING_DIR": str(WORKING),
        "TAAF_RUNPOD_NOTEBOOK_CODE": str(code_path),
        "PYTHONUNBUFFERED": "1",
    }
    log(f"running {notebook_path.name}")
    with open(WORKING / "runpod_notebook.log", "w", encoding="utf-8") as notebook_log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", _NOTEBOOK_RUNNER],
            cwd=WORKING,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            notebook_log.write(line)
        return process.wait()


def upload_outputs(*, owner: str, kernel_slug: str, pod_id: str, summary: dict[str, object]) -> None:
    ref = output_dataset_ref(owner, kernel_slug, pod_id)
    (WORKING / "runpod_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)
    with tarfile.open(OUTPUT_DIR / "working.tar.gz", "w:gz") as archive:
        for path in sorted(WORKING.iterdir()):
            if path.name.endswith("site-packages") or path.suffix == ".pid":
                continue
            archive.add(path, arcname=path.name)
    (OUTPUT_DIR / "dataset-metadata.json").write_text(
        json.dumps({"title": f"{kernel_slug} runpod {pod_id}"[:50], "id": ref, "licenses": [{"name": "CC0-1.0"}]}),
        encoding="utf-8",
    )
    for attempt in range(1, 4):
        try:
            run(["kaggle", "datasets", "create", "-p", str(OUTPUT_DIR), "-q"])
            log(f"outputs uploaded to {ref}")
            return
        except subprocess.CalledProcessError:
            log(f"output upload attempt {attempt} failed")
            time.sleep(30 * attempt)
    raise RuntimeError(f"could not upload outputs to {ref}")


def main() -> int:
    spec = json.loads(os.environ["TAAF_RUNPOD_SPEC"])
    pod_id = os.environ.get("RUNPOD_POD_ID", "local")
    bundle = INPUT / "datasets" / os.environ["TAAF_RUNPOD_SOURCE_DATASET"]
    started = time.time()
    summary: dict[str, object] = {"pod_id": pod_id, "started_at": started, "stage": "staging"}
    exit_code = 1
    try:
        subprocess.run(["nvidia-smi"], check=False)
        metadata = json.loads((bundle / "kernel" / "kernel-metadata.json").read_text(encoding="utf-8"))
        if metadata.get("kernel_sources"):
            raise RuntimeError(f"Kaggle utility-script sources are not available on RunPod: {metadata['kernel_sources']}")
        stage_inputs(bundle, metadata, spec.get("hf_models", {}))
        summary["stage"] = "notebook"
        summary["staging_seconds"] = round(time.time() - started)
        exit_code = run_notebook(bundle, bundle / "kernel" / metadata["code_file"])
        summary["stage"] = "finished"
    except Exception:
        traceback.print_exc()
        (WORKING / "runpod_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    summary.update({"notebook_exit_code": exit_code, "finished_at": time.time()})
    log(f"summary {json.dumps(summary)}")
    upload_outputs(owner=spec["kernel_owner"], kernel_slug=spec["kernel_slug"], pod_id=pod_id, summary=summary)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
