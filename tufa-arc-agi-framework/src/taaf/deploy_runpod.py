"""``RunPodTarget`` — run the Kaggle notebook bundle on a RunPod GPU pod.

Lifecycle:

1. ``deploy()`` builds the same source dataset and rendered notebook as
   ``KaggleTarget`` and uploads the dataset to Kaggle (private). The
   ``kernel/`` folder (rendered notebook + ``kernel-metadata.json``) rides
   along inside the dataset. Instead of pushing a Kaggle kernel, it creates
   a RunPod pod through the REST API.
2. The pod's start command (``_BOOTSTRAP``) installs the Kaggle and Hugging
   Face CLIs, downloads the source dataset to its Kaggle mount path, and
   runs ``taaf/runpod_pod.py`` from the bundle. That runner rebuilds
   ``/kaggle/input`` from ``kernel-metadata.json`` (competition files and
   datasets via the Kaggle API, models from Hugging Face where
   ``hf_models`` maps them, otherwise from Kaggle), executes the notebook's
   code cells, uploads ``/kaggle/working`` as a private Kaggle dataset, and
   deletes its own pod.
3. ``handle.wait()`` polls the pod until it is gone, then downloads the
   output dataset into ``job_dir``. ``handle.stop()`` deletes the pod.

A watchdog in the start command deletes the pod after ``hard_limit_s`` no
matter what, so a hung run cannot bill indefinitely.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import taaf.benchmark
import taaf.deploy
import taaf.deploy_kaggle
from taaf.runpod_pod import output_dataset_ref

RUNPOD_REST_URL = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_GPU_TYPE_IDS: tuple[str, ...] = (
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
)
_HARD_LIMIT_MARGIN_S = 90 * 60.0

# Pod start command. Runs before any bundle code exists on the pod, so it only
# fetches the bundle and hands over to taaf/runpod_pod.py. Every exit path
# deletes the pod; the background sleep is the hard spending cap.
_BOOTSTRAP = r"""
set -uo pipefail
mkdir -p /kaggle/working
exec > >(tee -a /kaggle/working/runpod_bootstrap.log) 2>&1
# SSH for debugging a live run: RunPod injects the account's PUBLIC_KEY.
if [ -n "${PUBLIC_KEY:-}" ] && [ -x /usr/sbin/sshd ]; then
  mkdir -p ~/.ssh /run/sshd && chmod 700 ~/.ssh
  echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  ssh-keygen -A >/dev/null 2>&1; /usr/sbin/sshd || echo "taaf.runpod: sshd failed to start"
fi
delete_pod() {
  echo "taaf.runpod: deleting pod $RUNPOD_POD_ID ($1)"
  curl -sS -X DELETE -H "Authorization: Bearer $TAAF_RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" || true
  sleep 600
  exit 1
}
( sleep "$TAAF_RUNPOD_HARD_LIMIT_S"; delete_pod "hard limit ${TAAF_RUNPOD_HARD_LIMIT_S}s reached" ) &
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip install -q 'kaggle>=2.2' 'huggingface_hub[hf_xet]>=0.34' || delete_pod "pip install failed"
bundle="/kaggle/input/datasets/$TAAF_RUNPOD_SOURCE_DATASET"
mkdir -p "$bundle"
kaggle datasets download -d "$TAAF_RUNPOD_SOURCE_DATASET" -p "$bundle" --unzip -q \
  || delete_pod "source dataset download failed"
runner="$(find "$bundle/src" -path '*/taaf/runpod_pod.py' | head -n 1)"
[ -n "$runner" ] || delete_pod "taaf/runpod_pod.py missing from the bundle"
python3 -u "$runner"
delete_pod "runner exited with $?"
"""


@dataclass
class RunPodHandle(taaf.deploy.DeploymentHandle):
    """Handle to a RunPod pod running a TAAF Kaggle bundle.

    Fields:

    - ``pod_id``: RunPod pod id ("" for dry runs).
    - ``output_dataset``: private Kaggle dataset the pod uploads its
      ``/kaggle/working`` to, ``owner/slug``.
    - ``poll_interval_s``: cadence for ``wait()``.
    """

    pod_id: str = ""
    output_dataset: str = ""
    poll_interval_s: float = 60.0

    @property
    def job_id(self) -> str:
        return self.pod_id

    def wait(self) -> taaf.benchmark.Benchmark:
        if not self.pod_id:
            raise RuntimeError("RunPod dry-run handle has no pod to wait for.")
        while not self.is_done:
            time.sleep(self.poll_interval_s)
        pull_output(self.output_dataset, self.job_dir)
        benchmark_json = self.job_dir / "benchmark.json"
        if not benchmark_json.is_file():
            raise RuntimeError(f"Pod {self.pod_id} finished, but its output has no benchmark.json; see {self.job_dir}.")
        return taaf.benchmark.Benchmark.from_json(benchmark_json)

    def stop(self) -> None:
        if self.pod_id:
            delete_pod(self.pod_id)

    @property
    def is_done(self) -> bool:
        if not self.pod_id:
            return True
        pod = get_pod(self.pod_id)
        return pod is None or str(pod.get("desiredStatus", "")).upper() in {"EXITED", "TERMINATED"}

    @classmethod
    def _attach(cls, job_dir: Path, meta: dict[str, Any]) -> RunPodHandle:
        cfg = cast(dict[str, Any], meta.get("target_config") or {})
        return cls(
            job_dir=job_dir,
            pod_id=str(meta.get("job_id") or cfg.get("pod_id") or ""),
            output_dataset=str(cfg.get("output_dataset") or ""),
        )


@dataclass
class RunPodTarget(taaf.deploy_kaggle.KaggleTarget):
    """Run the rendered Kaggle notebook on a RunPod pod instead of Kaggle.

    Inherits every ``KaggleTarget`` field (dataset/kernel naming, setup
    commands, ``public_games``, ``max_runtime_s``, ``make_share_version``,
    ...). RunPod fields:

    - ``gpu_type_ids`` / ``gpu_count`` / ``cloud_type``: pod hardware.
    - ``image``: container image; needs Python 3.12 (the wheelhouses are
      cp312), ``curl`` and a C++ compiler.
    - ``container_disk_gb``: room for models, wheels and the serving venv.
    - ``min_ram_per_gpu_gb``: host RAM; Flash Next offloads its PLE table there.
    - ``min_download_mbps``: skip hosts on slow links (models are large).
    - ``allowed_cuda_versions``: host driver CUDA versions; the Pennyroyal
      wheelhouse ships a CUDA 13 toolchain.
    - ``hf_models``: Kaggle model ref -> ``repo[@revision][:subdir]`` on
      Hugging Face. Mapped models are downloaded from Hugging Face into the
      model's Kaggle mount path; unmapped ones come from Kaggle.
    - ``hard_limit_s``: the pod deletes itself after this long. Defaults to
      ``max_runtime_s`` plus 90 minutes of staging and startup.
    - ``startup_timeout_s``: ``deploy()`` waits this long for the container to
      start, then deletes the pod and fails (a host can hang before the start
      command ever runs, which the in-pod watchdog cannot catch).
    - ``ssh``: expose port 22 with a public IP and start ``sshd`` with the
      account's SSH key, for inspecting a live run.
    """

    gpu_type_ids: list[str] = field(default_factory=lambda: list(DEFAULT_GPU_TYPE_IDS))
    gpu_count: int = 1
    cloud_type: str = "SECURE"
    image: str = DEFAULT_IMAGE
    container_disk_gb: int = 400
    min_ram_per_gpu_gb: int = 160
    min_vcpu_per_gpu: int = 8
    min_download_mbps: float | None = None
    allowed_cuda_versions: list[str] = field(default_factory=lambda: ["13.0"])
    data_center_ids: list[str] = field(default_factory=lambda: list[str]())
    hf_models: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    hard_limit_s: float | None = None
    startup_timeout_s: float = 900.0
    ssh: bool = True

    pod_id: str = field(default="", init=False)
    output_dataset: str = field(default="", init=False)

    def _extra_source_entries(self, kernel_bundle: Path) -> list[Path]:
        # The pod runs the rendered notebook and stages inputs from kernel-metadata.json.
        return [kernel_bundle]

    def _launch_notebook(self, kernel_bundle: Path, *, username: str, key: str) -> None:
        api_key = runpod_api_key()
        kaggle_token = os.environ.get("KAGGLE_API_TOKEN") or key
        owner, slug = self.kernel_id.split("/", 1)
        env = {
            "TAAF_RUNPOD_SOURCE_DATASET": self.source_dataset_ref,
            "TAAF_RUNPOD_SPEC": json.dumps(self._pod_spec(owner=owner, kernel_slug=slug)),
            "TAAF_RUNPOD_HARD_LIMIT_S": str(int(self._hard_limit_s())),
            "TAAF_RUNPOD_API_KEY": api_key,
            "KAGGLE_USERNAME": username,
            "KAGGLE_API_TOKEN": kaggle_token,
        }
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            env["HF_TOKEN"] = hf_token
        pod = create_pod(self.pod_payload(name=f"taaf-{slug}", env=env), api_key=api_key)
        self.pod_id = str(pod["id"])
        self.output_dataset = output_dataset_ref(owner, slug, self.pod_id)
        print(f"deploy.runpod: pod {self.pod_id} created; waiting for its container to start")
        if not wait_for_container(self.pod_id, timeout_s=self.startup_timeout_s, api_key=api_key):
            delete_pod(self.pod_id, api_key=api_key)
            raise RuntimeError(
                f"RunPod pod {self.pod_id} did not start its container within {self.startup_timeout_s:.0f}s; "
                "deleted it. Retry (another host) or widen gpu_type_ids / data_center_ids."
            )
        print(f"deploy.runpod: container running; output dataset will be {self.output_dataset}")
        ssh = ssh_command(self.pod_id, api_key=api_key) if self.ssh else None
        if ssh:
            print(f"deploy.runpod: inspect with `{ssh}` (log: /kaggle/working/runpod_bootstrap.log)")

    def _job_id(self) -> str:
        return self.pod_id

    def _make_handle(self, *, job_dir: Path, dataset_ref: str) -> RunPodHandle:
        return RunPodHandle(job_dir=job_dir, pod_id=self.pod_id, output_dataset=self.output_dataset)

    def _hard_limit_s(self) -> float:
        if self.hard_limit_s:
            return float(self.hard_limit_s)
        return float(self.max_runtime_s) + _HARD_LIMIT_MARGIN_S

    def _pod_spec(self, *, owner: str, kernel_slug: str) -> dict[str, Any]:
        """What the pod runner needs beyond kernel-metadata.json."""
        return {
            "kernel_owner": owner,
            "kernel_slug": kernel_slug,
            "hf_models": dict(self.hf_models),
        }

    def pod_payload(self, *, name: str, env: dict[str, str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name[:60],
            "imageName": self.image,
            "computeType": "GPU",
            "gpuTypeIds": list(self.gpu_type_ids),
            "gpuTypePriority": "custom",
            "gpuCount": int(self.gpu_count),
            "cloudType": self.cloud_type,
            "containerDiskInGb": int(self.container_disk_gb),
            "volumeInGb": 0,
            "minRAMPerGPU": int(self.min_ram_per_gpu_gb),
            "minVCPUPerGPU": int(self.min_vcpu_per_gpu),
            "allowedCudaVersions": list(self.allowed_cuda_versions),
            "ports": ["22/tcp"] if self.ssh else [],
            "supportPublicIp": bool(self.ssh),
            "env": env,
            "dockerEntrypoint": ["bash", "-c"],
            "dockerStartCmd": [_BOOTSTRAP],
        }
        if self.min_download_mbps:
            payload["minDownloadMbps"] = float(self.min_download_mbps)
        if self.data_center_ids:
            payload["dataCenterIds"] = list(self.data_center_ids)
        return payload


def runpod_api_key() -> str:
    taaf.deploy_kaggle._load_dotenv_upwards()
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing RUNPOD_API_KEY (set it in the environment or a .env file).")
    return key


def _request(method: str, path: str, *, api_key: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{RUNPOD_REST_URL}{path}",
        data=data,
        method=method,
        # Cloudflare in front of the API rejects urllib's default User-Agent (error 1010).
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "taaf-runpod/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body.strip() else None


def create_pod(payload: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], _request("POST", "/pods", api_key=api_key, payload=payload))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod pod creation failed ({exc.code}): {detail}") from exc


def get_pod(pod_id: str, *, api_key: str | None = None) -> dict[str, Any] | None:
    try:
        return cast(dict[str, Any], _request("GET", f"/pods/{pod_id}", api_key=api_key or runpod_api_key()))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def pod_runtime(pod_id: str, *, api_key: str | None = None) -> dict[str, Any] | None:
    """The pod's live container runtime (uptime, ports), or None until the container runs.

    Only RunPod's GraphQL API reports this; the REST pod record does not.
    """
    query = (
        "query($id: String!) { pod(input: {podId: $id}) { runtime { uptimeInSeconds "
        "ports { ip isIpPublic privatePort publicPort type } } } }"
    )
    request = urllib.request.Request(
        RUNPOD_GRAPHQL_URL,
        data=json.dumps({"query": query, "variables": {"id": pod_id}}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key or runpod_api_key()}",
            "Content-Type": "application/json",
            "User-Agent": "taaf-runpod/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        pod = (json.load(response).get("data") or {}).get("pod") or {}
    return cast(dict[str, Any] | None, pod.get("runtime"))


def wait_for_container(pod_id: str, *, timeout_s: float, api_key: str | None = None, poll_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if pod_runtime(pod_id, api_key=api_key):
                return True
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"deploy.runpod: runtime poll failed ({exc!r}); retrying")
        time.sleep(poll_s)
    return False


def ssh_command(pod_id: str, *, api_key: str | None = None) -> str | None:
    for port in (pod_runtime(pod_id, api_key=api_key) or {}).get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return f"ssh -i ~/.ssh/id_ed25519 -p {port['publicPort']} root@{port['ip']}"
    return None


def delete_pod(pod_id: str, *, api_key: str | None = None) -> None:
    try:
        _request("DELETE", f"/pods/{pod_id}", api_key=api_key or runpod_api_key())
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise


def pull_output(output_dataset: str, dest_dir: Path, *, timeout_s: float = 1800.0) -> Path:
    """Download a pod's output dataset into ``dest_dir`` and unpack ``working.tar.gz``.

    Kaggle needs a few minutes to process a new dataset, so retry until it
    downloads or ``timeout_s`` passes.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", output_dataset, "-p", str(dest_dir), "--unzip", "-q"],
            capture_output=True,
            text=True,
            check=False,
        )
        archive = dest_dir / "working.tar.gz"
        if result.returncode == 0 and archive.is_file():
            subprocess.run(["tar", "-xzf", str(archive), "-C", str(dest_dir)], check=True)
            archive.unlink()
            return dest_dir
        # Kaggle usually expands the uploaded archive into the dataset's own files.
        if result.returncode == 0 and (dest_dir / "runpod_summary.json").is_file():
            return dest_dir
        if time.monotonic() > deadline:
            raise RuntimeError(f"Could not download {output_dataset}:\n{result.stdout}\n{result.stderr}")
        time.sleep(30)
