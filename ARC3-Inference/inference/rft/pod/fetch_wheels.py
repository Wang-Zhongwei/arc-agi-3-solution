"""Download the pod's Python packages as wheels on a machine with PyPI, for pods whose host cannot reach it.

Some RunPod hosts time out on pypi.org and every mirror while Hugging Face and download.pytorch.org work
(a secure RTX PRO 6000, 10-19). Download the top-level packages without dependencies, then walk their
METADATA and fetch every dependency the pod's image lacks (torch, triton and CUDA libraries excluded), for
Linux x86_64 and CPython 3.12. On the pod: `pip install --no-index --no-deps WHEELS/*.whl`, then `pip check`.

    python fetch_wheels.py OUT_DIR "pkg1;pkg2;..." [PACKAGES_ON_POD]   # PACKAGES_ON_POD: `pip list --format=freeze` names
"""

import re
import subprocess
import sys
import zipfile
from pathlib import Path

PLATFORM = ["--only-binary=:all:", "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_17_x86_64",
            "--platform", "manylinux_2_28_x86_64", "--python-version", "3.12", "--implementation", "cp"]
ENV = {"python_version": "3.12", "python_full_version": "3.12.3", "sys_platform": "linux", "platform_system": "Linux",
       "platform_machine": "x86_64", "implementation_name": "cpython", "os_name": "posix", "extra": ""}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def download(out: Path, specs: list[str]) -> None:
    for spec in specs:
        r = subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", *PLATFORM, "-d", str(out), spec],
                           capture_output=True, text=True)
        if r.returncode:
            print("failed", spec, r.stderr.strip()[-200:])


def requirements(wheel: Path) -> set[str]:
    from packaging.markers import Marker

    z = zipfile.ZipFile(wheel)
    meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
    deps = set()
    for line in z.read(meta).decode().splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        spec, _, marker = line.split(":", 1)[1].strip().partition(";")
        if "extra" in marker or (marker.strip() and not Marker(marker.strip()).evaluate(ENV)):
            continue
        deps.add(norm(re.split(r"[ <>=!~\[(]", spec.strip())[0]))
    return deps


def main() -> None:
    out, top = Path(sys.argv[1]), [s for s in sys.argv[2].split(";") if s]
    have = {norm(n) for n in (sys.argv[3].split(";") if len(sys.argv) > 3 else []) if n}
    out.mkdir(parents=True, exist_ok=True)
    download(out, top)
    for _ in range(20):
        got = {norm(p.name.split("-")[0]): p for p in out.glob("*.whl")}
        missing = {d for p in got.values() for d in requirements(p)} - set(got) - have
        missing = {d for d in missing if not d.startswith(("torch", "nvidia", "triton"))}
        if not missing:
            break
        print("fetching", sorted(missing))
        download(out, sorted(missing))
    print(len(list(out.glob("*.whl"))), "wheels in", out)


if __name__ == "__main__":
    main()
