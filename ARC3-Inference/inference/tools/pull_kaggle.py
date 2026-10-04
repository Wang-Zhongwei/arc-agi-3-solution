"""Download a specific Kaggle notebook version's output files and kernel log.

Saves to runs/kaggle/<owner>/<slug>/vNN/, the folder the viewer reads. With
--source, also save that version's notebook to kernel/ and the TAAF source
dataset it ran against (src/, benchmark_initial.pkl, ...) into the run folder,
matching the layout TAAF deploys write.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

SOURCE_DATASET_PREFIX = "taaf-kaggle-source"


def _notebook(value: str) -> tuple[str | None, str]:
    if not re.fullmatch(r"(?:[A-Za-z0-9_-]+/)?[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("Use a notebook slug or owner/slug reference.")
    owner, _, slug = value.rpartition("/")
    return owner or None, slug


def _version(value: str) -> int:
    if not re.fullmatch(r"v?[0-9]+", value) or int(value.removeprefix("v")) < 1:
        raise argparse.ArgumentTypeError("Version must be a positive number, such as 17 or v17.")
    return int(value.removeprefix("v"))


def _default_owner() -> str:
    import taaf.deploy_kaggle

    taaf.deploy_kaggle._load_dotenv_upwards()
    username, _ = taaf.deploy_kaggle._resolve_kaggle_credentials(None)
    return username


def _download(url: str, destination: Path) -> None:
    temporary = None
    try:
        with requests.get(url, timeout=600, stream=True) as response:
            response.raise_for_status()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
            temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch(owner: str, slug: str, label: str, target: Path, keep: list[str]) -> bool:
    # Import only when fetching: Kaggle may authenticate during import, so --help
    # must remain available without credentials.
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    files = []
    log = ""
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.version_label = label
        request.page_size = 500
        seen_tokens = set()
        while True:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            log = response.log or log
            token = response.next_page_token
            if not token:
                break
            if token in seen_tokens:
                raise RuntimeError("Kaggle repeated an output page; download aborted.")
            seen_tokens.add(token)
            request.page_token = token
    print(f"[{label}] files={len(files)} log_chars={len(log)}")
    if not files and not log:
        # Kaggle exposes a version's output only once that session is terminal.
        print(f"[{label}] nothing to download -- version still running, or no such version")
        return False

    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        name = item.file_name
        if keep and not any(token in name for token in keep):
            continue
        out = target / name
        if not out.resolve().is_relative_to(target.resolve()) or Path(name).is_absolute():
            raise ValueError(f"Output filename escapes the destination: {name}")
        _download(item.url, out)
        print(f"   {name}  {out.stat().st_size:,}B")

    if log:
        (target / f"{slug}.log").write_text(log, encoding="utf-8")
        print(f"   {slug}.log  {len(log):,} chars")
    print(f"Saved to {target}")
    return True


def _utc(value: datetime) -> datetime:
    # Kaggle reports naive UTC times; benchmark.json start_time is naive UTC on Kaggle.
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


_ENV_PAIR = re.compile(r"""["']([A-Z][A-Z0-9_]*)["']\s*:\s*["']([^"']*)["']""")


def _bundle_matches_run(bundle: Path, target: Path) -> bool | None:
    """Whether the bundle's literal setup env agrees with the env the run applied.

    Setup commands embed the harness env as literal 'KEY': 'value' pairs, and the
    notebook saves what it applied as taaf_setup_env.json. None if either is missing.
    """
    try:
        applied = json.loads((target / "taaf_setup_env.json").read_text())
        commands = "\n".join(json.loads((bundle / "setup_commands.json").read_text()))
    except (OSError, ValueError, TypeError):
        return None
    pairs = [(key, value) for key, value in _ENV_PAIR.findall(commands) if key in applied]
    return all(str(applied[key]) == value for key, value in pairs) if pairs else None


def _run_start(target: Path) -> datetime | None:
    try:
        return _utc(datetime.fromisoformat(json.loads((target / "benchmark.json").read_text())["start_time"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def fetch_source(
    owner: str,
    slug: str,
    version: int,
    target: Path,
    dataset_version: int | None = None,
    dataset_ref: str | None = None,
) -> bool:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.version_label = f"v{version}"
        kernel = client.kernels.kernels_api_client.get_kernel(request)
        if kernel.metadata.current_version_number != version:
            raise RuntimeError(
                f"Kaggle returned notebook version {kernel.metadata.current_version_number}, not v{version}."
            )
        sources = kernel.metadata.dataset_data_sources or []
        if dataset_ref:
            refs = [dataset_ref]
        else:
            refs = [ref for ref in sources if ref.split("/")[-1].startswith(SOURCE_DATASET_PREFIX)]
            # A config's kaggle.dataset_slug renames the bundle; TAAF still uploads
            # it under the deploying owner, unlike external inputs.
            refs = refs or [ref for ref in sources if ref.split("/")[0] == owner]
        versions = []
        if len(refs) == 1:
            request = ApiGetDatasetRequest()
            request.owner_slug, request.dataset_slug = refs[0].split("/")
            versions = client.datasets.dataset_api_client.get_dataset(request).versions or []

    notebook = target / "kernel" / (slug + (".ipynb" if kernel.blob.kernel_type.lower() == "notebook" else ".py"))
    notebook.parent.mkdir(parents=True, exist_ok=True)
    notebook.write_text(kernel.blob.source, encoding="utf-8")
    print(f"[v{version}] notebook  {notebook}")
    if len(refs) != 1:
        print(f"[v{version}] expected one {SOURCE_DATASET_PREFIX}* or {owner}/* dataset source, found {sources}; pass --dataset-ref")
        return False

    start = _run_start(target)
    if dataset_version is not None:
        candidates, selection = [dataset_version], "explicit"
    elif start is None:
        print(f"[v{version}] no benchmark.json start_time in {target}; pass --dataset-version")
        return False
    else:
        # The kernel mounts its dataset at session start, minutes before benchmark.json's
        # start_time, so a version pushed in between can postdate the mount. Try versions
        # created before start_time, newest first, and keep the first whose setup env
        # matches the run's. Kaggle lists only recent versions; older runs need --dataset-version.
        earlier = sorted((item.version_number for item in versions if _utc(item.creation_date) <= start),
                         reverse=True)
        if not earlier:
            print(f"[v{version}] no listed {refs[0]} version predates {start}; pass --dataset-version")
            return False
        candidates = earlier[:3]
        selection = f"newest version before run start {start.isoformat()} whose setup env matches the run"

    if (target / "src").exists():
        dataset_version = candidates[0]
        print(f"[v{version}] kept existing {target / 'src'}; remove it to download {refs[0]} v{dataset_version}")
        matches = None
    else:
        matches = None
        for dataset_version in candidates:
            staging = Path(tempfile.mkdtemp(prefix=".source-dataset-", dir=target))
            try:
                api.dataset_download_files(f"{refs[0]}/{dataset_version}", path=str(staging), unzip=True, quiet=True)
                matches = _bundle_matches_run(staging, target)
                if matches is False and dataset_version != candidates[-1]:
                    print(f"[v{version}] {refs[0]} v{dataset_version} setup env differs from the run; trying older")
                    continue
                # Source files sit at the top of the run folder, like a Slurm run. Files the
                # notebook already wrote to its output (e.g. git_status.txt) are kept.
                for entry in staging.iterdir():
                    if not (target / entry.name).exists():
                        entry.rename(target / entry.name)
                break
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        print(f"[v{version}] source dataset  {refs[0]} v{dataset_version} -> {target}")
        print(f"[v{version}] bundle setup env matches run: {matches}")
    created = next((_utc(item.creation_date).isoformat() for item in versions
                    if item.version_number == dataset_version), None)

    (target / "source.json").write_text(json.dumps({
        "notebook": f"{owner}/{slug}", "notebook_version": version,
        "notebook_file": str(notebook.relative_to(target)),
        "dataset": refs[0], "dataset_version": dataset_version, "dataset_version_created": created,
        "selection": selection, "setup_env_matches_run": matches,
    }, indent=2) + "\n", encoding="utf-8")
    return matches is not False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=_notebook, help="Notebook slug or owner/slug; default owner: your Kaggle username.")
    parser.add_argument("version", type=_version, help="Exact version to fetch: 17 or v17.")
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"),
        help="Runs root; the run lands in <runs-dir>/kaggle/<owner>/<slug>/vNN/. Default: runs.",
    )
    parser.add_argument("--output-dir", "-o", type=Path, help="Exact destination; overrides --runs-dir.")
    parser.add_argument(
        "--filter", action="append", default=[], dest="keep", metavar="TEXT",
        help="Download filenames containing TEXT; repeat to match any. Always saves the kernel log.",
    )
    parser.add_argument(
        "--source", action="store_true",
        help="Also save this version's notebook to kernel/ and its TAAF source dataset into the run folder.",
    )
    parser.add_argument(
        "--dataset-version", type=_version, metavar="N",
        help="Source dataset version for --source; default: the version current at benchmark.json start_time.",
    )
    parser.add_argument(
        "--dataset-ref", metavar="OWNER/SLUG",
        help=f"Source dataset for --source; default: the notebook's {SOURCE_DATASET_PREFIX}* input.",
    )
    args = parser.parse_args(argv)
    if (args.dataset_version is not None or args.dataset_ref) and not args.source:
        parser.error("--dataset-version and --dataset-ref require --source")
    owner, slug = args.notebook
    try:
        owner = owner or _default_owner()
        target = args.output_dir or args.runs_dir / "kaggle" / owner / slug / f"v{args.version:02d}"
        ok = fetch(owner, slug, f"v{args.version}", target, args.keep)
        if args.source:
            ok = fetch_source(owner, slug, args.version, target, args.dataset_version, args.dataset_ref) and ok
        return 0 if ok else 1
    except Exception as exc:
        print(f"Kaggle download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
