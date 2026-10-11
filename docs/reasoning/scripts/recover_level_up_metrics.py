"""Stream exact Kaggle outputs for runs whose telemetry is not saved locally.

For each notebook seen in the local run folders, every version up to the
current one is considered. Versions saved locally without request logs get
their request-log metrics streamed; versions missing locally, or saved without
event files, get benchmark.json, events and request logs streamed into compact
per-trial digests (level wins, calibration notes, rebuilt clearing-call history
and request metrics for clearing steps). Signed URLs and conversations are not
saved. Does not launch or change notebooks. Re-running resumes; pass
--retry-failed to retry failed listings only.
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest

from level_up_context_all_runs import (DATA, benchmark_summary, compact_digest, digest_events, inventory,
                                       request_metrics_lines)

REQUESTS_OUT = DATA / "recovered_request_metrics.json"
REMOTE_OUT = DATA / "remote_runs.json"
MANIFEST = DATA / "recovery_coverage.json"
DONE = {"events streamed", "request logs streamed", "no request logs", "no benchmark or events"}
NAME = re.compile(r"(?:artifacts/)?([^/_]+)_p(\d+)_(events|requests)\.jsonl")


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    api = KaggleApi()
    api.authenticate()
    local, ownerless = {}, {}
    for relative, files in inventory().items():
        match = re.fullmatch(r"kaggle/([^/]+)/([^/]+)/v(\d+)", relative)
        if match and files["benchmarks"]:
            local[(match[1], match[2], int(match[3]))] = files
        match = re.fullmatch(r"kaggle/([^/]+)/v(\d+)", relative)
        if match and files["benchmarks"] and files["events"]:
            ownerless[(match[1], int(match[2]))] = files
    notebooks = sorted({(owner, slug) for owner, slug, _ in local})
    previous = {r["run"]: r for r in load(MANIFEST, [])}
    remote = {r["run"]: r for r in load(REMOTE_OUT, [])}
    recovered = load(REQUESTS_OUT, [])
    recovered_runs = {r["source_run"] for r in recovered}

    def client_call(fn):
        for attempt in range(5):
            try:
                with api.build_kaggle_client() as client:
                    return fn(client.kernels.kernels_api_client)
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status not in (429, 500, 502, 503) or attempt == 4:
                    raise
                time.sleep(5 * 2 ** attempt)

    def current_version(owner, slug):
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        try:
            return client_call(lambda c: c.get_kernel(request)).metadata.current_version_number or 0
        except Exception:
            return 0

    targets = []
    for owner, slug in notebooks:
        latest = max(current_version(owner, slug), max(v for o, s, v in local if (o, s) == (owner, slug)))
        for version in range(1, latest + 1):
            run = f"kaggle/{owner}/{slug}/v{version:02d}"
            files = local.get((owner, slug, version)) or ownerless.get((slug, version))
            if files is not None and files["events"] and files["requests"]:
                continue
            if run in remote or run in recovered_runs:
                continue
            if previous.get(run, {}).get("status") in DONE and "--retry-failed" not in sys.argv:
                continue
            if "--retry-failed" in sys.argv and previous.get(run, {}).get("status") != "listing failed":
                continue
            targets.append((run, owner, slug, version, files))
    print(f"{len(targets)} versions to list", flush=True)

    def listing(run, owner, slug, version):
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug, request.version_label = owner, slug, f"v{version}"
        request.page_size = 500
        files, seen = [], set()
        while True:
            result = client_call(lambda c: c.list_kernel_session_output(request))
            files.extend(result.files or [])
            token = result.next_page_token
            if not token:
                return files
            if token in seen:
                raise ValueError("Repeated output page")
            seen.add(token)
            request.page_token = token

    def stream_lines(url):
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            yield from response.iter_lines(chunk_size=1024 * 1024)

    def recover(target):
        run, owner, slug, version, files = target
        outputs = listing(run, owner, slug, version)
        by_kind = {"events": {}, "requests": {}}
        benchmark_url = None
        for item in outputs:
            match = NAME.fullmatch(item.file_name)
            if match:
                by_kind[match[3]][(match[1], int(match[2]))] = item.url
            elif item.file_name == "benchmark.json":
                benchmark_url = item.url
        need_events = files is None or not files["events"]
        item = dict(run=run, output_files=len(outputs), events=len(by_kind["events"]), request_logs=len(by_kind["requests"]))
        if need_events:
            if benchmark_url is None or not by_kind["events"]:
                return dict(item, status="no benchmark or events"), None, []
            benchmark = benchmark_summary(requests.get(benchmark_url, timeout=(30, 300)).json())
            trials = {}
            for (game, pass_index), url in sorted(by_kind["events"].items()):
                digest = digest_events(stream_lines(url))
                metrics = {}
                if (game, pass_index) in by_kind["requests"]:
                    metrics = request_metrics_lines(stream_lines(by_kind["requests"][(game, pass_index)]), f"{run}/{game}_p{pass_index}_requests.jsonl")
                trials[f"{game}|{pass_index}"] = compact_digest(digest, metrics)
            return dict(item, status="events streamed"), dict(run=run, benchmark=benchmark, trials=trials), []
        rows = []
        for (game, pass_index), url in sorted(by_kind["requests"].items()):
            if (game, pass_index) in files["requests"]:
                continue
            for step, metric in request_metrics_lines(stream_lines(url), f"{run}/{game}_p{pass_index}_requests.jsonl").items():
                rows.append(dict(source_run=run, game=game.split("-")[0], pass_index=pass_index, step=step, **metric))
        return dict(item, status="request logs streamed" if rows else "no request logs"), None, rows

    coverage = [r for r in previous.values() if r["run"] not in {t[0] for t in targets}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(recover, target): target[0] for target in targets}
        for future in as_completed(futures):
            run = futures[future]
            try:
                item, digest, rows = future.result()
            except Exception as exc:
                item, digest, rows = dict(run=run, status="listing failed", error_type=type(exc).__name__,
                                          http_status=getattr(getattr(exc, "response", None), "status_code", None)), None, []
            coverage.append(item)
            if digest:
                remote[run] = digest
                REMOTE_OUT.write_text(json.dumps(sorted(remote.values(), key=lambda r: r["run"])) + "\n")
            if rows:
                recovered.extend(rows)
                REQUESTS_OUT.write_text(json.dumps(recovered, indent=1) + "\n")
            MANIFEST.write_text(json.dumps(sorted(coverage, key=lambda r: r["run"]), indent=1) + "\n")
            print(f"{len(coverage)} {run}: {item['status']} (events {item.get('events', 0)}, requests {item.get('request_logs', 0)})", flush=True)
    print(f"Listed {len(targets)} versions; {len(remote)} streamed runs; {len(recovered)} recovered request turns.", flush=True)


if __name__ == "__main__":
    main()
