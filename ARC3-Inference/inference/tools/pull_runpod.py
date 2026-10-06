"""Fetch a RunPod run's outputs into its run folder.

Takes the run folder `make runpod-duck` created (runs/runpod/<owner>/<slug>/<pod-id>/).
While the pod is still up it reports the pod state and exits 1; once the pod
has deleted itself, it downloads the private Kaggle output dataset the pod
uploaded and unpacks /kaggle/working into the run folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import taaf.deploy
import taaf.deploy_runpod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run folder holding deploy_meta.json.")
    parser.add_argument("--wait", action="store_true", help="Poll until the pod is gone, then download.")
    parser.add_argument("--stop", action="store_true", help="Delete the pod now (outputs not uploaded are lost).")
    args = parser.parse_args(argv)

    handle = taaf.deploy.DeploymentHandle.attach(args.run_dir.resolve())
    if not isinstance(handle, taaf.deploy_runpod.RunPodHandle):
        parser.error(f"{args.run_dir} is not a RunPod run ({type(handle).__name__}).")
    if args.stop:
        handle.stop()
        print(f"Deleted pod {handle.pod_id}.")
        return 0
    if args.wait:
        handle.wait()
    elif not handle.is_done:
        pod = taaf.deploy_runpod.get_pod(handle.pod_id) or {}
        print(f"Pod {handle.pod_id} is still {pod.get('desiredStatus', 'running')}; outputs are not uploaded yet.")
        return 1
    else:
        taaf.deploy_runpod.pull_output(handle.output_dataset, handle.job_dir)
    print(f"Outputs from {handle.output_dataset} are in {handle.job_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
