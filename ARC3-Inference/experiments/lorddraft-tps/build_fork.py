"""Build a Kaggle fork of matijaludvig/arc3-vcintel-lorddraft-25x1 v1 with one experimental change.

The base notebook is the exact v1 source saved by `make pull-kaggle ... SOURCE=true`. Each variant is a
list of exact-match edits; an edit that does not match exactly once stops the build, so a fork can only
differ from v1 in the lines listed here. Inputs (datasets, models, image, machine) are v1's.

    python build_fork.py <variant> <base.ipynb> <out_dir>
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = "jonathanwang2022"

# matijaludvig/arc3-vcintel-lorddraft-25x1 v1 kernel metadata (fetched 2026-10-09).
BASE_METADATA = {
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "dataset_sources": [
        "lordhansolo/vllm-main-e975732-arc3",
        "dfranzen/taaf-kaggle-source-bundle-copy",
    ],
    "kernel_sources": [],
    "model_sources": [
        "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
        "lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1",
    ],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c",
    "machine_shape": "NvidiaRtxPro6000",
}

HEADER_ANCHOR = "### VC-INTEL + lord MTP drafter:"


def games15_edits() -> list[tuple[str, str]]:
    return [
        (
            "    'ARC3_MAX_ACTIVE_STREAMS': 14,   # VCINTEL:",
            "    'ARC3_MAX_ACTIVE_STREAMS': 15,   # RUN B: one game more than vLLM max_num_seqs (14); VCINTEL:",
        ),
        (
            "assert int(os.environ['ARC3_MAX_ACTIVE_STREAMS']) == DHL_STREAMS, 'harness streams must equal vLLM max_num_seqs'",
            "assert int(os.environ['ARC3_MAX_ACTIVE_STREAMS']) == DHL_STREAMS + 1, 'RUN B: one harness game more than vLLM max_num_seqs'",
        ),
    ]


def retain1664_edits() -> list[tuple[str, str]]:
    # RecoverSSM's commit already writes the exact GDN state at every 1664-token Mamba block
    # boundary crossed during decode; retention 0 drops those states, so each next turn
    # re-prefills the previous completion. A 1664 interval keeps one state per block boundary.
    return [
        (
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL = 0\n",
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL = 1664  # RUN A: keep decode-crossed Mamba block checkpoints (was 0)\n",
        ),
    ]


VARIANTS = {
    "retain1664": {
        "slug": "lorddraft-retain-decode-checkpoints",
        "title": "lorddraft retain decode checkpoints",
        "header": (
            "### RUN A (jonathanwang2022): lorddraft v1 with --prefix-cache-retention-interval 1664, so the GDN "
            "states RecoverSSM writes at Mamba block boundaries during decode stay cached and the next turn "
            "resumes inside the previous completion. Only VLLM_PREFIX_CACHE_RETENTION_INTERVAL (0 -> 1664) "
            "differs from matijaludvig/arc3-vcintel-lorddraft-25x1 v1."
        ),
        "edits": retain1664_edits,
    },
    "games15": {
        "slug": "lorddraft-15-games-14-seqs",
        "title": "lorddraft 15 games 14 seqs",
        "header": (
            "### RUN B (jonathanwang2022): lorddraft v1 with 15 harness games and vLLM max_num_seqs 14. "
            "Only ARC3_MAX_ACTIVE_STREAMS (14 -> 15) and the equality assert differ from "
            "matijaludvig/arc3-vcintel-lorddraft-25x1 v1."
        ),
        "edits": games15_edits,
    },
}


def apply_edits(nb: dict, edits: list[tuple[str, str]]) -> None:
    for old, new in edits:
        hits = [(i, c) for i, c in enumerate(nb["cells"]) if old in "".join(c["source"])]
        total = sum("".join(c["source"]).count(old) for _, c in hits)
        if total != 1:
            raise SystemExit(f"edit matched {total} times, expected 1: {old[:90]!r}")
        i, cell = hits[0]
        src = "".join(cell["source"]).replace(old, new)
        cell["source"] = src.splitlines(keepends=True)


def main() -> None:
    variant, base, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    spec = VARIANTS[variant]
    nb = json.loads(base.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        cell.pop("outputs", None) if cell["cell_type"] != "code" else cell.update(outputs=[], execution_count=None)
    apply_edits(nb, [(HEADER_ANCHOR, spec["header"] + "\n\n" + HEADER_ANCHOR)] + spec["edits"]())
    out.mkdir(parents=True, exist_ok=True)
    name = spec["slug"] + ".ipynb"
    (out / name).write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    meta = {"id": f"{OWNER}/{spec['slug']}", "title": spec["title"], "code_file": name, **BASE_METADATA}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / name} and kernel-metadata.json for {meta['id']}")


if __name__ == "__main__":
    main()
