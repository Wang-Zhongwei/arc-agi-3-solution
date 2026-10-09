"""Run a serving option's settings handoff offline.

Every Kaggle setup command ends the same way: it writes the settings the harness
process starts with to TAAF_KAGGLE_SETUP_ENV. These helpers render the real
command for the configured `kaggle.serving` and execute only that handoff, so a
test sees exactly what a Kaggle run would receive, without installing or
starting a server.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_exports(config_path: Path, keys: list[str], root: Path = ROOT) -> dict[str, str]:
    """The values Make exports for `keys` when run with this config."""
    code = f"import json, os; print(json.dumps({{k: os.environ[k] for k in {keys!r} if k in os.environ}}))"
    recipe = "print-test-env:\n\t@" + shlex.join([sys.executable, "-c", code]) + "\n"
    result = subprocess.run(
        ["make", "--no-print-directory", "-s", "-f", "Makefile", "-f", "-",
         f"CONFIG_PATH={config_path}", f"PYTHON={sys.executable}", "print-test-env"],
        cwd=root, input=recipe, text=True, capture_output=True, check=True,
        env={key: value for key, value in os.environ.items() if key not in keys},
    )
    return json.loads(result.stdout)


@contextmanager
def _environ(values: dict[str, str]):
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _heredoc_body(command: str) -> str:
    return command.split("\n", 1)[1].rsplit("\nPYSETUP", 1)[0]


def handoff(tmp_path: Path, root: Path = ROOT, namespace: dict | None = None) -> dict[str, str]:
    """Settings the harness starts with, for the serving ARC3_KAGGLE_JSON selects.

    `namespace`, if given, receives the executed launcher globals (SPEC, CFG, ...).
    """
    from inference.framework import kaggle

    setup_env = tmp_path / "setup_env.json"
    setup_env.unlink(missing_ok=True)
    bundle_vocab = tmp_path / "src" / root.name / "configs/draft_vocab_32k.json"
    bundle_vocab.parent.mkdir(parents=True, exist_ok=True)
    bundle_vocab.write_bytes((root / "configs/draft_vocab_32k.json").read_bytes())
    paths = {
        "TAAF_KAGGLE_WORKING_DIR": str(tmp_path),
        "TAAF_KAGGLE_BUNDLE_DIR": str(tmp_path),
        "TAAF_KAGGLE_SETUP_ENV": str(setup_env),
    }
    with _environ(paths):
        serving = kaggle.kaggle_serving_config()["serving"]
        namespace = {} if namespace is None else namespace
        if serving == "pennyroyal":
            assignment = shlex.split(kaggle.pennyroyal_setup_command())[0]
            spec = assignment.split("=", 1)[1]
            launcher = (root / "inference/framework/pennyroyal_launch.py").read_text()
            head = launcher.split("\ndef _dataset_dir", 1)[0]
            tail = launcher[launcher.index("# ---- hand the endpoint"):]
            with _environ({"ARC3_PENNYROYAL_SPEC": spec}):
                exec(head, namespace)
                exec(tail, namespace)
        else:
            if serving == "lordhansolo":
                source = _heredoc_body(kaggle.lordhansolo_setup_command())
                head = source.split("print(f'vLLM runtime dataset path:", 1)[0]
            else:
                source = _heredoc_body(kaggle.duck_kaggle_setup_command())
                head = source.split("print(f'vLLM wheelhouse path:", 1)[0]
            exec(head, namespace)
            exec(source[source.index("\nsetup_env = {") + 1:], namespace)
    return json.loads(setup_env.read_text())
