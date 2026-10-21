"""The Kaggle notebook's setup_env, read from kaggle/template.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE_NB = Path(__file__).resolve().parents[2] / "kaggle" / "template.ipynb"


def notebook_env() -> dict[str, str]:
    """setup_env exactly as the Kaggle template builds it (priority scheduling on)."""
    cells = json.loads(TEMPLATE_NB.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells if "setup_env = {" in "".join(c["source"]))
    start = source.index("setup_env = {")
    end = source.index("os.environ.update({k: str(v) for k,v in setup_env.items()})")
    namespace = {"SERVER_BASE_URL": "http://127.0.0.1:9/v1", "SERVED_MODEL_NAME": "flashnext",
                 "USE_PRIORITY_SCHEDULING": True}
    exec(source[start:end], namespace)  # noqa: S102 - our own committed template
    return {k: str(v) for k, v in namespace["setup_env"].items()}
