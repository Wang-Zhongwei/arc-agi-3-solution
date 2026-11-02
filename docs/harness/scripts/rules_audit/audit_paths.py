"""Where the audit data lives. Run artifacts stay out of Git, so the data sits in the Drive run folders.
Set RULES_AUDIT_DATA to read or write a copy instead."""
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNS = REPO / "ARC3-Inference/runs/kaggle/jonathanwang2022/main-tuned-drafter"
DATA = pathlib.Path(os.environ.get("RULES_AUDIT_DATA", RUNS / "v02/rules_audit"))
DATA_V01 = pathlib.Path(os.environ.get("RULES_AUDIT_DATA_V01", RUNS / "v01/rules_audit"))
