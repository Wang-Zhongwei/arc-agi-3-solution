"""Count python tool calls that mention pose_hash / shape_hash anywhere (nodes or frame_diff).

usage: python -I hash_mentions.py RUN_DIR...
"""
import glob
import json
import re
import sys

ARG = re.compile(r'^\s*"arguments":\s*(".*")\s*,?\s*$')

for run in sys.argv[1:]:
    calls = fd = 0
    c = {"pose_hash": 0, "shape_hash": 0, "shape_hash+frame_diff": 0}
    for path in glob.glob(run + "/transcripts/*.txt"):
        for line in open(path, encoding="utf-8", errors="replace"):
            m = ARG.match(line)
            if not m:
                continue
            try:
                code = json.loads(json.loads(m.group(1))).get("code")
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
            if not isinstance(code, str):
                continue
            calls += 1
            fd += "frame_diff" in code
            for f in ("pose_hash", "shape_hash"):
                c[f] += f in code
            c["shape_hash+frame_diff"] += "shape_hash" in code and "frame_diff" in code
    print(f"{run.split('kaggle/')[-1]}: calls={calls} frame_diff_calls={fd} {c}")
