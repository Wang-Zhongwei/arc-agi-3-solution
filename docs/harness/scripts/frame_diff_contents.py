"""What frame_diff returns, and how much of its displayed output is hashes.

usage: python -I frame_diff_contents.py RUN_DIR

1. Recomputes frame_diff with this checkout's code for every consecutive pair
   of action boards in RUN_DIR/artifacts/*_events.jsonl, and reports how often
   each category is non-empty, its entry count, and the share of str(diff)
   taken by hash fields.
2. Measures hash key/value text inside the python tool results of calls that
   call frame_diff in RUN_DIR/transcripts/*.txt (what the model was shown).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ARC3-Inference"))
from inference.utils.frame_diff import compute_frame_diff  # noqa: E402
from inference.utils.grid_utils import ARC_COLOR_CHARS  # noqa: E402
from inference.utils.segmentation import segment_layer  # noqa: E402

CATS = ["moved", "rotated", "appeared", "disappeared", "changed_color", "resized"]
HASH_KV = re.compile(r"""['"](?:hash|shape_hash|pose_hash|before_hash|after_hash)['"]:\s*['"][0-9a-f]+['"],?\s?""")
CALL = re.compile(r"^\[TOOL CALL: python\]\n<tool_call>\n<function=python>\n<parameter=code>\n(.*?)\n</parameter>",
                  re.S | re.M)
RESULT = re.compile(r"^\[TOOL RESULT: python\]\n(.*?)(?=^\[(?:MODEL RESPONSE META|THINKING|TOOL CALL: python|"
                    r"TOOL RESULT: python|ANALYZER STATUS|SYSTEM PROMPT|USER PROMPT)\]|\Z)", re.S | re.M)


def boards(path):
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if e.get("type") in ("initial", "action") and e.get("board"):
            yield e["board"]


def recompute(run):
    pairs = 0
    nonempty, entries, sizes = Counter(), Counter(), []
    hash_chars = total_chars = 0
    for path in sorted(Path(run, "artifacts").glob("*_events.jsonl")):
        prev = None
        for grid in boards(path):
            seg = segment_layer(grid, ARC_COLOR_CHARS)
            if prev is not None and len(prev[0]) == len(grid) and len(prev[0][0]) == len(grid[0]):
                d = compute_frame_diff(prev[0], grid, prev[1]["nodes"], seg["nodes"])
                if d["changed_cell_count"]:
                    pairs += 1
                    for c in CATS:
                        nonempty[c] += bool(d[c])
                        entries[c] += len(d[c])
                    text = str(d)
                    total_chars += len(text)
                    hash_chars += sum(len(m) for m in HASH_KV.findall(text))
                    sizes.append(len(text))
            prev = (grid, seg)
    print(f"action transitions with a changed cell: {pairs}")
    print(f"{'category':16}{'non-empty':>11}{'% trans':>9}{'entries':>9}{'per non-empty':>15}")
    for c in CATS:
        print(f"{c:16}{nonempty[c]:>11}{nonempty[c] / max(pairs, 1):>9.0%}{entries[c]:>9}"
              f"{entries[c] / max(nonempty[c], 1):>15.1f}")
    sizes.sort()
    if sizes:
        print(f"str(diff) chars: median {sizes[len(sizes) // 2]}, p90 {sizes[int(len(sizes) * .9)]}, "
              f"hash key/value share {hash_chars / max(total_chars, 1):.0%}")


def displayed(run):
    fd = re.compile(r"\bframe_diff\s*\(")
    calls = shown = 0
    hash_chars = total = 0
    for path in sorted(Path(run, "transcripts").glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        codes, results = CALL.findall(text), RESULT.findall(text)
        for code, res in zip(codes, results):
            if not fd.search(code):
                continue
            calls += 1
            h = sum(len(m) for m in HASH_KV.findall(res))
            shown += h > 0
            hash_chars += h
            total += len(res.rstrip())
    print(f"frame_diff calls {calls}; results showing a hash field {shown} ({shown / max(calls, 1):.0%}); "
          f"hash key/value chars {hash_chars} of {total} result chars ({hash_chars / max(total, 1):.0%})")


if __name__ == "__main__":
    recompute(sys.argv[1])
    displayed(sys.argv[1])
