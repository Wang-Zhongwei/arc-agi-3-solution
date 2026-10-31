"""Per-arm copies of the exact Kaggle setup scripts with the B200 edits.

    python3 make_arms.py RUNS_DIR OUT_DIR    # RUNS_DIR = runs/kaggle/jonathanwang2022

A = dflash2-k7 v3 (the k7 v2 script with prefix_match_unit 16), B = dflash2-k3 v2, C = porting-advaith v6 (MTP-3).
The scripts come from each run's setup_commands.json, i.e. what Kaggle executed. Each arm gets its own port
and the KV pool its Kaggle run reported on the RTX PRO 6000, so the B200's larger memory does not change
prefix-cache eviction, and the copy stops before the main flow, which run_arm.py replays. Every edit must
match exactly once.
"""
import json
import sys
from pathlib import Path

GIB = 2**30
ARMS = {
    # arm: (run folder, port, Kaggle "Available KV cache memory" GiB, extra edits)
    "A": ("dflash2-k7/v02", 8001, 21.91, [("VLLM_PREFIX_MATCH_UNIT = 128\n", "VLLM_PREFIX_MATCH_UNIT = 16\n")]),
    "B": ("dflash2-k3/v02", 8002, 22.23, []),
    "C": ("porting-advaith/v06", 8003, 21.73, []),
}


def rendered_setup(run_dir: Path) -> str:
    command = json.loads((run_dir / "setup_commands.json").read_text())[0]
    return command.split("<<'PYSETUP'\n", 1)[1].rsplit("\nPYSETUP", 1)[0]


def main():
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for arm, (run, port, kv_gib, extra) in ARMS.items():
        text = rendered_setup(runs / run)
        edits = [
            ("VLLM_PORT = 8001\n", f"VLLM_PORT = {port}\n"),
            ("VLLM_KV_CACHE_MEMORY_BYTES = 0\n", f"VLLM_KV_CACHE_MEMORY_BYTES = {round(kv_gib * GIB)}\n"),
            *extra,
        ]
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{arm}: {old!r} occurs {text.count(old)} times in {run}")
            text = text.replace(old, new)
        (out / f"setup_{arm}.py").write_text(text[: text.index("print(f'vLLM runtime dataset path")])
        print(arm, run, port, kv_gib, len(edits), "edits")


if __name__ == "__main__":
    main()
