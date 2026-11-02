#!/usr/bin/env python3
"""Print one audited claim with its labels and the transcript text around it. Usage: show_claim.py ID [DATA_DIR]"""
import json
import pathlib
import sys

from audit_paths import DATA as DEFAULT_DATA

DATA = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DATA
r = next(x for x in json.load(open(DATA / "joined_v02.json")) if x["id"] == sys.argv[1])
for k in ("id", "game", "level", "step", "kind", "in_section", "shown", "next1", "claim_verdict", "next_role", "usefulness", "evidence"):
    print(f"{k}: {r.get(k)}")
for reader in ("same", "later"):
    x = r.get(reader) or {}
    print(f"{reader} reader: meaningful={x.get('meaningful')} kind={x.get('info_type')} reason={x.get('reason')}")
name = pathlib.Path(r["transcript"]).name
line = r["thinking_line"]
print(f"\ntranscripts/{name} thinking block starts at line {line}")
probe = r["raw"].lstrip(" -*#>•").split("\n")[0][:60]
text = list(open(DATA.parent / "transcripts" / name, encoding="utf-8", errors="replace"))
hit = next((i for i in range(line - 1, min(line + 400, len(text))) if probe in text[i]), None)
if hit is None:
    print("claim text not found in the block")
else:
    print(f"claim at line {hit + 1}:")
    print("".join(text[max(0, hit - 2) : hit + 4]))
