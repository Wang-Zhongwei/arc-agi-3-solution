# How often the agent tried file and JSON I/O

Across six saved Kaggle versions (150 game transcripts), **26 of 13,098 Python
tool calls contained `open()`** (0.20%). Seventeen tool results reported
`NameError: name 'open' is not defined`. Only two calls contained `json.dump()`;
both also contained `open()` and failed there. No call contained `json.load()` or
`json.loads()`. The agent used `json.dumps()` in 104 calls, mostly to format
in-memory data for a compact printout; that is not file I/O.

The current and saved-run Python sandbox excludes `open` from `SAFE_BUILTINS`
([source](../../ARC3-Inference/inference/agent/python_tool_sandbox.py)), so the
confirmed errors are expected. The other nine `open()` occurrences cannot be
counted as successful I/O: some are in `if False` branches or code that did not
reach the call. We observed no successful file read or write through these APIs.

## Prompt change

The Python tool description now includes one sentence:

> Avoid file I/O: `open()` is unavailable, and relative files do not persist between Python calls.

Each Python call runs in a fresh temporary directory, which is deleted afterward.
There is no configuration toggle for file I/O. Allowed imports such as `pathlib`
can still access files; this sentence discourages file I/O rather than enforcing
a filesystem restriction. In-memory JSON formatting remains available.

Validation: reran the counting script on all six versions and reproduced the
table below. Python syntax and Git whitespace checks passed. No new gameplay run
or A/B comparison was performed, so any reduction in failed calls is unmeasured.

## Exact versions

All folders are under `ARC3-Inference/runs/kaggle/jonathanwang2022/`. A cell in
the API columns counts a Python tool call whose submitted code contains that
call, not the number of dynamic invocations inside a loop or helper function.

| Notebook/version | Games | Python calls | `open()` | Confirmed `open` errors | `json.load()` / `json.dump()` | `json.loads()` / `json.dumps()` |
|---|---:|---:|---:|---:|---:|---:|
| `new-baseline/v01` | 25 | 3,424 | 2 | 1 | 0 / 0 | 0 / 18 |
| `new-baseline/v02` | 25 | 604 | 0 | 0 | 0 / 0 | 0 / 16 |
| `new-baseline/v03` | 25 | 2,255 | 5 | 4 | 0 / 0 | 0 / 11 |
| `intermediate-frames/v01` | 25 | 2,350 | 6 | 4 | 0 / 0 | 0 / 16 |
| `animation-frames/v02` | 25 | 2,331 | 5 | 4 | 0 / 0 | 0 / 26 |
| `intermediate-animation-frames/v01` | 25 | 2,134 | 8 | 4 | 0 / 2 | 0 / 17 |
| **Total** | **150** | **13,098** | **26** | **17** | **0 / 2** | **0 / 104** |

In the combined-frame v1 run discussed here, the eight `open()` occurrences
appear in five games: `cn04` has four, while `ar25`, `m0r0`, `tr87`, and `tu93`
have one each. Four produced the confirmed `NameError`: twice in `cn04`, once
in `ar25`, and once in `m0r0`. For example, `cn04` submitted
`json.dump(..., open('/tmp/asm.json', 'w'))`; the [tool result](../../ARC3-Inference/runs/kaggle/jonathanwang2022/intermediate-animation-frames/v01/transcripts/cn04-2fe56bfb_p0.txt)
reports that `open` is undefined. The model spent a tool call preparing a file
write it could not perform. Other snippets used `open('/dev/null', ...) if False
else None`, which is present in code but never opens a file.

## Counting method

[`file_io_frequency.py`](scripts/file_io_frequency.py) extracts `[TOOL CALL:
python]` code from each saved transcript, parses it with Python's AST, and
counts direct `open` and `json` API calls. It counts each API at most once per
tool call; a separate scan of the tool results counts the exact `open`
`NameError`. Of 13,098 tool calls, 28 did not parse as Python; none of those
28 contained an `open(` or `json.` candidate. This is a frequency of **model-
submitted snippets**, not proof that every call site ran. The error count is
execution evidence, but snippets can execute an earlier game action before a
later file-access error.

Reproduce from the repository root:

```bash
python -I docs/harness/scripts/file_io_frequency.py \
  ARC3-Inference/runs/kaggle/jonathanwang2022/new-baseline/v01 \
  ARC3-Inference/runs/kaggle/jonathanwang2022/new-baseline/v02 \
  ARC3-Inference/runs/kaggle/jonathanwang2022/new-baseline/v03 \
  ARC3-Inference/runs/kaggle/jonathanwang2022/intermediate-frames/v01 \
  ARC3-Inference/runs/kaggle/jonathanwang2022/animation-frames/v02 \
  ARC3-Inference/runs/kaggle/jonathanwang2022/intermediate-animation-frames/v01
```
