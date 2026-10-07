# bp35 prompt branch with Lord Hansolo serving

v01 used ten active streams and ten server sequences; v02 raised both to fourteen.
The later versions of `jonathanwang2022/bp35-lordhansolo` differ as follows:

| Version | Source | Exposed modules | Generated-token cap per game | Public play budget |
| --- | --- | --- | --- | --- |
| v02 | `5222bc8` | Original 15 standard-library modules; no NumPy or NetworkX | 200,000 | 120 minutes |
| v03 | `565ad28` | Extra standard-library modules plus NumPy and NetworkX | 200,000 | 120 minutes |
| v05 (canceled) | `792813e`, including `565ad28` | Same modules as v03 | None | 10 minutes |
| v06 (canceled) | `be1939b` with the 10-minute/no-cap config override | Extra standard-library modules only; no NumPy or NetworkX | None | 10 minutes |
| v07 | `be1939b` with the 120-minute/no-cap config override | Extra standard-library modules only | None | 120 minutes |
| v08 | `565ad28` with the 120-minute/no-cap config override | Extra standard-library modules plus NumPy and NetworkX | None | 120 minutes |
| v09 | `194af38` | Same modules as v08; adds ordinary `open()`/`exec()`, workspace module imports, and persistent per-agent files | None | 120 minutes |
| v10 | `6303467` | Extra standard-library modules only; NumPy/NetworkX exposure, game priors, and workspace I/O removed | None | 120 minutes |
| v11 | `6303467` plus `configs/lordhansolo/v11.json` | Same as v10; four private waves and 8192 thinking tokens per gameplay request | None | 120 minutes |
| v12 | `6303467` plus config cleanup and thinking-budget handoff | Same as v11; removes duplicate settings and corrects the server image cap to 640 × 640 | None | 120 minutes |
| v13 | Harness `1b60bc3`, Lord serving from v12 | Original 15 standard-library modules and original animation/game-over prompts, before branch edits | None | 120 minutes |

v13 was accepted ([receipt](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v13/push_receipt.json)).
Its agent and utility source trees match `1b60bc3` exactly; the runtime config
matches v12, including Lord serving, 14 active streams, four private waves,
8192 thinking tokens, and the corrected 640 × 640 image cap. It was prepared in
`/private/tmp/arc3-bp35-v13-baseline` with [v13.json](../../../ARC3-Inference/configs/lordhansolo/v13.json).
Reproducing v13 requires that historical harness, not just its config. All 24
focused checks passed; the submitted run has not been evaluated.

v10 was accepted by Kaggle and its source snapshot verified against the launching
checkout ([receipt](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v10/push_receipt.json)).
It is submitted, not evaluated.

v11 was also accepted and its packaged config verified
([receipt](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v11/push_receipt.json)).
Public runs still use one wave; the total output cap remains 12288 tokens per request.

The extra standard-library modules are `builtins`, `glob`, `hashlib`, `importlib`,
`inspect`, `os`, `pathlib`, `pickle`, `sys`, `time`, `traceback`, and `types`.
v05 and v06 are therefore the NumPy/NetworkX-enabled and disabled variants,
respectively. v03 already exposed those packages.

Verified against the archived `src/`, `git_info.txt`, and `preamble.txt` in
[v02](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v02/),
[v03](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v03/),
[v05](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v05/), and
[v06](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v06/).
v06's source and `git_info.txt` identify `be1939b`; its preamble still names
the launching checkout's `792813e`.

v07 and v08 were accepted by Kaggle; their source snapshots and budgets were
verified against the respective checkouts. Receipts:
[v07](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v07/push_receipt.json),
[v08](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v08/push_receipt.json).
These are submitted runs, not evaluated results.

v09 adds brief I/O guidance and saved-file reminders in user prompts. Its initial
push reached Kaggle's two-session GPU limit; retrying the prepared bundle after
a two-hour wait succeeded on October 7, 2026. The source snapshot was verified
against `194af38`; the [receipt](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v09/push_receipt.json)
records version 9. It is submitted, not evaluated. All 14 focused workspace/package
tests passed, including file persistence after errors/timeouts, helper imports,
and separate workspaces; the two-agent `glob` isolation check also passed.

Config: [v11.json](../../../ARC3-Inference/configs/lordhansolo/v11.json).
Local cleanup after v11 removes duplicate analyzer and multimodal overrides from
`harness_env`. Both image rendering and the server image cap now use the same 10×
setting (409600 pixels). This cleanup was submitted as v12; the source and image
cap were verified ([receipt](../../../ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v12/push_receipt.json)).
The current branch retains the animation-region wording and history-revisit
guidance, with game priors, NumPy/NetworkX exposure, and persistent workspace I/O
removed through rebasing. The version records above describe the archived runs.
The serving source comes from `tonghuikang/daniel-franzen-arc-agi-3` snapshot
`f472820b24f85e4bfe7790c66bce7d6463b5d568`, using runtime
`lordhansolo/vllm-main-e975732-arc3`. Credit goes to Lord Hansolo and the upstream
authors.

Current public configuration: the 25 official games, one pass, one wave, priority
scheduling, fourteen active generation streams, a 120-minute play budget from
notebook start, and no per-game token cap.
Private: one pass of the live competition games, four waves, priority scheduling,
and fourteen active streams.
Freed game lanes admit the next game immediately; waves describe the lane
capacity, not a barrier between batches.

The private play budget ends 532 minutes after notebook start. Engine startup
is subtracted before dividing the remaining time between the four waves. The
server keeps Lord's 147072-token model window, three-token built-in MTP with
the 32k draft vocabulary, FP8 KV, BF16 Mamba state, pinned host embeddings,
and patched prefix-state retention. Server sequence capacity is fourteen to match
the harness admission gate. The agent keeps its 131072-token context setting,
12288-token output cap, sampling, tool settings, and memory behavior. The thinking
token budget is 8192 per gameplay request.

Prepare locally from `ARC3-Inference/`:

```sh
make kaggle-duck CONFIG_PATH=configs/lordhansolo/v11.json \
  KAGGLE_USERNAME=jonathanwang2022 KAGGLE_DRY_RUN=true \
  EXPERIMENT_DIR="$PWD/.cache/kaggle-prepared/bp35-lordhansolo"
```

The staged kernel is `kernel/taaf_kaggle_run.ipynb`; its source dataset snapshot
and setup/teardown commands live beside `kernel/`. These generated artifacts
are ignored by Git. A real launch still needs GPU startup and image/tool-call
validation; offline packaging does not establish serving speed or ARC score.

Offline checks passed: 29 serving/notebook and RunPod tests, Ruff on the changed
Python files and vendored serving modules, rendered notebook syntax, exact
runtime/model attachments, source snapshot equality, and preservation of the
complete harness settings. The private-wave tests cover both even and odd live
game counts and reject multiple waves with UCB.
