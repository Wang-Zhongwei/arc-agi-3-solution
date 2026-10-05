# bp35 prompt branch with Lord Hansolo serving

v01 used ten active streams and ten server sequences; v02 raised both to fourteen.
The later versions of `jonathanwang2022/bp35-lordhansolo` differ as follows:

| Version | Source | Exposed modules | Generated-token cap per game | Public play budget |
| --- | --- | --- | --- | --- |
| v03 | `565ad28` | Extra standard-library modules plus NumPy and NetworkX | 200,000 | 120 minutes |
| v05 (canceled) | `792813e`, including `565ad28` | Same modules as v03 | None | 10 minutes |
| v06 (canceled) | `be1939b` with the 10-minute/no-cap config override | Extra standard-library modules only; no NumPy or NetworkX | None | 10 minutes |
| v07 | `be1939b` with the 120-minute/no-cap config override | Extra standard-library modules only | None | 120 minutes |
| v08 | `565ad28` with the 120-minute/no-cap config override | Extra standard-library modules plus NumPy and NetworkX | None | 120 minutes |
| v09 | `194af38` | Same modules as v08; adds ordinary `open()`/`exec()`, workspace module imports, and persistent per-agent files | None | 120 minutes |

The extra standard-library modules are `builtins`, `glob`, `hashlib`, `importlib`,
`inspect`, `os`, `pathlib`, `pickle`, `sys`, `time`, `traceback`, and `types`.
v05 and v06 are therefore the NumPy/NetworkX-enabled and disabled variants,
respectively. v03 already exposed those packages.

Verified against the archived `src/`, `git_info.txt`, and `preamble.txt` in
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

Config: [lordhansolo.json](../../../ARC3-Inference/configs/1006-prompt-cleaning-engine-swap/lordhansolo.json).
The current branch retains the animation-region wording and history-revisit
guidance, with game priors, NumPy/NetworkX exposure, and persistent workspace I/O
removed through rebasing. The version records above describe the archived runs.
The serving source is pinned as
recorded in [SOURCE.md](../../../ARC3-Inference/inference/framework/lordhansolo/SOURCE.md).

Current public configuration: the 25 official games, one pass, one wave, priority
scheduling, fourteen active generation streams, a 120-minute play budget from
notebook start, and no per-game token cap.
Private: one pass of the live competition games, two waves, priority scheduling,
and fourteen active streams. For 110 games, each wave has capacity for 55 games.
Freed game lanes admit the next game immediately; waves describe the lane
capacity, not a barrier between batches.

The private play budget ends 532 minutes after notebook start. Engine startup
is subtracted before dividing the remaining time between the two waves. The
server keeps Lord's 147072-token model window, three-token built-in MTP with
the 32k draft vocabulary, FP8 KV, BF16 Mamba state, pinned host embeddings,
and patched prefix-state retention. Server sequence capacity is fourteen to match
the harness admission gate. The agent keeps its 131072-token context setting,
12288-token output cap, sampling, tool settings, and memory behavior.

Prepare locally from `ARC3-Inference/`:

```sh
make kaggle-duck CONFIG_PATH=configs/1006-prompt-cleaning-engine-swap/lordhansolo.json \
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
