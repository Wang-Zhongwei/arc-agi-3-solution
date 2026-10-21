# verify-prs-for-merge: verify:1-docstrings

Workflow run `wf_4b117a88-1cb`, phase Verify. Adversarially verify which open PRs are certainly correct to merge (builder, ledger-on-trim, docstrings, fork README, #4/#5 fixes) before merging

---

```json
{
 "pr": "#1 docs/inference-docstrings -> main (head 297b9b0, base 10882e3)",
 "certain": true,
 "merge_cleanly": "yes. It merges into the current origin/main (615944f) with 0 conflicts. Since 10882e3, main has only changed research/, runs/ and 2 lines of README.md, and PR #1 touches none of those.",
 "conditions": "CORRECTNESS: certain. Behaviour is preserved and no executed code changed. The one runtime-visible difference is a harness line number in a rare fallback traceback (see risks). It is not literally \"zero runtime effect\". I judge that cosmetic, but if the user's rule is read as zero runtime effect, it narrowly fails. CONFLICT COST: high, so I would not merge it now. Without PR #1, all 33 origin/arm/* branches merge into main cleanly (0 conflicts each, checked with git merge-tree). With PR #1 on main, 22 of the 33 gain 1 to 4 conflicted files. arm/stack-v2: 2 files, 4 hunks, about 89 lines inside conflict regions (python_tool_sandbox.py 3 hunks / 83 lines; tool_agent.py 1 hunk / 6 lines). arm/memory-v2: 2 files, 6 hunks, about 127 lines (python_tool_sandbox.py 3 / 83; tool_agent.py 3 / 44). These conflicts sit in behaviour-critical code: the fix-4 deadline loop in run_sandboxed_python (wall_cap and host_seconds), _state_hash, the memory-v2 turn-line injection, and the WALL_CAP_FACTOR import. In each hunk the HEAD (docs) side is only a docstring or comment, and the arm side carries code. Taking HEAD would silently drop arm behaviour, for example `message = None` / `if not isinstance(message, dict):`. Merge order doesn't help: merging the arm first and PR #1 second gives the same 2 conflicted files. Safe resolution rule, checked: take the arm side in every conflict hunk. For both stack-v2 and memory-v2, the result is AST-identical (docstrings stripped) to the arm head across all ARC3-Inference .py files (39 and 46 files). The only non-.py difference is README.md, which comes from main's own change. Recommendation: merge PR #1 after the arm stack lands, or rebase it then. Rebasing is cheap: re-take the arm side and update docstrings that the arms make stale, such as run_sandboxed_python's \"host handler time counts against the timeout\", which fix-4 changes. If it is merged now, every arm merge needs the take-arm-side rule.",
 "evidence": [
  "All 34 changed files are .py files with status M. There are no non-.py changes, no adds and no deletes.",
  "AST comparison of base 10882e3 vs head 297b9b0, with docstrings stripped from module, class and function bodies and attributes excluded, is identical for 34/34 files. Every file compiles on both sides.",
  "Under the project interpreter (venv Python 3.12.12, requires-python ==3.12.12), all 34 files plus the generated sandbox bootstrap compile with -W error::SyntaxWarning and -W error::DeprecationWarning.",
  "No code in ARC3-Inference reads __doc__ or calls inspect.getdoc, so no prompt or tool schema is built from docstrings. prompts.py string constants are AST-identical.",
  "The one runtime path that carries docs: python_tool_sandbox.py builds _SANDBOX_BOOTSTRAP by pasting inspect.getsource() of segmentation.py, frame_diff.py and retained_functions.py. The generated bootstrap was dumped from both trees and is AST-identical once all bare string statements are stripped. It grows from 1568 to 1938 lines and from 65,756 to 81,793 bytes. The injected modules contain no placeholder tokens that would break the .replace() chain.",
  "Live sandbox probe with run_sandboxed_python, main vs PR #1: user-code errors, SyntaxError via ast.parse, normal output and frame_diff.__doc__ give byte-identical results. Two fallback cases differ only in the harness line number: top-level `return` gives 'line 1525, in main' vs 'line 1895, in main', and repr of 10**5000 gives 'line 1529' vs 'line 1899'.",
  "Frequency of that fallback path in real runs: 0 of 249 tracebacks across 35 transcripts in runs/kaggle/v4-panel-control/v1 and v4-stack-v1/v1. Every one cites only <python_tool> user frames.",
  "Spot-checked docstring claims against the code, all accurate: CPU rlimit = timeout+1; ARC3_REPORT_GAMEPLAY_CHANGED defaults to False; LOCAL_ANALYZER_TOOL_CHOICE is lowercased; ARC3_STATE_IDENTITY_BORDER defaults to 0; state hash is SHA-1.",
  "Trial merges in detached temp worktrees: main+stack-v2 and main+memory-v2 are clean; (main+PR1)+stack-v2 has 2 files / 4 hunks; (main+PR1)+memory-v2 has 2 files / 6 hunks. git merge-tree over all origin/arm/* shows 22/33 branches go from 0 to 1-4 conflicted files.",
  "PR #49 (docs/fork-readme) and PR #1 don't conflict with each other.",
  "The only review on PR #1 is a Copilot quota-limit notice, so no human or tool has reviewed it."
 ],
 "risks": [
  "Not strictly zero runtime effect. When an exception has no <python_tool> frame (compile-time SyntaxError such as top-level return/break, or a repr/JSON failure in the harness), the model sees a harness line number shifted by up to +370. This can perturb a sampling trajectory. It has no systematic effect and wasn't observed in 249 real tracebacks.",
  "The PR description says the sandbox bootstrap string is byte-identical. That holds for the source literal only. The generated _SANDBOX_BOOTSTRAP grows by about 16 KB because the docstrings of the injected modules travel with it.",
  "Linux limits a single argv string to 131,072 bytes (MAX_ARG_STRLEN), and the bootstrap is passed via `python -c`. PR #1 cuts the headroom from about 65 KB to about 49 KB. Merged with memory-v2, the bootstrap is 86,694 bytes, still under the limit. Future arms that grow segmentation.py, frame_diff.py, retained_functions.py or the bootstrap now hit Popen E2BIG ('Sandbox process could not start') about 16 KB sooner. That failure is Kaggle-only and would not show on macOS.",
  "Merge-conflict cost: 22 of 33 arm branches gain conflicts in python_tool_sandbox.py, tool_agent.py and other files. The conflicts sit in behaviour-critical code (the fix-4 deadline loop, the state hash, the memory-v2 injection). Taking the HEAD (docs) side silently drops arm code. The rule 'take the arm side in every hunk' was verified only for stack-v2 and memory-v2.",
  "Some docstrings will go stale once the arms merge, for example run_sandboxed_python's timeout semantics after fix-4 and _state_hash after memory-v2. Stale docs can mislead future agents, though they don't affect runtime.",
  "Only a spot check of docstring accuracy was done (5 claims) out of about 9.3k added lines. Full factual accuracy of the docs is unverified."
 ]
}
```
