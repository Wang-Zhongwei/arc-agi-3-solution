# gsq-coder-gguf-runner: build:runner-gguf

Workflow run `wf_0ae4547f-80a`, phase Build. Serve the ISTA GSQ-RCO Coder GGUF the way its Hugging Face repo documents (llama.cpp or Strata, both shards, 256 experts, mmproj), validate the harness contract locally, build staging/engine/test notebooks and the 2-hour final-run notebook

---

**Status:** the llama.cpp runner for the GSQ-RCO Coder GGUF is built, tested and committed on `arm/runner-gguf`. All seven push folders are generated and checked. Nothing has been pushed to Kaggle or GitHub and no PR is open; that is your step. Two things were not validated end to end (see "Not done") because the laptop ran down to 2% battery during the local runs.

**Worktree:** `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/runner-gguf`, branched from local `arm/runner-gsq` at `2674188`, tree clean.

**Commits (15, oldest first):**
- `d5c9e08` stage_gsq: name the staging notebooks per file. A refactor; a test checks the shard-1 notebooks still build exactly as they were pushed.
- `8d066f5` Add the GGUF pins for shard 2 and the mmproj (`serving/gguf/__init__.py` and the embedded template). The opt-in HF API test confirms all three sizes and sha256s at `5348543e`.
- `93e6e2c` build_gsq_stage: stage shard 2 and the mmproj.
- `fac990e` Add the llama-server build script for sm_120a. Every check result now goes into the manifest, and `--version` must show commit 9881906.
- `90050a6` Add the engine build notebook generator with --check-manifest.
- `ac730aa` Add smoke_gguf with parallel and overflow checks.
- `327d6fb` Add gguf_serve: place, install, launch, checks, metrics.
- `d3184c5` Recognise llama-server context-overflow errors (in `tool_agent.py`).
- `d92571e` Share the GGUF runner's test stand-ins.
- `472dc7b` Add the F001 smoke and final notebook builder for the GGUF runner.
- `0580e0c` Add gguf_report.
- `cf5168c` Launch llama-server at log verbosity 4. Found locally: without `-lv 4`, b11476 logs none of the model-load lines the serve-log check needs.
- `655ed46` smoke_gguf: count a call cut at max_tokens as a wrong answer. Found locally: llama-server returns the partial arguments.
- `0afd450` Add the local llama.cpp validation script and its fixtures.
- `d64ab57` Document the GGUF runner: README, `serving/README.md`, `serving/CUSTOM.md` §8, `serving/gguf/README.md`, `kaggle/README.md`, `CONFIGURATION.md`.

**Tests:** `kaggle/tests` gives 440 passed, 9 skipped (the existing 352 plus 88 new), run niced with `FABLE_F001_NOTEBOOK` set to the real F001.
- The new suites are `test_build_gguf_engine`, `test_gguf_serve`, `test_smoke_gguf`, `test_context_error`, `test_build_fable_gguf`, `test_gguf_report`, plus additions to the two staging suites.
- Cell 14 and `bring_up` run end to end in the tests against a fake llama-server.
- All seven notebooks pass `nbcheck.py`, are under 1 MB, and their metadata is valid.

**Local validation (Metal, a 0.8B model from the Coder's family, the Kaggle flags, the Coder's own template):**
- **L1:** builds, and `--version` shows commit 9881906.
- **L2:** both model files match their pinned sha256.
- **L3:** 2 slots of 16,384 tokens; all 25 layers offloaded; `/props` shows the pinned template hash and vision on; the rendered prompt contains "Reasoning effort is set to xhigh".
- **L4:** all 9 smoke checks pass. One stream runs at 140 tok/s, a 640 px frame costs exactly 402 tokens, and a too-long prompt gets the 400 that the patched matcher recognises.
- **L5, one 6-minute run of the real harness:**
  - sk48: 88 requests, 86 python calls executed, an image on every frame request, reasoning sent back.
  - ft09: the 0.8B model never called the tool, which is the model, not the server.
  - No 5xx errors, and the Fable layer applied.
- **L6:** fixtures captured. The local server and harness are stopped.

**Not done:**
- The overflow path was not exercised through the real harness. The first run's prompts stayed under 14.1K tokens (the slot is 16,384), and two reruns with a larger window were invalid because battery throttling cut decode from 140 to about 17 tok/s. Covered instead by L4 (a real llama-server 400 against the patched matcher) and the unit tests (the unpatched F001 matcher does not match it).
- The edited build script was not re-run in a container. It passes `bash -n`, its embedded Python compiles, and I ran the manifest step on a fake build.
- `arms.md` is not updated, since there are no results yet.

**The exact llama-server command (final run; the smoke run uses `-cram 8192`):**
```
/tmp/lcpp/llama-b11476-cu130-sm120a/llama-server.sh -m /tmp/gguf/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --mmproj /tmp/gguf/mmproj-Qwen3.8-Flash-Next-BF16.gguf --host 127.0.0.1 --port 8001 --alias flashnext -ngl all -fit off -fa on -ctk f16 -ctv f16 --lazy-mode auto -np 10 --no-kv-unified -c 1392640 -b 2048 -ub 512 --jinja --reasoning-format deepseek --min-p 0 --no-prefill-assistant --chat-template-kwargs {"preserve_thinking":true} -ctxcp 8 -cram 32768 --metrics -to 3600 -lv 4
```

**Push order** (every push needs your approval; folders are in the scratchpad):
1. Push `runner-gsq` to GitHub first: it is 3 commits ahead of origin and is the PR's base. Then push `arm/runner-gguf` and open the PR against `arm/runner-gsq`.
2. Push the CPU notebooks, engine first, since it fails within seconds on the wrong image: `push-gguf-engine-build`, then `push-gguf-stage-shard2-p1`, `-p2`, `-p3`, then `push-gguf-stage-mmproj-p1`. Do not re-push `gsq-coder-stage-p1..p3`.
3. When they show COMPLETE, fetch only the manifests and run:
   - `build_gguf_engine.py --check-manifest`
   - `build_gsq_stage.py --check-manifests --file shard2`
   - the same with `--file mmproj`
4. Push `push-gguf-smoke`, then run `kaggle/gguf_report.py` on its outputs. Every go/no-go check must pass.
5. Push `push-gguf-final` with no other GPU session running. Score it and compare with control's 46.25 using `make significance`.

**Expected minutes (estimates):**

| Notebook | Minutes |
|---|---|
| engine build | 20–35 (75-minute cap) |
| each shard-2 part | 5–10 |
| mmproj | 2–4 |
| smoke | 12–18 of startup + 15 of play, about 35 GPU minutes |
| final | about 15 of startup + 121 of play, about 2.4 GPU hours |

**Risks:**
- **Speed:** a projected 90–400 tok/s aggregate against SGLang's ~760, so fewer tokens per game.
- **Multi-slot stability:** b11476's 10-slot attention cache code is days old. Fallbacks: `-np 8`, then Strata.
- **Engine build image:** whether Kaggle runs the pinned image in a CPU session is untested. If it doesn't, the build exits 3 within seconds; rebuild with `--gpu`, or `--image latest --bundle-glibc`.
- **Model quality:** users report repeat loops with the Coder, and nothing here mitigates that.
- **Kernel sources:** the GPU notebooks attach 8; Kaggle's cap is unknown.
- **Shard 2 mount:** if the staged shard 2 fails, `--shard2-from clothespin` mounts a competitor's dataset holding the identical file (hash `316b46f3`).
- **Truncated tool calls** come back with partial arguments; the harness drops them as malformed.
- **One run, three changes:** model, engine and Fable layer all change at once, so the final run cannot say which made the difference.

Files are in `…/scratchpad`:
- `push-gguf-engine-build/`
- `push-gguf-stage-shard2-p1/`, `-p2/`, `-p3/`
- `push-gguf-stage-mmproj-p1/`
- `push-gguf-smoke/`
- `push-gguf-final/`
- `gguf-local/` (local run logs)
