# throughput-and-alt-models: build:model-profiles

Workflow run `wf_f7f94969-5fa`, phase Build. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

I built the per-model profiles refactor, three new profiles, a CPU checkpoint checker and the Humming bench on `arm/model-profiles`. Every existing arm and bench config still launches exactly as before. Nothing has run on a GPU and nothing was pushed or opened as a PR.

**Neither candidate model can be served this week.**
- **GLM-5.3-Flash:** not feasible (175.6 GB against 96 GB of VRAM, no engine support) and would cost throughput, not add it. I built no profile for it.
- **The friend's GSQ Coder GGUF:** not servable as published; it needs a re-export first.

The `qwen38fn-gsq` profile is ready for that export once its Kaggle path is filled in (it is "TBD" for now, so pushes are refused).

**Worktree:** `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/model-profiles`, off `arm/fix-6-cold-start` with `arm/serving-bench` merged in (merge commit `3db0db4`).

**Commits:**
- `e891a19` Builder refactor so tests can build any arm the way a push does. Control, coldstart and bench notebooks are byte-identical.
- `9519beb` Goldens: the exact `sglang serve` command, server env, runtime patches and CFG for 3 arms, 7 synthetic cases and bench C0–C7, recorded by `kaggle/tests/launch_harness.py`. The NVFP4 reference was recorded from arm/nvfp4's own builder.
- `38bbfb5` Refactor only: the builder's edits moved into the launcher, which is now split into `install`, `apply_runtime_patches`, `build_env`, `prepare_views`, `frspec_token_map`, `build_args` and `launch`. All goldens identical.
- `37dadb6` Profile loader plus `serving/profiles/qwen38fn-w4a16.json`, which the launcher also keeps inline. Goldens identical.
- `b9bbf9a` The builder takes inputs, kernel sources and the drafter from the profile. Also adds the arms.json `"profile"` key, `--profile` for benches, the context-window-plus-8Ki = CTX check, and the push refusal for TBD inputs.
- `2ab41cd` `qwen38fn-nvfp4` profile and `nvfp4` arm. They launch the same flags, env and patches as arm/nvfp4's reference.
- `e2bb3d8` `qwen38fn-gsq` slot, arms `gsq-panel` and `gsq`, and a config view that keeps Humming from misreading BF16 layers.
- `eb69ec0` `serving/tools/check_checkpoint.py`: checks a checkpoint on the CPU using only config, safetensors headers, tokenizer and template.
- `8e17f78` Bench C8 (Humming vs Marlin on the same weights), the `humming` bench (C0 then C8) and the `baseline` bench (C0 alone, for `--profile`).
- `57ecdab` Docs: `serving/profiles/README.md`, `serving/CUSTOM.md` section 6, `kaggle/README.md`, `README.md`, `serving/README.md`. `CONFIGURATION.md` is unchanged because no harness flag was added.

I also logged `model-profiles` under In progress in `arms.md`, and the GLM and GSQ-GGUF verdicts under Not pursuing.

**Tests:** I ran `kaggle/tests` under nice with the main repo's venv: 187 passed. New files are `test_launch_golden.py`, `test_launcher.py`, `test_profiles.py`, `test_build_profiles.py` and `test_check_checkpoint.py`.
- I fetched Intel's W4A16 tokenizer from Hugging Face; its SHA matches the FR-Spec gate's `06b95093…`.
- I ran the checker on the W4A16's real safetensors headers (fetched header-only). It estimates 68.50 GiB of GPU weights and the 1,011,264-token KV pool the W4A16 actually got on Kaggle (69.85 GiB loaded).

**Local builds** (none pushed): all six arms (control, panel-control, coldstart-save, nvfp4, gsq-panel, gsq) built at about 338 KB each. The benches smoke, levers and humming built, and baseline built with the w4a16, nvfp4 and gsq profiles (about 596 KB each). One unintended event: `--arm gsq --push` ran once by mistake; it stopped at the TBD refusal before calling Kaggle.

**Launch commands:** full commands with Kaggle paths are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/mp-check/launch_commands.txt`. Differences from control (W4A16, same command as today):

| Profile | What changes |
|---|---|
| **nvfp4** | `--quantization modelopt_fp4`, the RadixArk rope override via `--json-model-override-args`, draft quantization `unquant`. No draft path, draft KV dtype or draft MoE runner. |
| **gsq, INT4 export** | Same as control except the model path, which is TBD. |
| **gsq INT3 export / C8** | `--quantization humming` and `--model-path /tmp/sgl-intel/target-view`. The drafter stays on compressed-tensors. |

**Where I departed from the plan:**
- **No cherry-pick of `fee2a6c`.** NVFP4 is a profile checked against arm/nvfp4's reference. Cherry-picking would have added a temporary CFG key and changed the JIT-cache config hash. The only differences from the reference are the order of the speculative flags and that the unused drafter is no longer mounted.
- **C8 is not `args_set --quantization humming`.** Reading Pennyroyal's source, its Humming loader ignores AutoRound's `extra_config`, so it would read Intel's BF16 layers (attention, router, shared experts and so on) as INT4. The launcher now serves a config view that turns those 2340 entries into 12 skip patterns. C8 switches profile per bench config, so CFG and the JIT-cache hash stay shared with C0.
- **Builder generalization came before the NVFP4 commit,** because the NVFP4 gate needs the model inputs from the profile.
- **The checker compares against the measured pool, not CTX × MAXREQ.** The full demand (1.39M tokens) never fits even for today's W4A16.
- **No build or convert notebook.** SGLang and Humming are already in the wheelhouse, and the GSQ export happens on the friend's side.

**Limitations:**
- These paths are unverified until a GPU run: the Humming view and humming-kernels 0.1.12, `gptq_marlin`, and compressed-tensors on the main model. Humming also only reads a compressed-tensors group that targets `Linear`.
- `nvfp4`, `gsq` and `gsq-panel` run the control harness, not stack-v2.
- The checker picks `qwen38fn-gsq` for any AutoRound INT4 checkpoint; it lists the other matching profiles too.
- The fork-overview table in the README lives on `docs/fork-readme`, so it still needs a `model-profiles` tooling row.
- **Merge conflicts:**
  - Into `arm/stack-v2`, `gate11`, `fix-3` and `stub-dead`: only `kaggle/arms.json`; keep both sides.
  - Into `arm/throughput-stack`: `launch_sglang.py`, `build_notebook.py`, `bench.json`, `arms.json` and both READMEs. To resolve: put `CUDAGRAPH_BS_EXTRA=[]` and its assert in CFG and add `*cfg["CUDAGRAPH_BS_EXTRA"]` to `graph_bs` inside `build_args`. Combine their arm `runtime_patches`, `--replica` and `--gate-passed` with the `profile` key in `_arm_config` and `main`. Then rerun `launch_harness.py --write` and review the golden diff (every golden's `cfg` gains the new key).

**Run order when quota returns:**
1. `python kaggle/build_notebook.py --bench humming --push`
2. The friend runs `python serving/tools/check_checkpoint.py <export>` from a checkout of this branch, then sets `inputs.model` in `serving/profiles/qwen38fn-gsq.json`.
3. `--bench baseline --profile qwen38fn-gsq`, then `gsq-panel`, then `gsq`.
