# throughput-and-alt-models: integrate

Workflow run `wf_f7f94969-5fa`, phase Integrate. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

Both arms are integrated, pushed and open as PRs (#68 and #69). Nothing ran on a GPU and nothing was pushed to Kaggle.

**PRs**
- https://github.com/adsdemaybe/daniel-franzen-sol/pull/68: `arm/model-profiles` into `arm/fix-6-cold-start`. Its diff also shows the serving-bench (#50) commits until #50 lands.
- https://github.com/adsdemaybe/daniel-franzen-sol/pull/69: `arm/throughput-stack` into `arm/stack-v2`.
- Both bodies quote the user's request and cover:
  - each candidate model's verdict, with links
  - what changed and the tests
  - the sim-v2 games-won estimate
  - the run order for when quota returns
  - "Not run: weekly GPU quota reached"
- Both end with the required attribution, and each links to the other.

**Suite results** (niced, one job at a time)
- **throughput-stack:**
  - Full `ARC3-Inference` suite: 69 passed. `kaggle/tests`: 125 passed (194 in total).
  - ruff finds only the 8 E702 errors in `inference/utils/frame_diff.py` that `arm/stack-v2` already has.
- **model-profiles:**
  - `kaggle/tests`: 193 passed.
  - The branch has no `ARC3-Inference/tests` and changes no harness code, so there was no harness suite to run.
  - ruff found two unused imports in the new test helpers. I removed them in one new commit, `eb7f340` ("Drop unused imports from the launcher test helpers"). The tests still pass and the F rules are now clean.

**Build results** (local only, never with `--push`)
- **throughput-stack:**
  - `tput-safe` (373,151 bytes), `tput-safe-lowm` (374,481), `tput-accept05` (373,216), `panel-tput-safe` (373,273), `tput-coldstart-save` (373,135), `stack-v2` (372,740), `control` (372,699).
  - Bench `tput-gates`: 639,449 bytes.
  - In every arm notebook the embedded harness patch equals `git diff bundle-base..HEAD -- ARC3-Inference`, and every arm env var is set.
  - The serve overrides carry `CUDAGRAPH_BS_EXTRA [3,5,6]`, plus 0.5/0.5 acceptance for `tput-accept05`.
  - Only `tput-safe-lowm` embeds runtime patches (m64.py and nosplitk.py), and they match `serving/bench_patches/` byte for byte. The two gated arms print their gates.
- **model-profiles:**
  - `control` (338,390 bytes), `panel-control` (338,557), `nvfp4` (338,474), `gsq-panel` (338,828), `gsq` (338,661). The two GSQ arms warn that their model input is TBD, so a push is refused.
  - Benches: `humming` 596,527 bytes, `baseline` 595,915 bytes.
  - Each notebook's harness patch equals the branch diff, and each mounts its profile's model and drafter. `nvfp4` mounts only the RadixArk checkpoint.

**Other pushes**
- I also pushed `arm/sim-v2` (`664b7e5..0200c0b`, three `throughput_sweep.py` commits) to its open PR #63. It was not one of the two listed arms, but both PRs and `arms.md` cite the sweep. Its tests pass (8).

**Doc changes**
- **`docs/fork-readme`:** pulled with rebase (already up to date). New commit `45e29c3` adds:
  - a Tooling row for model profiles (#68)
  - an Arms row for `gsq`/`gsq-panel` (#68)
  - the `tput-safe` row now points to #69 instead of "PR pending"
  - the sweep added to the `sim-v2` row
  
  Pushed together with the pending commit `4682b91`, as `77c4b1a..45e29c3`.
- **`/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md`:** backed up first to `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/arms.md.bak-tput`. Nothing was deleted.
  - The Goal line now gives the sizing: +2 offline games needs ×1.27 throughput; +2 of the 110 hidden games needs ×1.09.
  - **model-profiles** (In progress): now PR #68, with the sim-v2 estimate for the GSQ export and an integration note.
  - **throughput-stack** (In progress): now PR #69, with an integration note and the run order.
  - **Simulator v2:** new "Throughput sweep" sub-entry with the speed-to-games-won table. Over half of the offline gain falls on games no real run has won. It also has the gate11 simulation, the `tput-safe` and `tput-accept05` estimates, and the near-win analysis.
  - **Serving section:** Hugging Face links and PR #68 references on both model verdicts. I also added a correction: SGLang issue #37105 now has a posted workaround (TileLang with BF16 KV, shown on 8 GPUs). The GLM verdict stands, because the checkpoint still doesn't fit one card.

All worktrees are clean and in sync with origin. The PR body sources are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/integrate/` (`pr_model_profiles.md`, `pr_throughput_stack.md`).
