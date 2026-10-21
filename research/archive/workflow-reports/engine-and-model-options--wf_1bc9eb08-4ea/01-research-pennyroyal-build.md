# engine-and-model-options: research:pennyroyal-build

Workflow run `wf_1bc9eb08-4ea`, phase Research. Research Pennyroyal build optimization, alternative (pruned) models like Inkling Small, and alternative inference engines; verify and recommend

---

# pennyroyal-build: Pennyroyal's documented levers vs. what we run (read-only)

**Bottom line.** No newer Pennyroyal runtime exists.
- The latest release is `pennyroyal-v2.5.3`, published 2026-09-28 (`gh api repos/jpezzulli/sglang-rtxpro6000/releases`).
- Branch HEAD is `f9c4a89c` (2026-10-03). The four commits after d00d88ef are the WSL2 setup flag, docs and traffic-report scripts.
- Our bundle already pins d00d88ef (`build_bundle_pennyroyal.sh:29`), so every v2.4.0–v2.5.3 change (CHANGES.md:23-265) is already in the wheel.
- A rebuild of the same pin gains nothing.
- The remaining levers are env/flag toggles or runtime patches. Only a changed patch set, NIXL, or the NVMe reader forces a new wheelhouse.

## Optimizations (all citations under `competitors/sglang-rtxpro6000/` unless noted)

| Item | Measured gain, as written | W4A16 (Intel AutoRound), bs~10 | NVFP4 (RadixArk) at ~10 streams | Rebuild? |
|---|---|---|---|---|
| **Online MXFP8** `SGLANG_SM120_ONLINE_MXFP8=true` (v2.5.0; FP8.md:8-13, RESULTS.md:53-70) | C1 short 161.465→207.124 (+28.28%). 128K C1 154.703→195.634 (+26.46%). 490K C1 149.135→172.644 (+15.76%). 128K C4 367.127→422.037 (+14.96%). 490K C4 +39.01% (unequal token totals). Free VRAM after graphs 3.66→7.52 GiB. | We forced it off (`launch_sglang.py:275`, bundle `runtime_env.sh:1231`) after measuring decode −27% at full batch, 555 vs 765 tok/s (arms.md:92). | They never measured above C4. Independent [issue #4](https://github.com/jpezzulli/sglang-rtxpro6000/issues/4) (v2.5.0, RadixArk, 4 streams): 291.0→224.5 tok/s (−22.9%). It also reports FR-Spec accept 1.00 at bs1-2; the maintainer could not reproduce it. | No for the toggle. The M>32 cap (`sm120_online_fp8.py:20,331`) would be a runtime `.py` patch; it is the hypothesis from the prior research, unproven. |
| **HiCache host tier, no NIXL** (v2.5.1 fix for issue #3; CHANGES.md:107,113-116; RUN.md:262-270) | "0/9 to 9/9" prefix restores; median revisit 9.04 s→1.05 s (NVFP4, one request at a time, ~105K-token chats). | Our launcher has no `--enable-hierarchical-cache` (`launch_sglang.py:305-345`). Untested with the INT4 draft, Marlin and extra_buffer. Our 4.89M cache-miss tokens are not eviction-driven (prior research), so the ceiling is small. Host RAM cost is 32 GB default plus ~48 GiB PLE (README.md:33-36). | Their qualified path for decode and restore at C4; not measured at 10. | No, flags only. The bundle omits NIXL (`build_bundle:1354`), and the disk tier is not needed. |
| **NVMe PLE** (v2.5.0; NVME-PLE.md:138-141) | C4 median RAM 428.90 vs NVMe 369.73 (−13.8%). Community: C2 349→323, C4 411→403, 128K C1 176→164 (COMMUNITY-RESULTS.md:238-250). Saves 47.68 GiB RAM. | Only worth it if RAM-bound. Overlay and table-format checks are written for the 47.68 GiB FP8 table; our BF16-PLE / albucino compatibility is unverified. | Same speed cost. | Separate Rust reader build (`tools/ple_nvme/install.sh`) plus a 48 GiB overlay. Not an SGLang rebuild, but a new bundle component. |
| **FR-Spec map** (v2.3; RESULTS.md:277-280) | +9.7–16.8% C1, +4.5–7.0% C4. | Already used (`hot_tokens_64k.pt`). untcoder2 found full-vocab drafting beat the map on RU/EN prompts (COMMUNITY-RESULTS.md:394), so A/B only. | Same. | No (data only). |
| **v2.4.0 prefill, v2.4.1/2/3 fixes** (RESULTS.md:223-226; CHANGES.md:200,248) | 64K 13,506→14,842 tok/s (+9.90%). 490K 8,464→8,773 (+3.65%). v2.4.1 PLE kernel: "No net serving-speed gain". | Already in d00d88ef and quant-independent. No action. | Same. | n/a |
| **`SGLANG_FORWARD_UNKNOWN_TOOLS=true`** (v2.5.2; their launcher `serve-flash-next-frspec.sh:8`; default False at `environ.py:686`, used at `qwen3_coder_detector.py:120`) | Correctness, not speed: unknown-tool calls reach the client for error recovery. | Not set in our launcher (grep). Relevant to python-tool recovery. | Same. | No (env). |
| **Reasoning effort default "medium"** (PR #18; `reasoning-effort.sh`; template `froggeric-v22.5.jinja:17-27`) | 27B only: medium cut completion tokens 80.4% and summed request time 84.74% vs xhigh; reasoning 95.807 vs 98.26 (RESULTS.md:583-591). | Needs our chat template to honor `reasoning_effort` (unverified). Quality trade-off. | Same. | No (config). |
| **Swift (ukisai) fine-tune** (SWIFT-27B.md; issue #21) | 27B: generated tokens 43,599→38,205 (−12%), suite time −6.5%. | No W4A16 AutoRound of Swift Flash-Next is known (unverified). | Issue #21: loads with BF16 PLE; PP/TG "+/-3%" vs Radix. | No (model swap). |
| **Larger KV cap / C6 recipe** (CHANGES.md:110,121-122) | Capacity only: 1,034,176 fitted, community 1,114,304. | We already have 1,011,264. Skip. | n/a | No |

Not applicable:
- **GPU clocks:** the RESULTS.md:363-371 profile (power cap 450, core/mem offsets) needs driver control. I did not verify whether Kaggle allows it.
- **NIXL disk persistence and 27B/DFlash2:** C1 108 tok/s, C4 375 tok/s (README.md:175-181); not useful for us.
- **Overlap plan stream:** Pennyroyal warns against `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`. It broke native MTP until removed (COMMUNITY-RESULTS.md:318, RESULTS.md:440). The earlier launcher review proposed it as an A/B, so treat that arm as risky.

## Practical wheelhouse build without the laptop

**What I found**
- **Target image is private.** It is `gcr.io/kaggle-private-byod/python@sha256:57e612b4…`, machine shape NvidiaRtxPro6000, internet off (`daniel-franzen-sol/kaggle/template-metadata.json`).
- **The run log shows Python 3.12 and nvcc 13.0.88.** Python 3.12 is at `fz-arms.log:142` and nvcc 13.0.88 (pip overlay) at `:147-151`. Venv plus wheelhouse install took ~3.5 s (`:144-146`).
- **The builder needs no Docker and no GPU.** It needs python, git, curl, gcc, g++, make and pkg-config (`:46-48`), plus internet.
  - It fetches rustup (`:1484`), PyPI, pytorch cu130, `docs.sglang.ai/whl/cu130` and `pypi.nvidia.com` (`:1521-1524`), and `flashinfer.ai` (`:1545-1547`).
  - It compiles only SGLang's Rust extensions (`:1488-1494`). torch, sglang-kernel and FlashInfer are prebuilt wheels.
- **The only compatibility gate is `check_platform`** (`:1049-1060`). CPython major.minor, SOABI and arch must be equal, and runtime glibc must be ≥ build glibc.

**Steps**
1. Open a private Kaggle notebook with Internet ON (CPU or any GPU). Verify `python3 --version` is 3.12 and note `ldd --version`.
2. Put the script in the notebook and run `OUT=/kaggle/working/out WORK=/tmp/penny BUILD_JOBS=4 bash build_bundle_pennyroyal.sh` in one `%%bash` cell. On failure, resume with `RESUME_STAGE=` (`:24`).
3. Publish `bundle-pennyroyal` as a dataset, either with "Save Version → New Dataset from output" or `kaggle datasets create` using Kaggle Secrets credentials.
4. Prove ABI match in an offline notebook on the real image. Attach the dataset and run `install_offline.sh` (~4 s). It fails fast on a Python or glibc mismatch.
5. **Fallback if Kaggle's internet image does not match** (public `kaggle-gpu-images` may differ): run the same script on a GitHub Actions runner.
   - Use an ubuntu-22.04 container or runner with CPython 3.12, which has older glibc, so it is safe under the ≥ rule.
   - Upload with `kaggle datasets create`.
   - Pennyroyal's own CI builds the full image on ubuntu-24.04 with `timeout-minutes: 180` and frees disk first (`pennyroyal-container.yml:32,73-77`).

**Blockers**
- **Image tag:** the private digest cannot be pulled. The README says to prefer a fixed digest (`build_bundle:1286-1290`).
- **Internet:** Kaggle needs a phone-verified account. I did not verify whether internet can be enabled on the RTX Pro 6000 or competition-linked notebooks.
- **Disk and size:** unknown. It needs wheels plus a resolver venv plus an offline-check venv, roughly 25–40 GB scratch (my guess). The `/kaggle/working` cap is, from memory, about 20 GB. Check the size of the existing `dfranzen/pennyroyal-v253` dataset first.
- **nvcc and arch:** no GPU is needed. The sm_120 compile check runs at the end of the script. The JIT recompiles at first launch, as today. Pennyroyal qualified CUDA 13.3 and GCC 15 (BUILD.md:222-234); we already deviate (13.0.88, image GCC).
- **Hash guards:** patch helpers are pinned to d00d88ef with exact BEFORE/AFTER sha256, and there is `len(result) == 13` at `:1029`. Any new patch or revision needs the hash chain, README text and manifest edited.
- **Index drift:** a `docs.sglang.ai` or `flashinfer.ai` change could break the build. Keep `dfranzen/pennyroyal-v253` as the fallback.

**Effort estimate (mine, not measured for our script)**

| Task | Estimate |
|---|---|
| Compute | Pennyroyal's CI built the full image in 25–30 min: run 36368166760 in 27m28s (02:00:53→02:28:21), 36080605586 in 30m25s, 35156312504 in 25m28s. Our script also downloads and wheels every dependency and runs an offline reinstall, so budget 1–2 h. |
| Identical rebuild, human time | 2–3 h |
| Add one embedded patch to the builder | +1–2 h |
| Runtime patch, no rebuild | ~30 min via `SGLANG_RUNTIME_PATCHES` (`launch_sglang.py:199-220`, currently empty) |

**Recommendation:** do not rebuild. Test without a rebuild, in this order: `FORWARD_UNKNOWN_TOOLS=true`, a host-HiCache write-through arm, and reasoning effort. Revisit MXFP8 only with the cap-raise runtime patch, after the prior research's reproduce-at-bs≤4 step.

## Unverified
- Kaggle disk, RAM and internet limits for the RTX Pro 6000 shape.
- The glibc and Python of the private image beyond Python 3.12.
- Whether HiCache works with the INT4 g32 draft and Marlin.
- Our template honoring `reasoning_effort`.
- Existence of a W4A16 Swift Flash-Next checkpoint.
