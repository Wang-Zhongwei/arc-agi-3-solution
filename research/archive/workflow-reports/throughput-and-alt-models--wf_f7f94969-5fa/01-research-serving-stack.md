# throughput-and-alt-models: research:serving-stack

Workflow run `wf_f7f94969-5fa`, phase Research. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

Refactor map: model-specific items in the serving + harness stack, as a per-model profile

Sources: launcher line numbers are from `arm/nvfp4`, `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/nvfp4/serving/kaggle/launch_sglang.py`. `arm/custom-serving` is the same file without the MODEL_FORMAT branches (its lines 35-50, 302-306, 339 and 351-367 are the only diff). Harness line numbers are from `main` at 10882e3, extracted read-only to `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/mainsrc/ARC3-Inference/inference/`. "env:N" means line N of `kaggle/template.ipynb` cell 4.

**0. Decisive constraints, before any refactor**
- Pennyroyal SGLang cannot load an EXL3 checkpoint. EXL3 is ExLlamaV3's own format. GLM-5.3-Flash-EXL3 therefore needs either a second engine (ExLlamaV3 + TabbyAPI, offline wheels) or a different quant of the same base model that SGLang can load (AutoRound W4A16, NVFP4 or FP8). I did not check SGLang's GGUF path, but nothing in our launcher or wheelhouse supports it. It accepts only `--load-format safetensors` (launcher:317), and qwen4exp GGUF is a llama.cpp arch (#27742). So qwen38fn-gguf most likely means llama.cpp `llama-server`. Prior research lists it as "llama.cpp (MTP bugs)" (arms.md:290; engine-model-research.md:333 has #28286 cross-slot contamination at `--parallel >1` and #29148 MTP load failure).
- Images are effectively required. arms.md:110 has ASCII-only 4.63 vs ASCII+image 9.859, and Franzen's biggest image gain was 10x upscale (WRITEUP:203). A profile must declare `vision`. A text-only model, or a GGUF with no mmproj file, means turning off env:104-108 and expecting a large loss.

**1. Profile shape** (proposed `serving/profiles/<name>.json`; the builder injects it the same way it injects `_SERVE_OVERRIDES` now)
```
{engine: sglang|llamacpp|exllamav3, vision: bool,
 inputs: {model:{dir,source}, draft:{dir,source}|null, wheelhouse:{dir,source}, extra_datasets:[]},
 validate: {quant_method, bits|substr}, args: {quantization, reasoning_parser, tool_call_parser,
   chat_template_kwargs, json_model_override, chat_template_override, arch_flags:[...]},
 spec: {algorithm, steps, draft_view: albucino|native|none, draft_quant}, frspec: {map_sha, tokenizer_sha}|null,
 cfg: {...CFG defaults}, serve_env: {...}, harness_env: {...setup_env overrides}, startup_timeout_s}
```
arms.json then takes `"profile": "<name>"`, replacing today's `"model"` key (arms.json:58-69, `nvfp4-stack`).

**2. Launcher items** (all on nvfp4 except the 2c FR-Spec ones)

2a. Checkpoint format
- :4-5, :35-38: MODEL_FORMAT comment and key. Becomes the profile name.
- :48, :339: `NVFP4_OVERRIDES`, the RadixArk mrope/yarn rope override. Goes to `args.json_model_override`.
- :49-50: format assert and `NVFP4` bool. Replaced by profile lookup.
- :51-52 and :101-156: `prepare_draft_view`, `EXPERT_TARGET`, `DENSE_IGNORE`. These are albucino-MTP-specific (the regex is `mtp.layers.0.mlp.experts`, compressed-tensors INT4 g32). Becomes `spec.draft_view=albucino`.
- :300-306: quant validation (`auto-round` with bits 4, or `"fp4"` substring). Goes to `validate`.
- :308: logs the PLE dtype from `text_config`, which is Flash-Next-specific.
- :309-312: draft dir assert, run only when SPEC and not NVFP4. Goes to `spec.draft_view`.
- :321: `--quantization` (auto-round or modelopt_fp4). Goes to `args.quantization`.
- :342-343: chat template taken from MODEL_DIR/chat_template.jinja. Add `args.chat_template_override` for a prefix-stable template (see harness :1141-1160).

2b. Qwen3.8-Flash-Next architecture flags, which go to `arch_flags` and the hybrid `cfg` keys
- :17 `MAMBA_CACHE`, :27-30 `KVDTYPE`/`SSM_DTYPE`/`MAMBA_RADIX`/`GDN_MTP_CACHE_MODE`, :26 `LINEAR_BACKEND`.
- :328-332: `--mamba-*` and `--linear-attn-*`.
- :333: `--ple-offload-embedding` (per-layer embeddings), `--trust-remote-code`.
- :334: `--mm-feature-transport cpu`, `--image-processor-backend pil` (vision models only).
- :345: `--gdn-mtp-cache-mode`.
- A full-attention model such as GLM must drop all of these.

2c. Tokenizer- and template-coupled
- :43-46: `TOKEN_MAP_SHA` becfa41d…, `TOKENIZER_SHA` 06b95093… (w4a16), `NVFP4_TOKENIZER_SHA` 0997f410…. These are the Qwen3.8 vocabulary.
- :358-367: FR-Spec map lookup and sha gates. Any other vocabulary needs `frspec=null` or a newly built hot-token map.
- :335: `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`.
- :336: `--default-chat-template-kwargs {"preserve_thinking":true}`.
- All three go to `args.*`.

2d. Speculative decoding
- :346-357: NEXTN with 3 steps. w4a16 uses a separate draft path with compressed-tensors quant and the draft KV dtype. nvfp4 uses the native head with draft quant `unquant`.
- :21 SPEC_STEPS=3 is capped by "Qwen QSA caps at 4 draft tokens" (CUSTOM.md:21).
- All of this goes to `spec`.

2e. Engine-bound: the whole install path assumes SGLang
- :170-174: exactly one `sglang-*.whl`, plus `requirements.lock`.
- :180-200: offline install with uv.
- :233-259: CUDA 13 from `nvidia/cu13` plus the libcuda symlink. Reusable by any engine.
- :267-289: env including the `SGLANG_SM120_*` knobs. `SGLANG_SM120_ONLINE_MXFP8` is "Flash-Next-only" per the bundle's environ patch.
- :317-338: `sglang serve` args.
- :383: `/health` URL.
- A second engine needs its own `install()`, `build_args()` and health URL behind an `engine` switch. Keep this block's CUDA setup shared.

2f. Builder anchors in `kaggle/build_notebook.py` that any rewrite must keep, or fold into the launcher, since it is our file now
- :111-112 `DRAFT_KV_OLD`, which must occur exactly once in the launcher text.
- :121-147 `ATTN_ARGS_ANCHOR` (adds `TEXT_CONFIG_OVERRIDE` and `SPEC_ADAPTIVE`, both SGLang-only).
- :151 `SERVER_DIED_OLD`, :159 `SERVE_ENV_ANCHOR`, :161 `SERVE_ANCHOR`.
- :192-204 wait-loop anchors.
- `_replace_once` fails the build when an anchor is missing (:262-266). Recommendation: move those edits into `launch_sglang.py` first, as a refactor-only commit with identical served args, so that an engine switch cannot break the anchors.

**3. Notebook, builder and metadata items**
- env:16 `SERVER_STARTUP_TIMEOUT=12*60`. Load is 472-637 s now, and a different engine or size changes it. Goes to `startup_timeout_s`.
- env:21-25: WHEELHOUSE_DIR, MODEL_DIR, DRAFT_MODEL_DIR, SERVED_MODEL_NAME 'flashnext'. The builder can only swap MODEL_DIR (`MODEL_DIR_LINE`, build_notebook:63, :321-323). WHEELHOUSE has no hook.
- build_notebook:104-107: `_resolve_input(DRAFT_MODEL_DIR)` raises if the albucino drafter is not mounted, so a profile cannot drop it today. Make the call conditional on `inputs.draft`.
- build_notebook:64, :398-400: `build_metadata` replaces only `W4A16_MODEL_SOURCE`. The drafter source (template-metadata.json:24) and `dfranzen/pennyroyal-v253` (:16) are never removed. Generalise this to `inputs.*.source` plus `extra_datasets`.
- Template cell 6, last line: `precache(MODEL_DIR, DRAFT_MODEL_DIR)` reads the drafter even when it is unused. Make it profile-driven.
- `kaggle/serve_report.py:33-37`: the regexes parse SGLang's "Decode batch … accept len" lines, so they are engine-specific. Other engines need their own parser or the `/metrics` endpoint.
- `serving-bench` replays a corpus (`kaggle/bench_corpus.json.gz`) of Qwen-format request messages with `reasoning_content` and `tool_calls`. It is reusable only if the other model's template accepts those keys.

**4. Harness (client) items** (most are already env knobs; profile puts them in `harness_env`)
- Reasoning echo:
  - env:72 `ARC3_REASONING_HISTORY_KEY='reasoning_content'` (code default `reasoning`, tool_agent:1254-1270). Previous reasoning is written back under these keys at :6906-6908.
  - The reply's reasoning is read from `reasoning`, falling back to `reasoning_content` (:3431; streaming at openai_compat:151-154).
  - Must match what the server parser emits and what the template reads.
- Thinking switch:
  - env:73 sets `enable_thinking`, which openai_compat:91-94 sends as `chat_template_kwargs` on the "vllm" provider path (env:65).
  - `preserve_thinking` comes from `ARC3_PRESERVE_THINKING` (default 1, :1317-1330), merged at :5200-5258.
  - `reasoning_effort` comes from the ladder (:1301-1315).
  - These are Qwen3.8 template kwargs. Preserve-thinking is what keeps the history prefix byte-stable (reuse 93%, arms.md:39). A GLM template uses different kwargs (GLM-4.7 calls it `clear_thinking`; check the 5.3 template). Profile key: `harness_env.ARC3_PRESERVE_THINKING` plus `args.chat_template_kwargs`.
- Summary-prefix note (:1141-1160): the Qwen template's `enable_thinking` conditional moves the system-prompt prefix (reuse 75% → 45%). Every new template needs the same check.
- Tool-call format:
  - The server parser `qwen3_coder` must emit OpenAI `tool_calls`.
  - The harness fallback `_TOOL_CALL_BLOCK_RE`/`_TOOL_CALL_PARAMETER_RE` (:85-91), used in `_recover_tool_calls_from_markup` (:140-170) at call site :6843-6849, matches only Qwen XML `<tool_call><function=…><parameter=…>`. A GLM-style `<arg_key>/<arg_value>` format would never be recovered.
  - `_render_tool_call_markup` (:3183-3201) is used only by the logs and transcript (:3289, :7068), so it is harmless.
  - Refactor: an env `ARC3_TOOL_MARKUP=qwen3_coder|glm|none` with one regex pair per format. `LOCAL_ANALYZER_TOOL_CHOICE` (:2502) covers servers without `auto`.
- Sampling: env:78-80 (0.7 / 0.95 / 20) are Qwen-recommended values; code defaults are at :201-204. Per-model.
- Context:
  - env:83-85 (window 128Ki, max output 12Ki, drain 58Ki) must satisfy launcher `CTX = window + 8Ki` (:13).
  - env:128 `ARC3_MAX_ACTIVE_STREAMS=10` must equal `MAXREQ` (:15).
  - Profile should derive both sides from one value.
- Token estimates:
  - Text: seed of 3 chars/token (:2862), calibrated from server `usage` (:2955, :6077-6100) and clamped by `ARC3_TEXT_TOKEN_CHARS_MAX=3.3`. That ceiling was fitted on Qwen's tokenizer (22 games, 2.75-3.69, :2898-2919). Re-fit per tokenizer.
  - Vision: `_VISION_PATCH_PIXELS=32` ("Qwen3.6: 16px patches, 2x2 merge"), `_VISION_SENTINEL_TOKENS=2`, `_IMAGE_TOKENS_FALLBACK=402` (:2856-2858). These are not env knobs. Only `ARC3_IMAGE_TOKENS_FLAT` (:2959) can override them. For a non-Qwen vision tower, set FLAT per profile or add a `ARC3_VISION_PATCH_PIXELS` env (a small code commit).
- Token-denominated tuning, all calibrated on this model's verbosity: env:76 `YIELD_TOKENS=2048`, env:91 `TOOL_OUTPUT_TOKENS=3072`, `ARC3_PRIORITY_TOKEN_SCALE=62000` (:2171), and `priority_pace_reference.json` (Kaplan-Meier token costs per level, :1968). Treat as a profile `token_scale` multiplier.
- Context-overflow detection (:3438-3449) matches SGLang and vLLM wording only. llama.cpp and TabbyAPI errors need their strings added.
- Stop tokens: none are sent by the harness, so EOS comes from the checkpoint's generation_config (draft view links it at launcher:145). With GGUF, EOS comes from the GGUF metadata.
- Degenerate-reasoning guard (:1665, :6919-6935) is model-agnostic.
- Not on our path: `inference/framework/kaggle.py:11-12,327-350` and `solver.py:1378-1496` (Tufa's vLLM Qwen3.6-27B notebook generator). Franzen's template unpickles `bm` instead.

**5. Profiles**
- `qwen38fn-w4a16` (current): Intel AutoRound plus the albucino view. FR-Spec 06b95093. Parsers qwen3 / qwen3_coder. `preserve_thinking`. All hybrid flags. `vision=true`. Harness env as in template cell 4.
- `qwen38fn-nvfp4`: `modelopt_fp4`, native NEXTN head, draft quant `unquant`. FR-Spec only when the tokenizer sha is 0997f410. `NVFP4_OVERRIDES`. Source `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1`. Harness unchanged. Drafter input should be dropped (blocked by build_notebook:106).
- `qwen38fn-gguf` (slot):
  - Engine llamacpp. Harness side ≈ current: same Qwen tokenizer and template family, but check whether the "Coder" fine-tune changed `chat_template`.
  - Needs: `llama-server` built for CUDA sm_120, `--jinja`, `--reasoning-format` emitting `reasoning_content`, `--parallel 10`, an mmproj file for vision (I have not checked the repo for one), and the overflow-error string.
  - No FR-Spec. MTP is unreliable (bugs above).
- `glm53-flash` (slot):
  - The EXL3 artifact needs engine exllamav3 (TabbyAPI), or a re-quant to SGLang-loadable safetensors with GLM parsers.
  - Everything in 2b and 2c flips: no mamba/GDN/PLE flags, a new vocabulary (FR-Spec off), GLM reasoning and tool parsers, a GLM tool-markup regex, different template kwargs, re-fitted char/token ceiling and vision patch size (if it has vision at all), a different MTP path, and a re-fitted token scale.
  - The "Ablit" (abliterated) weights are unmeasured on ARC.

**6. Wheelhouse and adding a new engine offline**
- `serving/build_bundle_pennyroyal.sh` must be run online in the Kaggle GPU image. It is pinned to Pennyroyal d00d88ef / 0.5.19+gd00d88efc8d6 (:29-31). Its steps:
  1. Applies five hash-chained source patches. `verify_bundle.py` enforces exactly 13 patched files (:1029) and exactly one SGLang wheel (:1034).
  2. Builds the SGLang wheel with Rust extensions (:1479-1495).
  3. Resolves dependencies in an empty venv against the pytorch cu130, sglang and nvidia indexes, with the cu13.0.88 toolkit pins (:1519-1543) and flashinfer-jit-cache/cubin 0.6.17 (:1544-1547).
  4. Locks every wheel to `packages.json` (the manifest asserts no extra or missing wheels, :1611-1619).
  5. Copies the FR-Spec map (:1576).
  6. Proves an offline reinstall plus an sm_120 nvcc compile (:1640-1653).
- So adding a wheel to this bundle means editing the lock, manifest and hash guards. Effort estimate from prior research is 1-2 h of compute plus 2-3 h of human time.
- Cheaper route: a separate engine dataset installed after the base install, at launcher :197-200: `uv pip install --no-index --find-links <extra>/wheels`, plus `--no-deps` for anything that would move torch.
  - ExLlamaV3 + TabbyAPI wheels must be built against the exact torch in Pennyroyal's `requirements.lock`, with `TORCH_CUDA_ARCH_LIST=12.0`, in the same image (Python 3.12, glibc).
  - llama.cpp needs no Python: bundle a `llama-server` binary plus `libggml*.so` built with CUDA 13.0 for sm_120, and reuse `pennyroyal-v253`'s cu13 libraries through the launcher's `LD_LIBRARY_PATH` (:273).
- Smallest commit sequence:
  1. Fold the builder's string-replace edits into the launcher (refactor only; must serve identical args).
  2. Profile loader, plus the two Qwen profiles, plus build_notebook input/metadata generalisation, including the optional drafter.
  3. Harness env knobs `ARC3_TOOL_MARKUP` and `ARC3_VISION_PATCH_PIXELS`.
  4. Engine switch plus `llamacpp`/`exllamav3` installers.
  5. Docs: CUSTOM.md §6 ("changing the base model is out of scope"), CONFIGURATION.md, kaggle/README.
