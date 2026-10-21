# arc3-bottleneck-audit: audit:perception

Workflow run `wf_4971e9bd-207`, phase Audit. Audit the whole Franzen harness (inference, serving, TAAF, notebook, configs) for bottlenecks; verify each; classify DEFINITELY vs MAYBE for GitHub issues

---

```json
{
 "findings": [
  {
   "title": "Resumption openers re-send an identical board image (18% of all images)",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::_resume_prompt_images_enabled (L908-915), ToolAgent.analyze resume_frame/opening_message (L6651-6694), _yield_resume_prompt_mode (L776-791); vision_context.py::current_grid_image_part (L93-101)",
   "resource": "tokens",
   "problem": "In the competition config LOCAL_ANALYZER_YIELD_TOKENS=2048 makes ~30% of analyzer invocations end with a token-budget yield. The resumed invocation rebuilds the FULL opener (ARC3_YIELD_RESUME_PROMPT unset -> 'full') and re-attaches the current-grid image (ARC3_RESUME_PROMPT_IMAGES defaults True). A resumption only happens when no action executed, so the board is bit-identical to the image already in the retained opener 3-5 messages earlier. ARC3_PRUNE_CONTROL_CONTEXT / ARC3_PRUNE_RESUME_PROMPTS are off, so these duplicates then stay in history until evicted.",
   "evidence": "fz-arms-v4-control transcripts (same notebook config): 466 of 1551 invocations end 'Yielded control to solver: turn_token_budget'; 462 resumption openers, each carrying exactly 1 image and ~2,544 chars (~700 tok) of restated opener text. Total images sent = 2,509, so 462 = 18.4% are duplicates. Each image = 640x640 -> 20x20 merged patches + 2 = 402 tokens (tool_agent.py L2856-2974). Duplicates therefore = 186K image tokens + ~320K text tokens = ~0.5M of 10.6M total new-prefill tokens (serve.log ReqTimeStats: input 166.4M, cached 155.7M, new 10.6M). ~4.4 of the ~24 images in an average retained history are such duplicates (~1.8K tokens).",
   "estimated_cost": "~1.8% of prefill tokens for the image alone (~4.8% with the repeated opener text); ~2.5% of average context; <1% wall-clock. Small but pure waste.",
   "fix": "Set ARC3_RESUME_PROMPT_IMAGES=0 (the code documents this exact purpose), or only attach the image on a resume when current_frame.grid differs from the last opener's grid. Optionally also ARC3_PRUNE_RESUME_PROMPTS=1. Caveat: WRITEUP says resume-prompt variants showed no consistent score gain, and the image-drop was only validated with 'short' mode; A/B it in 'full' mode.",
   "proposed_class": "DEFINITELY"
  },
  {
   "title": "Old board images are never dropped: ~24 images (~9.7K tok, 13%) per request",
   "location": "ARC3-Inference/inference/agent/tool_agent.py::_history_image_keep (L892-905), _apply_history_image_window (L944-960), _persistent_history_messages (L6313-6350, keep_images at L6317-6321)",
   "resource": "tokens",
   "problem": "ARC3_HISTORY_IMAGE_KEEP is unset in the notebook, so _history_image_keep() returns None and the window is never applied. Every retained opener keeps its 1-3 images (current + diff) for its whole life. They count against the context/drain budget at full vision cost (ARC3_IMAGE_TOKEN_ESTIMATE=True) and are re-prefilled at every drain, where the whole retained history changes prefix and is recomputed.",
   "evidence": "Reconstructed from transcripts (history_messages and messages per turn): mean 24.1 images in the retained window (median 24, max 57) = 9.7K tokens (max ~23K) vs mean prompt length 72K (serve.log) = 13.4%. serve.log ReqTimeStats: 173 of 2,311 requests have >=8K new tokens and carry 5.93M of the 10.6M new prefill tokens (56%); 92 of them are >=32K (drain re-prefills). Image share of those ~58K retained tokens = ~7.8K x 92 = ~0.7M tokens (~6.8% of all new prefill). WRITEUP L87: only the final prefill mamba checkpoint is kept, so a mid-history prefix change falls back to the system-prompt checkpoint (full re-prefill) - which is why per-turn windowing is unsafe.",
   "estimated_cost": "~0.7M prefill tokens (~7% of new tokens, ~1% wall via 8K-chunk decode stalls) and ~13% of context capacity that could hold ~13% more text turns before the next drain.",
   "fix": "Do NOT just set ARC3_HISTORY_IMAGE_KEEP (it would strip a mid-history message every turn and break the cache each turn). Instead apply _apply_history_image_window(messages, keep=K~4-6) only inside the drain commit (when token_trim_dropped / force_drain is true in _persistent_history_messages), where the prefix is already being invalidated, so it costs no extra cache. Needs an A/B for score since old diff images may help the model.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Two 640x640 images per opener = ~21% of non-drain prefill tokens",
   "location": "ARC3-Inference/inference/agent/vision_context.py::frame_to_png_data_url (L71-90), diff_image_part (L287-332), current_grid_image_upscale (L61-68); tool_agent.py::_build_user_message (L4660-4700), _prepare_auto_diff (L3954-4030)",
   "resource": "tokens",
   "problem": "MULTIMODAL_UPSCALE=10 renders 64x64 -> 640x640 = 402 vision tokens per image, and every post-action opener carries current image + diff image (ARC3_DIFF_IMAGE=1) = ~804 tokens plus labels. 10px cells straddle the 16px ViT patches (cells not patch-aligned); 8x (512px, 2 cells per patch) would be 258 tokens. diff_image_part is attached whenever any cell differs (L4014-4017), including HUD-only budget-bar ticks (the composite-image code dims the border for exactly this reason, but diff_image_part does not).",
   "evidence": "Openers: 625 with 1 image, 910 with 2, 17 with 3 -> 2,509 images x 402 = 1.0M tokens = 9.5% of all 10.6M new prefill tokens and 21.5% of the 4.68M new tokens in sub-8K (non-drain) requests. 912 diff images = 367K tokens. 305 'NO-OP (executed, no change inside the board area)' actions in 5,632 (5.4%) bound the HUD-only diff images at ~123K tokens. At 8x: saves 144 tok x 2,509 = 361K tokens (3.4% of prefill) and ~3.5K of the 9.7K average history image tokens (~4.8% of context). WRITEUP says 4x->10x was the biggest image gain; 8x untested on this model.",
   "estimated_cost": "Upper bound ~3-4% of prefill tokens and ~5% of context for 8x; ~1% for suppressing HUD-only diffs. <1% wall-clock.",
   "fix": "A/B MULTIMODAL_UPSCALE=8 (patch-aligned, -36% image tokens). Skip the diff image when the diff is confined to the ARC3_NOOP_GUARD_BORDER band (gameplay_changed False), mirroring the animation composite's border dimming.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Server re-decodes/preprocesses/hashes every history image on every request",
   "location": "Notebook cell 12 launch args (notebook_cells.py L10698: --mm-feature-transport cpu --image-processor-backend pil); cell 4 L10180 TRANSFORMERS_NO_TORCHVISION=1 inherited via env=dict(os.environ) (L10630); harness side vision_context.py::frame_to_png_data_url resends data URLs every request",
   "resource": "wall-clock-cpu-io",
   "problem": "Images travel inline as base64 in the history, so SGLang's tokenizer manager must decode, preprocess (slow numpy/PIL processor, since the fast processor needs torchvision and NO_TORCHVISION=1 is exported) and hash all ~24 history images on every request even though the KV prefix is cached. server_args in serve.log show enable_prefix_mm_cache=False, mm_preprocess_cache_size_mb=None, mm_processor_worker_num=0; startup log: 'Multimodal processor concurrency enabled with 2 isolated worker threads'. This latency is in front of the scheduler, so it is not in ReqTimeStats queue/prefill numbers.",
   "evidence": "2,311 logged requests x 24.1 avg images = ~56K image preprocesses per 25-game run (~245K for a 110-game scored rerun). A 640x640 image becomes 1,600 patches x 1,536 floats (~9.8 MB float32) to normalize and hash. My estimate (UNMEASURED) is ~30-70 ms per image, i.e. ~0.7-1.7 CPU-s per request. Throughput impact is bounded: decode throughput is 711 tok/s at 10 running vs 696 at 9 (serve.log), so a delayed stream costs ~2% at most.",
   "estimated_cost": "Unmeasured; likely hundreds of stream-seconds of added TTFT per run and 1-2 CPU cores of tokenizer-side work; <=2% throughput.",
   "fix": "Measure tokenizer-manager CPU/TTFT vs image count. Reduce images per request (findings 1-2), or enable SGLang's multimodal preprocess/feature cache if the pinned build supports it; install torchvision in the server env (or un-set TRANSFORMERS_NO_TORCHVISION for the server) to get the fast processor.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "segment_layer does a full-grid flood fill per component (O(n*4096)), recomputed every python call",
   "location": "ARC3-Inference/inference/utils/segmentation.py::segment_layer containment pass (L191-219, 'for b in range(n)' at L194, inner 'for a in range(n)' L214); python_tool_sandbox.py FrameView.segmentation (L167-173, lazy, per fresh subprocess); frame_diff() L365-381 segments both frames",
   "resource": "wall-clock-cpu-io",
   "problem": "For each of n components it floods the whole complement of that component from the border (4,096 cells x 4 neighbours in pure Python) to find enclosing parents, then scans all n components again. Cost is ~1 ms per component. Each python tool call starts a fresh `python -I -S` subprocess, so current_frame.segmentation (which the prompt tells the model to use as its primary view) is rebuilt from scratch every call, and frame_diff() builds it for two frames.",
   "evidence": "Benchmark (M-series, py3.12): 82 nodes 86 ms, 195 -> 192 ms, 376 -> 369 ms, 697 -> 696 ms, noisy 2,131 nodes 2.3 s (a 4,096-component dither frame ~4.5 s; x3 frames inside frame_diff+previous). Real frames in the run print 57-146 nodes (r11l 146/95/80, lp85 57-107, re86 72, lf52 77) = ~60-150 ms per access (Kaggle CPU likely 1.5-2x slower). 2,270 python calls in 25 games, 451 reference .segmentation directly (more via retained helpers) -> est. 100-300 s of sandbox CPU per run (<0.5% of 10 streams x 7,300 s). frame_diff hungarian is not an issue (400 identical moved dots: 147 ms). PNG pipeline (PIL resize + PNG + b64) measured 2-12 ms per image, animation _transient_cells ~1.6 ms per frame (median animation 13 frames, max 152): both negligible.",
   "estimated_cost": "<0.5% of stream time on typical frames; tail risk of a 30 s tool timeout (RLIMIT_CPU 31 s) on very noisy frames on a slow CPU.",
   "fix": "Compute parents in one pass from the adjacency graph (component touching the border => root; enclosure via BFS on the component adjacency tree), or restrict the per-component flood to its bounding box expanded by 1. Optionally pass nodes from the host (cache keyed by grid hash) instead of recomputing in each subprocess.",
   "proposed_class": "MAYBE"
  },
  {
   "title": "Entire game history (every frame) rewritten to disk and re-sent to each sandbox",
   "location": "ARC3-Inference/inference/agent/runtime_state.py::write_runtime_state (L127-140, json.dumps(indent=2) at L139), load_runtime_state (L113-124); framework/solver.py::_execute_action (L1342), _play_inner (L628, L669); tool_agent.py::_serialized_runtime_state (L5730-5765), _ascii_history_view_payload (L2843-2853)",
   "resource": "wall-clock-cpu-io",
   "problem": "history_entries is never trimmed (append at solver.py L1339). After every executed action and every analysis step the whole history (grid as 4,096 numbers per frame, indent=2 = one number per line) is rewritten to tool_runtime_state.json; every python call then re-parses it (L5727) and rebuilds ascii + grid for every history frame to ship to the sandbox subprocess, again after each action() inside a snippet (L5735). All of it holds the GIL of the single harness process that runs the 10 game threads.",
   "evidence": "Benchmark (py3.12): N=300 frames -> 19.1 MB file, 212 ms dump, 51 ms load; sandbox payload 5.1 MB, ~140 ms build+dumps+loads. Per-run action counts (summary.txt: 5,632 actions, wa30 545) give mean history depth ~152 frames. Estimated ~790 s of JSON writes + ~250 s of load/serialize host CPU per 25-game run (~15% of one core) and ~70 GB of rewritten data in /kaggle/working. Mean running requests is 8.9/10 (serve.log) so decode loss from client gaps is only ~2% - this is a scaling cost, not a current throughput limiter.",
   "estimated_cost": "~1,000 s host CPU per 25 games (~14% of one core, GIL-shared), growing with game length and ~4x for a 110-game rerun; ~1-2% of decode throughput at most.",
   "fix": "Keep only the last K (e.g. 8-16) history frames in the file and the sandbox payload (history[-k] is all the prompt documents), write compact JSON (no indent), or append incrementally; cache the parsed state between the load calls within one python call.",
   "proposed_class": "MAYBE"
  }
 ]
}
```
