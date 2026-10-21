# arc3-idea-sweep: verify-gap:1

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "gap-0",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 3,
   "effort_days": 0.5,
   "confidence": "medium",
   "first_step": "Edit ARC3-Inference/inference/agent/tool_agent.py _harness_template_kwargs() (~line 7225): read env ARC3_REASONING_EFFORT and set kwargs['reasoning_effort'] when no ladder rung is active. Then run a 20-minute tput-style smoke arm with ARC3_REASONING_EFFORT=low and measure mean output_len per request from serve.log ReqTimeStats (script at scratchpad/reqstats.py). Only run the full 121-min arm (x2) if mean tokens per request drops by at least 25%. Compare score vs cumulative generated tokens against control 46.25.",
   "reason": "KEEP, NARROWED to one arm: static reasoning_effort=low. The mechanism is verified. The raw HF chat_template.jinja is byte-identical to the Intel W4A16 template the server uses (diff run). Line 47 defaults effort to xhigh, lines 51-54 emit the xhigh or low sentence, and lines 59-60 and 80 place it at the start of the system message before '# Tools', so any mid-game change invalidates the whole prefix. Franzen's competition config (user's franzen/kaggle/template.ipynb cell 4) sets no ARC3_REASONING_EFFORT_LADDER and passes only preserve_thinking=true, so the model runs at xhigh for the whole game. Static effort is absent, and the ladder code (tool_agent.py:1778-1800, 7225-7233, 9412-9423) is present but off.\n\nMeasured evidence that tokens are the currency, from kaggle-outputs/fz-arms-v4-control/serve.log (2309 requests):\n- Decode is about 64,065 request-seconds against about 1,264 for initial prefill, so the run is about 91% decode and about 2% prefill. 93.6% of prompt tokens are cached.\n- Mean output is 1990 tokens, median 1182, p90 4644. The 31% of requests at or above 2048 tokens produce 70% of all generated tokens.\n- From benchmark.json, weighted level completions are about 52% at 60 min, 75% at 90 min and 100% at 121 min. Progress is still roughly linear at the end, so more turns per minute should convert to score.\n\nCorrections to the idea:\n(a) The truncation share is overstated. The 4% in the prompts.py:336-349 comment does not match the competition run: only 26 of 2309 requests (1.1%) hit the 12,288 cap, wasting 7% of generated tokens.\n(b) Drop the cache-safe ladder fix. The ladder is off, and 1.1% truncations times two re-prefills of about 8-12 s is negligible against a 2% prefill share.\n(c) Drop the temperature=1.0 factor. The user's temp06 arm showed no effect (45.92 vs 46.25), and arms.md notes that lower temperature raises MTP acceptance, so temperature 1.0 would cost throughput.\n(d) Skip the medium arm. arms.md and GB10-OVERNIGHT-RESULTS.md line 47 already show 'medium does not shorten thinking'. In the template, medium only removes the xhigh sentence. That test covered medium only, so low is the materially different untested variant.\n\nCounter-evidence the idea omitted: the Qwen3.8-Flash-Next model card says 'In multi-turn agentic tasks, lower reasoning effort does not always reduce overall task completion time... can lead to insufficient analysis, more failures, and repeated retries' (https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/README.md, line 348). The user's streams14 arm (+15% throughput with a smaller context) gave no gain. Risk is real, so this needs the A/B, but the upside is the largest cheap lever found: if low cuts tokens per request by 40%, that is about +60% turns, versus streams14's +15%. A related system-prompt-only, cache-safe flag, ARC3_PREFER_TOOL_CALLS, is also off in the competition config and untested by the user (arms.md); it is worth a second cheap arm.\n\nThe other cited URLs check out: the unsloth page lists the xhigh/medium/low levels, and SGLang issue 25536 exists (closed; thinking_budget not enforced on Qwen3.6)."
  },
  {
   "id": "gap-1",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "high",
   "first_step": "Optional 15-minute version, not a score lever: after setup_env in the notebook (or the user's build_notebook.py), assert EXPOSE_UNDO=='on', MULTIMODAL_CONTEXT=='current_grid', MULTIMODAL_UPSCALE=='10' and ARC3_HISTORY_ASSISTANT_TURNS=='150'. Print the effective ARC3_* env once, and diff the submitted notebook's setup_env against franzen/kaggle/template.ipynb cell 4.",
   "reason": "DROP: the code facts are real, but none of the listed traps is active in Franzen's competition config. Checked against the user's franzen/kaggle/template.ipynb cell 4:\n- EXPOSE_UNDO='on' (the code-level trap at action_names.py:57-71 is real but not triggered).\n- MULTIMODAL_CONTEXT='current_grid' and MULTIMODAL_UPSCALE='10'. The kaggle.py:199 default of 4 belongs to Tufa's vLLM setup path.\n- ARC3_HISTORY_ASSISTANT_TURNS=150 with ARC3_HISTORY_TURN_DRAIN=30 and ARC3_HISTORY_DRAIN_COALESCE=1, so the idea's 'defaults slide the window every turn' does not apply. At roughly 3-4K tokens per turn a 150-message cap never binds before the 116K token window does.\n- ARC3_PRIORITY_TAIL_FADE=1 with no ENDGAME variable, so there is no silent override.\n- The numeric-parse traps (solver.py:167, :215) are not set.\n\nTrap (6) is refuted by data. In fz-arms-v4-control/serve.log the maximum real prompt is 119,530 tokens (p99 117,357) against a 116K-plus-margin harness budget, so the 3.3 chars-per-token clamp is not causing early trims.\n\nThe lint is also largely redundant. The user's arms builder already records a fingerprint of each run in runs.json and refuses duplicate builds. The notebook prints the solver, the full SGLang command, and server_args (including chat template kwargs) in serve.log. The only gap is that the effective ARC3_* env is not echoed in the run log. The 20.87 vs 27.89 leaderboard gap is better explained by game-sample and run noise than by a misconfiguration this lint could catch.\n\nNo score upside is attached, so it is not a score lever."
  },
  {
   "id": "gap-2",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 2,
   "confidence": "medium",
   "first_step": "No replay rig. For serving questions, parse existing serve.log lines instead (ReqTimeStats gives input_len, cached_input_len, output_len and initial_prefill_elapsed; Decode batch gives accept len and gen throughput). The scratchpad/reqstats.py script already does this from any run's serve.log.",
   "reason": "DROP: the payoff is bounded by the user's own results, and part of the claimed 'absent' tooling already exists.\n\n(1) Metrics already exist. serve.log from every Kaggle run already carries per-request ReqTimeStats (input_len, cached_input_len, output_len, initial_prefill_elapsed, post_prefill_elapsed) plus per-batch 'accept len', 'accept rate' and 'gen throughput', because the launcher passes --enable-request-time-stats-logging. Mode A's TTFT, cache and accept-length metrics come free from any normal run.\n\n(2) Payoff is small. The serving candidates it would screen are the same class the user already tried: streams14 gave +15% throughput for +0.13 score (noise), and spec-adaptive, MXFP8, experts16, spec4 and nvfp4-KV were all negative or crashed (arms.md). experts8 is the only untested one, and experts16 cost about 17% throughput, so top-8 would gain perhaps 10% throughput at some quality loss. 'bf16 SSM state' in its list is already on (see gap-3).\n\n(3) Cost is understated. A GPU matching the RTX Pro 6000 means a Kaggle session anyway, with about 10 min of server startup (load_weight 387 s plus CUDA graph capture 145 s in serve.log). requests.jsonl is not retained in the user's kaggle-outputs, and each line carries the full history (mean prompt about 72K tokens plus inline base64 images), so a run is multi-GB. Teacher-forced prompt logprobs with spec decoding and the hybrid Mamba radix cache are unverified in the Pennyroyal fork. Realistic effort is 2 days, not 1, which is 40% of the remaining window.\n\n(4) A real 121-min arm already costs only 2 hours of Kaggle with the user's builder, and it caught experts16's -7.6 directly.\n\nThe cited code references check out (run.py:1989-1994 flag, tool_agent.py:4808-4880 schema). Expected score impact is about +0 to +1."
  },
  {
   "id": "gap-3",
   "keep": false,
   "status_in_franzen": "present-and-on",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "high",
   "first_step": "None: the SSM state is already bfloat16. If extra memory is wanted, KV pool headroom is the place to look (the pool of 1,011,264 tokens reaches 'full token usage' 0.99 in places), not the Mamba pool.",
   "reason": "DROP: the premise is false. The idea says nothing sets the SSM dtype, and that is wrong. The Kaggle launcher (user's franzen/kaggle/template.ipynb cell 12) has CFG SSM_DTYPE='bfloat16' and passes '--mamba-ssm-dtype bfloat16'. The real run's serve.log shows server_args with 'mamba_ssm_dtype': 'bfloat16' and 'max_mamba_cache_size': 60. serving/build_bundle_pennyroyal.sh builds the wheelhouse only, so the launch arguments are not there, which is why a grep of the bundle missed it.\n\nThe pool is also small. serve.log says 'Mamba Cache is allocated... ssm_state size: 3.22GB, conv_state 0.13GB' for 60 slots, about 54 MB per slot, already at the bf16 size. The idea's 153.9 MB fp32 and 78.4 MB bf16 per-slot figures come from the 27B geometry. The whole Mamba pool is about 3.4 GB. Peak 'mamba usage' in the control run is 0.67 (about 40 of 60 slots), so slots are not the binding constraint either.\n\nNo change is possible here. The 'rerun on the replay rig' follow-up is moot."
  }
 ]
}
```
