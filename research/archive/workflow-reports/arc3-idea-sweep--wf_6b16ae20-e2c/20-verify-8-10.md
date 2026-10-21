# arc3-idea-sweep: verify:8/10

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "level-inventory-flag",
   "keep": false,
   "reason": "Drop. I ran the real compute_frame_diff + _render_auto_frame_diff (tool_agent.py:3689, 6073-6146) on 33 real level-start pairs from franzen/example-run/artifacts/*_events.jsonl. START-vs-START boards differ by 175-3983 cells, so 'appeared' has 3-347 entries (typically 10-90) and 'moved' up to 55. The 300-token budget forces the aggregated rung, which lists wall 'resized' blobs and 1px fragments (e.g. ar25 L1->L2: 'appeared: 16x color W 1px', 'resized: color b 1845->2124px'). That is not a clean new-mechanic signal. The cited cost holds: median first-action tokens on levels >=2 is 4,775 in the control benchmark.json (idea said 3,853), and 19-22% of cleared-level tokens. But that is mostly necessary perception and planning that a noisy 150-290 token list will not compress. Franzen also decoupled the knob and left it off in the final notebook (template cell 4 env has no ARC3_LEVEL_INVENTORY; patch has default False). Expected effect is about zero with unknown sign, and a one-run offline test (noise about 3 points) cannot resolve it.",
   "status_in_franzen": "present-but-off",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "medium",
   "first_step": "No action. If forced: add 'ARC3_LEVEL_INVENTORY': '1' to setup_env in arc-agi-3-v3/franzen/kaggle/template.ipynb cell 4."
  },
  {
   "id": "budget-bar-readout",
   "keep": false,
   "reason": "Drop. Evidence is real but the prize is small. Control transcripts do contain 2,299 thinking blocks (my regex gives 460-1,160 mentioning bar/timer/budget, so 371 is plausible) and 19 'GAME OVER occurred' events across 25 games and 5,632 actions. Only about 8 of the 19 are model-diagnosed budget deaths (sc25, tn36, tu93, vc33, wa30 mentions), concentrated in tu93 (9) and sc25 (3). The model already gets the death-frame image, a fatal-step diff image (ARC3_GAMEOVER_DIFF_IMAGE=1) and the BAR RULE prompt. A readout can only avoid a fraction of about 8 level-restarts per 25-game run, worth well under 1 offline point. Scott Le Grand's 25/25 public-game survey (competitors/kaggle/taaf-flashnext-sheetu12b-0922/taaf-flashnext-sheetu12b-0922.txt:280-300) shows a bar is detectable. But a robust cross-game detector is more than half a day, and a wrong 'N actions left' line would mislead the model on private games. Thinking-token savings are small.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 1,
   "confidence": "medium",
   "first_step": "No action. If revisited, start with a text-only readout limited to the death frame (extend _game_over_diff_lines, tool_agent.py ~6217)."
  },
  {
   "id": "hud-band-mask",
   "keep": false,
   "reason": "Drop. The proposed pre-check was cheap, so I ran it. In the control run's transcripts there are 4,017 '[action]' echo lines. Only 19 end in 'stop_reason: no_op_action' (wa30 5, sp80 4, others 1-2) and 3 are stale_state. At most 22 guard events in the whole run could be border false-positives (about 0.5% of actions), and several are plainly SPACE/no-effect moves. Scott Le Grand's F5 (competitors/kaggle/taaf-flashnext-sheetu12b-0922/taaf-flashnext-sheetu12b-0922.txt:289-312, 380-420) is real, and the bar is 1-2 cells/action in 25/25 games, but the existing ARC3_NOOP_GUARD_BORDER=4 already masks that band. A learned-HUD detector is a 1-day build with an upside bounded by those 22 events, so impact is at most 1.",
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 1,
   "confidence": "high",
   "first_step": "No action."
  },
  {
   "id": "zoom-crops",
   "keep": false,
   "reason": "Drop. The evidence does not support the transfer. The VISTA page (vista-research.github.io, fetched) does describe a zoom/inspect tool and a read_pixels tool, and reports 100.00 RHAE for Claude Opus 5.0 and 98.27 for GPT-5.6 Sol. But those are frontier models with a different harness, and the page states no zoom ablation. In the Qwen control transcripts, 'zoom' appears in only 8 of 2,299 thinking blocks and 'hard to see/too small/blurry' in 82 (3.6%). The model already has exact pixel access via current_frame.ascii and segmentation. Wang-era upscale-12 + overlay arm scored 5.6 (confounded). WRITEUP.md:301 reports image variants (composite animation images) hurting some games. No zoom code exists (grep zoom: 0 hits). The small-change-crop-instead-of-diff variant saves only about 330 context tokens per opener (about 5-8% of new context per turn) and loses global position. Cost is 1 day for an effect that is at best noise.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 1,
   "confidence": "medium",
   "first_step": "No action."
  },
  {
   "id": "reshow-level-start-board",
   "keep": true,
   "reason": "Keep as a low-priority add-on to the user's #1 backlog item (port ledger-on-trim), not as a standalone arm. It is not implemented: _cur_level_start_frame (tool_agent.py:5215, 5757-5770) is used only by the off inventory. The 'scene card' in the idea does not exist in this repo, so only the start-board image part is real. The mechanism is plausible. 20% of level attempts exceed 60k generated tokens (levels.csv, 172 of 869), and those are the long, hard levels where the oldest turns, including the start layout, have been trimmed. There are about 3.6 trims per game (89 history drops over 25 control transcripts), so the cost is about 400 tokens per trim (about 1.4k tokens per game). The ledger evidence is verified (findings.md section 3: 52.56 vs 46.25 offline, 38.7k vs 42.7k tokens per level), but it supports re-showing model-written notes, not raw boards. It also rests on one run, so the user's own arms.md calls it the one clear win but unrepeated. Expected effect is small, and Franzen reported no consistent gain from resume-prompt reinsertion variants. Share the insertion point with the ledger port.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "low",
   "first_step": "In ARC3-Inference/inference/agent/tool_agent.py, _persistent_history_messages (~8723-8760): after _drop_until_first_user_message, when history != original_history, prepend one labelled user message ('reference: starting layout of the CURRENT level, not the current board') carrying current_grid_image_part(self._cur_level_start_frame). Gate behind a new env flag and implement it in the same commit as the ledger-on-trim port."
  },
  {
   "id": "level-aware-trim",
   "keep": false,
   "reason": "Drop. I ran the gating test the idea asked for: rellik13's analysis/sawtooth.py on all 7 Franzen-era transcript sets (9,829 turns). Pooled stratified clear-rate difference for more visible in-level history is +5.6 pp at W=8 and +3.6 pp at W=15, larger for k>=16 (0.385 vs 0.277 at W=8). So the gate passes, but only for 'more history'. It says nothing about keeping the first K turns instead of the middle. Affected turns are small: k>=16 is 27% of turns and k>=32 only 4%. The measured history size does not grow past about 16 turns, and the history assistant-turn cap (150) is never binding (max 136 total messages). The user's own streams14 (96k/42k drain) matched control (46.38 vs 46.25), so the surface around the 42-58k drain is flat. Plain oldest-first pop(0) (tool_agent.py:8564-8617) already drops previous levels' turns first. rellik13's 14.49->22.53 (WRITEUP.md:8, 90-105) was 37k->27k to 57k->45k trims plus FP8 KV, and Franzen's 118k->59k already exceeds that. Realistic upside is about +0.5 offline for 2-3 days of work on the cache-sensitive trimmer. It also overlaps the cheaper reshow-level-start-board idea.",
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 2.5,
   "confidence": "medium",
   "first_step": "No action; fold any benefit into reshow-level-start-board."
  },
  {
   "id": "drain-depth-retune",
   "keep": false,
   "reason": "Drop. It is capacity-limited. serve.log of the control run shows a KV pool of 1,011,264 tokens (line 61) and 'full token usage' of p50 0.69, p90 0.86, p99 0.95, max 0.99 over 4,703 decode samples at the current ~59k post-trim window. Moving the post-trim window from 59k to 80k raises average per-stream context by about 12% (87k to 98k), which would put p90 around 0.96 and cause eviction of idle prefixes (Franzen's whole serving design depends on prefix reuse). Trim frequency also rises about 25%, and each re-prefill is about 36% larger. The sawtooth gate passes (+3.6 to +5.6 pp), but the neighbouring points are null: Franzen's 52k and 58k drains, and the user's streams14 (46.38 vs 46.25, +15% throughput cancelled by less context). A symmetric fewer-streams-plus-deeper-drain trade is likely to be about zero too. tong-m2-comparison/kaggle/SUMMARY.md:33 does confirm the competitor's 126k to 81,536 target, but there is no evidence it scored better. A single 2-hour run cannot resolve a sub-3-point effect.",
   "status_in_franzen": "tried-no-gain",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "medium",
   "first_step": "No action."
  },
  {
   "id": "scheduler-recalibration",
   "keep": true,
   "reason": "Keep. This is the best idea of the seven, with sim-only evidence and an env-only fix. Verified: score is min(115, (baseline/actions)^2*100) x level weight, capped at the cleared-weight fraction (tufa-arc-agi-framework/src/taaf/game.py:381-413). Franzen's notebook leaves ARC3_PRIORITY_HUMAN_ACTIONS at the default 25 (template cell 4 setup_env), so A decays to 0.09 at 60 actions, while measured efficiency is 1.05-1.12 at 10-60 actions and 83% of cleared levels have actions <= baseline (levels.csv: 742 cleared of 869; idea said 838). The env knobs are wired (tool_agent.py:2896, 3156-3166). I re-ran the user's replay simulator, and the idea's mysim5 reproduces. With 40 reps: h1000+remaining x0.5 gives +5.34 (40/40) and env h1000 base2/cap4 gives +5.60. With a fresh seed set and 30 reps, env-only h=300, TAIL_LOOKUP unset, TAIL_BASE=TAIL_CAP=4 gives +5.07 / +3.15 / +1.55 for NEVER (unclearable fraction) = 0 / 0.4 / 0.8. It beats Franzen in 26-30 of 30 reps, on a broad plateau h 200-3000 x B 3-6, so it is not knife-edge. Note that even plain round-robin is only -1.8 vs Franzen in the same sim, so the large delta comes from fixing the flat B=8 against a collapsed A. Caveats: (a) sim-only, and the user's replays are validated against absolute scores, not policy deltas; (b) the gain falls to +0.4 to +2 as unclearable share rises, and I found a regime where it flips negative (800 min-equivalent budget with NEVER=0.5, -1.9, 2/30 better), while 250-532 min gives +2.0 to +5.0; (c) the idea's 'TAIL_SCALE' code change is unnecessary. Real fresh games still go first (acquire uses the 2,000,000 band, tool_agent.py ~3000), as in the sim. Also, my sim shows nofreshfirst changes nothing. Expect +1 to +3 LB points (offline +2 to +5, times 0.56), not the claimed full +5.3.",
   "status_in_franzen": "partial",
   "impact": 3,
   "effort_days": 0.1,
   "confidence": "medium",
   "first_step": "In arc-agi-3-v3/franzen/kaggle/template.ipynb cell 4 (setup_env, USE_PRIORITY_SCHEDULING block) set ARC3_PRIORITY_HUMAN_ACTIONS='300', delete ARC3_PRIORITY_TAIL_LOOKUP, add ARC3_PRIORITY_TAIL_BASE='4' and ARC3_PRIORITY_TAIL_CAP='4', and keep SCORE_NORMALIZATION=1, REFRESH_QUEUE=1, TAIL_FADE=1, TAIL_FADE_FRACTION=0.2. Re-validate first with analysis/replay_scheduler.py at several --never values (0, 0.4, 0.8) and --minutes, then run one offline arm."
  }
 ]
}
```
