# prefill-v2: sizing the prompt tokens the server re-processes

2026-10-08. Branch `arm/prefill-v2`. Sources: request logs of `fz-ledger-on-trim-r2`,
`fz-arms-v4-control` and `v4-stack-v1`; sim2 (from `arm/sim-v2`, extended here); issues #11,
#17, #19, #21, #22; `failure-taxonomy.md`; `leaderboard-35-45.md`.

## Verdict

- **The yield waste is ten times smaller than reported.** The failure taxonomy said resumed
  requests carry 24.3M uncached tokens (31.7% of prefill). Almost all of that is the previous
  reply and its tool result, which the next request prefills whether or not the turn yields.
  What a yield adds is its opener: about 1,000 tokens (≈600 text + a 400-token image), 0.45M
  per run, 4.3–5.0% of prefill. A 6,144-token yield removes 0.28–0.35M per run (2.8–3.2%).
- **Every prefill lever together removes 7.4–7.9% of prefill and 11–12% of context growth**
  (yield 6,144 + opener dedup: 0.76–0.86M of 10.1–10.9M uncached tokens per run).
- **At matched wall-clock, sim2 puts every one of them within ±0.7 of the control.** Tokens
  generated rise by 0.1–0.3% (no credit for freed prefill time) to 1.8% (full credit). The
  score difference is −0.24 to +0.27 on no-ledger level costs and +0.11 to +0.67 on ledger
  costs. The run-to-run sd of a 25-game score is about 2.4, so no Kaggle run can see this.
- **The drain lever hurts and is dropped.** Keeping 80k instead of 59k after a trim means a
  third more trims, each re-prefilling more. Prefill rises 3.6M (+31%), tokens fall 1.2–5.2%,
  and the score falls 0.4–1.8. Keeping 70k costs half that. The sim cannot value a deeper
  history, so its only upside is untested.
- **Built:** `ARC3_OPENER_DEDUP` (default off), with unit tests, a loop test through the real
  `analyze()` and mutation checks. Arms: `prefill-v2`, `opener-dedup`, `yield-6k` and the
  panel smoke `panel-prefill-v2`.
- **Recommendation.** prefill-v2 is close to free and safe to stack, but it is not worth its
  own repeats: a ≤0.7-point effect needs far more than two runs to confirm. Run it inside a
  stack, or run it once as a crash and behaviour check (yield rate, actions per token)
  against `control`. The real unknown is behavioural (below), not throughput.

## 1. Log-measured sizing (`analysis/prefill_levers.py`)

The script aligns each request with the game's previous one (trims drop from the front), so it
knows exactly which messages each request adds. Fresh openers average 1,373–1,402 tokens, of
which 362 are the static block. Resumption openers average 976–1,040 tokens with one image.
A larger yield budget is replayed on the observed completions; at 2,048 the replay reproduces
every real resumption (449 / 527 / 466).

Prompt tokens removed per run (share of the run's uncached prefill / of its context growth):

| Lever | fz-ledger-on-trim-r2 | v4-stack-v1 | fz-arms-v4-control |
|---|---|---|---|
| Uncached prefill, run total | 10.12M | 10.93M | 10.52M |
| `static` (opener tail once per window) | 0.52M (5.1% / 7.5%) | 0.56M (5.2% / 7.8%) | 0.52M (4.9% / 7.5%) |
| `resume_image` | 0.16M (1.6% / 2.3%) | 0.19M (1.8% / 2.6%) | 0.18M (1.7% / 2.6%) |
| `dedup` (both) | 0.68M (6.7% / 9.8%) | 0.75M (6.9% / 10.4%) | 0.69M (6.6% / 10.1%) |
| yield 4,096 | 0.20M (2.0%), 246 resumes | 0.24M (2.2%), 288 | 0.21M (1.9%), 256 |
| yield 6,144 | 0.28M (2.8%), 165 resumes | 0.35M (3.2%), 189 | 0.31M (2.9%), 154 |
| yield 12,288 | 0.38M (3.7%), 70 resumes | 0.47M (4.3%), 72 | 0.39M (3.7%), 72 |
| yield 120 s (tokens off) | 0.31M (3.1%), 133 resumes | no entry times | no entry times |
| yield 6,144 + dedup | 0.76M (7.5% / 11.0%) | 0.86M (7.9% / 11.9%) | 0.77M (7.4% / 11.2%) |

- Removing a resumption saves its opener. If the reply before it had no tool call, the
  no-tool-call nudge is sent instead; only 31 of 2,245 r2 replies were bare.
- The time-based yield counts from the invocation start, and gate waits after a trim count too.
  The replay uses the server entry times in `calibration/requests.csv`, which include those
  waits.
- Issue #17's 490–535 tokens for the static block assumed 4 characters per token. The
  Flash-Next tokenizer gives 362.
- Not sized here: #19 (old images, about 13% of context) and #22 (shared-prefix Mamba
  eviction). Dropping old images rewrites history, and Scott Le Grand stripping them all cost
  29% of actions. #22 is an engine change.

## 2. Simulator sizing (`analysis/prefill_sizing.py`)

The sweep uses sim2 with three extensions, each in its own commit and tested:
- `--lever` applies the per-request table above to the calibrated request shapes.
- The trim level now follows `budget − min(drain, budget/2)`. The old scaling made a drain to
  80k behave like a drain to 36k; the default and the calibration runs move by under 1%.
- The client gap is fitted separately for fresh turns and resumptions. Pooled, the turn-start
  term is mostly fresh turns, whose gap includes the action's env step (r2 means: 5.05 s fresh,
  2.62 s resumption, 2.08 s mid-invocation). The pooled model made removing a resumption look
  about four times too valuable: yield 6,144 came out at +0.09 / +0.22 against −0.06 / +0.09.

The setup: 25 games, 10 slots, 121 min, the Franzen scheduler, and paired replicates on the
same seeds as the control. sim2's decode curve already includes the stalls today's prefill
causes, so each setting runs twice:
- **no credit**: the curve as calibrated, the lower bound;
- **full credit**: the curve sped up by the prefill seconds saved over the decode time
  (1 + ΔP/(T − P), at 11.6k prefill tok/s). This assumes a prefill chunk stalls every
  decoding stream, the upper bound.

**No-ledger level costs** (the template control has no ledger), 1,000 paired reps. Control:
prefill 11.77M (1,015 s), 4.507M tokens, score 45.68. Differences, with se in brackets:

| Setting | Prefill | Trims | Switch-outs | Tokens (none / full) | tok/s (none / full) | Levels (none / full) | Score, no credit | Score, full credit |
|---|---|---|---|---|---|---|---|---|
| static | −0.94M | −8.1 | −3.4 | +4k / +54k | +0.6 / +7.4 | −0.12 / +0.37 | −0.15 (0.05) | +0.16 (0.05) |
| image | −0.28M | −2.4 | −1.0 | +2k / +16k | +0.2 / +2.3 | −0.01 / +0.14 | +0.00 (0.04) | +0.09 (0.04) |
| dedup | −1.25M | −10.8 | −4.5 | +5k / +70k | +0.7 / +9.7 | −0.19 / +0.46 | −0.24 (0.05) | +0.18 (0.05) |
| yield 4,096 | −0.34M | −2.9 | −1.3 | +5k / +23k | +0.7 / +3.2 | +0.04 / +0.23 | +0.02 (0.04) | +0.15 (0.04) |
| yield 6,144 | −0.48M | −4.0 | −1.5 | +8k / +33k | +1.1 / +4.5 | −0.05 / +0.19 | −0.06 (0.04) | +0.09 (0.04) |
| yield 8,192 | −0.55M | −4.6 | −1.9 | +9k / +37k | +1.2 / +5.1 | −0.01 / +0.26 | −0.03 (0.04) | +0.14 (0.04) |
| yield 12,288 | −0.62M | −5.2 | −2.2 | +10k / +42k | +1.4 / +5.8 | −0.01 / +0.36 | −0.04 (0.04) | +0.19 (0.04) |
| yield 120 s | −0.54M | −4.6 | −1.9 | +8k / +37k | +1.2 / +5.0 | +0.01 / +0.27 | −0.02 (0.04) | +0.14 (0.04) |
| yield 150 s | −0.59M | −5.0 | −1.9 | +10k / +40k | +1.3 / +5.6 | −0.06 / +0.28 | −0.07 (0.04) | +0.15 (0.04) |
| **yield 6,144 + dedup** | −1.36M | −11.8 | −4.8 | +11k / +81k | +1.5 / +11.2 | −0.11 / +0.60 | −0.18 (0.05) | +0.27 (0.05) |
| yield 120 s + dedup | −1.38M | −11.9 | −4.8 | +11k / +83k | +1.5 / +11.4 | −0.16 / +0.57 | −0.22 (0.05) | +0.25 (0.05) |
| drain to 70k | +1.49M | +12.3 | +4.7 | −26k / −103k | −3.5 / −14.1 | −0.30 / −1.14 | −0.21 (0.05) | −0.73 (0.05) |
| drain to 80k | +3.59M | +29.7 | +11.2 | −53k / −236k | −7.2 / −32.4 | −0.60 / −2.82 | −0.43 (0.05) | −1.76 (0.05) |
| drain to 80k + dedup | +1.83M | +14.4 | +4.9 | −42k / −136k | −5.8 / −18.7 | −0.50 / −1.53 | −0.36 (0.04) | −1.00 (0.05) |

**Ledger level costs**, 600 paired reps. Control score 51.19:

| Setting | Tokens (none / full) | Levels (none / full) | Score, no credit | Score, full credit |
|---|---|---|---|---|
| dedup | +7k / +70k | +0.23 / +0.98 | +0.11 (0.07) | +0.55 (0.07) |
| yield 6,144 | +9k / +32k | +0.11 / +0.45 | +0.03 (0.06) | +0.26 (0.06) |
| yield 120 s | +10k / +37k | +0.14 / +0.53 | +0.06 (0.06) | +0.31 (0.07) |
| **yield 6,144 + dedup** | +13k / +82k | +0.29 / +1.20 | +0.13 (0.08) | +0.67 (0.08) |
| drain to 80k | −52k / −229k | −0.41 / −2.89 | −0.17 (0.07) | −1.68 (0.07) |

How to read this:
- **Throughput.** Without credit, the prefill levers add only about 0.1–0.3% tokens: they
  shorten each game's own prefill wait, and that wait is small. The real gain is GPU time freed
  for decoding, between the two columns. If prefill chunks run apart from decode batches (SGLang's
  default without mixed chunking), it is nearer the full-credit column: up to +1.8% tokens, or +11 tok/s on
  about 620.
- **Why fewer trims can cost score.** A trim is the only point where a game hands its slot over,
  and dedup removes 11 of 89 trims and 4.5 of 45 switch-outs. On no-ledger costs, slower
  rotation costs about as much as the throughput gains. On ledger costs it does not. sim2 is
  not trusted for compute reallocation (its README), so treat the sign of the no-credit column
  as unknown and the effect as about zero.
- **The drain lever loses in every column.** It is the one lever the sim says hurts, so it is
  dropped. Competitors keep more (Lord Han Solo trims 126k → 81.5k, rellik13 57k → 45k), but
  they run other harnesses. What deeper history is worth to the model is not in the sim.
  Longer contexts also put more pressure on the KV cache (#22), which the sim does not model.

## 3. Behaviour risks the sim cannot see

- **Yield.** Franzen chose the 2,048-token yield on purpose. WRITEUP §6 lists "Resume prompts and
  yielding" among the changes that did not clearly improve the score. He kept the ordinary
  resume prompt and moved the yield trigger from seconds to 2,048 generated tokens, so that
  setting is his configuration. A 6,144 budget (or a 120 s yield) reverses that choice.
  - The resume opener may work as a decisiveness nudge ("call `action(actions)` with your best
    next action").
  - Against that, resumed invocations act only 50% of the time against 77% for fresh turns, and
    their first reply is longer (failure taxonomy).
  - Check in a run: actions per generated token, and reply length after the 2,048th token of a
    turn.
- **Static block once per window.** The block repeats the system prompt word for word or nearly
  so (tool list, frame semantics, `segmentation` as the primary view, the tool-call format, the
  MOUSE arguments). It still appears once in every window an invocation starts from.
  - Within a turn, a mid-turn trim can drop the carrier. The block then returns with the next
    opener; the system prompt still says the same.
  - A 2026-09 prompt-dedup arm on the old A3 harness was the worst run on record (5.25 against
    6.98). It removed different text, and it was a single run.
- **Resumption image.** The caption says the board is unchanged and the picture is above.
  Franzen's notes say the `short` resume mode degraded without images. Here the image is left
  out only when the identical image is the most recent one in context.

## 4. What was built (`ARC3_OPENER_DEDUP`, default off)

- **`static`.** `_build_user_prompt` records its fixed tail, the lines from "Only tool:" to the
  end, and records nothing when a world-model block sits inside it. After the opener-time trim,
  `analyze()` removes `"\n" + block` from the new opener, but only if a message the request
  retains already contains the identical block.
- **`resume_image`.** A resumption, which by construction executed nothing, leaves out its
  current-grid image when the most recent retained image-bearing message contains the
  byte-identical image. A one-line caption stands in its place.
- **Scope.** Only the new opener changes. Nothing already sent is edited, the system prompt and
  tools are untouched, and with the flag off the requests and transcript are byte-identical to
  `origin/main`.
- **Refactor (separate commit).** The `USER PROMPT` transcript line moved after the opener-time
  trim. The trim writes nothing to the transcript, so the transcript is unchanged.

Verification (`ARC3-Inference/tests/`, 62 tests, about 25 s):
- **Unit tests (27):** flag parsing; the exact text removed; carrier rules (none, a nudge, a
  different variant, a longer variant containing it); the same-board rule (a different last
  image, the same board further back, extra images, fresh turns); the control tag kept;
  nothing mutated; the builder's recorded tail (with MOUSE, across turns, the first-turn
  variant, a world-model block).
- **Loop test (35)** (`opener_dedup_loop_sim.py`, the pattern of `arm/reasoning-scaffold`):
  - Setup: the real `ToolAgent.analyze()` under the notebook's `setup_env`, driven as
    `solver.py` does (same analysis_step on a resumption). A fake SGLang renders each request
    through the Flash-Next chat template. A 19-reply script plays 3 levels: 6 resumptions,
    two in a row and one after a bare reply. The 128k window never trims; a 25k window forces
    trims and real gate handovers.
  - Checks:
    - every prompt extends the previous one unless history was evicted, and no earlier message
      changes;
    - the system prompt and tools are identical across all requests and configurations;
    - against flags off, the dedup removes exactly the documented text, 12 static blocks and
      6 images;
    - the static block is in context at every invocation start, in both windows;
    - board images are dropped only on a resumption whose last retained image is the same
      board;
    - flags off is byte-identical to `git archive origin/main`: messages, rendered prompts and
      transcript.
- **Mutation checks (6 mutants), each caught:**

  | Mutant | Caught by |
  |---|---|
  | No carrier check | unit tests, block-in-context, exact-dedup |
  | Rewrites history (strips old images) | prefix extension, images, exact-dedup |
  | Leaves a stray newline | unit tests, exact-dedup |
  | Drops the image whatever the board | unit tests |
  | Touches the system prompt | system check, exact-dedup |
  | Default on | unit tests, byte-identical check |

## 5. Arms (`kaggle/arms.json`)

| Arm | Env | Purpose |
|---|---|---|
| `prefill-v2` | `ARC3_OPENER_DEDUP=1`, `LOCAL_ANALYZER_YIELD_TOKENS=6144` | All the levers the sim does not reject; full 25 games, 121 min; pair with `control` |
| `opener-dedup` | `ARC3_OPENER_DEDUP=1` | Code lever alone |
| `yield-6k` | `LOCAL_ANALYZER_YIELD_TOKENS=6144` | Env lever alone (reverses Franzen's 2,048) |
| `panel-prefill-v2` | as `prefill-v2`, balanced panel, 20 min, 4 streams | Smoke test |

Yield 6,144 rather than 12,288 or 120 s: every budget sizes the same within the sim's noise
(±0.05). 6,144 is the failure taxonomy's proposal and stays closest to Franzen's
behaviour. The drain arms were not added.

Reproduce (from `analysis/`):

    nice -n 15 python3 prefill_levers.py ../runs/kaggle/fz-ledger-on-trim-r2/v1 --tokenizer ... \
        --requests-csv calibration/requests.csv --table calibration/prefill_levers.csv
    nice -n 15 python3 prefill_sizing.py --reps 1000 --ledger off
    nice -n 15 python3 prefill_sizing.py --reps 600 --ledger on
