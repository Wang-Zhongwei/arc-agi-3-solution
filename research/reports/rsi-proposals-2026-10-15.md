# Putting the gains in the weights: recursive self-improvement proposals (2026-10-15)

**What the user asked:**
- "can you also draft a RSI way to retrain/improve the model thats most implicit in the weights? give me some
  proposals for how we could approach this"
- "also in the RSI system, implement version control (a simplified version of it.)"

**Short answer:**
- The harness has stopped moving the number that matters. The weights are the next lever.
- Six proposals follow, ranked by what we can verify before the 2026-11-02 deadline.
- The safest first step is **R1**, a self-distilled draft head. It is lossless, it trains on one Kaggle GPU, and it
  has a measurable target (more tokens per second at the same output).
- The real "implicit in the weights" loop is **R2 + R3**: expert iteration on our own trajectories, selected for
  fewer tokens per level. Its first blocker is plumbing: training a LoRA through the int4 experts.
- The version control is built: `game-reasoning/gamereason/rsi/` on branch `arm/rsi` (PR #177), with tests. It runs the loop
  end to end on a small model today, and the same store versions the big model's adapters later.

## 1. Why the weights, now

All numbers are from the mem-repair run (MR, 54.69, 10 wins) and the 12 earlier full runs.

| What we measured | Value | What it rules out |
|---|---|---|
| Points lost to levels never cleared | 43.4 of 100 | |
| Points lost to extra actions on cleared levels | 3.9 (median 0.65× the human baseline's actions) | Action efficiency is not the lever |
| Levels per million tokens, across 13 harness arms | 21.3–24.1, no trend | The harness arms shifted which levels clear, not how many |
| Tokens over time, every run | ~1.27M at 30 min, ~2.5M at 60 | |
| Server occupancy | Decode peaks at 12–13 running requests (~926 tok/s); MR averaged 12.3 | Throughput is near its ceiling |
| Scheduling bound | Oracle +4.9, realistic +1.1 (`archive/reports/scheduling-oracle.md`) | |
| Stalled levels per run | 24–36, holding 2.4–3.2M of ~5M tokens | |
| Generated tokens that are thinking | Nearly all; median ~2.2k per reply | |
| Draft accept length (MTP) | 2.68 | |

The model clears about 23 levels per million tokens whatever we put in its prompt. Half its tokens go to levels it
is stuck on, and almost all its output is thinking. A change in the weights can move all three numbers at once:
- fewer tokens per decision;
- fewer stalls;
- more accepted draft tokens per step.

Prompt lines can only nudge them.

## 2. Ground rules for every proposal

**Never train on the 25 public games.** They are the eval set. The store refuses any training row whose game name
starts with a never-train id (set at `init`).
- `ARC-AGI-3-Training-Data` ships augmented copies of public games (ar25, ls20, sb26 and others). Their ids go on
  the never-train list.
- That repo has no license file, so check before using it at all.

**Training environments (no public games):**
- arc-interactive: 249 games, MIT.
- DreamTeam's 25 generated games and its `game_creator`. Check the license: the file opens with an NVIDIA copyright
  line.
- `game-reasoning`'s procedural tasks.
- Games the model writes itself (R6).

**What a change is judged on:**
- The tool-level gate: paired, repeated evals on a frozen suite.
- Then the 25-game, 121-min Kaggle run, repeated, with `make significance` (the user's alpha rule).

**The model**, Qwen3.8-Flash-Next W4A16 (from the model-prong notes, checked 2026-10-04):
- 125B core parameters, ~6B active, 512 experts (top-10 + 1 shared), 48 layers.
- Only the routed experts are int4. Attention, GDN, the shared expert, the router, MTP and the PLE table are BF16.
- So a LoRA on non-expert modules merges into BF16 shards with no re-quantization.

**Compute:**
- Kaggle RTX Pro 6000 96 GB: 45 h a week on mellamoa and 30 h on snowiers.
- The M3 Pro laptop, for small models only.
- ms-swift documents LoRA and GRPO for this model on about 8 × 135 GB GPUs.
- **Renting GPUs costs money, so it needs the user's go-ahead.**

**Deadline:** 2026-11-02, 18 days away.

**Still to check:**
- Whether the Kaggle rules let a notebook use private custom weights during the competition. Prizes require
  open-sourcing.
- Any rule on training data produced by other models. All data in these proposals is the model's own, which avoids
  that question.

## 3. The proposals

### R1. Self-distilled draft head (lossless speed; the safest weights change)

**What changes in the weights:** only the MTP draft layer. The trunk stays fixed, so outputs are unchanged and every
gain is speed.

**Loop:**
1. Serve the champion and log the trunk's hidden states and tokens on our own harness transcripts. SGLang can return
   hidden states. Use transcripts from training games plus the public-run logs. Drafter training only sees the
   model's own text, not labels, but keep public-game text out anyway to be safe.
2. Train the MTP layer to predict the next tokens from those states.
3. Measure accept length offline on held-out transcripts.
4. Swap in the new draft head and run a 20-min throughput panel.

**Why it pays:**
- Accept length is 2.68 now. Reaching 3.0–3.2 would give roughly +10–15% tok/s.
- Sim-v2 sized +21% tok/s at about +2 offline games, so this is roughly +1 game. It holds on the private run too.

**RSI tie-in:** every new trunk (R2–R5) lowers draft acceptance, so each version re-distills its own drafter. The
store versions trunk adapter and drafter as one model version.

**Cost and risk:**
- No backward pass through the trunk; one layer trains on one GPU, so it fits Kaggle.
- The risk is plumbing: dumping hidden states, and loading a custom MTP layer into Franzen's SGLang build.
- Outputs are bit-for-bit the same with speculative decoding off, which makes it easy to verify.

### R2. Expert iteration on our own trajectories (the core RSI loop)

**What changes in the weights:** a LoRA on the non-expert modules. It encodes how this model explores and infers
goals in these games. The harness facts currently supply some of that from outside.

**Loop** (`gamereason/rsi/loop.py` runs this shape today on small models):
1. The champion plays training games through the real harness with a fixed token budget, 8 rollouts per game.
2. Keep trajectory segments that end in a level-up, cut at the level-up. Score them by tokens spent, against the
   group mean on that level.
3. Train a fresh LoRA on the kept segments from the champion's whole lineage. Rejected rounds' data never re-enters.
4. Gate it on held-out games: tool-level suite first, then the Kaggle run.
5. Promote or reject, then repeat with the new champion generating the next round's data.

**Why it fits this problem:** stalls are 97–120k-token levels against ~25k normally, and the model "acts half as
often per token" in them. Training on the rollouts that cleared a level quickly teaches exactly the decisions that
avoid a stall.

**Data per round:** one 2-hour session produces ~5M generated tokens and ~115 level clears. Across 8 rollouts per
game that is enough distinct successful segments for a low-rank LoRA.

**Blocker (gate M0 in arms.md):**
- Backpropagating to the LoRA through the int4 experts needs a backward pass through the AutoRound int4 matmul. No
  trainer confirms that for this checkpoint on SM120.
- A custom autograd function would work, because the experts are frozen. It dequantizes per expert and multiplies
  `grad_out @ W` with no weight gradient. It is slow, but it fits in memory with short segments and activation
  checkpointing.
- The alternative is renting 8 large GPUs for ms-swift's documented path, which needs the user's go-ahead.

**Risk:** generalization to unseen games is the open question. Naestro's QLoRA on generated games did not publish
transfer numbers. The gate on held-out games is the guard.

### R3. Shorter thinking at the same decision ("think less, act more")

**What changes in the weights:** reply length per decision, at unchanged decisions. This is the most direct lever
on levels per token, the number that has not moved.

**Loop:**
1. From R2's rollouts (or any run's logs on training games), take every decision point where the model's action was
   on a successful path.
2. For each, sample several replies at the same state. Keep the shortest reply whose action matches the successful
   action.
3. Train on those.

**How to verify it without a full run:**
- Replay frozen states, the state lab's method: the right next move out of 20 at stuck states, plus tokens per
  decision.
- A drop in tokens at an equal right-move rate is a clean win, measurable in an hour on one GPU.

**Why it's separate from R2:** R2 picks the successful paths. R3 compresses the thinking within them. They stack, and
R3 is the half with an offline measurement.

**Risk:** shorter thinking can mean worse decisions on hard states. The right-move rate at stuck states is the
guard, so sample those states more heavily.

### R4. Hindsight rationales at stalls (rationalization)

**What changes in the weights:** what to notice at a stuck state, learned from the model's own later solution.

**Loop:**
1. For a stalled level that the model eventually cleared (in that run or another), take the decisive stuck state.
2. Ask the model itself, given a hint (its own later recap and the route that worked), to write the reasoning that
   reaches the right hypothesis from the evidence available at that state.
3. Keep a rationale only if its action replays toward the solution in the engine.
4. Train on it with the hint removed.

This is STaR's rationalization step. The 25 public games can't be used, so this needs stalls on training games,
which R2's rollouts produce.

**Why:** the stall reads keep finding the same thing. The evidence was on screen and the model held a wrong
hypothesis, for example su15's "knight-like jumper with parity". A harness fact fixes one game's version of that.
Weights trained on many such corrections might fix the habit.

**Risk:**
- Hint-conditioned rationales can be post-hoc. The replay check keeps only those that act correctly.
- Hold out whole games, not states, when judging.

### R5. Distilling the harness's facts into the weights

**What changes in the weights:** the model learns to compute what the harness now tells it: moved objects,
untouched objects, its own confirmed rules.

**Loop:**
- The teacher is the champion with every memory line on.
- The student is the same model with a lean prompt (Franzen's lines only).
- On training-game states, train the student on the teacher's replies wherever the teacher's action was on a
  successful path.

**Why:**
- In the lab, facts moved decisions: moved objects 0 → 16 of 20, touching split 5 → 10, own rules 3 → 9.
- Facts in the weights can't be cut by a trim or a budget, can't be wrong the way a heuristic line is (su15's
  hit-background line), and cost no prompt tokens.

**Risk:**
- Some facts aren't computable from the frames the model sees.
- Keep the harness lines in serving either way; this only makes the model less dependent on them.
- Measure on the frozen stuck states, student without lines against teacher with them.

### R6. A self-generated curriculum (the "recursive" half, after 11-02 unless R2 works early)

**Loop:**
1. The model writes new games with the arcengine API (DreamTeam's `game_creator` is a template).
2. A verifier keeps only games that load, are solvable (a scripted or BFS solver) and come with a baseline action
   count.
3. The champion plays them. Games it solves 20–80% of the time are the learning frontier, and they feed R2's data.
4. The next champion writes harder games.

**Why:** this removes the data ceiling. 249 + 25 games run out after a few rounds.

**Risk:** generated games may drift from ARC's core-knowledge style. Keep a fixed share of human-made training games
in every round.

### Not proposed before 11-02

These are carried over from the model-prong plan:
- online multi-turn GRPO through the full harness;
- full-parameter training;
- test-time training on Kaggle;
- world-model GRPO (M3), since better transition prediction did not give better decisions in the cited studies.

## 4. The version control (built: `arm/rsi`, `game-reasoning/gamereason/rsi/`)

A simplified git for models. Every object is immutable and content-addressed (sha256). Only two things change:
pointers ("refs") and an append-only log.

**Objects:**

| Object | What it holds |
|---|---|
| blob | bytes of one file: a data file, one adapter file, an eval's episodes |
| tree | a directory: an adapter or drafter, as {path: blob} |
| suite | the frozen eval suite (tasks, envs, seeds, sampling settings). Its id defines "better" |
| dataset | a training-data file + its line-aligned meta + `generated_by` (the version that produced it) |
| version | a commit: base model, adapter tree, training datasets, training config, parent version, code git revision |
| eval | one version's episodes on the pinned suite |

**Refs:** `champion` (HEAD, the current best model) and `suite`. A version's status (candidate, promoted or rejected)
is read from the log and never written into the version.

**Commands** (`uv run python -m gamereason.rsi ...`):
- `init`: pin the suite and the never-train list.
- `baseline`: version 0, the untrained base, becomes the first champion.
- `round`: one loop round: collect with the champion, train, eval, gate.
- `status`, `log`, `lineage`, `diff A B`, `checkout`.
- `promote`, with `--force` logged as forced.
- `rollback`: by default back to the previous champion, step by step, or `--to` any earlier promoted version.

**Guards built in:**
- **Promotion gate (the user's alpha rule):**
  - paired per-seed differences against the champion, with a bootstrap 95% CI above 0;
  - every eval repeat higher;
  - at least `min_repeats` repeats;
  - no single game significantly worse.
- **No eval leakage.** Training rows must come from seeds ≥ 1,000,000 (eval seeds are below), and never from a
  never-train game. The eval suite can't be re-pinned once versions were evaluated on it.
- **No leakage from rejected rounds.** A candidate trains on the datasets in the champion's lineage plus its own new
  one, never on a rejected round's data.
- **Traceability.** Every dataset names the model that generated it, and every version names its parent, data,
  config and code revision. `lineage` and `diff` answer "what made this model?"
- **Integrity.** Objects are re-hashed on read; a corrupted file is refused.

**Mapping to Kaggle:**
- A promoted version's adapter (and drafter) is checked out, merged into the BF16 shards and uploaded as a new
  Kaggle Model version. Its store id goes into the upload's notes and into `runs.json`.
- Rolling back means pointing the notebook's `model_sources` at the previous Kaggle version, and `rollback` records
  that in the log.

**Tested** (56 tests pass, 9 of them for RSI, offline, ~11 s):
- objects, trees and integrity;
- suite freezing;
- dataset guards;
- the gate (gain, loss, too few repeats, one game significantly worse, forced);
- rollback walking back promotion by promotion;
- fake-backend rounds that promote gains, reject losses and keep rejected data out;
- a skipped round when nothing verifies;
- a real round on the tiny offline model: actual LoRA training, the adapter stored as a tree, and actual eval.

A command-line smoke run on the tiny model also showed the cold-start problem: a model that solves nothing
produces no data, so the round is skipped and logged. R2 needs a champion that clears some training levels, and
Flash-Next does.

## 5. Suggested order

1. **Now, no money needed:**
   - R1's offline half (dump hidden states from existing logs' prompts on one Kaggle session; train the draft head;
     measure accept length);
   - the M0 plumbing check for R2 (one LoRA forward/backward through the int4 experts on one 96 GB GPU, with the
     custom dequantizing backward).
2. **If M0 passes:**
   - one R2 + R3 round on arc-interactive games, at most 2 Kaggle sessions for rollouts;
   - judged on frozen states first (right moves and tokens per decision), then one 121-min run.
3. **If M0 fails:** decide with the user whether to rent GPUs for ms-swift's path, and how many hours.
4. **R4 and R5** reuse R2's rollouts and run once R2's loop works. **R6** after the deadline.

Each step is a version in the store and an arm in `arms.md`, and promotion still follows the repeat rule.
