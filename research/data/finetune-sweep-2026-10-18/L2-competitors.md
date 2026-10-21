# L2: Competitor code, public notebooks and model artifacts: who trains, on what, and what the top teams likely do (2026-10-18)

Paths starting `competitors/` are under `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/` (read-only clones). "Scratch" means this session's scratchpad (`/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/ce155f4d-514a-4385-ac65-a7e7e5a23e1c/scratchpad/`), where pulled cards and pages were saved. Kaggle CLI calls used `env -u KAGGLE_API_TOKEN -u KAGGLE_CONFIG_DIR` (mellamoa), read-only. Facts I could not check are marked UNVERIFIED. INFERENCE marks my reasoning.

Corrections to the 2026-10-17 notes (`research/data/finetune-research-2026-10-17/landscape.md`, `models.md`, `research/archive/reports/leaderboard-35-45.md`):
- **Yi-Chia Chen's `threerabbits/opd-32b-v33-s200-gptq-w4a16` and the `dflash-32b-draft-*` models are not ARC artifacts.** They were uploaded 2026-06-24 for the AIMO "proof-pilot" competition: the kernel `threerabbits/submission-32b-fix4` lists `competition_sources: ["ai-mathematical-olympiad-proof-pilot"]` and those three models as `model_sources` (pulled metadata in scratch `k-threerabbits/kernel-metadata.json`). They show she can train (on-policy distillation of a 32B, plus custom DFlash drafters). They are not evidence about her ARC system.
- **Metir's "Tufa runs Qwen3.6-27B FP8" describes the Milestone-1 entry, not the 55.89 or 68.91 one.** The sentence sits in Metir's list of M1 prize winners ([Metir](https://www.metirai.com/blog/arc-agi-3-kaggle-scores-surge-harnesses-arc-prize-2026), "Tufa Labs ('The Duck'), $25K: a locally run Qwen 3.6 27B (FP8)").
- richardcsaky's `arc-expert-gptq-arc-v1` / `arc-expert-int3-g16-v1` (cited 2026-10-08) **no longer turn up** in a Kaggle model or dataset search (2026-10-18). They were deleted or made private. UNVERIFIED now.

## 1. Folder-by-folder: model, fine-tuned weights, training code or data, scores

### Takeaway
None of the 21 local competitor folders or the 27 pulled notebooks loads fine-tuned ARC weights or a LoRA adapter. Every notebook that scores 20 or more runs stock Qwen3.8-Flash-Next (dfranzen W4A16 + albucino drafter, or an NVFP4 variant). Training code appears in only three places: Tufa's dead LoRA-serving guard, Simon Ouellette's small goal-model LoRA, and DreamTeam's vendored `beam/fine_tune` library.

### Cited Findings
- **duck-harness (Tufa, M1, commit 7652836, 2026-07-01).**
  - The model is `vrfai/Qwen3.6-27B-FP8` with a 32,768-token context and `max_tokens` 2048 (`competitors/duck-harness/ARC3-Inference/configs/inference.json:3,6,69`).
  - The Kaggle notebook mounts `driessmit1/arc3-vllm-h100-wheelhouse-v3` and `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` (`competitors/duck-harness/taaf-duck-harness-kaggle-share.ipynb:220`).
  - It contains `inference/tools/vllm_runtime_lora_guard.py`, a guard for loading LoRA adapters at runtime in vLLM. It maps packed modules including the Gated-DeltaNet `in_proj_ba -> (in_proj_b, in_proj_a)` (`competitors/duck-harness/ARC3-Inference/inference/tools/vllm_runtime_lora_guard.py:6-13,91-109`). Nothing in any bundle imports `install_runtime_lora_warning_guard` (grep across `competitors/`), so it is dead code left over from Tufa's private repo.
  - M1 public score was 1.21 ([Tufa blog](https://tufalabs.ai/research/duck-harness/): "built around Qwen 3.6 27B FP8 ... mean score across all public games is 1.6002").
- **tong-m2-comparison** (Tong Hui Kang's mirror of Franzen M2) and our fork's base: Flash-Next W4A16 AutoRound + albucino MTP. Franzen: "This was inference and testing compute; I did not train or fine-tune the models" (`competitors/tong-m2-comparison/WRITEUP.md:318`).
- **rellik13-milestone2** (sirikilohit, M2 3rd, 22.53):
  - Flash-Next W4A16 + FP8 PLE.
  - Swift 1.5 "No better per token and about 35% slower to decode" (`competitors/rellik13-milestone2/WRITEUP.md:148`).
  - "LoRA on Qwen3.8-27B on Kaggle's TPU, trained on games from an open game pool ... it scored badly" (`WRITEUP.md:154`).
  - Team LB now 29.28 (scrape below).
- **kaggle-datasets/lordhansolo-taaf-kaggle-source** (M2 2nd): `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8`, vLLM e975732. No adapter (no `lora` in `setup_commands.json` or the bundle). Team LB 37.07, 96 submissions.
- **The 27 notebooks in `competitors/kaggle/`**:
  - Every one with `kernel-metadata.json` uses `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` + `dfranzen/albucino-qwen3-8-flash-next-drafter`, e.g. sujanmajhisuzan, sigeward, df-wm, dprime, kinetic, win-v2 (metadata dump in this session).
  - The others: keithtyser/RadixArk NVFP4 (duck-qwen3-8-*, sheetu12b, duck-18-1gc), boristown REAP-448E NVFP4 (arc3-b32-reap), jakobbrggen Qwen3.8-27B FP8 (taaf-model-20260815), and richardcsaky's sha-pinned "arena" runtime tarball (`arc-agi-3-milestone-2-submission`).
  - "duck-qwen3-8-tuned" means serving tuning only: "My changes are limited to model serving and performance" (notebook markdown cell 1).
  - kinetic-v8 says it runs "without external neural LoRA weights" (`competitors/kaggle/kinetic-learning-by-playing-v8/kinetic-learning-by-playing-v8.txt:4`).
  - yukailiu "champion-solution" hard-codes per-game action plans for public games (lp85, s5i5), which is worthless on hidden games.
- **ARC3-solution (StochasticGoose, Tufa/Dries Smit, preview winner):** a CNN that learns online at test time to predict which actions change the frame (`competitors/ARC3-solution/README.md`). It shows Tufa's lineage in RL and test-time learning.
- **dream-team (NVIDIA):** a frontier-API multi-agent harness plus `game_creator` (section 2). `beam/fine_tune/` is part of the vendored BeamDS library and is not used for ARC.
- **ARC-AGI-3-Training-Data** (Simon Ouellette): data generators and solvers (section 3). It also has a small LoRA "goal hypothesizer" trainer: "Train Olmo on policy tensors" (`competitors/ARC-AGI-3-Training-Data/utils/goal_numeric_training.py:1`). Author's LB 0.29, 3 submissions.
- **solve-arc-agi-3** (leejianrong): a plan-first repo on Qwen3.6-27B FP8/vLLM.
  - "Fine-tuning ... not required for the September submission ... Only run a small QLoRA pilot when one cluster is both common and plausibly trainable" (`competitors/solve-arc-agi-3/docs/adr/0012-gate-fine-tuning-on-measured-harness-failures.md`).
  - Test-time training is excluded (`docs/adr/0005-exclude-test-time-training.md`).
  - No training result.
- **occam:** "Pure algorithmic ARC-AGI-3 solver. No LLM ... 57.60% RHAE on 25 public games" (`competitors/occam/README.md`). Public games only, no LB evidence.
- **polyphony-arc-3:** "training-free ARC-AGI-3 agent on open models" (commit message, 2026-07-06), Qwen3.6-27B on vLLM.
- **arc-agi-3-prime-agent:** Claude Opus 5 xhigh, 95.24% median on the public 25 at $1,059 per run (`competitors/arc-agi-3-prime-agent/README.md`). API only.
- **leoprovorov-fable-astra:** "Nobody has shown that fine-tuning on strong agents' logs works here" (`competitors/leoprovorov-fable-astra/narrative.md:324`). Team LB 31.73.
- **sglang-\* repos** (Pennyroyal, mratsim turbo, gabrielolympie): serving only.
- **ARC-AGI-3-Agents:** the official template.
- **arc-agi-3 (sonpham) and arc-interactive:** game corpora (section 3).

### Inferences
- The public record is all serving and harness work. Whatever separates the top two is in private code or private Kaggle models; Kaggle lets a team attach its own private models to its submissions.

### Gaps
- No competitor folder holds a trained ARC policy we could evaluate.

## 2. DreamTeam game_creator and "Workspace Optimization: How to Train Your Agent"

### Takeaway
DreamTeam "trains" the workspace, not the weights. Its game creator is a frontier-LLM coding agent that writes 8-level arcengine games with a built-in solver. The repo ships 25 curated games under Apache-2.0. Making hundreds more is feasible on our own node if a local coder model can stand in for GPT-5.5; that is untested.

### Cited Findings
- **Paper:**
  - arXiv 2605.09650, submitted 2026-05-10. Authors Elad Sarafian, Gal Kaplun, Ron Banner, Daniel Soudry, Boris Ginsburg ([abs](https://arxiv.org/abs/2605.09650), `citation_author` meta).
  - Abstract: "Modern agents built on frontier language models often cannot adapt their weights. What, then, remains trainable? ... the agent's workspace". It raises the protocol-matched SOTA "from 36% to 38.4%, while using 31% fewer environment actions per game" on the 25 public games, averaged over 2 runs.
  - Roles run on Opus 4.6 and GPT-5.5 (paper's per-role cost table, [HTML](https://arxiv.org/html/2605.09650)).
- **Licence:** Apache-2.0, "Copyright (c) 2026 NVIDIA" (`competitors/dream-team/LICENSE:1-3`). Repo pushed 2026-07-15 (GitHub API).
- **Generator:**
  - "Generates standalone 64x64 arcade-game artifacts from seed words using LLM agents. Each game has 8 levels with evolving difficulty" (`competitors/dream-team/arc_agi_3/game_creator/README.md:3`).
  - Inputs are 12 BIP39 seed words (`README.md:42`). Variety comes from 15 genres (maze, sokoban, chase, ..., cellular, falling), 12 cognitive skills, roughly 30 feature tags (fog, ice, teleport, gravity, keys, ...), 15 palettes and 4 element sizes (README tables).
  - Quality checks cover structure (7 required functions including `solve_level`), relational win conditions, genre and skill diversity, and "progression consistency" (`README.md` Quality Metrics).
  - Timeout is "1800 # 30 minutes per game creation", and the summarizer is `arc3-flash` (`arc_agi_3/game_creator/config.yaml:7,18`).
  - LLM profiles are `arc3-gpt5.5` (openai gpt-5.5) and `arc3-flash` (gcp/google/gemini-3-flash-preview) (`arc_agi_3/llm_configs.yaml:5-7,51-53`); the CLI also lists opus.
- **Shipped games:** 25 games, all 8 levels, tagged click (13), keyboard (9) or keyboard_click (3). The `baseline_actions` totals run from 53 to 345 per game (metadata.json of each `environment_files_generated/*/*/`), 15,821 lines of Python in total. "every game's exported solution replays to a WIN through the SDK (`--verify`)" (`competitors/dream-team/README.md`, Generated games). The curation yield (how many generated per kept) is not stated.
- **Who:** Sarafian and Kaplun are on NVARC3 (LB #9, 43.94, 41 submissions) together with cpmpml (Jean-François Puget) and sorokin (Ivan Sorokin), the 2025 ARC-AGI-2 winners (roster from the board scrape; [markbarney 10-13](https://arc.markbarney.net/news/2026-10-13-morning-arc-3)).

### Inferences
- **Cost to generate more (INFERENCE, unmeasured):** with a frontier model it is up to 30 min per game, so about 25 to 50 GPU-equivalent games per hour across 8 to 16 parallel workers. Locally we could drive the creator with BF16 Flash-Next at TP8 on the 8xH200; its base LiveCodeBench v6 is 88.4% per UkisAI's table. The open-model yield of games that pass `--verify` and are not trivial is unknown. Pilot 20 games before counting on hundreds.
- The team best placed to train on generated games (NVARC3: generator authors plus ARC-AGI-2 synthetic-data winners) is at 43.94, not 60+. Whether NVARC3 trains at all is unknown.

### Gaps
- No public count of games generated versus curated. No measurement of an open model as the generator.

## 3. Other data sources and public ARC-AGI-3 fine-tunes

### Takeaway
The data is plentiful (about 250 MIT games, about 430 unlicensed games, about 60 unlicensed AI-written games, an Apache generator). Every public ARC-AGI-3 fine-tune we found scores at or below a stock Flash-Next copy, and most score near zero. That is negative evidence, but it comes from low-resource attempts.

### Cited Findings
- **ARC-AGI-3-Training-Data** (pushed 2026-10-08, no licence per the GitHub API):
  - "It contains 429 distinct games, together with solvers and demonstration generators ... 64×64 matrix" (`competitors/ARC-AGI-3-Training-Data/README.md:5`). It includes augmented versions of public games (ar25, ls20, sb26).
  - Episodes are JSON with per-action `phase` in {reset, explore, burst, expert} and an `optimal` label (`README.md:122-149`). They come from `master_data_generator.py --episodes 1000`.
- **arc-interactive:** 249 games per the README badge; 252 environment dirs locally. MIT, pushed 2026-10-01, 57 stars (GitHub API).
- **sonpham-org/arc-agi-3:** about 62 Claude-written games (`GAMES_PROGRESS.md`), no licence, last push 2026-05-07. Its author sonphamorg is on "Son & Mark & Ronen" (LB #5, 48.92, 81 submissions). That team serves Flash-Next with FP8 PLE (Kaggle datasets `sonphamorg/arc3-flashnext-serving-part-{a,b,c}-v1`, `arc3-flashnext-ple-fp8-v1`, 2026-08-29 and 2026-09-27). The site owner `markbarney` is also on this team (board JSON `ourTeamId: 15605182`), so treat arc.markbarney.net commentary as a competitor's.
- **Public fine-tunes and their LB** (team best, submissions; LB from the board scrape, verified against the CLI for the top 11):

| Artifact | What | LB (team) |
|---|---|---|
| `cputer/naestro-agi3-27b` = HF `star-ga/naestro-agi3-27b` | QLoRA r16 on Qwen3.6-27B, "procedurally-generated ARC-AGI-3-style simulations (the STARGA 'v3' corpus)", frame to goal plus plan; no scores on card | STARGA 0.17 (11) |
| `mariogemoll/arc-prize-2026-arc-agi-3-vlm` | Qwen3.6-27B "with ARC-AGI-3 perception LoRA merged in", synthetic perception data | 0.43 (8) |
| `zerabytex/grid-world-qwen2.5-7b-grpo` | GRPO Qwen2.5-7B "2D Grid World" | 0.31 (4) |
| `mikedan7/qwen2-vl-2b-arc3-schema-qlora-bf16` | QLoRA on public schema traces; card: "four-game ID holdout ... completed zero levels and scored 0.0" | 28.48 (36), almost certainly from a different notebook |
| HF `ericmao/arc-agi3-qwen36-{sft,sft-v2,sft-v3,grpo,human100,raw-control,spatial-continuation-100}` (2026-07-21) | LoRA SFT and GRPO on Qwen3.6-27B, empty cards | not found on LB |
| `huikang/qwen-38-flash-next-finetune` (v15, "Merged run 09-29-21-15", 83,987,919,497 B, 2026-09-30) | the only public Flash-Next ARC fine-tune; no description | Tong Hui Kang 32.48 (106) |
| `ukisai/Swift-1.5-Qwen3.8-Flash-Next-*` (mirrors `phuongncn/arc3-qwen38-swift-w4a16-autoround`, `lordhansolo/swift-1-5-...-nvfp4`) | general "efficient-thinking" RL fine-tune (section 4) | Fususu team 35.62 (196); felixwang1m 26.0 (26) |

  Sources: Kaggle `models get` cards (scratch `models/*/model-metadata.json`), HF API, `kaggle models instances list huikang/qwen-38-flash-next-finetune --csv`, board JSON.

### Inferences
- Nobody has shown a public fine-tune beating the base. The Flash-Next-based attempts (Tong Hui Kang, Swift users) land inside the base model's best-of-N range (26 to 38). The 27B and small-model attempts score about 0.
- Whether Tong Hui Kang's best submission used his fine-tune is unknown.

### Gaps
- Tong Hui Kang's fine-tune method and data are not public; Kaggle won't list the instance files (404). STARGA's v3 corpus isn't public.

## 4. Public custom model artifacts used in notebooks

### Takeaway
Apart from Swift 1.5 and Tong Hui Kang's merge, the Flash-Next artifacts in public notebooks are re-quantizations, prunings or mirrors. None is a weight-trained ARC policy.

### Cited Findings
- Re-quantizations, prunings and mirrors:
  - `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (Intel AutoRound mirror).
  - `dfranzen/albucino-qwen3-8-flash-next-drafter`, plus the new `dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple`.
  - `keithtyser/qwen3-8-flash-next-nvfp4` ("Pinned RadixArk ModelOpt NVFP4"), `lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8`, `boristown/qwen3-8-flash-next-nvfp4-reap-448e` ("REAP-pruned ... 512->448 routed experts. For pushing Kaggle serving context to 128K"), and `nvidia/qwen3-8-flash-next-nvfp4`.
  - Source: `kaggle models list --search flash-next`, saved at `~/.claude/projects/.../tool-results/bvcd1ce0j.txt`.
- **Swift 1.5 Qwen3.8-Flash-Next** (UkisAI, created 2026-09-22, gated, `base_model_relation: finetune`, dataset `ukisai/Qwen3.8-27B-multi-turn-agent-sft`; HF API):
  - "63.4% fewer thinking tokens, with a 1.8x speed up while keeping the accuracy loss <1%" (card, scratch `hf/swift15-fn.md:38`).
  - Method: "figuring out which tokens were linked to pathological overthinking and penalizing them without 'attacking' the reasoning length directly then regained the accuracy with RL and OPD" (`:54`).
  - GPQA-D 89.80 vs 89.60, mean thinking tokens 17,683 vs 7,823.
  - On the one agentic benchmark, Terminal-Bench 2.1, mean output tokens rose 40,591 to 45,428 (+11.9%) while the median fell 17.9% (`:118`).
  - Licence: Swift Open License v1.0 plus Qwen; commercial use is limited above a "gross revenue of one million United States dollars" threshold (scratch `hf/swift-license.txt:87-88`).
- **richardcsaky (artificialagencylab.com, #10, 41.74, 56 submissions):** private "arena" runtime tarballs, e.g. `arc-arena-hero-v2-blocks2-p37-int3-runtime`, "Private benchmark runtime ... 20 public games, 300 s per trial" (`kaggle datasets files` and metadata). The name suggests int3 expert quantization (INFERENCE). Teammate cmechevalier hosts `face-of-agi-qwen36-35b-fp8-weights`.
- `davidaylward/whittle-next-27b-a3b`: a 27B-total / 3B-active qwen4_exp body "distilled from Qwen3.8-27B"; "not a finished, general-purpose model" (card). Not an ARC model.

### Inferences
- Swift's token savings come from short-answer benchmarks. In the one multi-turn agentic setting its mean tokens rose, which matches rellik13's in-harness null result. A general "think less" fine-tune does not transfer to our harness; an on-distribution one (R3) might.

### Gaps
- Swift in our harness at matched wall-clock: still unrun (`arms.md:674`).

## 5. Tufa Labs (#1), Yi-Chia Chen (#2), #3-#10: what is public

### Takeaway
Tufa is the only top team with a primary-source statement about training. On 2026-07-01 they described an end-to-end RL pipeline with reward shaping on self-generated games, and they own 48 B200 GPUs (144 B300 expected). Yi-Chia Chen has trained models before (on-policy distillation, GRPO) but has published nothing about ARC. Teams #3 to #10 have no public method.

### Cited Findings
- **Tufa's RL pipeline** (MLST transcript, "ARCAGI3 Goal Acquisition, Human Priors, and Why Games Are Harder Than Grids", July 1, 2026, [PDF](https://app.rescript.info/api/public/sessions/23f8cebdc5ce8f5f/pdf); local copy scratch `dl-rescript/mlst.txt`):
  - Dries Smit, 00:54:29: "the most naive implementation is just use level transitions and then the ARC-AGI score for that specific level transition ... per level ... add various other types of rewards".
  - Dries Smit, 00:54:54 (the transcript writes "oral trading pipeline", which reads as "RL training pipeline", `mlst.txt:1597`): "we do reward shaping. We have 25 games. We actually have a lot more games that we've generated ourselves and we can for example train on them to ... make the agent better for example exploration, finding goals, achieving those goals ... through end to end RL on just a bunch of games ... you need to train over ... 100,000, 200,000 tokens, which is extremely difficult ... adding reward shaping, training on shorter sequences, trying to generalize to longer sequences."
  - Stefano Viel, 01:12:42: "starting from language and then fine tuning to the specific games, you can get there much faster." Viel, 01:08:36: "training is also hard".
  - Jeroen Cottaar, 01:21:46: "What you need to get into the model is the basic thinking patterns ... without having memorized every game".
- **Tufa compute and priorities:** "Current: 6x 8 DGX B200 ... 48 B200", "Expected summer 2026: 2x NVL72 GB300 ... 144 B300". Research priorities: "RLVR, GRPO, SDPO", secondary "Investigate ARC v3" ([tufalabs.ai](https://tufalabs.ai/), Compute table). Other Tufa posts: Orbit Wars self-play RL win, and an external ICML-2026-workshop paper on on-policy self-distillation from world feedback ([arXiv 2605.30070](https://arxiv.org/abs/2605.30070), 2026-05-28, authors He, Sieber, Saponati) ([tufalabs.ai/research](https://tufalabs.ai/research/)).
- **Tufa LB:** 68.91 on 2026-10-17 (Kaggle CLI leaderboard). Jump 56.52 to 68.91 seen 2026-10-13T07:10Z. 166 submissions, 6 members (dlorah, driessmit1, infinitecreativity, jeroencottaar, pressman1, stefano1283) (board scrape `https://arc.markbarney.net/api/kaggle/arc-prize-2026-arc-agi-3/board`, fetched 2026-10-18T09:24Z; `events` list).
- **Tufa's public Kaggle artifacts:** none newer than the M1 inputs. `driessmit1` datasets `vrfai-qwen3-6-27b-fp8-hf-snapshot` (29.05 GB) and `arc3-vllm-h100-wheelhouse-v3` both show lastUpdated 2026-10-05, the day after Tufa reached 55.89. These are the M1 notebook's inputs, re-versioned.
- **Tufa's 10-09 post:** per a secondary source, Tufa posted on 2026-10-09 about a "temporary setback" and staying "dedicated to crushing competitors with overwhelming resources and compute" ([markbarney competitor page](https://arc.markbarney.net/news/competitors/arc-3-15486995)). The X post itself was not read; UNVERIFIED wording.
- **Yi-Chia Chen (threerabbits):**
  - Solo, 34 submissions. Steady climbs: 48.59, 55.77 (10-08), 59.17 (10-09), 62.96 (10-10), 67.49 (10-12) (board `events`).
  - Past public training work: `crystal-math-gpt-oss-20b-grpo` (2026-02-09) and the AIMO OPD-32B and DFlash models (2026-06-24) (`kaggle kernels list --user threerabbits`; `models instances list`). No ARC artifact is public.
- **#3 to #10** (board JSON, 2026-10-18): mtg (michaeltgao, solo, 72 subs) 50.37; the last dance (gklambauer, lukasaichberger, fses91, dwellement0baser; 85) 49.85; Son & Mark & Ronen 48.92 (Flash-Next); 復活の混テキスト (8 members; 88) 45.11; Grigoriy Zatravkin (70) 45.02; Majkel1337 (21) 44.84; NVARC3 (41) 43.94; artificialagencylab.com (56) 41.74; ARC Slayers (keithtyser; 121) 41.19; Daniel Franzen (98) 38.85. None of these members has a public ARC-3 model on Kaggle (owner listings for all members, scratch `web/top-users-artifacts.txt`).

### Inferences
- **Best-of-N inflates top scores.** Kaggle's public LB shows the best submission. Our own same-notebook draws have SD 5.0 (38.22, 28.73, 29.54, 26.90). The expected maximum of N independent draws is mean + 2.09 SD at N=34 and mean + 2.68 SD at N=166 (simulated, 20k trials). If top-team noise is like ours, a fixed system scoring 68.91 at best of 166 would have a mean near 56 to 60. For Chen at N=34 the mean would be near 57 to 60. Upper bound on the inflation, since submissions are different systems: so "consistent 60" may be above even the leaders' typical draw.
- **Tufa's model is unknown.** Two readings fit. (a) Still the Qwen3.6-27B family plus RL or LoRA weights kept private, supported by the vLLM LoRA guard with Qwen3.5/3.6-style `in_proj_ba` and the July RL statement. (b) A switch to Flash-Next like everyone else. Tension: our Qwen3.8-27B scored 14.51 offline against 54.69 for Flash-Next (`arms.md:101`). If Tufa is on a 27B, training or harness work must close an enormous gap.
- The steady +4 steps of a solo GRPO and OPD practitioner fit iterated training or harness rounds equally well. Unknown.

### Gaps
- Tufa's and Chen's current base model, whether they fine-tuned, and their harness are all undisclosed. Tufa promises release after 11-02 (prior notes).

## 6. Bottom line: does fine-tuning separate 60+ from about 35 to 50?

### Takeaway
There is no direct evidence; it is plausible. The only primary-source signal is Tufa's July description of RL on self-generated games, backed by over 20 times our compute. Everything verifiable in the 35 to 50 band is serving, harness and best-of-N. Every public fine-tune lands at or below base.

### Cited Findings
- For training as the separator:
  - Tufa's RL pipeline quotes (section 5).
  - Tufa's LoRA-serving guard (section 1).
  - Tufa's 48 B200 / 144 B300 ([tufalabs.ai](https://tufalabs.ai/)).
  - Chen's training track record (section 5).
  - The 18-point gap between #2 (67.49) and #3 (50.37) on 2026-10-17 (Kaggle CLI). That gap is too large for best-of-N alone, which is about 2.7 SD, or about 13 points.
- Against, or neutral:
  - Franzen's M2 win without training (`competitors/tong-m2-comparison/WRITEUP.md:318`).
  - Public fine-tunes at 0.17 to 35.62 (section 3).
  - Swift's agentic token *increase* on Terminal-Bench (section 4).
  - Rellik13's failed 27B LoRA (`WRITEUP.md:154`).
  - NVARC3, with the strongest synthetic-data pedigree, at 43.94.

### Inferences
- The likely recipe for 65+ is a strong harness plus heavy compute for rollouts, RL on generated games, and many submissions. That is about 25 times our training compute and 5 times our submission count. Tufa itself says it is hard.

## Implications for our plan

1. **Do: build the generated-game pipeline first.** It is the one ingredient Tufa named, and it also serves as our leave-family-out validation set. Run DreamTeam's `game_creator` (Apache-2.0) with a local coder model on the Azure node. Test first: 20 games with BF16 Flash-Next, then count `--verify` passes and non-trivial ones (our model clears some but not all levels). Do not plan on hundreds until that yield is known. arc-interactive (MIT) is the second source. Training-Data and sonpham (no licence) are for validation only.
2. **Do: copy Tufa's stated RL tricks into any RL stage:**
   - per-level reward equal to the ARC level score at the level transition;
   - reward shaping;
   - training on shorter windows and generalizing to long ones.
   This matches our 8 to 16k decision windows. Tufa trains at 100 to 200k tokens per game; we cannot, so a short-window design is required, not optional.
3. **Don't: expect an off-the-shelf "think less" fine-tune to work.** Swift showed mean tokens +11.9% on agentic Terminal-Bench, rellik13 found it no better and 35% slower, and its users score 26 to 35.6. R3, shortest-correct on our *own* harness decisions, is the on-distribution version. Gate it on tokens per cleared level at matched wall-clock, not on token count. Swift's recipe ("penalize overthinking tokens, regain accuracy with RL + OPD") is a useful design reference for R3. If spare quota exists, a 20-min Swift W4A16 smoke stays a cheap negative control.
4. **Do: reframe the target.** The public LB is best-of-N. At our SD of about 5, a system averaging about 55 to 58 would post 60 to 65 bests over 15 daily submissions. Report the *mean* of repeated draws, as the user's rule already requires. Spend the 15 remaining submissions on the best candidate's repeats, not one-offs.
5. **Test-first: model choice.** No public evidence shows any 27B reaching 30 on the LB, and Tufa's base is unknown. Keep Flash-Next (only base with 30+ LB evidence). Re-check if Tufa's code appears after 11-02.
6. **Don't: count on competitor artifacts.** No public trained ARC policy exists to distil from. The richardcsaky overlays are gone. Tong Hui Kang's Flash-Next merge has no description and scores within base noise.

## Fact-check (independent verifier, 2026-10-18)

How I checked: I did not use the browser. I downloaded each primary source again into scratch `fc-L2-verifier/`: the MLST PDF (HTTP 200, the same 251,910 bytes), the tufalabs.ai home and research pages, the board JSON (fetched again 2026-10-18T09:54:36Z), the HF API and README for `ukisai/Swift1.5-Qwen3.8-Flash-Next`, and the arXiv abs pages for 2605.09650 and 2605.30070. I also ran the Kaggle CLI read-only as mellamoa (leaderboard, kernel metadata, model cards, instance versions, dataset lists). I read the local repo claims at the cited lines.

### Claims

1. **Tufa described an end-to-end RL pipeline on self-generated games, with per-level ARC-score rewards, reward shaping and training on shorter sequences.** CONFIRMED. The quotes at 00:54:29 (p. 34) and 00:54:54 (p. 35) are verbatim. Context the plan should keep:
   - Smit gives the per-level ARC score as "the most naive implementation".
   - Later in the same answer he lists the rewards Tufa used: "level rewards, ARC-AGI scores, whether the code gets executed or not, length of the reasoning steps". The claim leaves out the code-execution and reasoning-length terms.
   - The generated-games part is hedged: "we can for example train on them".
   - On 2026-07-01 Tufa's M1 submission ran stock `vrfai/Qwen3.6-27B-FP8`. The transcript describes a research pipeline. It does not say that a submitted model was RL-trained.
2. **48 B200 (6x DGX B200) and 144 B300 (2x NVL72 GB300) expected by summer 2026.** CONFIRMED. The tufalabs.ai Compute table reads "Current: 6x 8 DGX B200 ... 48 B200 (8 per cluster)" and "Expected summer 2026: 2x NVL72 GB300 ... 144 B300 (72 per cluster)", plus "Extra rented capacity: On-demand external clusters". The page still says "Expected" on 2026-10-18 and does not say whether the B300s arrived.
3. **Leaderboard scores (2026-10-17) and submission counts.** CONFIRMED.
   - The CLI (rerun 2026-10-18) gives exactly 68.91 / 67.49 / 50.37 / 49.85 / 48.92 / 43.94, all with submissionDate 2026-10-17.
   - The board JSON gives Tufa 166 submissions and 6 members, and Chen 34 submissions, solo (`threerabbits`).
   - The board's `ourTeamId` 15605182 is Son & Mark & Ronen (LB #5), so the site is run by a competitor, as the notes say.
4. **Chen's `opd-32b-v33-s200-gptq-w4a16` and the `dflash-32b` drafters were built for AIMO proof-pilot.** CONFIRMED.
   - Pulled again, `threerabbits/submission-32b-fix4` lists `competition_sources: ["ai-mathematical-olympiad-proof-pilot"]` and those three models as `model_sources`.
   - All three are v1 "Initial release" builds created 2026-06-24 (16:23, 17:29 and 20:23 UTC). That competition's deadline was 2026-06-25 10:00 (`kaggle competitions list -s proof-pilot`).
   - "Built for AIMO" is an inference from this, but a strong one.
5. **Swift 1.5 GPQA-D and Terminal-Bench 2.1 numbers.** CONFIRMED.
   - GPQA-Diamond at xhigh: 89.80% vs 89.60%, mean thinking tokens 17,683 vs 7,823 (55.8% fewer by mean, 63.4% fewer by median).
   - Terminal-Bench 2.1: accuracy 67.64% vs 69.66%, mean tokens 40,591 vs 45,428 (+11.9%), median 17.9% lower.
   - Caveats:
     - The Terminal-Bench column counts *total generated output tokens*, not thinking tokens.
     - Swift's Terminal-Bench accuracy rose by 2 points.
     - A footnote on the card says the Flash-Next Terminal-Bench runs used default task timeouts.
     - A higher mean with a lower median fits a few long successful episodes as well as it fits "doesn't carry over to agentic use". The why-it-matters inference is therefore weaker than stated.
   - The repo id without the hyphen, `Swift1.5-...`, is the right one. `ukisai/Swift-1.5-Qwen3.8-Flash-Next` does not exist (401).
6. **Every public ARC-AGI-3 fine-tune scores at or below base Flash-Next copies.** PARTIALLY CONFIRMED. All five LB numbers and counts match the board: STARGA 0.17 (11), Mario Gemoll 0.43 (8), Zerabyte_X 0.31 (4), Tong Hui Kang 32.48 (106), and "Matija L & Zhongwei W & Fususu" 35.62 (196). Corrections:
   - (a) These are team-best scores. No source shows that a team's best submission used its fine-tune. The notes already concede this for Tong Hui Kang and mikedan7.
   - (b) Swift-1.5 is not an ARC-AGI-3 fine-tune. It is UkisAI's general efficient-thinking fine-tune (dataset `ukisai/Qwen3.8-27B-multi-turn-agent-sft`), and phuongncn's card calls itself a "verbatim copy" of UkisAI's W4A16 repo.
   - (c) Two of the near-zero bests are from before Flash-Next was in use: Zerabyte_X on 2026-06-04 and Mario Gemoll on 2026-07-03. Tufa's own base-27B M1 harness averaged only 1.6 offline at that time. Comparing these scores with Flash-Next copies mixes eras.
   - (d) STARGA's adapter is only the fallback inside a symbolic "mechanic-class cascade" (card), so 0.17 scores that whole agent, not the LoRA.
   - (e) Tong Hui Kang's model does carry a description; see other error 3.
   - What the evidence supports is "no public evidence that a fine-tune beat the base". It does not give measured scores for the fine-tunes.
7. **DreamTeam game creator: 30-minute budget, 8-level games with `solve_level`, 25 curated Apache-2.0 games that replay to a WIN.** CONFIRMED.
   - `config.yaml:7` reads `timeout: 1800 # 30 minutes per game creation`, with `depth: 75`.
   - `game_creator/README.md:3` says "8 levels". `README.md:229` names "all 7 required functions (including `solve_level`)".
   - `llm_configs.yaml:5-7` defines `arc3-gpt5.5` and `:51-53` defines `arc3-flash` (gemini-3-flash-preview). The Quick Start calls gpt5.5 the "default high-capacity model".
   - LICENSE: Apache 2.0, "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES".
   - There are 25 game dirs, each with 8 `baseline_actions` entries.
   - The replay-to-WIN claim is as the README states it. I did not rerun `--verify`.
8. **Franzen's M2 solution used no training or fine-tuning.** CONFIRMED. `WRITEUP.md:318` is quoted verbatim, and the same line is in our fork's `WRITEUP.md`. Scope: the statement covers development up to the M2 write-up, whose notebook scored 27.89. Franzen's team's current 38.85 (2026-10-17) is later and outside the statement. Our own untrained 38.22 supports the 28-39 band independently.

### Other errors in these notes

1. §1, line 20: "M1 public score was 1.21" cites the Tufa blog, but the blog says only "mean score across all public games is 1.6002 +/- 0.4475" (25 public games, 20 tries each, offline). The number 1.21 is not on that page. Its source is unverified.
2. §5, line 153: "Tufa's public Kaggle artifacts: none newer than the M1 inputs" is wrong.
   - `kaggle datasets list --user driessmit1` also lists `driessmit1/qwen3-8-27b-fp8-hf-017b9c7a` ("Qwen3.8-27B-FP8 HF 017b9c7a", 25.3 GB, lastUpdated 2026-08-18, 712 downloads).
   - That upload is after M1 and suggests Tufa moved at least to Qwen3.8-27B. It bears on readings (a) and (b) on line 162.
   - Separately, "the day after Tufa reached 55.89" can't be checked: the board history starts at 2026-10-04T18:30:56Z with Tufa already at 55.89.
3. §3 lines 102 and 112, and Implications 6: Tong Hui Kang's fine-tune does have a description.
   - The instance metadata (`kaggle models instances get huikang/qwen-38-flash-next-finetune/transformers/default`) says "LoRA checkpoint 09-29-21-15, merged into its pinned NVFP4 base".
   - The base is `RadixArk/Qwen3.8-Flash-Next-NVFP4` at revision 7b719225. It is meant to be mounted with keithtyser's NVFP4 model for the PLE, and it refers to a `training_run.json`.
   - So the method (a LoRA merged into NVFP4 Flash-Next) is public. The data and the contents of `training_run.json` are not, and the file listing still returns 404.
4. §2, line 72: the 25 generated games are tagged click 14, keyboard 8 and keyboard_click 3 (each `metadata.json` `tags`), not 13 / 9 / 3. The 53 to 345 action range, the 8 levels each and the 15,821 lines of Python are correct.
5. §6, line 179: 67.49 minus 50.37 is 17.12, so the gap is about 17 points, not 18.
6. §5, line 158: the list is headed "#3 to #10" but includes ARC Slayers (#11) and Daniel Franzen (#16).
7. Implications 2, line 197: "Tufa trains at 100 to 200k tokens per game" misreads the transcript.
   - Smit: "you need to train over ... 100,000, 200,000 tokens, which is extremely difficult to do. So we try and improve this by ... training on shorter sequences, trying to generalize to longer sequences."
   - Tufa itself tries to avoid training at full length. The short-window design copies their approach; a compute gap is not the only reason for it.
8. §4, line 125: "gated". The README front matter says `gated: true`, but the HF API returns `gated: false` for `ukisai/Swift1.5-Qwen3.8-Flash-Next`, and the README downloads without auth. This is minor. The card's headline "63.4% fewer thinking tokens" is the *median* cut at xhigh; the mean cut is 55.8%.
